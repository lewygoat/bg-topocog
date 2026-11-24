import os, sys, json, time, math, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
INTER = Path(os.environ.get("BGTC_DATA", REPO / "data" / "intermediate"))
RERUN = Path(os.environ.get("BGTC_RESULTS", REPO / "results"))
OUT = RERUN
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Monitor:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def log(self, stage, msg, **kw):
        rec = {"t": round(time.time() - self.t0, 2), "stage": stage, "msg": msg}
        for k, v in kw.items():
            if isinstance(v, (np.floating, np.integer)):
                v = float(v)
            elif isinstance(v, torch.Tensor):
                v = float(v.item())
            rec[k] = v
        line = json.dumps(rec, ensure_ascii=False)
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def build_prior():
    w = np.load(INTER / "s9_skill_weight.npy").astype(np.float32)
    w_web = np.load(INTER / "s9_skill_weight_1x_web.npy").astype(np.float32)
    w_rk = np.load(INTER / "s9_skill_weight_rk.npy").astype(np.float32)
    w_bd = np.load(INTER / "s9_skill_weight_1x_bigdata.npy").astype(np.float32)

    def ln(x):
        return np.log1p(x) / max(np.log1p(x.max()), 1e-8)

    return torch.tensor(np.stack([ln(w), ln(w_web), ln(w_rk), ln(w_bd)], axis=1), dtype=torch.float32)


def build_adj(b, n_c):
    rows, cols = [], []
    for key in ("prereq_predicted", "prereq_adopted"):
        ei = b[key].long()
        rows.append(ei[0]); cols.append(ei[1])
    r = torch.cat(rows); c = torch.cat(cols)
    a = torch.zeros(n_c, n_c)
    a[r, c] = 1.0
    a = a + a.t() + torch.eye(n_c)
    deg = a.sum(1).clamp(min=1e-8).pow(-0.5)
    return (a * deg.unsqueeze(1) * deg.unsqueeze(0)).to(DEVICE)


class GateGCN(nn.Module):
    def __init__(self, n_u, n_c, d, prior, mode, g_init=-2.9444):
        super().__init__()
        self.mode = mode
        self.has_prior = mode != "none"
        self.emb_c = nn.Embedding(n_c, d)
        self.emb_u = nn.Embedding(n_u, d)
        nn.init.normal_(self.emb_c.weight, std=0.1)
        nn.init.normal_(self.emb_u.weight, std=0.1)
        self.W_p = nn.Linear(prior.shape[1], d, bias=False)
        self.g = nn.Parameter(torch.tensor(float(g_init)))
        self.register_buffer("prior", prior)

    def gate_value(self):
        if self.mode == "sigmoid":
            return 2.0 * torch.sigmoid(self.g)
        if self.mode == "tanh":
            return 1.0 + torch.tanh(self.g)
        if self.mode == "clip":
            return self.g.clamp(0.0, 2.0)
        if self.mode == "unbounded":
            return self.g
        if self.mode == "none":
            return torch.zeros((), device=self.g.device)
        raise ValueError(self.mode)

    def propagate(self, adj):
        h = adj @ self.emb_c.weight
        h = adj @ h
        return h

    def forward(self, adj):
        h = self.propagate(adj)
        v = h + self.gate_value() * self.W_p(self.prior)
        return v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)

    @torch.no_grad()
    def diagnostics(self, adj):
        h = self.propagate(adj)
        prior_term = self.gate_value() * self.W_p(self.prior)
        return {
            "norm_ratio": float(prior_term.norm(dim=1).mean() / h.norm(dim=1).mean().clamp(min=1e-8)),
            "h_norm": float(h.norm(dim=1).mean()),
            "prior_term_norm": float(prior_term.norm(dim=1).mean()),
            "gate": float(self.gate_value()),
            "gate_logit": float(self.g),
        }


@torch.no_grad()
def evaluate(v, emb_u, ei, n_c, k=10):
    u = ei[0]; c = ei[1]
    users = u.unique()
    scores = v @ emb_u.weight[users].t()
    pos = torch.zeros(n_c, users.numel(), dtype=torch.bool, device=v.device)
    lut = torch.full((int(u.max()) + 1,), -1, dtype=torch.long, device=v.device)
    lut[users] = torch.arange(users.numel(), device=v.device)
    pos[c, lut[u]] = True
    kk = min(k, n_c)
    top = torch.topk(scores, kk, dim=0).indices
    hit = pos[top, torch.arange(users.numel(), device=v.device).unsqueeze(0)]
    any_hit = hit.any(0)
    hr = float(any_hit.float().mean())
    ranks = torch.where(hit, torch.arange(1, kk + 1, device=v.device).unsqueeze(1).float(), torch.full_like(hit, 1e9, dtype=torch.float))
    first = ranks.min(0).values
    ndcg = torch.where(any_hit, 1.0 / torch.log2(first.clamp(min=1) + 1), torch.zeros_like(first))
    mrr = torch.where(any_hit, 1.0 / first.clamp(min=1), torch.zeros_like(first))
    return hr, float(ndcg.mean()), float(mrr.mean()), int(users.numel())


def bpr_loss(u, i, j, v, emb_u):
    vu = emb_u(u)
    pos = (vu * v[i]).sum(-1)
    neg = (vu * v[j]).sum(-1)
    return -F.logsigmoid(pos - neg).mean()


def train_one(seed, mode, epochs, batch, lr, monitor, eval_users=None, wd=1e-6):
    torch.manual_seed(seed)
    np.random.seed(seed)
    b = torch.load(INTER / "s6_graph.pt", weights_only=False)
    n_u, n_c = int(b["n_users"]), int(b["n_concepts"])
    train = b["train_ei"].long(); valid = b["valid_ei"].long(); test = b["test_ei"].long()
    adj = build_adj(b, n_c)
    prior = build_prior().to(DEVICE)

    model = GateGCN(n_u, n_c, 64, prior, mode).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    pop = np.bincount(train[1].numpy(), minlength=n_c).astype(np.float64) + 1.0
    pop = pop / pop.sum()
    nt = train.shape[1]
    best = -1.0; best_ep = -1; best_state = None

    for ep in range(epochs):
        perm = torch.randperm(nt)
        tot = 0.0; nb = 0
        for st in range(0, nt, batch):
            sel = perm[st:st + batch]
            if sel.numel() < 2:
                continue
            u = train[0][sel].to(DEVICE)
            i = train[1][sel].to(DEVICE)
            neg = torch.randint(0, n_c, (sel.numel(),), device=DEVICE)
            v = model(adj)
            loss = bpr_loss(u, i, neg, v, model.emb_u)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        v = model(adj)
        dg = model.diagnostics(adj)
        hr, ndcg, mrr, nu = evaluate(v, model.emb_u, valid.to(DEVICE), n_c)
        monitor.log("EPOCH", f"seed={seed} mode={mode} ep={ep+1}", loss=tot / max(nb, 1),
                    val_hr=hr, val_ndcg=ndcg, val_mrr=mrr, **dg)
        if ndcg > best:
            best = ndcg; best_ep = ep + 1
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    v = model(adj)
    dg = model.diagnostics(adj)
    hr, ndcg, mrr, nu = evaluate(v, model.emb_u, test.to(DEVICE), n_c)
    res = {"seed": seed, "mode": mode, "best_epoch": best_ep,
           "test_HR@10": hr, "test_NDCG@10": ndcg, "test_MRR": mrr, "n_eval_users": nu, **dg}
    return res, v.cpu()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="sigmoid,unbounded,tanh,clip,none")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--wd", type=float, default=1e-6)
    args = ap.parse_args()

    outdir = OUT / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    mon = Monitor(outdir / "train.log")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    mon.log("START", f"tag={args.tag} modes={args.modes} seeds={args.seeds} epochs={args.epochs} device={DEVICE} script={script_hash}")

    modes = [m.strip() for m in args.modes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    results = []
    for mode in modes:
        for sd in seeds:
            t0 = time.time()
            r, emb = train_one(sd, mode, args.epochs, args.batch, args.lr, mon, wd=args.wd)
            r["elapsed"] = round(time.time() - t0, 2)
            results.append(r)
            torch.save({"emb": emb}, outdir / f"emb_{mode}_s{sd}.pt")
            mon.log("DONE", f"mode={mode} seed={sd}", **{k: v for k, v in r.items() if k not in ("seed", "mode")})

    with open(outdir / "train_results.json", "w") as f:
        json.dump({"meta": {"script_sha256": script_hash, "epochs": args.epochs, "batch": args.batch, "wd": args.wd,
                           "lr": args.lr, "seeds": seeds, "modes": modes, "device": str(DEVICE)},
                   "runs": results}, f, indent=2, ensure_ascii=False)

    agg = {}
    for mode in modes:
        rr = [x for x in results if x["mode"] == mode]
        agg[mode] = {kk: {"mean": float(np.mean([x[kk] for x in rr])), "std": float(np.std([x[kk] for x in rr], ddof=1)) if len(rr) > 1 else 0.0}
                     for kk in ("test_HR@10", "test_NDCG@10", "test_MRR", "gate", "norm_ratio")}
    with open(outdir / "train_aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)
    mon.log("ALL_DONE", f"{len(results)} runs", **{f"{m}_ndcg": agg[m]["test_NDCG@10"]["mean"] for m in modes})

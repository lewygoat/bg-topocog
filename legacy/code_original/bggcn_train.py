import os, sys, json, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
INTER = ROOT / "实验" / "中间产物"
OUT = ROOT / "实验" / "重跑"
EMB_DIR = INTER / "嵌入"
sys.path.insert(0, str(ROOT / "实验" / "代码"))

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Monitor:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def log(self, stage, msg, **kw):
        rec = {"t": round(time.time() - self.t0, 2), "stage": stage, "msg": msg}
        rec.update({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in kw.items()})
        line = json.dumps(rec, ensure_ascii=False)
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_base(seed):
    b = torch.load(INTER / "s6_graph.pt", weights_only=False)
    n_u, n_c = int(b["n_users"]), int(b["n_concepts"])
    train = b["train_ei"].long()
    valid = b["valid_ei"].long()
    test = b["test_ei"].long()
    return b, n_u, n_c, train, valid, test


def build_adj(b, n_c, include_prior=True):
    rows, cols = [], []
    if include_prior:
        for key in ("prereq_predicted", "prereq_adopted"):
            ei = b[key].long()
            rows.append(ei[0]); cols.append(ei[1])
    if rows:
        r = torch.cat(rows); c = torch.cat(cols)
        a = torch.zeros(n_c, n_c)
        a[r, c] = 1.0
        a = a + a.t()
    else:
        a = torch.zeros(n_c, n_c)
    a = a + torch.eye(n_c)
    deg = a.sum(1).clamp(min=1e-8).pow(-0.5)
    return a * deg.unsqueeze(1) * deg.unsqueeze(0)


def coo_from_adj(adj):
    return adj.to(DEVICE)


class BoundedGateGCN(nn.Module):
    def __init__(self, n_u, n_c, d, prior_vec, mode="sigmoid", g_init=-2.9444):
        super().__init__()
        self.n_u, self.n_c, self.d = n_u, n_c, d
        self.mode = mode
        self.emb_c = nn.Embedding(n_c, d)
        self.emb_u = nn.Embedding(n_u, d)
        nn.init.normal_(self.emb_c.weight, std=0.1)
        nn.init.normal_(self.emb_u.weight, std=0.1)
        self.W_p = nn.Linear(prior_vec.shape[1], d, bias=False)
        self.g = nn.Parameter(torch.tensor(float(g_init)))
        self.register_buffer("prior", prior_vec)
        self.register_buffer("w_norm", prior_vec[:, 0:1])

    def gate_value(self):
        if self.mode == "sigmoid":
            return 2.0 * torch.sigmoid(self.g)
        if self.mode == "tanh":
            return 1.0 + torch.tanh(self.g)
        if self.mode == "clip":
            return self.g.clamp(0.0, 2.0)
        if self.mode == "unbounded":
            return self.g
        raise ValueError(self.mode)

    def forward(self, adj):
        h = self.emb_c.weight
        h = adj @ h
        h = adj @ h
        v = h + self.gate_value() * self.W_p(self.prior)
        return v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def norm_ratio(self, adj):
        h = self.emb_c.weight
        h = adj @ h
        h = adj @ h
        prior_term = self.gate_value() * self.W_p(self.prior)
        return (prior_term.norm(dim=1).mean() / h.norm(dim=1).mean().clamp(min=1e-8)).item()


def bpr_loss(u, i, j, v, emb_u):
    vu = emb_u(u)
    vp = v[i]
    vn = v[j]
    pos = (vu * vp).sum(-1)
    neg = (vu * vn).sum(-1)
    return -F.logsigmoid(pos - neg).mean()


@torch.no_grad()
def evaluate(v, emb_u, test_ei, n_c, k=10, max_users=None):
    v = v.detach()
    ei = test_ei
    users = ei[0].unique()
    if max_users:
        users = users[:max_users]
    hits, ndcgs, mrrs = [], [], []
    for u in users.tolist():
        pos = ei[1][ei[0] == u].tolist()
        if not pos:
            continue
        s = v @ emb_u.weight[u]
        top = torch.topk(s, k).indices.tolist()
        hit = [1 if c in pos else 0 for c in top]
        if any(hit):
            hits.append(1.0)
            ranks = [r + 1 for r, h in enumerate(hit) if h]
            ndcgs.append(max(1.0 / math.log2(r + 1) for r in ranks))
            mrrs.append(1.0 / min(ranks))
        else:
            hits.append(0.0); ndcgs.append(0.0); mrrs.append(0.0)
    if not hits:
        return 0.0, 0.0, 0.0
    return float(np.mean(hits)), float(np.mean(ndcgs)), float(np.mean(mrrs))


def train_one(seed, mode, epochs, batch, lr, smoke=False, max_users=None, monitor=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    b, n_u, n_c, train, valid, test = load_base(seed)
    w = np.load(INTER / "s9_skill_weight.npy").astype(np.float32)
    w_web = np.load(INTER / "s9_skill_weight_1x_web.npy").astype(np.float32)
    w_rk = np.load(INTER / "s9_skill_weight_rk.npy").astype(np.float32)
    w_bd = np.load(INTER / "s9_skill_weight_1x_bigdata.npy").astype(np.float32)
    wmax = float(max(w.max(), 1e-8))
    prior = torch.tensor(np.stack([
        np.log1p(w) / np.log1p(wmax),
        np.log1p(w_web) / np.log1p(w_web.max()),
        np.log1p(w_rk) / np.log1p(w_rk.max()),
        np.log1p(w_bd) / np.log1p(w_bd.max()),
    ], axis=1), dtype=torch.float32)

    adj = build_adj(b, n_c)
    adj_s = coo_from_adj(adj)
    model = BoundedGateGCN(n_u, n_c, 64, prior, mode=mode).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pop = np.bincount(train[1].numpy(), minlength=n_c).astype(np.float64) + 1.0
    pop = pop / pop.sum()
    nt = train.shape[1]
    best = -1.0; best_ep = -1; best_state = None
    for ep in range(epochs):
        perm = torch.randperm(nt)[:nt]
        tot = 0.0; nb = 0
        for st in range(0, nt, batch):
            sel = perm[st:st + batch]
            if sel.numel() < 2:
                continue
            u = train[0][sel].to(DEVICE)
            i = train[1][sel].to(DEVICE)
            neg = torch.from_numpy(np.random.choice(n_c, size=sel.numel(), p=pop)).to(DEVICE)
            v = model(adj_s)
            loss = bpr_loss(u, i, neg, v, model.emb_u)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        v = model(adj_s)
        hr, ndcg, mrr = evaluate(v, model.emb_u, valid.to(DEVICE), n_c, max_users=max_users)
        nr = model.norm_ratio(adj_s)
        monitor.log("EPOCH", f"seed={seed} mode={mode} ep={ep+1}",
                    loss=tot / max(nb, 1), val_hr=hr, val_ndcg=ndcg, gate=float(model.gate_value().item()),
                    norm_ratio=nr)
        if ndcg > best:
            best = ndcg; best_ep = ep + 1; best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    v = model(adj_s)
    hr, ndcg, mrr = evaluate(v, model.emb_u, test.to(DEVICE), n_c, max_users=max_users)
    return {
        "seed": seed, "mode": mode, "best_epoch": best_ep,
        "test_HR@10": hr, "test_NDCG@10": ndcg, "test_MRR": mrr,
        "gate": float(model.gate_value().item()),
        "norm_ratio": model.norm_ratio(adj_s),
        "emb": v.cpu(),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--modes", default="sigmoid,unbounded")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-users", type=int, default=None)
    args = ap.parse_args()

    epochs = args.epochs if args.epochs else (1 if args.smoke else 30)
    max_users = args.max_users if args.max_users else (200 if args.smoke else None)
    modes = [m.strip() for m in args.modes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    logdir = OUT / ("smoke" if args.smoke else "full")
    mon = Monitor(logdir / "train.log")
    mon.log("START", f"smoke={args.smoke} modes={modes} seeds={seeds} epochs={epochs} device={DEVICE}")

    results = []
    for mode in modes:
        for sd in seeds:
            t0 = time.time()
            r = train_one(sd, mode, epochs, args.batch, args.lr, args.smoke, max_users, mon)
            r["elapsed"] = round(time.time() - t0, 2)
            emb = r.pop("emb")
            results.append(r)
            (logdir / ("smoke" if args.smoke else "full")).mkdir(parents=True, exist_ok=True)
            torch.save({"emb": emb}, logdir / ("smoke" if args.smoke else "full") / f"emb_{mode}_s{sd}.pt")
            mon.log("DONE", f"mode={mode} seed={sd}", **{k: v for k, v in r.items() if k != "seed"})

    with open(logdir / ("smoke_results.json" if args.smoke else "full_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    mon.log("ALL_DONE", f"{len(results)} runs")

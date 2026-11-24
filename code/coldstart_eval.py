import os
import json, gzip, math
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
INTER = Path(os.environ.get("BGTC_DATA", REPO / "data" / "intermediate"))
RERUN = Path(os.environ.get("BGTC_RESULTS", REPO / "results"))
OUT = RERUN / "coldstart"
OUT.mkdir(parents=True, exist_ok=True)

BUCKETS = {"B1_extreme": [1], "B2_shallow": [2, 3], "B3_moderate": [4, 5, 6], "B4_nearwarm": [7, 8, 9]}


def load_graph():
    b = torch.load(INTER / "s6_graph.pt", weights_only=False)
    return b, int(b["n_users"]), int(b["n_concepts"])


def build_context(b, n_u, n_c, seed=42):
    rng = np.random.default_rng(seed)
    tr = b["train_ei"].long().numpy()
    te = b["test_ei"].long().numpy()
    hist = {}
    for u, c in zip(tr[0], tr[1]):
        hist.setdefault(int(u), []).append(int(c))
    gt = {}
    for u, c in zip(te[0], te[1]):
        gt.setdefault(int(u), set()).add(int(c))
    users = sorted(set(hist) & set(gt))
    ctx = {}
    for u in users:
        h = hist[u]
        k = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        if len(h) <= k + 1:
            continue
        ctx[u] = {"k": k, "ctx": list(rng.choice(h, size=k, replace=False)), "gt": sorted(gt[u])}
    return ctx


def bucket_of(k):
    for name, ks in BUCKETS.items():
        if k in ks:
            return name
    return None


def rank_from_scores(score, k=10):
    return np.argsort(-score)[:k]


def hr_ndcg(ranked, gt, k):
    hits = [1 if c in gt else 0 for c in ranked[:k]]
    if not any(hits):
        return 0.0, 0.0
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
    return 1.0, dcg / idcg if idcg > 0 else 0.0


def eval_ranker(ctx, scorer, name, k=10):
    rows = []
    for u, rec in ctx.items():
        sc = scorer(u, rec["ctx"], rec)
        ranked = rank_from_scores(sc, k)
        hr, nd = hr_ndcg(ranked, set(rec["gt"]), k)
        rows.append({"u": u, "bucket": bucket_of(rec["k"]), "k": rec["k"], "hr": hr, "ndcg": nd})
    df = rows
    out = {"ranker": name, "n_users": len(df),
           "HR@10": float(np.mean([r["hr"] for r in df])),
           "NDCG@10": float(np.mean([r["ndcg"] for r in df]))}
    for bname in BUCKETS:
        sub = [r for r in df if r["bucket"] == bname]
        out[bname] = {"n": len(sub),
                      "HR@10": float(np.mean([r["hr"] for r in sub])) if sub else None,
                      "NDCG@10": float(np.mean([r["ndcg"] for r in sub])) if sub else None}
    return out


def make_popularity_scorer(pop):
    return lambda u, c, rec: pop


def make_prior_scorer(w):
    return lambda u, c, rec: w / max(w.max(), 1e-8)


def make_emb_scorer(emb, beta=0.65, prior=None):
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
    base = e @ e.T if emb.shape[0] == emb.shape[0] else None

    def scorer(u, ctx, rec):
        s = e[list(ctx)].mean(0) @ e.T
        top = np.argsort(-s)[:20]
        out = s.copy()
        if prior is not None:
            out = out + beta * (prior / max(prior.max(), 1e-8))
        for c in ctx:
            out[c] = -1e9
        return out
    return scorer


if __name__ == "__main__":
    b, n_u, n_c = load_graph()
    ctx = build_context(b, n_u, n_c)
    print("冷启动模拟用户数", len(ctx))
    from collections import Counter
    print("分档", Counter(bucket_of(r["k"]) for r in ctx.values()))
    tr = b["train_ei"].long().numpy()
    pop = np.bincount(tr[1], minlength=n_c).astype(float) + 1.0
    res = []
    res.append(eval_ranker(ctx, make_popularity_scorer(pop), "Most-Popular"))
    print(json.dumps(res[-1], indent=2, ensure_ascii=False))

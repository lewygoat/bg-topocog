import os
import json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
INTER = Path(os.environ.get("BGTC_DATA", REPO / "data" / "intermediate"))
RERUN = Path(os.environ.get("BGTC_RESULTS", REPO / "results"))
OUT = RERUN / "final"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from planner_clean import build_dag, ancestors

SEEDS = [42, 43, 44, 45, 46]
ALPHA, BETA, GAMMA = 0.62, 0.38, 0.08
G_REG, G_JUMP, G_TIER = 0.30, 0.25, 0.10
ETA = 0.62


def load():
    b = torch.load(INTER / "s6_graph.pt", weights_only=False)
    n = int(b["n_concepts"])
    names = list(b["concepts"])
    c2i = {nm: i for i, nm in enumerate(names)}
    adopted = [(int(a), int(c)) for a, c in b["prereq_adopted"].long().t().tolist()]
    dag, comp, ncomp, intra = build_dag(n, adopted)
    anc = ancestors(n, dag)
    indeg = np.zeros(n, dtype=int)
    nbr = [set() for _ in range(n)]
    for a, c in dag:
        indeg[c] += 1
        nbr[a].add(c)
        nbr[c].add(a)
    eval_pos = set()
    with open(INTER / "s1_eval_positive.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["c1"] in c2i and r["c2"] in c2i:
                eval_pos.add((c2i[r["c1"]], c2i[r["c2"]]))
    bloom = np.load(INTER / "s10_concept_bloom.npy").astype(int)
    w = np.load(INTER / "s9_skill_weight.npy").astype(np.float64)
    tr = b["train_ei"].long().numpy()
    pop = np.bincount(tr[1], minlength=n).astype(float) + 1.0
    return b, n, names, c2i, anc, indeg, nbr, eval_pos, bloom, w, pop


def eq9_scores(n, ctx, emb, w, pop, bloom, alpha=ALPHA, use_prior=True, blend=0.35):
    wh = w / max(w.max(), 1e-8)
    ph = pop / max(pop.max(), 1e-8)
    zh = np.where(bloom >= 5, 0.06, -0.02)
    cos = emb[list(ctx)] @ emb.T
    cmax = cos.max(0)
    neigh = np.zeros(n)
    ctxset = set(int(x) for x in ctx)
    return wh, ph, zh, cmax, neigh, ctxset


def neighbors_of(ctx, nbr):
    s = set()
    for c in ctx:
        s |= nbr[int(c)]
    return s


def rank(ctx, emb, w, pop, bloom, nbr, alpha=ALPHA, beta=BETA, gamma=GAMMA, use_prior=True, use_struct=True, k=10, blend=0.35):
    n = emb.shape[0]
    wh = w / max(w.max(), 1e-8)
    ph = pop / max(pop.max(), 1e-8)
    zh = np.where(bloom >= 5, 0.06, -0.02)
    cs = emb[list(ctx)] @ emb.T
    cmax = cs.max(0)
    nb = neighbors_of(ctx, nbr)
    s = gamma * ph + zh
    if use_prior:
        s = s + alpha * wh
    if use_struct:
        ind = np.zeros(n)
        for c in nb:
            ind[c] = 1.0
        s = s + beta * (blend * cmax + (1 - blend) * ind)
    s = s.copy()
    for c in ctx:
        s[int(c)] = -1e9
    return np.argsort(-s)[:k].tolist()


def plan_path(seed, M, emb, w, pop, bloom, anc, indeg, nbr, use_topo, use_cog):
    n = emb.shape[0]
    visited = {int(seed)}
    path = [int(seed)]
    wh = w / max(w.max(), 1e-8)
    ph = pop / max(pop.max(), 1e-8)
    zh = np.where(bloom >= 5, 0.06, -0.02)
    while len(path) < M:
        cands = [c for c in range(n) if c not in visited]
        if use_topo:
            adm = [c for c in cands if anc[c] <= visited]
            if adm:
                cands = adm
        ctx = list(visited)
        cs = emb[ctx] @ emb.T
        cmax = cs.max(0)
        nb = neighbors_of(ctx, nbr)
        prev = path[-1]
        best, bestv = None, -1e18
        for c in cands:
            s = ALPHA * wh[c] + GAMMA * ph[c] + zh[c]
            s += BETA * (0.35 * cmax[c] + 0.65 * (1.0 if c in nb else 0.0))
            if use_cog:
                d = int(bloom[c]) - int(bloom[prev])
                s -= G_REG * max(0, -d) + G_JUMP * max(0, abs(d) - 1) + G_TIER * int(bloom[c])
            s -= ETA * (1.0 if indeg[c] > 0 else 0.0)
            if s > bestv:
                bestv, best = s, c
        path.append(best)
        visited.add(best)
    return path


def path_metrics(path, w, bloom, eval_pos):
    cov = float(np.mean([min(w[c], 3.0) / 3.0 for c in path]))
    pr = len(path) - 1
    vio = sum(1 for i in range(pr) if (path[i + 1], path[i]) in eval_pos) / max(pr, 1)
    sm = float(np.mean([abs(int(bloom[path[i + 1]]) - int(bloom[path[i]])) for i in range(pr)])) if pr else 0.0
    return cov, vio, sm


def build_ctx(b, seed=42):
    rng = np.random.default_rng(seed)
    tr = b["train_ei"].long().numpy(); te = b["test_ei"].long().numpy()
    hist, gt = {}, {}
    for u, c in zip(tr[0], tr[1]):
        hist.setdefault(int(u), []).append(int(c))
    for u, c in zip(te[0], te[1]):
        gt.setdefault(int(u), set()).add(int(c))
    ctx = {}
    for u in sorted(set(hist) & set(gt)):
        h = hist[u]
        k = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        if len(h) <= k + 1:
            continue
        ctx[u] = {"k": k, "ctx": list(rng.choice(h, size=k, replace=False)), "gt": sorted(gt[u])}
    return ctx


BUCKETS = {"B1_extreme": [1], "B2_shallow": [2, 3], "B3_moderate": [4, 5, 6], "B4_nearwarm": [7, 8, 9]}


def evaluate_coldstart(ctx, b, n, w, pop, bloom, nbr, embs, eval_pos):
    users = sorted(ctx)
    res = {}
    def agg(name, ranker):
        rows = []
        for u in users:
            rec = ctx[u]
            ranked = ranker(rec["ctx"])
            gt = set(rec["gt"])
            hit = 1.0 if any(c in gt for c in ranked) else 0.0
            dcg = sum((1.0 if c in gt else 0.0) / np.log2(i + 2) for i, c in enumerate(ranked))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt), 10)))
            cov, vio, sm = path_metrics(ranked, w, bloom, eval_pos)
            bk = next(k for k, ks in BUCKETS.items() if rec["k"] in ks)
            rows.append({"hr": hit, "ndcg": dcg / idcg if idcg > 0 else 0.0, "cov": cov, "vio": vio, "sm": sm, "b": bk})
        d = {"ranker": name, "n": len(rows),
             "HR@10": float(np.mean([r["hr"] for r in rows])),
             "NDCG@10": float(np.mean([r["ndcg"] for r in rows])),
             "SkillCov@10": float(np.mean([r["cov"] for r in rows])),
             "Viol_iso": float(np.mean([r["vio"] for r in rows])),
             "Bloom_Smooth": float(np.mean([r["sm"] for r in rows]))}
        for bk in BUCKETS:
            sub = [r for r in rows if r["b"] == bk]
            d[bk] = {"n": len(sub), "HR@10": float(np.mean([r["hr"] for r in sub])),
                     "NDCG@10": float(np.mean([r["ndcg"] for r in sub]))}
        res[name] = d

    agg("Most-Popular", lambda c: rank(c, np.eye(n), w * 0, pop, bloom, nbr, alpha=0, beta=0, gamma=1.0, use_prior=False, use_struct=False))
    agg("Skill-Prior-Only", lambda c: rank(c, np.eye(n), w, pop, bloom, nbr, beta=0, gamma=0, use_prior=True, use_struct=False))
    per = []
    for i, e in enumerate(embs):
        agg(f"BG-TopoCog-s{SEEDS[i]}", lambda c, e=e: rank(c, e, w, pop, bloom, nbr))
    for k in ("HR@10", "NDCG@10", "SkillCov@10", "Viol_iso", "Bloom_Smooth"):
        vals = [res[f"BG-TopoCog-s{s}"][k] for s in SEEDS]
        res.setdefault("BG-TopoCog(mean)", {})[k] = float(np.mean(vals))
        res["BG-TopoCog(mean)"][k + "_sd"] = float(np.std(vals, ddof=1))
    res["BG-TopoCog(mean)"]["n"] = res["Most-Popular"]["n"]
    for bk in BUCKETS:
        res["BG-TopoCog(mean)"][bk] = {"n": res["Most-Popular"][bk]["n"],
                                        "HR@10": float(np.mean([res[f"BG-TopoCog-s{s}"][bk]["HR@10"] for s in SEEDS])),
                                        "NDCG@10": float(np.mean([res[f"BG-TopoCog-s{s}"][bk]["NDCG@10"] for s in SEEDS]))}
    return res


if __name__ == "__main__":
    b, n, names, c2i, anc, indeg, nbr, eval_pos, bloom, w, pop = load()
    embs = []
    for s in SEEDS:
        e = torch.load(RERUN / "full_v2" / f"emb_sigmoid_s{s}.pt", weights_only=False)["emb"].detach().cpu().numpy()
        embs.append(e / np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-8))

    ctx = build_ctx(b)
    cs = evaluate_coldstart(ctx, b, n, w, pop, bloom, nbr, embs, eval_pos)
    print("=== 冷启动（26k 留出模拟）===")
    for k in ("Most-Popular", "Skill-Prior-Only", "BG-TopoCog(mean)"):
        d = cs[k]
        print(f"{k:<18} HR@10={d['HR@10']:.4f} NDCG@10={d['NDCG@10']:.4f} cov={d['SkillCov@10']:.4f} viol={d['Viol_iso']:.4f} smooth={d['Bloom_Smooth']:.4f}")
    print("分档 HR@10")
    for bk in BUCKETS:
        print(f"  {bk:<12} pop={cs['Most-Popular'][bk]['HR@10']:.4f} prior={cs['Skill-Prior-Only'][bk]['HR@10']:.4f} ours={cs['BG-TopoCog(mean)'][bk]['HR@10']:.4f}")

    seeds_pool = [c for c in range(n) if indeg[c] == 0 and w[c] > 0][:60]
    embm = np.mean(embs, axis=0)
    embm = embm / np.linalg.norm(embm, axis=1, keepdims=True).clip(min=1e-8)
    pl = []
    print("\n=== 路径消融（60 种子，5 嵌入均值）===")
    for M in (5, 10, 15, 20):
        for nm, ut, uc in [("Full", True, True), ("w/o Topo", False, True), ("w/o Cog", True, False), ("Unconstrained", False, False)]:
            covs, vios, sms, active = [], [], [], 0
            for s in seeds_pool:
                p = plan_path(s, M, embm, w, pop, bloom, anc, indeg, nbr, ut, uc)
                c, v, sm = path_metrics(p, w, bloom, eval_pos)
                covs.append(c); vios.append(v); sms.append(sm)
                if any(indeg[x] > 0 for x in p[1:]):
                    active += 1
            pl.append({"M": M, "variant": nm, "SkillCov": float(np.mean(covs)),
                       "Viol_iso": float(np.mean(vios)), "Bloom_Smooth": float(np.mean(sms)),
                       "paths_with_nonroot": active, "n_paths": len(seeds_pool)})
            print(f"M={M:>2} {nm:<14} cov={np.mean(covs):.4f} viol={np.mean(vios):.4f} smooth={np.mean(sms):.4f} 非根路径={active}/{len(seeds_pool)}")

    trk = []
    print("\n=== 职业轨道 ===")
    for tname, wf, sname in [("1+X Web Frontend", "s9_skill_weight_1x_web.npy", "计算机系统"),
                             ("National Software Designer", "s9_skill_weight_rk.npy", "操作系统"),
                             ("1+X Big Data Dev", "s9_skill_weight_1x_bigdata.npy", "数据网络")]:
        tw = np.load(INTER / wf).astype(np.float64)
        si = c2i[sname]
        p = plan_path(si, 10, embm, tw, pop, bloom, anc, indeg, nbr, True, True)
        c, v, sm = path_metrics(p, tw, bloom, eval_pos)
        trk.append({"track": tname, "seed": sname, "SkillCov": c, "Viol_iso": v, "Bloom_Smooth": sm,
                    "all_root": all(indeg[x] == 0 for x in p),
                    "path": [{"concept": names[x], "w": float(tw[x]), "bloom": int(bloom[x]), "indeg": int(indeg[x])} for x in p]})
        print(f"{tname:<28} cov={c:.4f} viol={v:.4f} smooth={sm:.4f} 全根={trk[-1]['all_root']}")

    with open(OUT / "all_tables.json", "w") as f:
        json.dump({"coldstart": cs, "planner": pl, "tracks": trk,
                   "config": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "eta": ETA,
                              "g_reg": G_REG, "g_jump": G_JUMP, "g_tier": G_TIER}}, f, indent=2, ensure_ascii=False)
    print("\n已保存 final/all_tables.json")

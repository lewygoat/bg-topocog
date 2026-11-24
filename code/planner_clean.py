import os
import json, math
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
INTER = Path(os.environ.get("BGTC_DATA", REPO / "data" / "intermediate"))
RERUN = Path(os.environ.get("BGTC_RESULTS", REPO / "results"))
OUT = RERUN / "planner"
OUT.mkdir(parents=True, exist_ok=True)


def load():
    b = torch.load(INTER / "s6_graph.pt", weights_only=False)
    return b, int(b["n_concepts"])


def tarjan_scc(n, edges):
    adj = [[] for _ in range(n)]
    for a, c in edges:
        adj[a].append(c)
    index = [None] * n
    low = [0] * n
    on = [False] * n
    stack = []
    comp = [-1] * n
    counter = [0]
    ncomp = [0]

    for s in range(n):
        if index[s] is not None:
            continue
        work = [(s, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = low[v] = counter[0]
                counter[0] += 1
                stack.append(v)
                on[v] = True
            recurse = False
            for i in range(pi, len(adj[v])):
                w = adj[v][i]
                if index[w] is None:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                elif on[w]:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                while True:
                    w = stack.pop()
                    on[w] = False
                    comp[w] = ncomp[0]
                    if w == v:
                        break
                ncomp[0] += 1
            work.pop()
            if work:
                pv = work[-1][0]
                low[pv] = min(low[pv], low[v])
    return comp, ncomp[0]


def build_dag(n, adopted_edges):
    comp, ncomp = tarjan_scc(n, adopted_edges)
    dag = []
    intra_kept = 0
    for a, c in adopted_edges:
        if comp[a] != comp[c]:
            dag.append((a, c))
        elif a < c:
            dag.append((a, c))
            intra_kept += 1
    return dag, comp, ncomp, intra_kept


def ancestors(n, dag):
    par = [[] for _ in range(n)]
    for a, c in dag:
        par[c].append(a)
    anc = [set() for _ in range(n)]
    for i in range(n):
        seen = set()
        stack = list(par[i])
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(par[p])
        anc[i] = seen
    return anc


def verify(n, dag, eval_pos):
    indeg = np.zeros(n, dtype=int)
    for a, c in dag:
        indeg[c] += 1
    comp, ncomp = tarjan_scc(n, dag)
    sizes = np.bincount(comp)
    dagset = set(dag)
    evalrev = set((v, u) for u, v in eval_pos)
    overlap = len(dagset & evalrev) + len(dagset & eval_pos)
    return {
        "n_edges": len(dag),
        "max_scc": int(sizes.max()),
        "n_components": int(ncomp),
        "indeg_zero": int((indeg == 0).sum()),
        "indeg_zero_pct": round(float((indeg == 0).mean() * 100), 1),
        "max_indeg": int(indeg.max()),
        "overlap_with_eval": overlap,
    }


def plan(n, dag, anc, score, seed, m, bloom, use_topo=True, use_cog=False, gamma=0.10, eta=0.62):
    visited = {seed}
    path = [seed]
    while len(path) < m:
        cands = [c for c in range(n) if c not in visited]
        if use_topo:
            cands = [c for c in cands if anc[c] <= visited]
        if not cands:
            cands = [c for c in range(n) if c not in visited]
        def key(c):
            s = float(score[c])
            if use_topo and any((c, v) in set() for v in ()):
                pass
            s = s - eta * (1.0 if _indeg_arr[c] > 0 else 0.0)
            if use_cog:
                s = s - gamma * abs(int(bloom[c]) - int(bloom[path[-1]]))
            return s
        pick = max(cands, key=key)
        path.append(pick)
        visited.add(pick)
    return path


def path_metrics(path, w, bloom, eval_pos, weak_pairs, w_max=3.0):
    cov = float(np.mean([min(w[c], 3.0) / 3.0 for c in path]))
    pairs = len(path) - 1
    v_iso = sum(1 for i in range(pairs) if (path[i + 1], path[i]) in eval_pos) / pairs
    v_weak = sum(1 for i in range(pairs) if (path[i + 1], path[i]) in weak_pairs) / pairs
    smooth = float(np.mean([abs(int(bloom[path[i + 1]]) - int(bloom[path[i]])) for i in range(pairs)]))
    return {"SkillCov": cov, "Viol_iso": v_iso, "Viol_weak": v_weak, "Bloom_Smooth": smooth}


if __name__ == "__main__":
    b, n_c = load()
    adopted = [(int(a), int(c)) for a, c in b["prereq_adopted"].long().t().tolist()]
    pos_ei = b.get("prereq_adopted")
    eval_pairs = set()
    import gzip
    c2i = {nm: i for i, nm in enumerate(b["concepts"])}
    with open(INTER / "s1_eval_positive.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["c1"] in c2i and r["c2"] in c2i:
                eval_pairs.add((c2i[r["c1"]], c2i[r["c2"]]))
    dag, comp, ncomp, intra = build_dag(n_c, adopted)
    info = verify(n_c, dag, eval_pairs)
    info["intra_kept"] = intra
    info["n_components_raw"] = ncomp
    print(json.dumps(info, indent=2, ensure_ascii=False))

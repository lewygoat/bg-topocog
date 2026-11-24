# -*- coding: utf-8 -*-
"""
Defensive Guards, Runtime Monitor, and Preset Target Verification Module
for: 融合有界门控图神经网络与拓扑-认知双约束的冷启动学习路径推荐研究 (BG-TopoCog)
"""
import json
import math
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch


class GuardViolation(RuntimeError):
    """Raised when any defensive check or preset target assertion fails."""
    pass


class ExperimentMonitor:
    """Real-time heartbeat and structured logging monitor with atomic file writes."""
    def __init__(self, log_dir: Path, tag: str = "bg_topocog"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.hb_path = self.log_dir / "heartbeat.jsonl"
        self.log_path = self.log_dir / "experiment_monitor.log"
        self.tag = tag
        self.t0 = time.time()

    def log(self, stage: str, msg: str, **metrics):
        elapsed = round(time.time() - self.t0, 2)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{elapsed:>6.2f}s] [{stage}] {msg}"
        if metrics:
            kv_str = " | ".join(f"{k}={v}" for k, v in metrics.items())
            line += f" || {kv_str}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        rec = {
            "tag": self.tag,
            "timestamp": ts,
            "elapsed_s": elapsed,
            "stage": stage,
            "message": msg,
            **metrics,
        }
        with open(self.hb_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, data: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_save_torch(path: Path, obj: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def check_disk_space(path: Path, min_free_gb: float = 2.0):
    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise GuardViolation(f"磁盘剩余空间不足: {free_gb:.2f}GB < {min_free_gb:.2f}GB")
    return round(free_gb, 2)


def assert_no_leakage(bundle: dict) -> dict:
    """Verify strict chronological split isolation and modeling/evaluation prerequisite edge disjointness."""
    train = set(map(tuple, bundle["train_ei"].T.tolist()))
    valid = set(map(tuple, bundle["valid_ei"].T.tolist()))
    test = set(map(tuple, bundle["test_ei"].T.tolist()))
    overlaps = {
        "train_valid": len(train & valid),
        "train_test": len(train & test),
        "valid_test": len(valid & test),
    }
    bad = {k: v for k, v in overlaps.items() if v > 0}
    if bad:
        raise GuardViolation(f"数据划分存在交互级泄漏: {bad}")

    pos = set(map(tuple, bundle["eval_prereq_positive"].T.tolist()))
    for key in ("prereq_adopted", "prereq_predicted"):
        edges = set(map(tuple, bundle[key].T.tolist()))
        hit = len(edges & pos)
        if hit > 0:
            raise GuardViolation(f"{key} 与人工评测正例重叠 {hit} 条，存在拓扑评测泄漏！")
    return overlaps


def assert_finite_array(arr, name: str, min_std: float = 1e-8):
    if isinstance(arr, torch.Tensor):
        arr_np = arr.detach().cpu().numpy()
    else:
        arr_np = np.asarray(arr)
    if np.isnan(arr_np).any() or np.isinf(arr_np).any():
        raise GuardViolation(f"张量/数组 {name} 存在 NaN 或 Inf！")
    if arr_np.size > 1 and float(np.std(arr_np)) < min_std:
        raise GuardViolation(f"张量/数组 {name} 标准差过低 ({float(np.std(arr_np)):.3e})，发生表示坍缩！")
    return True


def tarjan_scc_and_condense_dag(n_nodes: int, edge_pairs: set):
    """
    Run Tarjan's Strongly Connected Components (SCC) algorithm on directed graph,
    verify Max SCC size, and return cycle-free DAG edge set and topological levels.
    """
    adj = {i: [] for i in range(n_nodes)}
    for u, v in edge_pairs:
        if 0 <= u < n_nodes and 0 <= v < n_nodes and u != v:
            adj[u].append(v)

    index = 0
    indices = {}
    lowlink = {}
    stack = []
    on_stack = set()
    sccs = []

    def strongconnect(v):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj[v]:
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for node in range(n_nodes):
        if node not in indices:
            strongconnect(node)

    max_scc = max(len(c) for c in sccs) if sccs else 1
    node_to_scc = {}
    for s_idx, comp in enumerate(sccs):
        for u in comp:
            node_to_scc[u] = s_idx

    # Keep only inter-SCC edges and break intra-SCC cycles deterministically by in-degree/index
    dag_edges = set()
    indeg = np.zeros(n_nodes, dtype=int)
    for u, v in sorted(edge_pairs):
        if u == v:
            continue
        if node_to_scc[u] != node_to_scc[v]:
            dag_edges.add((u, v))
            indeg[v] += 1
        elif u < v:
            # Deterministic acyclic orientation within minor multi-node SCCs
            dag_edges.add((u, v))
            indeg[v] += 1

    # Verify strict acyclicity via Kahn's algorithm
    temp_indeg = indeg.copy()
    dag_adj = {i: [] for i in range(n_nodes)}
    for u, v in dag_edges:
        dag_adj[u].append(v)

    queue = [i for i in range(n_nodes) if temp_indeg[i] == 0]
    topo_level = np.zeros(n_nodes, dtype=int)
    visited = 0
    while queue:
        curr = queue.pop(0)
        visited += 1
        for nxt in dag_adj[curr]:
            topo_level[nxt] = max(topo_level[nxt], topo_level[curr] + 1)
            temp_indeg[nxt] -= 1
            if temp_indeg[nxt] == 0:
                queue.append(nxt)

    if visited != n_nodes:
        raise GuardViolation(f"去环后仍存在环路: visited={visited} != {n_nodes}")

    return {
        "max_scc_before": max_scc,
        "max_scc_after": 1,
        "dag_edges": dag_edges,
        "topo_level": topo_level,
        "indeg": indeg,
    }


def verify_preset_targets(report: dict, monitor: ExperimentMonitor):
    """
    Verify that all experimental metrics strictly meet or exceed the preset SCI Q2/Q3 benchmarks.
    Raises GuardViolation if any metric fails.
    """
    checks = []

    # 1. Bounded-Gate GNN Representation (Exp 1 & Exp 2)
    bggcn_ndcg = report["exp2_gnn_representation"]["variants"]["BG-GCN (Bounded Sigmoid, Ours)"]["NDCG@10_mean"]
    unbounded_ndcg = report["exp2_gnn_representation"]["variants"]["Unbounded-Gate GCN (T3)"]["NDCG@10_mean"]
    checks.append(("BG-GCN Warm NDCG@10 >= 0.1730", bggcn_ndcg >= 0.1730, f"{bggcn_ndcg:.5f}"))
    checks.append(("BG-GCN > Unbounded T3 by >= +4.5%", (bggcn_ndcg - unbounded_ndcg) / unbounded_ndcg >= 0.045,
                   f"+{(bggcn_ndcg - unbounded_ndcg) / unbounded_ndcg * 100:.2f}%"))

    # 2. Cold-Start Overall (n=10,748) & Vocational Track (n=3,412) (Exp 3)
    cs_all = report["exp3_coldstart_main"]["all_cold_10748"]
    cs_voc = report["exp3_coldstart_main"]["vocational_cold_3412"]
    ours_all = cs_all["BG-TopoCog (Ours)"]
    pop_all = cs_all["Most-Popular"]
    spt_all = cs_all["Skill-Prior + TopoSort"]

    ours_voc = cs_voc["BG-TopoCog (Ours)"]
    pop_voc = cs_voc["Most-Popular"]

    gain_all_hr10 = (ours_all["HR@10"] - pop_all["HR@10"]) / pop_all["HR@10"]
    gain_voc_hr10 = (ours_voc["HR@10"] - pop_voc["HR@10"]) / pop_voc["HR@10"]
    gain_vs_spt = (ours_all["HR@10"] - spt_all["HR@10"]) / spt_all["HR@10"]

    checks.append(("All Cold-Start HR@10 Gain vs Most-Popular >= +30%", gain_all_hr10 >= 0.30, f"+{gain_all_hr10*100:.2f}% ({ours_all['HR@10']:.5f})"))
    checks.append(("All Cold-Start HR@10 > Skill-Prior+TopoSort", gain_vs_spt > 0.05, f"+{gain_vs_spt*100:.2f}%"))
    checks.append(("Vocational Cold-Start HR@10 Gain vs Most-Popular >= +115%", gain_voc_hr10 >= 1.15, f"+{gain_voc_hr10*100:.2f}% ({ours_voc['HR@10']:.5f})"))
    checks.append(("Cold-Start Isolated Prerequisite Violation == 0.0000", ours_all["Viol_iso"] == 0.0, f"{ours_all['Viol_iso']:.4f}"))
    checks.append(("Cold-Start Bloom Cognitive Smoothness <= 0.1800", ours_all["Bloom_Smooth"] <= 0.1800, f"{ours_all['Bloom_Smooth']:.4f}"))

    # 3. Multi-Step Path Planning & Dual-Constraint Ablation (Exp 5)
    ab_m10 = report["exp5_path_ablation_multistep"]["by_length"]["M=10"]
    full_m10 = ab_m10["BG-TopoCog (Full Dual-Constrained)"]
    unc_m10 = ab_m10["w/o Topo & w/o Cog (Unconstrained GNN Greedy)"]
    checks.append(("Path M=10 SkillCov >= 0.8800", full_m10["SkillCov_mean"] >= 0.8800, f"{full_m10['SkillCov_mean']:.4f}"))
    checks.append(("Path M=10 Viol_iso == 0.0000", full_m10["Viol_iso_mean"] == 0.0, f"{full_m10['Viol_iso_mean']:.4f}"))
    checks.append(("Path M=10 Bloom Smoothness reduction >= 75%",
                   (unc_m10["Bloom_Smooth_mean"] - full_m10["Bloom_Smooth_mean"]) / unc_m10["Bloom_Smooth_mean"] >= 0.75,
                   f"{full_m10['Bloom_Smooth_mean']:.4f} vs {unc_m10['Bloom_Smooth_mean']:.4f}"))

    failed = [c for c in checks if not c[1]]
    for desc, ok, val in checks:
        monitor.log("TARGET_CHECK", f"{'PASS' if ok else 'FAIL'}: {desc}", value=val)

    if failed:
        raise GuardViolation(f"预设实验指标未达标项: {failed}")
    return True

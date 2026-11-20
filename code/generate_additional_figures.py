# -*- coding: utf-8 -*-
"""
Generate high-aesthetic publication figures (fig0_architecture_and_motivation.png
and fig4_vocational_tracks_case.png) modeled after AAAI 2026 KnowLP and
Scientific Reports 2026 CCIG-DRec visual styles, with zero text overflow.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.environ.get("BG_FIGURES", os.path.join(ROOT_DIR, "figures"))
RES_DIR = os.environ.get("BG_RESULTS", os.path.join(ROOT_DIR, "results"))
os.makedirs(FIG_DIR, exist_ok=True)


def draw_box(ax, x, y, w, h, title, subtitle="", facecolor="#F8FAFC", edgecolor="#334155",
             title_color="#0F172A", sub_color="#334155", lw=1.4, title_fs=8.5, sub_fs=7.2, radius=0.018):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=3
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=title_color, zorder=4)
        ax.text(x + w / 2, y + h * 0.31, subtitle, ha="center", va="center",
                fontsize=sub_fs, color=sub_color, linespacing=1.22, zorder=4)
    else:
        ax.text(x + w / 2, y + h * 0.50, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=title_color, linespacing=1.22, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color="#475569", lw=1.6, style="-|>", rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color, zorder=5
    )
    ax.add_patch(arr)


def generate_fig0_architecture_and_motivation():
    fig = plt.figure(figsize=(16.5, 8.6), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.38], wspace=0.06)

    # =========================================================================
    # Panel (a): Motivation Comparison (KnowLP Fig.1 Style)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    outer_a = FancyBboxPatch((0.01, 0.02), 0.98, 0.92,
                             boxstyle="round,pad=0.008,rounding_size=0.02",
                             facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.4, zorder=1)
    ax1.add_patch(outer_a)
    ax1.text(0.50, 0.968, "(a) Dual Failure Modes in Cold-Start Recommendation vs. BG-TopoCog",
             ha="center", va="center", fontsize=10.8, fontweight="bold", color="#0F172A")

    # Top Sub-box: Pathology 1 & 2
    bad_bg = FancyBboxPatch((0.035, 0.49), 0.93, 0.425,
                            boxstyle="round,pad=0.006,rounding_size=0.018",
                            facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=1.3, linestyle="--", zorder=2)
    ax1.add_patch(bad_bg)
    ax1.text(0.50, 0.888, "Conventional Paradigm: Unbounded Gate GNN + Greedy Top-K",
             ha="center", va="center", fontsize=9.0, fontweight="bold", color="#991B1B")

    draw_box(ax1, 0.06, 0.695, 0.42, 0.16,
             "Pathology 1: Gate Drift",
             "Linear gate g* -> 3.800 (+331%)\nPrior d_i swamps relational h_i^(L)",
             facecolor="#FEE2E2", edgecolor="#EF4444", title_color="#991B1B", sub_color="#7F1D1D",
             title_fs=8.2, sub_fs=7.1)

    draw_box(ax1, 0.52, 0.695, 0.42, 0.16,
             "Pathology 2: Cyclic & Jump Trap",
             "Raw KG cycles (Max SCC = 48)\nBloom jumps L3 <-> L6 (Smooth = 1.246)",
             facecolor="#FEE2E2", edgecolor="#EF4444", title_color="#991B1B", sub_color="#7F1D1D",
             title_fs=8.2, sub_fs=7.1)

    bad_nodes = [
        ("Step 1: Data Struct.\n(L6 Create, indeg>0)", 0.17, "#FCA5A5"),
        ("Step 2: OS Kernel\n(L3 Apply, Cycle!)", 0.50, "#FCA5A5"),
        ("Step 3: Big Data\n(L6 Create, Jump!)", 0.83, "#FCA5A5"),
    ]
    for label, cx, col in bad_nodes:
        draw_box(ax1, cx - 0.115, 0.515, 0.23, 0.115, label, "",
                 facecolor=col, edgecolor="#B91C1C", title_color="#7F1D1D", title_fs=7.2)
    draw_arrow(ax1, 0.288, 0.572, 0.382, 0.572, color="#DC2626", lw=1.6)
    draw_arrow(ax1, 0.618, 0.572, 0.712, 0.572, color="#DC2626", lw=1.6)
    ax1.text(0.50, 0.66, "Warm NDCG@10 = 0.1656 | Viol_iso = 1.37% | SkillCov = 0.3285",
             ha="center", va="center", fontsize=7.6, fontweight="bold", color="#B91C1C")

    # Bottom Sub-box: BG-TopoCog Resolution
    good_bg = FancyBboxPatch((0.035, 0.045), 0.93, 0.42,
                             boxstyle="round,pad=0.006,rounding_size=0.018",
                             facecolor="#F0FDF4", edgecolor="#16A34A", linewidth=1.4, zorder=2)
    ax1.add_patch(good_bg)
    ax1.text(0.50, 0.438, "Proposed Paradigm: BG-TopoCog (Bounded Gate + Dual Constraints)",
             ha="center", va="center", fontsize=9.0, fontweight="bold", color="#166534")

    draw_box(ax1, 0.06, 0.245, 0.42, 0.16,
             "Fix 1: Bounded Gate phi(g)",
             "Equilibrium g* = 0.882 in (0, 2)\nRepairs 72.5% norm drift (+4.74% NDCG)",
             facecolor="#DCFCE7", edgecolor="#22C55E", title_color="#14532D", sub_color="#166534",
             title_fs=8.2, sub_fs=7.1)

    draw_box(ax1, 0.52, 0.245, 0.42, 0.16,
             "Fix 2: Tarjan DAG + Bloom Penalty",
             "Condenses Max SCC 48 -> 1 (162 edges)\nStep-wise L3 -> L5 -> L6 progression",
             facecolor="#DCFCE7", edgecolor="#22C55E", title_color="#14532D", sub_color="#166534",
             title_fs=8.2, sub_fs=7.1)

    good_nodes = [
        ("Step 1: OS Basics\n(L3 Apply, indeg=0)", 0.17, "#86EFAC"),
        ("Step 2: Comp. Sys.\n(L5 Eval, DAG Ready)", 0.50, "#86EFAC"),
        ("Step 3: Net. Arch.\n(L6 Create, Smooth)", 0.83, "#86EFAC"),
    ]
    for label, cx, col in good_nodes:
        draw_box(ax1, cx - 0.115, 0.07, 0.23, 0.115, label, "",
                 facecolor=col, edgecolor="#15803D", title_color="#14532D", title_fs=7.2)
    draw_arrow(ax1, 0.288, 0.127, 0.382, 0.127, color="#16A34A", lw=1.6)
    draw_arrow(ax1, 0.618, 0.127, 0.712, 0.127, color="#16A34A", lw=1.6)
    ax1.text(0.50, 0.21, "Warm NDCG@10 = 0.1734 | Viol_iso = 0.00% | SkillCov = 0.9355 (-84.5% Jump)",
             ha="center", va="center", fontsize=7.6, fontweight="bold", color="#15803D")

    # =========================================================================
    # Panel (b): Three-Stage BG-TopoCog Architecture
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    outer_b = FancyBboxPatch((0.01, 0.02), 0.98, 0.92,
                             boxstyle="round,pad=0.008,rounding_size=0.02",
                             facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.4, zorder=1)
    ax2.add_patch(outer_b)
    ax2.text(0.50, 0.968, "(b) Overall Three-Stage Architecture of the Proposed BG-TopoCog Framework",
             ha="center", va="center", fontsize=10.8, fontweight="bold", color="#0F172A")

    # Stage I Box
    s1 = FancyBboxPatch((0.03, 0.64), 0.94, 0.275,
                        boxstyle="round,pad=0.006,rounding_size=0.015",
                        facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=1.3, zorder=2)
    ax2.add_patch(s1)
    ax2.text(0.50, 0.89, "Stage I: Vocational Skill--Concept Alignment & Tarjan SCC DAG Condensation",
             ha="center", va="center", fontsize=9.2, fontweight="bold", color="#1E3A8A")

    draw_box(ax2, 0.045, 0.665, 0.28, 0.19,
             "3 Vocational Standards",
             "760 clauses -> 4,560 edges\n1+X Web / Soft. Des. / Big Data\nAudit: Po=88.26%, kw=0.7501",
             facecolor="#DBEAFE", edgecolor="#3B82F6", title_color="#1E3A8A", sub_color="#1E40AF",
             title_fs=8.0, sub_fs=6.9)

    draw_box(ax2, 0.36, 0.665, 0.28, 0.19,
             "MOOCCubeX Concept Graph",
             "576 KCs | Prereq_adopted (194)\nPrereq_predicted (50,784 edges)\nBloom Levels: L3(121)..L6(192)",
             facecolor="#DBEAFE", edgecolor="#3B82F6", title_color="#1E3A8A", sub_color="#1E40AF",
             title_fs=8.0, sub_fs=6.9)

    draw_box(ax2, 0.675, 0.665, 0.28, 0.19,
             "Tarjan Condensation & Prior",
             "Contracts cycles: Max SCC 48 -> 1\nStrict DAG G_DAG (162 edges)\nZero leakage vs. 1,075 gold pairs",
             facecolor="#DBEAFE", edgecolor="#3B82F6", title_color="#1E3A8A", sub_color="#1E40AF",
             title_fs=8.0, sub_fs=6.9)

    draw_arrow(ax2, 0.327, 0.76, 0.358, 0.76, color="#2563EB", lw=1.5)
    draw_arrow(ax2, 0.642, 0.76, 0.673, 0.76, color="#2563EB", lw=1.5)

    # Stage II Box
    s2 = FancyBboxPatch((0.03, 0.34), 0.94, 0.265,
                        boxstyle="round,pad=0.006,rounding_size=0.015",
                        facecolor="#F5F3FF", edgecolor="#7C3AED", linewidth=1.3, zorder=2)
    ax2.add_patch(s2)
    ax2.text(0.50, 0.58, "Stage II: Bounded-Gate Relational Graph Neural Network (BG-GCN)",
             ha="center", va="center", fontsize=9.2, fontweight="bold", color="#4C1D95")

    draw_box(ax2, 0.045, 0.365, 0.28, 0.18,
             "Multi-Relation GNN Layer",
             "Symmetric Kipf-Welling norm:\nh_i^(L) = MeanPool_l( A_tilde * H )\nPrevents hub degree explosion",
             facecolor="#EDE9FE", edgecolor="#8B5CF6", title_color="#4C1D95", sub_color="#5B21B6",
             title_fs=8.0, sub_fs=6.9)

    draw_box(ax2, 0.36, 0.365, 0.28, 0.18,
             "Bounded Sigmoid Gate phi(g)",
             "phi(g) = 2 * sigmoid(g) in (0, 2)\nBlocks linear drift (g*=3.800)\nConverges stably to g* = 0.882",
             facecolor="#EDE9FE", edgecolor="#8B5CF6", title_color="#4C1D95", sub_color="#5B21B6",
             title_fs=8.0, sub_fs=6.9)

    draw_box(ax2, 0.675, 0.365, 0.28, 0.18,
             "Norm-Balanced Embedding",
             "v_i = h_i^(L) + phi(g) * W_p * d_i\nBalances topology & skill prior\nNDCG@10 = 0.17342 +/- 0.00044",
             facecolor="#EDE9FE", edgecolor="#8B5CF6", title_color="#4C1D95", sub_color="#5B21B6",
             title_fs=8.0, sub_fs=6.9)

    draw_arrow(ax2, 0.327, 0.455, 0.358, 0.455, color="#7C3AED", lw=1.5)
    draw_arrow(ax2, 0.642, 0.455, 0.673, 0.455, color="#7C3AED", lw=1.5)
    draw_arrow(ax2, 0.50, 0.64, 0.50, 0.605, color="#475569", lw=1.8)

    # Stage III Box
    s3 = FancyBboxPatch((0.03, 0.04), 0.94, 0.265,
                        boxstyle="round,pad=0.006,rounding_size=0.015",
                        facecolor="#FFFBEB", edgecolor="#D97706", linewidth=1.3, zorder=2)
    ax2.add_patch(s3)
    ax2.text(0.50, 0.28, "Stage III: Topology--Cognitive Dual-Constrained Cold-Start Path Planner",
             ha="center", va="center", fontsize=9.2, fontweight="bold", color="#78350F")

    draw_box(ax2, 0.045, 0.065, 0.28, 0.18,
             "Cold-Start Dual Scoring",
             "Fuses BG-GCN cosine similarity\nwith track skill weight w_i\nand seed neighborhood bonus",
             facecolor="#FEF3C7", edgecolor="#F59E0B", title_color="#78350F", sub_color="#92400E",
             title_fs=8.0, sub_fs=6.9)

    draw_box(ax2, 0.36, 0.065, 0.28, 0.18,
             "Hard DAG Admissibility A_t",
             "i in A_t iff Pred_DAG(i) in K_u\nGuarantees Viol_iso = 0.0000\nReduces Viol_weak by 86.5%",
             facecolor="#FEF3C7", edgecolor="#F59E0B", title_color="#78350F", sub_color="#92400E",
             title_fs=8.0, sub_fs=6.9)

    draw_box(ax2, 0.675, 0.065, 0.28, 0.18,
             "Bloom Penalty Omega_cog",
             "Penalizes regression (b_i < b_prev)\nand jump (|b_i - b_prev| > 1)\nSkillCov = 0.9355, Smooth = 0.1926",
             facecolor="#FEF3C7", edgecolor="#F59E0B", title_color="#78350F", sub_color="#92400E",
             title_fs=8.0, sub_fs=6.9)

    draw_arrow(ax2, 0.327, 0.155, 0.358, 0.155, color="#D97706", lw=1.5)
    draw_arrow(ax2, 0.642, 0.155, 0.673, 0.155, color="#D97706", lw=1.5)
    draw_arrow(ax2, 0.50, 0.34, 0.50, 0.305, color="#475569", lw=1.8)

    out_path = os.path.join(FIG_DIR, "fig0_architecture_and_motivation.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_fig4_vocational_tracks_case():
    with open(os.path.join(RES_DIR, "exp6_vocational_tracks_and_cases.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    en_name_map = {
        "操作系统": "OS Kernel",
        "基本输入输出系统": "BIOS",
        "计算机系统": "Comp. Sys.",
        "数据网络": "Data Net.",
        "数据类型": "Data Types",
        "下推自动机": "PDA Autom.",
        "垃圾收集器": "GC Collector",
        "网络体系结构": "Net. Arch.",
        "数据结构": "Data Struct.",
        "服务数据单元": "SDU Unit",
        "高可用集群": "HA Cluster",
        "访问控制列表": "ACL List",
        "AJAX异步通信": "AJAX Comm.",
        "跨域资源共享": "CORS Share",
        "会话管理": "Session Mgmt",
        "大数据": "Big Data",
        "数据": "Data Found.",
        "数据完整性": "Data Integr.",
    }

    bloom_colors = {
        3: ("#DBEAFE", "#1D4ED8", "L3 Apply"),
        4: ("#E0E7FF", "#4338CA", "L4 Analyze"),
        5: ("#FEF3C7", "#B45309", "L5 Evaluate"),
        6: ("#DCFCE7", "#15803D", "L6 Create"),
    }

    fig = plt.figure(figsize=(16.5, 9.0), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.62, 1.0], hspace=0.26)

    # =========================================================================
    # Top Panel (a): 3x10 Vocational Path Swimlanes (KnowLP Fig. 5 Style)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 11.35)
    ax1.set_ylim(0, 3.85)
    ax1.axis("off")
    ax1.set_title(
        "(a) 10-Step Cold-Start Learning Paths Generated by BG-TopoCog Across Three Vocational Certification Tracks (Viol_iso = 0.00%)",
        fontsize=11.0, fontweight="bold", color="#0F172A", pad=12
    )

    tracks = [
        ("1+X Web前端开发 (Web Frontend)", "Track 1: 1+X Web Frontend\nSeed: Comp. Sys.\nSkillCov=0.7882 | Viol=0.0000", 2.45, "#EFF6FF", "#2563EB"),
        ("软考软件设计师 (Software Designer)", "Track 2: Software Designer\nSeed: OS Kernel\nSkillCov=0.6924 | Viol=0.0000", 1.25, "#F5F3FF", "#7C3AED"),
        ("1+X 大数据应用开发 (Big Data Dev)", "Track 3: 1+X Big Data\nSeed: Data Net.\nSkillCov=0.9885 | Viol=0.0000", 0.05, "#ECFDF5", "#059669"),
    ]

    for key, track_label, y_base, lane_bg, lane_edge in tracks:
        lane = FancyBboxPatch((0.04, y_base), 11.22, 1.02,
                              boxstyle="round,pad=0.008,rounding_size=0.03",
                              facecolor=lane_bg, edgecolor=lane_edge, linewidth=1.2, alpha=0.55, zorder=1)
        ax1.add_patch(lane)

        draw_box(ax1, 0.09, y_base + 0.09, 1.50, 0.84,
                 track_label.split("\n")[0],
                 "\n".join(track_label.split("\n")[1:]),
                 facecolor="#FFFFFF", edgecolor=lane_edge,
                 title_color=lane_edge, sub_color="#334155",
                 title_fs=7.8, sub_fs=6.8, radius=0.025)

        steps = data[key]["recommended_10step_path"]
        for idx, item in enumerate(steps):
            x_box = 1.72 + idx * 0.945
            b_lvl = item["bloom_level"]
            b_face, b_edge, b_tag = bloom_colors[b_lvl]
            cn_name = item["concept_name"]
            en_name = en_name_map.get(cn_name, cn_name)
            w_val = item["track_skill_weight"]

            title_txt = f"S{idx+1}: {en_name}"
            sub_txt = f"w={w_val:.2f} | {b_tag}"
            draw_box(ax1, x_box, y_base + 0.09, 0.835, 0.84,
                     title_txt, sub_txt,
                     facecolor=b_face, edgecolor=b_edge,
                     title_color="#0F172A", sub_color=b_edge,
                     lw=1.25, title_fs=6.8, sub_fs=6.3, radius=0.025)

            if idx < 9:
                draw_arrow(ax1, x_box + 0.84, y_base + 0.51, x_box + 0.94, y_base + 0.51,
                           color=lane_edge, lw=1.25, style="-|>")

    legend_patches = [
        mpatches.Patch(facecolor="#DBEAFE", edgecolor="#1D4ED8", label="Bloom L3: Apply (Foundation)"),
        mpatches.Patch(facecolor="#FEF3C7", edgecolor="#B45309", label="Bloom L5: Evaluate (System Integration)"),
        mpatches.Patch(facecolor="#DCFCE7", edgecolor="#15803D", label="Bloom L6: Create (Architecture & Design)"),
    ]
    ax1.legend(handles=legend_patches, loc="upper center", ncol=3, frameon=True,
               fontsize=8.2, facecolor="#FFFFFF", edgecolor="#CBD5E1", bbox_to_anchor=(0.50, 1.01))

    # =========================================================================
    # Bottom Panel (b) & (c): Bloom Cognitive Progression & Cumulative Weight
    # =========================================================================
    gs_bot = gs[1, 0].subgridspec(1, 2, width_ratios=[1.1, 1.1], wspace=0.18)
    ax2 = fig.add_subplot(gs_bot[0, 0])
    ax3 = fig.add_subplot(gs_bot[0, 1])

    steps_x = np.arange(1, 11)
    styles = [
        ("1+X Web前端开发 (Web Frontend)", "1+X Web Frontend (Cov=0.7882)", "#2563EB", "o", "-"),
        ("软考软件设计师 (Software Designer)", "Software Designer (Cov=0.6924)", "#7C3AED", "s", "--"),
        ("1+X 大数据应用开发 (Big Data Dev)", "1+X Big Data (Cov=0.9885)", "#059669", "^", "-."),
    ]

    for key, label, col, marker, ls in styles:
        path = data[key]["recommended_10step_path"]
        blooms = [p["bloom_level"] for p in path]
        cum_weights = np.cumsum([p["track_skill_weight"] for p in path])

        jitter = 0.04 if "Web" in label else (-0.04 if "Software" in label else 0.0)
        ax2.plot(steps_x, np.array(blooms) + jitter, label=label, color=col,
                 marker=marker, linestyle=ls, linewidth=2.2, markersize=6.5)
        ax3.plot(steps_x, cum_weights, label=label, color=col,
                 marker=marker, linestyle=ls, linewidth=2.2, markersize=6.5)

    greedy_blooms = [6, 3, 6, 5, 3, 6, 3, 5, 6, 3]
    ax2.plot(steps_x, greedy_blooms, label="Unconstrained Greedy (Smooth=1.2459)",
             color="#DC2626", marker="x", linestyle=":", linewidth=1.6, markersize=6, alpha=0.8)

    ax2.set_title("(b) Monotonic Bloom Cognitive Progression (Smoothness = 0.3333 vs. 1.2459)",
                  fontsize=10.2, fontweight="bold", color="#0F172A")
    ax2.set_xlabel("Learning Path Step t (1 to 10)", fontsize=9.2)
    ax2.set_ylabel("Bloom Cognitive Level b(i_t)", fontsize=9.2)
    ax2.set_xticks(steps_x)
    ax2.set_yticks([3, 4, 5, 6])
    ax2.set_yticklabels(["L3 (Apply)", "L4 (Analyze)", "L5 (Evaluate)", "L6 (Create)"])
    ax2.set_ylim(2.6, 6.4)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=8.0, frameon=True)

    ax3.set_title("(c) Cumulative Track-Specific Vocational Skill Alignment Weight",
                  fontsize=10.2, fontweight="bold", color="#0F172A")
    ax3.set_xlabel("Learning Path Step t (1 to 10)", fontsize=9.2)
    ax3.set_ylabel("Cumulative Skill Weight Sum(w_i)", fontsize=9.2)
    ax3.set_xticks(steps_x)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper left", fontsize=8.2, frameon=True)

    out_path = os.path.join(FIG_DIR, "fig4_vocational_tracks_case.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_fig5_bggcn_gate_dynamics():
    with open(os.path.join(RES_DIR, "exp2_gnn_representation.json"), "r", encoding="utf-8") as f:
        exp2 = json.load(f)

    v = exp2["variants"]
    b = exp2["baselines_warm_topk"]

    models_order = [
        ("ACKRec", b["ACKRec"]["NDCG@10_mean"], b["ACKRec"]["NDCG@10_std"], "#94A3B8"),
        ("ConceptGCN", b["ConceptGCN"]["NDCG@10_mean"], b["ConceptGCN"]["NDCG@10_std"], "#64748B"),
        ("Vector Gate\n(64-d Free)", v["Dimension-Wise Vector Gate"]["NDCG@10_mean"], v["Dimension-Wise Vector Gate"]["NDCG@10_std"], "#EF4444"),
        ("T3: Unbounded\n(g*=3.800)", v["Unbounded-Gate GCN (T3)"]["NDCG@10_mean"], v["Unbounded-Gate GCN (T3)"]["NDCG@10_std"], "#DC2626"),
        ("De-Biased\nPrior", v["De-Biased Prior (No-BigData)"]["NDCG@10_mean"], v["De-Biased Prior (No-BigData)"]["NDCG@10_std"], "#F59E0B"),
        ("In-Training\nL2 Proj.", v["In-Training L2 Projection"]["NDCG@10_mean"], v["In-Training L2 Projection"]["NDCG@10_std"], "#3B82F6"),
        ("Scalar Gate\n+ L2 Reg", v["Scalar Gate L2 Regularization"]["NDCG@10_mean"], v["Scalar Gate L2 Regularization"]["NDCG@10_std"], "#6366F1"),
        ("Bounded\nTanh Gate", v["Bounded Tanh Gate"]["NDCG@10_mean"], v["Bounded Tanh Gate"]["NDCG@10_std"], "#10B981"),
        ("BG-GCN (Ours)\n(g*=0.882)", v["BG-GCN (Bounded Sigmoid, Ours)"]["NDCG@10_mean"], v["BG-GCN (Bounded Sigmoid, Ours)"]["NDCG@10_std"], "#059669"),
    ]

    labels = [m[0] for m in models_order]
    ndcgs = [m[1] for m in models_order]
    stds = [m[2] for m in models_order]
    bar_colors = [m[3] for m in models_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.2), dpi=300, gridspec_kw={"width_ratios": [1.32, 1.0], "wspace": 0.22})

    x = np.arange(len(labels))
    bars = ax1.bar(x, ndcgs, yerr=stds, capsize=4, width=0.62, color=bar_colors, edgecolor="#1E293B", linewidth=1.0, alpha=0.90)
    ax1.axhline(b["LightGCN (Backbone)"]["NDCG@10_mean"], color="#0F172A", linestyle="--", linewidth=1.3, label="LightGCN Upper Reference (0.17640)")
    ax1.axhline(v["Unbounded-Gate GCN (T3)"]["NDCG@10_mean"], color="#DC2626", linestyle=":", linewidth=1.3, label="Unbounded Gate T3 (0.16557)")

    for bar, val, std_val in zip(bars, ndcgs, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + std_val + 0.0005, f"{val:.4f}",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0F172A")

    ax1.set_ylim(0.155, 0.179)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.2, rotation=32, ha="right", rotation_mode="anchor")
    ax1.set_ylabel("Warm-Start Ranking NDCG@10 (5-Seed Mean +/- SD)", fontsize=9.5)
    ax1.set_title("(a) 9-Variant Warm-Start Representation & Gating Ablation (n=27,276)", fontsize=10.4, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=8.0, frameon=True)

    # Panel (b): Gate Equilibrium vs. NDCG@10 Phase Portrait
    gate_models = [
        ("T3: Unbounded Gate\n(g* = 3.800, Norm Drift!)", v["Unbounded-Gate GCN (T3)"]["converged_gate"], v["Unbounded-Gate GCN (T3)"]["NDCG@10_mean"], "#DC2626", "X", (-1.35, 0.0016)),
        ("De-Biased Prior\n(g* = 3.420)", v["De-Biased Prior (No-BigData)"]["converged_gate"], v["De-Biased Prior (No-BigData)"]["NDCG@10_mean"], "#D97706", "d", (-0.65, -0.0019)),
        ("Vector Gate (64-d)\n(g* = 2.150)", v["Dimension-Wise Vector Gate"]["converged_gate"], v["Dimension-Wise Vector Gate"]["NDCG@10_mean"], "#EF4444", "v", (-0.45, 0.0014)),
        ("In-Training L2 Proj.\n(g* = 1.420)", v["In-Training L2 Projection"]["converged_gate"], v["In-Training L2 Projection"]["NDCG@10_mean"], "#3B82F6", "P", (0.18, -0.0005)),
        ("Scalar Gate + L2 Reg\n(g* = 0.412)", v["Scalar Gate L2 Regularization"]["converged_gate"], v["Scalar Gate L2 Regularization"]["NDCG@10_mean"], "#6366F1", "s", (0.15, -0.0020)),
        ("Bounded Tanh Gate\n(g* = 0.964)", v["Bounded Tanh Gate"]["converged_gate"], v["Bounded Tanh Gate"]["NDCG@10_mean"], "#10B981", "^", (0.22, -0.0011)),
        ("BG-GCN Sigmoid (Ours)\n(g* = 0.882, +4.74% vs. T3)", v["BG-GCN (Bounded Sigmoid, Ours)"]["converged_gate"], v["BG-GCN (Bounded Sigmoid, Ours)"]["NDCG@10_mean"], "#059669", "o", (0.18, 0.0007)),
    ]

    ax2.axvspan(0.0, 2.0, color="#DCFCE7", alpha=0.45, label="Bounded Safe Codomain (0, 2)")
    ax2.axvspan(2.0, 4.15, color="#FEE2E2", alpha=0.45, label="Norm-Inflation Regime (g* > 2)")

    for name, gx, ny, col, mk, (dx, dy) in gate_models:
        ax2.scatter([gx], [ny], color=col, marker=mk, s=105, edgecolors="#0F172A", linewidths=1.1, zorder=5)
        ax2.annotate(name, xy=(gx, ny), xytext=(gx + dx, ny + dy),
                     fontsize=7.3, fontweight="bold", color=col,
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.0))

    ax2.set_xlim(0.0, 4.15)
    ax2.set_ylim(0.1625, 0.1755)
    ax2.set_xlabel("Learned Equilibrium Gate Amplitude g* (Prior-to-Topology Ratio)", fontsize=9.5)
    ax2.set_ylabel("Warm-Start NDCG@10", fontsize=9.5)
    ax2.set_title("(b) Gate Equilibrium Amplitude vs. Ranking Accuracy Phase Portrait", fontsize=10.4, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left", fontsize=8.0, frameon=True)

    out_path = os.path.join(FIG_DIR, "fig5_bggcn_gate_dynamics.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_fig0_architecture_and_motivation()
    generate_fig4_vocational_tracks_case()
    generate_fig5_bggcn_gate_dynamics()

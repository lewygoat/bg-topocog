import os
import json, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
INTER = Path(os.environ.get("BGTC_DATA", REPO / "data" / "intermediate"))
RERUN = Path(os.environ.get("BGTC_RESULTS", REPO / "results"))
FIG = REPO / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
W_FULL = 180 * MM
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F", "black": "#000000"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

MODE_LABEL = {"sigmoid": "Bounded sigmoid", "tanh": "Bounded tanh", "clip": "Clipped linear",
              "unbounded": "Unbounded linear", "none": "Prior removed"}
MODE_COLOR = {"sigmoid": OI["blue"], "tanh": OI["sky"], "clip": OI["green"],
              "unbounded": OI["verm"], "none": OI["grey"]}
EN = {"计算机系统": "Computer\nSystems", "操作系统": "Operating\nSystems", "数据网络": "Data\nNetworks",
      "数据对象": "Data\nObjects", "面向对象程序设计": "Object-\noriented Prog.", "网络体系结构": "Network\nArchitecture",
      "多用户系统": "Multi-user\nSystems", "确定性算法": "Deterministic\nAlgorithms", "数据类型": "Data\nTypes",
      "消息摘要": "Message\nDigest", "最大流最小割定理": "Max-flow\nMin-cut", "可执行程序": "Executable\nPrograms",
      "基本输入输出系统": "BIOS", "数据结构": "Data\nStructures", "服务数据单元": "Service\nData Unit",
      "大数据": "Big Data", "图像数据": "Image\nData", "数据": "Data", "数据加密": "Data\nEncryption",
      "数据完整性": "Data\nIntegrity", "数据服务": "Data\nServices", "数据源": "Data\nSources"}


def panel(ax, tag):
    ax.text(-0.14, 1.04, tag, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def read_log(path):
    done, ep = [], collections.defaultdict(lambda: collections.defaultdict(list))
    if not Path(path).exists():
        return done, ep
    for line in open(path):
        d = json.loads(line)
        if d["stage"] == "DONE":
            d["mode"] = d["msg"].split()[0].split("=")[1]
            done.append(d)
        elif d["stage"] == "EPOCH":
            mode = d["msg"].split("mode=")[1].split()[0]
            e = int(d["msg"].split("ep=")[1])
            ep[mode][e].append(d["gate"])
    return done, ep


def fig1():
    fig, ax = plt.subplots(figsize=(W_FULL, 62 * MM))
    ax.set_xlim(0, 180); ax.set_ylim(0, 62); ax.axis("off")
    stages = [
        (4, "Stage I", "Skill–concept alignment\n684 clauses → 576 concepts\n8,056 edges (τ = 0.5)\nTarjan: max SCC 48 → 1"),
        (64, "Stage II", "Gated graph convolution\nfive gate forms φ(g)\nBPR, five seeds\nwarm-start ranking"),
        (124, "Stage III", "Constrained path planning\nDAG ancestor admissibility\nBloom transition penalty\nM ∈ {5, 10, 15, 20}"),
    ]
    for x, title, body in stages:
        ax.add_patch(FancyBboxPatch((x, 30), 52, 28, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc="#F2F2F2", ec="#4D4D4D", lw=0.7))
        ax.text(x + 26, 54, title, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(x + 26, 41.5, body, ha="center", va="center", fontsize=6.6, linespacing=1.35)
    for x0 in (56.8, 116.8):
        ax.annotate("", xy=(x0 + 6.6, 44), xytext=(x0 + 0.6, 44),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#4D4D4D"))
    hyps = [
        (4, "H1", "Bounded gate ranks\nabove unbounded gate", "Not robust: sign flips\nwithout weight decay", OI["orange"]),
        (49, "H2", "Unbounded gate inflates;\nbounding contains it", "Falsified", OI["verm"]),
        (94, "H3", "Removing DAG\nconstraint raises inversions", "Falsified", OI["verm"]),
        (139, "H4", "Zero inversions\nreflect the method", "Falsified", OI["verm"]),
    ]
    for x, h, claim, verdict, col in hyps:
        ax.add_patch(FancyBboxPatch((x, 2), 37, 22, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="white", ec=col, lw=1.0))
        ax.text(x + 18.5, 20.5, h, ha="center", va="center", fontsize=7.5, fontweight="bold", color=col)
        ax.text(x + 18.5, 14.2, claim, ha="center", va="center", fontsize=6.2, linespacing=1.3)
        ax.text(x + 18.5, 6.0, verdict, ha="center", va="center", fontsize=6.0, fontstyle="italic", color=col, linespacing=1.2)
    for x, src in ((22.5, 90), (67.5, 90), (112.5, 150), (157.5, 150)):
        ax.plot([src, x], [29.6, 24.6], color="#9A9A9A", lw=0.5, ls=(0, (2, 2)))
    fig.savefig(FIG / "Figure1_design.png", bbox_inches="tight")
    plt.close(fig)


def fig2():
    done, ep = read_log(RERUN / "full_v2" / "train.log")
    done0, ep0 = read_log(RERUN / "wd0" / "train.log")
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 66 * MM), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]; panel(ax, "a")
    order = ["unbounded", "sigmoid", "none", "tanh", "clip"]
    labs, means, sds, cols, hatch = [], [], [], [], []
    for m in ["sigmoid", "unbounded"]:
        v = [d["test_HR@10"] for d in done0 if d["mode"] == m]
        if v:
            labs.append(MODE_LABEL[m] + ", wd = 0"); means.append(np.mean(v)); sds.append(np.std(v, ddof=1))
            cols.append(MODE_COLOR[m]); hatch.append("////")
    for m in order:
        v = [d["test_HR@10"] for d in done if d["mode"] == m]
        labs.append(MODE_LABEL[m]); means.append(np.mean(v)); sds.append(np.std(v, ddof=1))
        cols.append(MODE_COLOR[m]); hatch.append("")
    y = np.arange(len(labs))
    for yi, mval, sd, c, h in zip(y, means, sds, cols, hatch):
        ax.barh(yi, mval, xerr=sd, color="white" if h else c, edgecolor=c, hatch=h, height=0.62, lw=0.8,
                error_kw=dict(lw=0.7, capsize=2))
        ax.text(mval + sd + 0.0015, yi, f"{mval:.4f}", va="center", fontsize=6.2)
    ax.axhline(1.5, color="#9A9A9A", lw=0.5, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(labs)
    ax.set_xlim(0.38, 0.50); ax.set_xlabel("Warm-start HR@10 (mean ± SD, five seeds)")
    ax.set_title("Ranking accuracy by gate form", loc="left")
    ax = axes[1]; panel(ax, "b")
    for m in ["sigmoid", "tanh", "clip", "unbounded"]:
        es = sorted(ep[m]); mu = [np.mean(ep[m][e]) for e in es]
        lo = [np.min(ep[m][e]) for e in es]; hi = [np.max(ep[m][e]) for e in es]
        ax.plot(es, mu, color=MODE_COLOR[m], lw=1.1, label=MODE_LABEL[m])
        ax.fill_between(es, lo, hi, color=MODE_COLOR[m], alpha=0.15, lw=0)
    if ep0.get("unbounded"):
        es = sorted(ep0["unbounded"]); mu = [np.mean(ep0["unbounded"][e]) for e in es]
        ax.plot(es, mu, color=MODE_COLOR["unbounded"], lw=1.1, ls="--", label="Unbounded linear, no weight decay")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(3.8, color="#9A9A9A", lw=0.7, ls=":")
    ax.text(20, 3.62, "previously reported drift (3.800)", ha="right", va="top", fontsize=6, color="#6B6B6B")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Effective gate scale φ(g)")
    ax.set_xlim(1, 20); ax.set_ylim(-3.2, 4.1)
    ax.set_title("Gate trajectory during BPR training", loc="left")
    ax.legend(loc="center right", frameon=False, ncol=1, bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout(w_pad=2.2)
    fig.savefig(FIG / "Figure2_gate.png", bbox_inches="tight")
    plt.close(fig)


def fig3():
    d = json.load(open(RERUN / "final" / "all_tables.json"))
    cs = d["coldstart"]
    keys = [("Most-Popular", "Most-Popular", OI["grey"], "s"),
            ("Skill-Prior-Only", "Skill-prior only", OI["orange"], "^"),
            ("BG-TopoCog(mean)", "BG-TopoCog", OI["blue"], "o")]
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 64 * MM))
    ax = axes[0]; panel(ax, "a")
    for k, lab, c, mk in keys:
        ax.scatter(cs[k]["SkillCov@10"], cs[k]["HR@10"], s=34, color=c, marker=mk, label=lab, zorder=3)
        ax.annotate(f"{lab}\nHR@10 {cs[k]['HR@10']:.3f}, cov {cs[k]['SkillCov@10']:.3f}",
                    (cs[k]["SkillCov@10"], cs[k]["HR@10"]), textcoords="offset points",
                    xytext=(6, -12) if k != "Skill-Prior-Only" else (-8, 8),
                    ha="left" if k != "Skill-Prior-Only" else "right", fontsize=6)
    ax.set_xlim(0.4, 1.06); ax.set_ylim(0.30, 0.60)
    ax.set_xlabel("Vocational skill coverage (top-10)"); ax.set_ylabel("HR@10")
    ax.set_title("Retrieval–coverage trade-off (n = 26,445)", loc="left")
    ax = axes[1]; panel(ax, "b")
    bks = ["B1_extreme", "B2_shallow", "B3_moderate", "B4_nearwarm"]
    xl = ["k = 1", "k = 2–3", "k = 4–6", "k = 7–9"]
    x = np.arange(4)
    for k, lab, c, mk in keys:
        ax.plot(x, [cs[k][b]["HR@10"] for b in bks], color=c, marker=mk, ms=4, lw=1.0, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_xlabel("Context size (seed interactions)")
    ax.set_ylabel("HR@10"); ax.set_ylim(0.30, 0.60)
    ax.set_title("HR@10 by sparsity stratum", loc="left")
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(FIG / "Figure3_coldstart.png", bbox_inches="tight")
    plt.close(fig)


def fig4():
    d = json.load(open(RERUN / "final" / "all_tables.json"))
    pl = d["planner"]
    vs = [("Full", "Full dual constraint", OI["blue"], "o", "-"),
          ("w/o Topo", "w/o topological (identical to full)", OI["orange"], "x", "--"),
          ("w/o Cog", "w/o cognitive", OI["green"], "s", "-"),
          ("Unconstrained", "Unconstrained (identical to w/o cognitive)", OI["grey"], "x", "--")]
    Ms = [5, 10, 15, 20]
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 58 * MM))
    specs = [("Bloom_Smooth", "Mean Bloom tier change", "Cognitive pacing", (0, 0.65)),
             ("SkillCov", "Skill coverage (path)", "Coverage", (0.78, 1.0)),
             ("Viol_iso", "Held-out inversion rate", "Prerequisite inversions", (0, 0.12))]
    for ax, (key, yl, tt, lim), tag in zip(axes, specs, "abc"):
        panel(ax, tag)
        for v, lab, c, mk, ls in vs:
            vals = []
            for M in Ms:
                r = next(r for r in pl if r["M"] == M and r["variant"] == v)
                val = r[key] * (100.0 / r["n_paths"]) if key == "paths_with_nonroot" else r[key]
                vals.append(val)
            ax.plot(Ms, vals, color=c, marker=mk, ms=3.5, lw=1.0, ls=ls, label=lab)
        ax.set_xticks(Ms); ax.set_xlabel("Horizon M"); ax.set_ylabel(yl); ax.set_ylim(*lim)
        ax.set_title(tt, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.8)
    fig.savefig(FIG / "Figure4_planner.png", bbox_inches="tight")
    plt.close(fig)


def fig5():
    d = json.load(open(RERUN / "final" / "all_tables.json"))
    tr = d["tracks"]
    tier_col = {3: "#DCE9F5", 4: "#A6C8E6", 5: "#5E9CD3", 6: "#1F5F9E"}
    tier_txt = {3: "black", 4: "black", 5: "white", 6: "white"}
    fig, ax = plt.subplots(figsize=(W_FULL, 66 * MM))
    nT, nS = len(tr), 10
    for ti, t in enumerate(tr):
        yy = nT - 1 - ti
        for si, p in enumerate(t["path"]):
            ax.add_patch(plt.Rectangle((si, yy + 0.06), 0.94, 0.88, fc=tier_col[p["bloom"]], ec="white", lw=0.6))
            ax.text(si + 0.47, yy + 0.56, EN.get(p["concept"], p["concept"]), ha="center", va="center",
                    fontsize=5.0, color=tier_txt[p["bloom"]], linespacing=1.1)
            ax.text(si + 0.47, yy + 0.19, f"L{p['bloom']}", ha="center", va="center", fontsize=5.4,
                    color=tier_txt[p["bloom"]], fontweight="bold")
    ax.set_xlim(0, nS); ax.set_ylim(0, nT)
    ax.set_xticks(np.arange(nS) + 0.47); ax.set_xticklabels([f"S{i+1}" for i in range(nS)])
    ax.set_yticks(np.arange(nT) + 0.5)
    ax.set_yticklabels([f"{t['track']}\ncov {t['SkillCov']:.3f}, smooth {t['Bloom_Smooth']:.3f}, inv {t['Viol_iso']:.3f}" for t in tr][::-1])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=tier_col[k]) for k in (3, 4, 5, 6)]
    ax.legend(handles, ["L3 Apply", "L4 Analyze", "L5 Evaluate", "L6 Create"], ncol=4, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Ten-step paths per vocational track (S1 is the seed concept)", loc="left")
    fig.savefig(FIG / "Figure5_tracks.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
        print("ok", f.__name__)

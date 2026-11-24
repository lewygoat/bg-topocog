# BG-TopoCog

Code, results, and manuscript for **Bounded-Gate Graph Representation and Topology–Cognitive Constrained Planning for Cold-Start Learning Path Recommendation on MOOCCubeX: A Pre-Registered Evaluation**.

Target journal: *Scientific Reports*.

## What this repository contains

The current pipeline lives in `code/` and produces every table and figure in `paper/`. The original experiment script, its results, and the first manuscript version are kept in `legacy/` for audit only. They should not be used to reproduce the paper (see [Known issues in the original script](#known-issues-in-the-original-script)).

```
.
├── paper/
│   ├── SCI_Paper_BG_TopoCog.md     manuscript source
│   └── figures_v2/                 five manuscript figures (300 dpi)
├── code/
│   ├── bggcn_train_v2.py           five gate forms × five seeds, BPR training
│   ├── coldstart_eval.py           held-out cold-start protocol
│   ├── planner_clean.py            Tarjan condensation, DAG ancestry
│   ├── rerun_all.py                Tables 5–7 and the η = 0 diagnostic
│   └── make_figures.py             Figures 1–5
├── results/
│   ├── full_v2/                    training log, per-seed results, embeddings (weight decay 1e-6)
│   ├── wd0/                        robustness run without weight decay
│   └── final/                      all_tables.json, diag_eta0.json
├── figures/                        copies of the manuscript figures
├── data/
│   ├── intermediate/               alignment, annotation, Bloom and split artefacts
│   └── README.md                   how to obtain s6_graph.pt (149 MB, not tracked)
├── docs/
│   ├── PREREGISTRATION.md          hypotheses and decision rules fixed before the rerun
│   └── RERUN_RECORD_zh.md          verification log of the rerun
└── legacy/                         original script, results, figures and manuscript v1
```

## Environment

```
Python      3.9+
torch       2.8.0
numpy       2.0.2
scipy       1.13.1
matplotlib  3.9.4
```

```bash
pip install -r requirements.txt
```

## Reproducing the paper

Place `s6_graph.pt` in `data/intermediate/` (see `data/README.md`). Paths can be overridden with `BGTC_DATA` and `BGTC_RESULTS`.

```bash
python code/bggcn_train_v2.py --modes sigmoid,tanh,clip,unbounded,none --seeds 42,43,44,45,46 --epochs 20 --lr 0.01 --tag full_v2
python code/bggcn_train_v2.py --modes unbounded,sigmoid --seeds 42,43,44,45,46 --epochs 20 --lr 0.01 --wd 0 --tag wd0
python code/rerun_all.py
python code/make_figures.py
```

Training takes roughly 20 minutes per gate form on an Apple M4 (MPS). `rerun_all.py` and `make_figures.py` take about a minute each and read the embeddings already stored in `results/full_v2/`, so Tables 5–7 and all figures can be regenerated without retraining. Running `rerun_all.py` against the released embeddings reproduces all 203 numeric entries of `results/final/all_tables.json` exactly.

| Paper item | Source |
| :--- | :--- |
| Table 3 | `data/intermediate/s7_skills*.json`, `s8_align*.json`, `code/planner_clean.py` |
| Table 4, Figure 2 | `results/full_v2/train.log`, `results/wd0/train.log` |
| Table 5, Figure 3 | `results/final/all_tables.json` → `coldstart` |
| Table 6, Figure 4 | `results/final/all_tables.json` → `planner`; `diag_eta0.json` |
| Table 7, Figure 5 | `results/final/all_tables.json` → `tracks` |

## Key results

| Hypothesis | Outcome |
| :--- | :--- |
| H1 Bounded gate ranks above unbounded gate | Not robust: sign reverses without weight decay |
| H2 Unbounded gate inflates; bounding contains it | Falsified: unbounded gate shrinks in magnitude |
| H3 Removing the DAG constraint raises inversions | Falsified: constraint inert; with η = 0 it raises inversions |
| H4 Zero inversions reflect the method | Falsified: full planner inverts 2.9–8.0% of adjacent held-out pairs |

The Bloom transition penalty is the one component with a robust positive effect, cutting mean tier change by about 90% at M = 10 for an 11% coverage cost.

## Known issues in the original script

`legacy/code_original/run_bg_topocog_experiments.py` does not train any model. It contains:

1. Literal constants in place of BG-GCN gate values and ranking metrics.
2. A cold-start context generator that draws held-out targets with probability 0.14–0.44 and feeds their nearest neighbours back as input.
3. Path-ordering routines that consult the held-out evaluation pairs, which forces the reported inversion rate to zero.
4. Hyperparameter selection on test HR@10 against a preset threshold.
5. A synthetic cold-start cohort (96.6% `U_COLD_*` identifiers).

Results produced by that script are in `legacy/results_original/` and are superseded by `results/`. Corrections to the corpus description: the three standards contain 684 clauses (260/224/200), alignment uses `bge-small-zh-v1.5` with top-20 candidates and τ = 0.5 (8,056 retained edges), not the figures stated in manuscript v1.

## License and data

Code is released for research use. `MOOCCubeX` is distributed by Tsinghua University under its own license; vocational competency standards are public national standards.

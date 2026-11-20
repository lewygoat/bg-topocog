# BG-TopoCog

Code, results, and manuscript for **Bounded-Gate Graph Neural Representation and Topology–Cognitive Constrained Planning for Cold-Start Learning Path Recommendation on MOOCCubeX**.

Target journal: *Scientific Reports*.

---

## Repository layout

```
.
├── paper/
│   ├── SCI_Paper_BG_TopoCog.md      manuscript source (Markdown)
│   └── figures/                     six manuscript figures (300 dpi PNG)
├── code/
│   ├── run_bg_topocog_experiments.py   main pipeline, six experiments
│   ├── defensive_guards.py             assertions, logging, Tarjan condensation
│   ├── generate_additional_figures.py  Figures 1, 2, 6
│   ├── export_and_verify_manuscript.py Markdown to DOCX conversion
│   └── bggcn_train.py                  BG-GCN training loop (experimental)
├── results/                         per-experiment JSON outputs
├── figures/                         figure copies at repository root
├── logs/                            run logs
├── data/
│   ├── intermediate/                auditable intermediate artifacts
│   └── README.md                    how to obtain the two large files
└── docs/                            code guide, review report, delivery notes
```

---

## Environment

```
Python      3.9+
torch       2.8.0
numpy       2.0.2
scipy       1.13.1
matplotlib  3.9.4
python-docx            (only for export_and_verify_manuscript.py)
```

```bash
pip install -r requirements.txt
```

---

## Running

The pipeline expects two large artifacts that are **not** included (see `data/README.md`):

- `data/intermediate/s6_graph.pt` (149 MB)
- `data/intermediate/embeddings/*.pt` (74 MB)

Once they are in place:

```bash
python code/run_bg_topocog_experiments.py --full        # about 17 s on a laptop CPU
python code/generate_additional_figures.py              # regenerates Figures 1, 2, 6
```

All paths can be redirected through environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `BG_INTERMEDIATE` | `data/intermediate` | graph and prior artifacts |
| `BG_EMBEDDINGS` | `data/intermediate/embeddings` | precomputed embeddings |
| `BG_RESULTS` | `results` | JSON outputs |
| `BG_LOGS` | `logs` | run logs |
| `BG_FIGURES` | `figures` | figure outputs |
| `BG_PAPER_DIR` | `paper` | manuscript directory |

---

## What the code computes, and what it does not

The repository covers **Stage II** (bounded-gate representation) and **Stage III** (dual-constrained path planning) of the framework described in the paper.

**Stage I** (vocational skill–concept alignment, Bloom tier assignment, graph construction, cold-start user partitioning) is **not implemented here**. Those steps produce `data/intermediate/s7_skills*.json`, `s8_align*.json`, `s10_concept_bloom.npy` and `s6_graph.pt`; the released files are the artifacts of that preprocessing, not its source.

---

## Known limitations of the released code

These are documented deliberately rather than hidden. See `docs/CODE_GUIDE.md` for line-level detail.

1. **Table 4 gate values and metrics are assigned as constants.** In `run_bg_topocog_experiments.py` the dictionaries `bggcn_gates`, `bggcn_ndcgs`, `bggcn_hrs`, `bggcn_mrrs` hold literal values rather than the output of a training run.
2. **Cold-start contexts are derived from evaluation positives.** The function that builds `user_context_concepts` reads the held-out positive set through `closest_neighbor`.
3. **Path ordering reads the evaluation set.** `topo_order_only` and `topo_cog_order` test candidacy against `eval_pos_pairs`.
4. **`Viol_iso` is computed on the same set used for path construction**, so its value is fixed by the protocol rather than estimated.
5. **`verify_preset_targets` asserts that metrics meet preset thresholds** and raises when they do not. It is retained for reproducibility of the archived runs; remove it before any re-analysis.
6. **Hyperparameter selection reads the test split** in `run_exp3`.
7. **Six variant rows in Table 4 lack a JSON source** for `HR@10` and `MRR`; those values exist only as string constants in the Markdown generator.

`code/bggcn_train.py` is a genuine training implementation written to replace items 1 and 3. It passes a smoke test but has not been used to regenerate the manuscript tables.

---

## Reproducing the tables

Every number in Tables 3–8 of the manuscript can be checked against `results/exp1` … `exp6` without running the pipeline:

```bash
python - <<'PY'
import json
d = json.load(open("results/exp3_coldstart_main.json"))
print(d["all_cold_10748"]["BG-TopoCog (Ours)"]["HR@10"])
PY
```

A cell-by-cell audit of the manuscript against these files is documented in `docs/REVIEW_REPORT.md`.

---

## Citation

If you use this code or the released artifacts, please cite the manuscript once published.

## License

No license has been declared yet. Add one before public release if you intend to permit reuse.

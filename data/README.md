# Data

## Included

`intermediate/` holds every artifact small enough to version normally (6.8 MB total):

| File | Contents |
|---|---|
| `s7_skills.json`, `s7_skills_rk.json`, `s7_skills_bigdata.json` | competency clauses per vocational standard |
| `s8_align*.json` | skill–concept alignment edges with similarity scores |
| `s8_align*_annotation.json` | the stratified human audit sample with rater labels |
| `s9_skill_weight*.npy` | per-concept vocational importance vectors |
| `s9_skill_weight_meta.json` | normalization metadata for the above |
| `s10_concept_bloom.npy`, `s10_concept_bloom_meta.json` | Bloom tier per concept and its construction metadata |
| `s1_eval_positive.jsonl`, `s1_eval_negative.jsonl` | held-out prerequisite evaluation pairs |
| `s4_train.jsonl.gz` | warm-start interaction split |
| `s5_coldstart.jsonl.gz` | cold-start users with their held-out targets |

## Not included

Two artifacts exceed what belongs in a git repository and are omitted.

### `intermediate/s6_graph.pt` — 149 MB

The consolidated graph object: 27,276 users, 576 concepts, train/valid/test interaction tensors totalling 1,819,473 edges, the four prerequisite relation sets, the containment map, and Bloom and skill priors keyed by concept.

It exceeds GitHub's 100 MB per-file limit. Three ways to supply it:

1. **Git LFS.** The repository already declares `*.pt filter=lfs` in `.gitattributes`. Run `git lfs install` before adding the file.
2. **External archive.** Deposit the file on Zenodo or Figshare and record the DOI here; this is the option a journal data-availability statement usually expects.
3. **Regenerate.** Set `BG_INTERMEDIATE` to a directory containing your own build of the same object. The required keys are listed in `code/run_bg_topocog_experiments.py`, function `load_and_verify_foundation`.

### `intermediate/embeddings/*.pt` — 74 MB

Fifteen files: `emb_bggcn_s{42..46}.pt` (0.3 MB each) and `emb_t1_s{n}.pt`, `emb_t3_s{n}.pt` (7 MB each of ten). They are the precomputed baseline representations consumed by Experiment 2.

## Provenance

The underlying interactions come from the public MOOCCubeX repository (Yu et al., CIKM 2021). The vocational standards are public Chinese national standards (`1+X` and the National Software Designer syllabus). Redistribution of MOOCCubeX is governed by its original licence; check that licence before making any derived file public.

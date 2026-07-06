# Caching policy — what gets re-trained, what doesn't

> 2026-06-08 cleanup note: use `paper/CANONICAL_ARTIFACTS.md` and
> `paper/CANONICAL_PAPER_CHECKLIST.md` as the current source of truth.
> Some older scripts/results named below may now live under `paper/archive/`.

The repo is split cleanly into **TRAIN scripts** (write to `paper/models/`)
and **EVAL scripts** (read from `paper/models/`, write to `paper/results/`).
Every train script is cache-aware: it skips a stage that's already on disk
unless you pass `--force-retrain`. Eval scripts never train.

## Artifacts and when they get (re)built

| Artifact | Location | When to re-build | Trigger |
|---|---|---|---|
| QC CSV | `paper/results/corpus_qc_v12_uncurated.csv` | Corpus files change OR QC rules change | `python -m paper._scan_corpus_qc` |
| v12 Stage 1 (per seed) | `paper/models/v12_gentle_seed<N>/cell_type_classifier.pkl` | Stage 1 algorithm/hyperparams change | `python -m paper._retrain_v12_gentle_seed --seed N --force-retrain` |
| v12 Stage 2 (per seed) | `paper/models/v12_gentle_seed<N>/branch_classifier.pkl` | Stage 2 algorithm/hyperparams change OR QC change | same script |
| v12 GNN (per seed) | `paper/models/v12_gentle_seed<N>/gnn_apical_basal.pt` | GNN algorithm/hyperparams change OR QC change | same script |
| Branch3 rescue GNN | `paper/models/v12_gentle_seed123/gnn_branch3_rescue.pt` | Branch3 architecture/features change OR Stage 2/GNN predictions change | `python -m paper.gnn_branch3_rescue --model-dir paper/models/v12_gentle_seed123 --seed 123` |
| Branch3 graph cache | `paper/models/v12_gentle_seed123/branch3_cache/` | Branch3 features, Stage 2 model, or old GNN checkpoint change | delete cache dir before rerun |
| Branch3 learned gate | `paper/models/v12_gentle_seed123/branch3_gate.joblib` | Branch3 checkpoint/cache changes OR gate features change | `python -m paper.train_branch3_gate --model-dir paper/models/v12_gentle_seed123` |
| Branch3 flag dataset | `paper/results/heldout_per_cell_f1_seed123_branch3.csv`, `paper/results/flag_confidence_features_seed123_branch3.csv` | Accepted labeler changes OR heldout split changes | `python -m paper._score_model_split_for_flag --model-dir paper/models/v12_gentle_seed123 --split test --out-labels paper/results/heldout_per_cell_f1_seed123_branch3.csv --out-features paper/results/flag_confidence_features_seed123_branch3.csv --out-summary paper/results/flag_seed123_branch3_dataset_summary.json` |
| Current flag model | `paper/models/flag_model_seed123_branch3_pyramidal/flag_model_f060.joblib` | Branch3 flag dataset or flag features change | `python -m paper._train_flag_model --input paper/results/heldout_per_cell_f1_seed123_branch3.csv --confidence-features paper/results/flag_confidence_features_seed123_branch3.csv --stage1-disagreement paper/results/stage1_disagreement_seed123_missing.csv --model-dir paper/models/flag_model_seed123_branch3_pyramidal --out-stem paper/results/flag_model_seed123_branch3_pyramidal_eval --selection-reject-rate 0.01 --reject-rates 0.005,0.01,0.02,0.03,0.05 --precision-targets 0.6,0.7,0.8,0.9 --target-thresholds 0.5,0.6 --cell-type-filter pyramidal --seed 2026` |
| v12 QC gate (per seed) | `paper/models/v12_gentle_seed<N>/qc_gate.pkl` | Stage 1 features change OR QC change | same script |
| Baseline models | `paper/models/baselines/<method>.pkl` | QC change OR baseline algorithm change | `python -m paper._eval_baselines_on_v12 --force-retrain` |

## Train scripts (idempotent — skip cached stages)

| Script | Trains | Caching behavior |
|---|---|---|
| `paper/_scan_corpus_qc.py` | QC CSV | Overwrites — always re-scans |
| `paper/_retrain_v12_gentle_seed.py --seed N` | Stage 1 + Stage 2 + GNN + OOD/QC gate | Skips each stage whose artifact exists. `--force-retrain` overrides. |
| `paper/gnn_branch3_rescue.py --model-dir <dir>` | Branch3 rescue head only | Reuses cached Branch3 graphs if present; writes `gnn_branch3_rescue.pt`. |
| `paper/train_branch3_gate.py --model-dir <dir>` | Optional Branch3 accept/abstain gate | Reuses Branch3 graph cache; writes `branch3_gate.joblib`. Current artifact is opt-in only, not production default. |
| `paper/_purge_non_qc_result_rows.py` | Result CSV cleanup | Filters result CSVs with a `file` column to current `qc_pass=True` corpus. |
| `paper/_score_model_split_for_flag.py --model-dir <dir> --split test` | Current flag labels + features | Scores a no-leak model split with the accepted conservative-Branch3 labeler and writes Branch3-aligned flag inputs. |
| `paper/_train_flag_model.py` | Current flag model | Trains rank model + calibrated display probability. Use the Branch3 files above for the accepted labeler. |

To re-train ONE stage only (e.g., GNN), delete just that file and re-run:
```
rm paper/models/v12_gentle_seed42/gnn_apical_basal.pt
python -m paper._retrain_v12_gentle_seed --seed 42
# Skips Stage 1, Stage 2, OOD (cached). Re-trains only the GNN.
```

## Eval scripts (pure inference, no training, ever)

| Script | What it evaluates | Reads | Writes |
|---|---|---|---|
| `paper/_eval_per_seed_own_test.py` | All v12 seeds on their own held-out test sets, full pipeline (Stage 1 → 2 → 3) | Stage 1 + 2 + GNN per seed | `paper/results/per_seed_own_test.json` |
| `paper/_eval_v12_gt_celltype.py --seed N` | v12 seed=N with GT cell-type override (Stage 1 bypassed) | Stage 1 + 2 + GNN for seed N | `paper/results/v12_gt_celltype_seed<N>.json` |
| `paper/_eval_baselines_on_v12.py --seed N` | 4 external baselines on the same QC hash split as v12 seed=N | Cached baseline `.pkl` files (or trains on first run) | `paper/results/baselines_on_v12.json` |
| `paper/_eval_ensemble_full.py` | 4-seed v12 ensemble + per-branch flag analysis | Stage 1 + 2 + GNN for all 4 seeds | `paper/results/ensemble_eval_full.json` |
| `paper/_compile_comparison.py` | Stitches `per_seed_own_test.json` + `baselines_on_v12.json` into a single markdown table | both JSONs | `paper/v12_vs_baselines_comparison.md` |

## What you DON'T need to re-run on a v12 algorithm change

If you change Stage 2 hyperparameters and want to re-measure v12:
- ✓ Re-train v12 stage 2 (per seed)
- ✓ Re-train v12 GNN per seed if Stage 2 features change (e.g., subtree-owner)
- ✗ Don't re-run QC (corpus didn't change)
- ✗ Don't re-train baselines (they don't depend on v12)
- ✗ Don't re-train Stage 1 (cell-type prediction is independent of Stage 2)

Concretely:
```
# Change Stage 2 code, then:
rm paper/models/v12_gentle_seed42/branch_classifier.pkl   # invalidate cache for this stage
rm paper/models/v12_gentle_seed42/gnn_apical_basal.pt     # GNN depends on Stage 2 subtree-owner
python -m paper._retrain_v12_gentle_seed --seed 42        # re-trains Stage 2 + GNN only
python -m paper._eval_per_seed_own_test                   # re-evaluates v12
python -m paper._compile_comparison                       # re-builds comparison MD with new v12
# Baselines and QC are untouched — saves ~2.5 hours per iteration
```

## Naming convention

- Anything in `paper/models/` is a **trained artifact** — load only, don't recompute.
- Anything in `paper/results/` is a **measurement** — re-runnable from artifacts.
- TRAIN scripts may write to either. EVAL scripts write only to `paper/results/`.

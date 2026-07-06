# Model backup — 2026-05-24

**DO NOT MODIFY.** This directory is a frozen snapshot of all trained models
at a known-good state. Use it to restore if anything in `paper/models/` gets
corrupted, deleted, or accidentally retrained.

## Contents

### baselines/
| File | Size | Notes |
|---|---:|---|
| `neurom_rf.pkl` | 8.58 GB | NeuroM-style per-branch features → sklearn RF (n_estimators=400) |
| `sholl_rf.pkl` | 255 MB | Sholl-derived per-subtree features → RF |
| `lmeasure_rf.pkl` | 205 MB | L-Measure-style per-subtree features → RF (best baseline) |
| `sholl_mlp.pkl` | 53.9 KB | Same Sholl features → small sklearn MLP |

Trained on seed=42's 80% (10,016 cells of QC-passed corpus).
Evaluated on seed=42's 20% (2,468 cells).
Used to produce `paper/results/baselines_on_v12.json`.

### v12_gentle_seed42/
| File | Notes |
|---|---|
| `cell_type_classifier.pkl` | Stage 1 (XGB cell-type classifier) |
| `branch_classifier.pkl` | Stage 2 (per-cell-type branch RF bundle) |
| `gnn_apical_basal.pt` | Stage 3 GNN (GraphSAGE apical-vs-basal head). Held-out branch macro F1 = 0.9404 |
| `qc_gate.pkl` | OOD detector + structural QC checks |
| `train_test_split.json` | Seed=42 hash-bucket split (which files in train vs test) |
| `eval_split_for_gnn.json` | Subset of split used during GNN training |
| `training_metrics.json` | Training timing + counts |

### v12_gentle_seed789/
Same structure as seed=42 but for the seed=789 hash-bucket split.
Held-out branch macro F1 = 0.8881 (slightly lower than seed=42 due to seed variance).

## How to restore

If anything in `paper/models/` is damaged:

```bash
cd D:\Desktop\swc-autolabel-ml
cp -p paper/models_backup_2026_05_24/baselines/*           paper/models/baselines/
cp -p paper/models_backup_2026_05_24/v12_gentle_seed42/*   paper/models/v12_gentle_seed42/
cp -p paper/models_backup_2026_05_24/v12_gentle_seed789/*  paper/models/v12_gentle_seed789/
```

## Disk size

Total backup ≈ 8.7 GB (dominated by neurom_rf.pkl at 8.58 GB).

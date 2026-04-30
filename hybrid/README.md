# Hybrid Auto-Labeling Pipeline

ML + rule-based hybrid pipeline for per-node neuron morphology labeling
(soma, axon, basal dendrite, apical dendrite) across pyramidal and
interneuron cells.

## Architecture

```
Stage 1: Cell-type detection (whole-cell ML, 49 features) → pyramidal | interneuron
Stage 2: Per-branch classifier  (cell-type-conditioned ML, 57 features) → axon | basal | apical
Stage 3: Topology refinement    (rules) → enforce one-apical, subtree consistency, etc.
```

When Stage 1's confidence falls below 0.65, the pipeline runs Stage 2+3 for
**both** cell types and picks whichever produces sharper predictions
("soft handoff").

## Layout

| Module | Role |
|---|---|
| `features.py`            | Stage 1 whole-cell feature extraction |
| `branch_features.py`     | Stage 2 per-branch features |
| `subtree_features.py`    | Primary-subtree features |
| `cell_type_detector.py`  | Stage 1 logic |
| `stage3_refine.py`       | Stage 3 topology rules |
| `pipeline.py`            | End-to-end orchestrator (with soft handoff) |
| `train_stage1.py`        | Train Stage 1 cell-type classifier |
| `train_stage2.py`        | Train Stage 2 per-branch classifier |
| `evaluate.py`            | Full Stage 1+2+3 evaluation with clean train/test split |
| `evaluate_stage1.py`     | Stage 1 only — deeper recurring-error analysis |
| `qc.py`                  | Dataset QC toolbox: diagnose / clean-soma / filter / prune |
| `data_prep.py`           | Data prep toolbox: download / build-source-pool / build-benchmark |

Paper deliverables (baselines, slides, speaker notes) live in **`paper/`**, a sibling directory. See `paper/__init__.py` for the layout.

## Quick start

### Build / clean a dataset (one-time)

```bash
# Download raw morphologies
python -m hybrid.data_prep download-allen
python -m hybrid.data_prep download-neuromorpho --types purkinje granule interneuron

# Filter to a usable source pool (canonical labels, soma required)
python -m hybrid.data_prep build-source-pool

# Assemble the benchmark
python -m hybrid.data_prep build-benchmark --per-type 1000

# Soma cleanup + diagnostic-driven pruning
python -m hybrid.qc clean-soma
python -m hybrid.qc prune --source data/benchmark_pyramidal_interneuron_v1_soma_clean \
                          --scores hybrid/models/per_file_scores.csv \
                          --out   data/benchmark_pyramidal_interneuron_v1_qc_diag_pruned \
                          --apply
```

### Diagnose a low-F1 file

```bash
python -m hybrid.qc diagnose path/to/file.swc
python -m hybrid.qc diagnose --from-csv hybrid/models/per_file_scores.csv --top 15
```

### Train and evaluate

```bash
# Default --data-dir is data/benchmark_pyramidal_interneuron_v1_qc_diag_pruned
python -m hybrid.train_stage1
python -m hybrid.train_stage2
python -m hybrid.evaluate
```

`evaluate.py` does its own internal train/test split and trains fresh
models in `hybrid/models/eval_tmp/` to avoid data leakage. The scripts
above (`train_stage1`, `train_stage2`) are for shipping production
models trained on all data.

## Latest results

See `hybrid/models/evaluation_results.json` for the full numbers.

| Metric                 | Stage 2+3 |
|------------------------|-----------|
| Overall accuracy       | 0.992     |
| Pooled F1              | 0.976     |
| Per-file mean F1       | 0.963     |
| Per-file P10           | 0.901     |
| Apical F1              | 0.937     |
| Basal F1               | 0.972     |

## Public Python API

```python
# Pipeline
from hybrid.pipeline import run_pipeline_on_nodes

# QC
from hybrid.qc import diagnose, clean_benchmark, filter_benchmark, prune_benchmark

# Data prep
from hybrid.data_prep import (
    download_allen, download_neuromorpho,
    build_source_pool, build_benchmark,
)
```

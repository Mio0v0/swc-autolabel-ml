# Final Multi-Seed Training Run

Started: 2026-06-08

Purpose: build fresh final-paper v12+Branch3 artifacts for seeds 42 and 789,
matching the accepted seed123 recipe on the cleaned QC corpus.

Runner:

- `paper/_run_final_multiseed_training.ps1`
- background PowerShell PID: `239060`
- status log: `paper/results/logs/final_multiseed_training.status.log`

Sequence:

1. `python -m paper._retrain_v12_gentle_seed --seed 42`
2. `python -m paper.gnn_branch3_rescue --model-dir paper/models/v12_gentle_seed42 --ckpt paper/models/v12_gentle_seed42/gnn_branch3_rescue.pt --seed 42`
3. `python -m paper._retrain_v12_gentle_seed --seed 789`
4. `python -m paper.gnn_branch3_rescue --model-dir paper/models/v12_gentle_seed789 --ckpt paper/models/v12_gentle_seed789/gnn_branch3_rescue.pt --seed 789`

Logs:

- `paper/results/logs/final_multiseed_train_seed42.log`
- `paper/results/logs/final_multiseed_branch3_seed42.log`
- `paper/results/logs/final_multiseed_train_seed789.log`
- `paper/results/logs/final_multiseed_branch3_seed789.log`

Expected final artifacts:

- `paper/models/v12_gentle_seed42/cell_type_classifier.pkl`
- `paper/models/v12_gentle_seed42/branch_classifier.pkl`
- `paper/models/v12_gentle_seed42/gnn_apical_basal.pt`
- `paper/models/v12_gentle_seed42/gnn_branch3_rescue.pt`
- `paper/models/v12_gentle_seed42/qc_gate.pkl`
- `paper/models/v12_gentle_seed42/train_test_split.json`
- `paper/models/v12_gentle_seed789/cell_type_classifier.pkl`
- `paper/models/v12_gentle_seed789/branch_classifier.pkl`
- `paper/models/v12_gentle_seed789/gnn_apical_basal.pt`
- `paper/models/v12_gentle_seed789/gnn_branch3_rescue.pt`
- `paper/models/v12_gentle_seed789/qc_gate.pkl`
- `paper/models/v12_gentle_seed789/train_test_split.json`

Previous seed42/seed789 folders were archived before this run:

- `paper/archive/models/v12_gentle_seed42_pre_final_20260608_145324/`
- `paper/archive/models/v12_gentle_seed789_pre_final_20260608_145324/`


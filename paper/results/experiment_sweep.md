# Wave A+B experiment sweep results

Baseline reference: **v12 (Stage 2+3, GT cell-type override) on cleaned 11862-corpus, 4263 pyramidal test cells, seed 2024**.
All experiments use the same split.

## Results table

| Experiment | suffix | Per-node acc | Apical F1 corpus | Basal F1 corpus | F1 mean | **F1 P10** | apical_F1 mean | basal_F1 mean | Δ P10 (pp) | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| **v12 baseline** | `_clean` | 0.9814 | 0.9008 | 0.9404 | 0.9148 | **0.5908** | 0.8241 | 0.9491 | — | reference |
| **Branch3 rescue** | `_branch3` | 0.9823 | 0.9081 | 0.9437 | 0.9209 | **0.6214** | 0.8481 | 0.9517 | **+3.06** | **KEEP** |
| Branch3 loose gate | `_branch3_loose` | 0.9825 | 0.9075 | 0.9432 | 0.9243 | **0.6377** | 0.8660 | 0.9519 | **+4.69** | optional; more regressions |
| Branch3 learned gate | `_branch3_gate` | 0.9843 | 0.9117 | 0.9482 | 0.9283 | **0.6463** | 0.8663 | 0.9556 | **+5.55** | opt-in only; too many good-cell regressions |
| L-Measure-RF | (separate) | 0.9830 | 0.9032 | 0.9414 | 0.8893 | 0.4921 | 0.7694 | 0.9438 | −9.87 | reference |
| A3 balance=1.0 | `_balance1` | 0.9818 | 0.9053 | 0.9432 | 0.9151 | **0.5770** | 0.8175 | 0.9496 | **−1.38** | **REVERT** |
| A4 contractionOnly | `_contractionOnly` | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| B1 GAT | `_gat` | — | — | — | — | — | — | — | — | not yet launched |
| B2 hard-cells 2-pass | `_hardcells` | — | — | — | — | — | — | — | — | not yet launched |
| B3 GNN curriculum | `_curriculum` | — | — | — | — | — | — | — | — | needs code |
| B4 topology-aware GNN loss | `_topoloss` | — | — | — | — | — | — | — | — | needs code |
| B5 per-cell rare-apical weight | `_rareweight` | — | — | — | — | — | — | — | — | needs code |

## Decision rule

**Keep** if Δ P10 > +0.5 pp AND apical_F1 mean ≥ baseline.
**Revert** otherwise.

## Reference baselines

- `v12 baseline`: comprehensive_metrics.build_full_report on v12_pyramidal_only_seed2024_clean with override_cell_type='pyramidal'
- `L-Measure-RF`: same split, lmeasure_rf baseline trained pyramidal-only

## Comparison files

- `paper/results/v12_pyramidal_only_seed2024_clean_eval.{json,csv,txt}`
- `paper/results/v12_pyramidal_only_seed2024_clean_branch3_eval.{json,csv,txt}`
- `paper/results/v12_pyramidal_only_seed2024_clean_branch3_loose_eval.{json,csv,txt}`
- `paper/results/v12_pyramidal_only_seed2024_clean_branch3_gate_eval.{json,csv,txt}`
- `paper/results/lmeasure_pyramidal_only_eval.{json,csv,txt}`
- `paper/results/comparison_v12_vs_lmeasure.{json,txt}`
- `paper/results/rejection_sweep.{json,txt}` (Wave A1)

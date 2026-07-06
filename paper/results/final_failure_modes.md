# Final failure-mode analysis

Input: `paper\results\final_flag_multiseed_labels.csv`
Rows: 7023 row-seed pairs, 5739 unique files
Bad target: per-cell neurite F1 < 0.60
Bottom slice: lowest 10.0% = 703 rows, F1 <= 0.6620

Flag catch-rate columns use LOSO `baseline_oof` flagger, scope `all`, fixed reject rate 0.10.

## Category summary

| Category | Bottom rows | Bottom bad rows | Unique bad files | Enrichment | Flagged bad | Flag recall in category | False flags in category |
|---|---:|---:|---:|---:|---:|---:|---:|
| APICAL_OVERPREDICTED | 15 | 15 | 13 | 9.99x | 0 | 0.000 | 0 |
| STAGE1_PROPAGATION | 89 | 85 | 78 | 9.77x | 54 | 0.635 | 1 |
| APICAL_UNDERPREDICTED | 287 | 265 | 234 | 9.22x | 185 | 0.698 | 8 |
| APICAL_MISSED | 280 | 261 | 232 | 9.20x | 185 | 0.709 | 7 |
| BASAL_LOST | 131 | 123 | 103 | 8.96x | 63 | 0.512 | 3 |
| AXON_DENDRITE_CONFUSION | 331 | 214 | 185 | 8.96x | 113 | 0.528 | 16 |
| WHOLE_CELL_MISLABEL | 12 | 11 | 10 | 4.61x | 3 | 0.273 | 6 |
| TRUNCATED_APICAL | 118 | 108 | 96 | 2.93x | 67 | 0.620 | 21 |
| APICAL_INVERTED | 218 | 158 | 138 | 1.60x | 95 | 0.601 | 34 |
| DISCONNECTED | 0 | 0 | 0 | 0.00x | 0 | 0.000 | 0 |

## Worst examples

### WHOLE_CELL_MISLABEL

| file | seed | cell type | F1 | axon F1 | basal F1 | apical F1 | flagged | other categories |
|---|---:|---|---:|---:|---:|---:|---|---|
| `neuromorpho__KO1-4-3-9.swc` | 123 | pyramidal | 0.2047 | 0.0000 | 0.6141 | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__KO1-4-3-9.swc` | 789 | pyramidal | 0.2047 | 0.0000 | 0.6141 | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__KO4-4-5-1.swc` | 123 | pyramidal | 0.2480 | 0.0000 | 0.7441 | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, STAGE1_PROPAGATION |
| `neuromorpho__april9s1-cell-1.swc` | 123 | pyramidal | 0.2969 | - | 0.5938 | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__20180109C01.swc` | 123 | pyramidal | 0.3244 | - | 0.6488 | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, STAGE1_PROPAGATION |
| `neuromorpho__Mouse-CA2-Ma-Pyramidal-Cell.swc` | 789 | pyramidal | 0.3536 | - | 0.7073 | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, STAGE1_PROPAGATION |
| `neuromorpho__d3s2-cell-2.swc` | 123 | pyramidal | 0.3656 | - | 0.7313 | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__d2s2-cell-3.swc` | 123 | pyramidal | 0.4256 | - | 0.8512 | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__p120105-01-AAC.swc` | 123 | interneuron | 0.4480 | - | 0.4480 | - | no | BASAL_LOST, STAGE1_PROPAGATION |
| `allen__H16.06.007.01.07.04_531526539.swc` | 789 | pyramidal | 0.4781 | 1.0000 | 0.4343 | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST, STAGE1_PROPAGATION |

### APICAL_MISSED

| file | seed | cell type | F1 | axon F1 | basal F1 | apical F1 | flagged | other categories |
|---|---:|---|---:|---:|---:|---:|---|---|
| `neuromorpho__150420_2.swc` | 123 | interneuron | 0.0000 | - | - | 0.0000 | yes | APICAL_UNDERPREDICTED, APICAL_INVERTED |
| `neuromorpho__WT3-neuron5.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | no | APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__WT3-neuron1.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__WT2-Neuron-9.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__WT2-Neuron-8.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__C1054-whole-3.swc` | 42 | pyramidal | 0.0000 | - | - | 0.0000 | yes | APICAL_UNDERPREDICTED, APICAL_INVERTED |
| `neuromorpho__KO1-Neuron5.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__C1005-2KO-apical-2.swc` | 123 | pyramidal | 0.0000 | - | - | 0.0000 | yes | APICAL_UNDERPREDICTED, TRUNCATED_APICAL |
| `neuromorpho__10month-Arctic-C5aR1-KO-52280-LR_4.swc` | 789 | pyramidal | 0.0000 | - | - | 0.0000 | yes | APICAL_UNDERPREDICTED, APICAL_INVERTED, TRUNCATED_APICAL |
| `neuromorpho__KO1-Neuron7.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL, STAGE1_PROPAGATION |

### APICAL_UNDERPREDICTED

| file | seed | cell type | F1 | axon F1 | basal F1 | apical F1 | flagged | other categories |
|---|---:|---|---:|---:|---:|---:|---|---|
| `neuromorpho__WT3-neuron1.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__10month-Arctic-C5aR1-KO-52280-LR_4.swc` | 789 | pyramidal | 0.0000 | - | - | 0.0000 | yes | APICAL_MISSED, APICAL_INVERTED, TRUNCATED_APICAL |
| `neuromorpho__WT2-Neuron-9.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__150420_2.swc` | 123 | interneuron | 0.0000 | - | - | 0.0000 | yes | APICAL_MISSED, APICAL_INVERTED |
| `neuromorpho__C1005-2KO-apical-2.swc` | 123 | pyramidal | 0.0000 | - | - | 0.0000 | yes | APICAL_MISSED, TRUNCATED_APICAL |
| `neuromorpho__KO1-Neuron5.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__KO1-Neuron7.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__C1054-whole-3.swc` | 42 | pyramidal | 0.0000 | - | - | 0.0000 | yes | APICAL_MISSED, APICAL_INVERTED |
| `neuromorpho__sc17.swc` | 123 | interneuron | 0.0000 | - | - | 0.0000 | no | APICAL_MISSED, TRUNCATED_APICAL |
| `neuromorpho__WT2-Neuron-8.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |

### APICAL_OVERPREDICTED

| file | seed | cell type | F1 | axon F1 | basal F1 | apical F1 | flagged | other categories |
|---|---:|---|---:|---:|---:|---:|---|---|
| `neuromorpho__241-2-59LU.swc` | 123 | pyramidal | 0.1120 | - | 0.2239 | 0.0000 | no | BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__iPRCp_09.swc` | 123 | pyramidal | 0.1120 | - | 0.2239 | 0.0000 | no | BASAL_LOST |
| `neuromorpho__242-2-1VN.swc` | 123 | pyramidal | 0.1798 | 0.0000 | 0.5393 | 0.0000 | no | AXON_DENDRITE_CONFUSION |
| `neuromorpho__X046_S1_Z2_C05.swc` | 42 | pyramidal | 0.1934 | - | 0.3868 | 0.0000 | no | APICAL_INVERTED, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__X046_S1_Z2_C05.swc` | 789 | pyramidal | 0.1934 | - | 0.3868 | 0.0000 | no | APICAL_INVERTED, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__C4_3.swc` | 789 | pyramidal | 0.2126 | 0.0000 | 0.6377 | 0.0000 | no | AXON_DENDRITE_CONFUSION |
| `neuromorpho__C10-28.swc` | 123 | pyramidal | 0.2500 | - | 0.5000 | 0.0000 | no | APICAL_INVERTED, TRUNCATED_APICAL |
| `neuromorpho__mPFC_w1_L3_01.swc` | 123 | pyramidal | 0.2557 | 0.0000 | 0.7670 | 0.0000 | no | AXON_DENDRITE_CONFUSION |
| `neuromorpho__mPFC_w1_L3_01.swc` | 789 | pyramidal | 0.2570 | 0.0000 | 0.7711 | 0.0000 | no | AXON_DENDRITE_CONFUSION |
| `neuromorpho__X043_S5_Z2_C05_20x.swc` | 789 | pyramidal | 0.2688 | - | 0.5377 | 0.0000 | no | APICAL_INVERTED, TRUNCATED_APICAL |

### APICAL_INVERTED

| file | seed | cell type | F1 | axon F1 | basal F1 | apical F1 | flagged | other categories |
|---|---:|---|---:|---:|---:|---:|---|---|
| `neuromorpho__WT2-Neuron-9.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__C1054-whole-3.swc` | 42 | pyramidal | 0.0000 | - | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED |
| `neuromorpho__150420_2.swc` | 123 | interneuron | 0.0000 | - | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED |
| `neuromorpho__KO1-Neuron7.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__WT3-neuron5.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__10month-Arctic-C5aR1-KO-52280-LR_4.swc` | 789 | pyramidal | 0.0000 | - | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, TRUNCATED_APICAL |
| `neuromorpho__WT2-Neuron-12.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__Nov-9-IR-1-4d-40x-low-res.swc` | 42 | pyramidal | 0.1026 | - | 0.2053 | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__241-1-22-AW.swc` | 42 | pyramidal | 0.1354 | 0.0000 | 0.4062 | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__WT4-neuron7.swc` | 123 | pyramidal | 0.1584 | 0.0000 | 0.4753 | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, BASAL_LOST, TRUNCATED_APICAL, STAGE1_PROPAGATION |

### AXON_DENDRITE_CONFUSION

| file | seed | cell type | F1 | axon F1 | basal F1 | apical F1 | flagged | other categories |
|---|---:|---|---:|---:|---:|---:|---|---|
| `neuromorpho__KO1-Neuron7.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__WT3-neuron5.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, TRUNCATED_APICAL |
| `neuromorpho__WT3-neuron1.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, TRUNCATED_APICAL |
| `neuromorpho__WT2-Neuron-9.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, TRUNCATED_APICAL |
| `neuromorpho__WT2-Neuron-8.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, TRUNCATED_APICAL |
| `neuromorpho__WT2-Neuron-12.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__KO1-Neuron5.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__Kole_MPG05042019_3_StreptavidinA488_63x.swc` | 789 | interneuron | 0.0000 | 0.0000 | - | - | yes | - |
| `neuromorpho__Cell16.swc` | 123 | interneuron | 0.0128 | 0.0000 | 0.0255 | - | yes | BASAL_LOST |
| `neuromorpho__C2_5.swc` | 789 | pyramidal | 0.0582 | 0.0000 | 0.1745 | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST, STAGE1_PROPAGATION |

### BASAL_LOST

| file | seed | cell type | F1 | axon F1 | basal F1 | apical F1 | flagged | other categories |
|---|---:|---|---:|---:|---:|---:|---|---|
| `neuromorpho__170321_1.swc` | 42 | interneuron | 0.0000 | - | 0.0000 | 0.0000 | no | - |
| `neuromorpho__Red_4weeks_Neuron9.swc` | 123 | pyramidal | 0.0000 | - | 0.0000 | - | yes | - |
| `neuromorpho__GGN_20170309_sc.swc` | 789 | interneuron | 0.0000 | - | 0.0000 | - | no | - |
| `neuromorpho__hippo-104-6.swc` | 42 | pyramidal | 0.0000 | - | 0.0000 | - | no | - |
| `neuromorpho__GGN_20170309_sc.swc` | 123 | interneuron | 0.0000 | - | 0.0000 | - | no | - |
| `neuromorpho__C912-WT-basal-2.swc` | 123 | pyramidal | 0.0000 | - | 0.0000 | - | yes | - |
| `neuromorpho__C912-WT-basal-2.swc` | 42 | pyramidal | 0.0000 | - | 0.0000 | - | yes | - |
| `neuromorpho__hippo-104-6.swc` | 123 | pyramidal | 0.0000 | - | 0.0000 | - | no | - |
| `neuromorpho__hippo-104-6.swc` | 789 | pyramidal | 0.0000 | - | 0.0000 | - | no | - |
| `neuromorpho__38a.swc` | 123 | pyramidal | 0.0000 | - | 0.0000 | - | yes | - |

### TRUNCATED_APICAL

| file | seed | cell type | F1 | axon F1 | basal F1 | apical F1 | flagged | other categories |
|---|---:|---|---:|---:|---:|---:|---|---|
| `neuromorpho__WT3-neuron5.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION |
| `neuromorpho__KO1-Neuron7.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, STAGE1_PROPAGATION |
| `neuromorpho__KO1-Neuron5.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, STAGE1_PROPAGATION |
| `neuromorpho__10month-Arctic-C5aR1-KO-52280-LR_4.swc` | 789 | pyramidal | 0.0000 | - | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED |
| `neuromorpho__WT3-neuron1.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION |
| `neuromorpho__WT2-Neuron-12.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, STAGE1_PROPAGATION |
| `neuromorpho__C1005-2KO-apical-2.swc` | 123 | pyramidal | 0.0000 | - | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED |
| `neuromorpho__WT2-Neuron-9.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION |
| `neuromorpho__WT2-Neuron-8.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION |
| `neuromorpho__sc17.swc` | 123 | interneuron | 0.0000 | - | - | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED |

### STAGE1_PROPAGATION

| file | seed | cell type | F1 | axon F1 | basal F1 | apical F1 | flagged | other categories |
|---|---:|---|---:|---:|---:|---:|---|---|
| `neuromorpho__WT2-Neuron-12.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__KO1-Neuron7.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__KO1-Neuron5.swc` | 123 | pyramidal | 0.0000 | 0.0000 | - | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__C2_5.swc` | 789 | pyramidal | 0.0582 | 0.0000 | 0.1745 | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__PDE1c_vGluT2-ctrl_IN08.swc` | 42 | interneuron | 0.0674 | - | 0.0674 | - | yes | BASAL_LOST |
| `neuromorpho__11220c3.swc` | 42 | pyramidal | 0.0911 | - | 0.1822 | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__Nov-9-IR-1-4d-40x-low-res.swc` | 42 | pyramidal | 0.1026 | - | 0.2053 | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__389-02-04.swc` | 123 | pyramidal | 0.1153 | 0.0000 | 0.3458 | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__20180109B01.swc` | 42 | pyramidal | 0.1174 | 0.0000 | 0.3521 | 0.0000 | yes | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__389-01-05.swc` | 123 | pyramidal | 0.1178 | 0.0000 | 0.3535 | 0.0000 | no | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, BASAL_LOST |

## Unmatched bad rows

32 bad rows matched no category.

| file | seed | cell type | F1 | axon F1 | basal F1 | apical F1 | flagged |
|---|---:|---|---:|---:|---:|---:|---|
| `neuromorpho__iPRCp_05.swc` | 42 | pyramidal | 0.2562 | - | 0.5125 | 0.0000 | no |
| `neuromorpho__het_1_51.swc` | 42 | pyramidal | 0.5050 | - | 0.5050 | - | yes |
| `neuromorpho__11C2neuron4.swc` | 123 | pyramidal | 0.5094 | - | 0.5094 | - | yes |
| `neuromorpho__B2-F3.swc` | 123 | interneuron | 0.5128 | - | 0.5128 | - | yes |
| `neuromorpho__Green_6weeks_Neuron2_addl.swc` | 42 | pyramidal | 0.5155 | - | 0.5155 | - | yes |
| `neuromorpho__HIP_CA1_KO_13.swc` | 42 | pyramidal | 0.5205 | - | 0.5205 | - | yes |
| `neuromorpho__het_14_33.swc` | 123 | pyramidal | 0.5320 | - | 0.5320 | - | yes |
| `neuromorpho__3_den_ax.swc` | 123 | pyramidal | 0.5337 | 0.9997 | 0.6014 | 0.0000 | no |
| `neuromorpho__Green_4weeks_Neuron1.swc` | 42 | pyramidal | 0.5376 | - | 0.5376 | - | yes |
| `neuromorpho__het_9-2.swc` | 123 | pyramidal | 0.5473 | - | 0.5473 | - | yes |
| `neuromorpho__HIP_CA1_WT_2.swc` | 789 | pyramidal | 0.5478 | - | 0.5478 | - | no |
| `neuromorpho__HIP_CA1_WT_2.swc` | 123 | pyramidal | 0.5478 | - | 0.5478 | - | yes |
| `neuromorpho__SW150113-01RabiesPHALPFC_x772_y876_z296_8.swc` | 42 | pyramidal | 0.5485 | - | 0.5485 | - | no |
| `neuromorpho__crop0_1_seg_split_2_1-55.swc` | 789 | pyramidal | 0.5501 | - | 0.5501 | - | yes |
| `neuromorpho__1103-1-Sec3-BSL1.swc` | 123 | pyramidal | 0.5524 | - | 0.5524 | - | yes |
| `neuromorpho__C912-WT-basla-1.swc` | 123 | pyramidal | 0.5574 | - | 0.5574 | - | yes |
| `neuromorpho__NAS14.swc` | 789 | interneuron | 0.5585 | - | 0.5585 | - | no |
| `neuromorpho__Exp_V1B-L5_cell4.swc` | 123 | pyramidal | 0.5613 | - | 0.5613 | - | yes |
| `neuromorpho__HIP_CA1_KO_14.swc` | 42 | pyramidal | 0.5714 | - | 0.5714 | - | yes |
| `neuromorpho__het_7_40.swc` | 123 | pyramidal | 0.5767 | - | 0.5767 | - | yes |

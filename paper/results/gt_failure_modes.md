# GT failure-mode taxonomy

**Input**: `paper\results\heldout_per_cell_f1.csv` (4358 held-out cells)
**Bottom slice**: lowest 10% by per-cell F1 = **435 cells** (F1 < 0.5912)

**By cell type in bottom slice**: 381 pyramidals, 54 interneurons

## Category summary

| Category | Bottom-10% count (pyr / int) | Full-corpus count | Bottom enrichment |
|---|---|---|---|
| **WHOLE_CELL_MISLABEL** | 1  (1 pyr / 0 int) | 8 | 1.3x |
| **APICAL_MISSED** | 264  (262 pyr / 2 int) | 300 | 8.8x :rocket: |
| **APICAL_UNDERPREDICTED** | 276  (274 pyr / 2 int) | 312 | 8.9x :rocket: |
| **APICAL_OVERPREDICTED** | 9  (9 pyr / 0 int) | 9 | 10.0x :rocket: |
| **APICAL_INVERTED** | 149  (149 pyr / 0 int) | 897 | 1.7x |
| **AXON_DENDRITE_CONFUSION** | 142  (101 pyr / 41 int) | 239 | 6.0x :rocket: |
| **BASAL_LOST** | 90  (72 pyr / 18 int) | 105 | 8.6x :rocket: |
| **DISCONNECTED** | 0  (0 pyr / 0 int) | 0 | infx :rocket: |
| **TRUNCATED_APICAL** | 125  (123 pyr / 2 int) | 316 | 4.0x :rocket: |
| **STAGE1_PROPAGATION** | 77  (66 pyr / 11 int) | 83 | 9.3x :rocket: |

_Enrichment = (% of bottom slice in this category) / (% of all cells in this category). Values > ~2x are reliable failure-mode signals; 1x means the rule fires equally on bottom and top._

## Categories ranked by enrichment in bottom slice

### APICAL_OVERPREDICTED  (enrichment = 10.0x, n = 9 bottom cells, 9 pyramidals)

**Rule**: Model predicted >1.5x as many apical nodes as GT has (and apical_F1 low).

**Examples** (worst 9 of 9):

| file | cell_type | F1 | apical_F1 | basal_F1 | n_nodes | pred_a/b/p | other categories |
|---|---|---|---|---|---|---|---|
| `neuromorpho__X046_S1_Z2_C05.swc` | pyramidal | 0.2052 | 0.0000 | 0.4104 | 660 | 47/308/304 | APICAL_INVERTED, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__C4_3.swc` | pyramidal | 0.2126 | 0.0000 | 0.6377 | 1226 | 0/596/629 | AXON_DENDRITE_CONFUSION |
| `neuromorpho__mPFC_w1_L3_01.swc` | pyramidal | 0.2570 | 0.0000 | 0.7711 | 2540 | 0/1294/1245 | AXON_DENDRITE_CONFUSION |
| `neuromorpho__X043_S5_Z2_C05_20x.swc` | pyramidal | 0.2688 | 0.0000 | 0.5377 | 942 | 0/540/401 | APICAL_INVERTED, TRUNCATED_APICAL |
| `neuromorpho__241-2-93LU.swc` | pyramidal | 0.3088 | 0.0000 | 0.6176 | 586 | 3/260/322 | - |
| `neuromorpho__D1-9.swc` | pyramidal | 0.3171 | 0.0000 | 0.6342 | 352 | 0/238/113 | APICAL_INVERTED |
| `neuromorpho__inv-6-23-08-s2-cell-1.swc` | pyramidal | 0.3514 | 0.0000 | 0.7029 | 2116 | 0/1410/705 | APICAL_INVERTED, TRUNCATED_APICAL |
| `neuromorpho__A4_slide2_neuron1_II_III.swc` | pyramidal | 0.4733 | 0.0000 | 0.4200 | 871 | 80/348/442 | APICAL_INVERTED, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__236-2-28LS.swc` | pyramidal | 0.4980 | 0.0000 | 0.4939 | 758 | 13/431/313 | APICAL_INVERTED, BASAL_LOST |

### STAGE1_PROPAGATION  (enrichment = 9.3x, n = 77 bottom cells, 66 pyramidals)

**Rule**: Stage 1 was wrong AND per-cell F1 is low -- error propagated downstream.

**Examples** (worst 15 of 77):

| file | cell_type | F1 | apical_F1 | basal_F1 | n_nodes | pred_a/b/p | other categories |
|---|---|---|---|---|---|---|---|
| `neuromorpho__KO3-neuron8.swc` | pyramidal | 0.0000 | 0.0000 | - | 193 | 0/192/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__110E_1_L_Ngr2KO_1.swc` | pyramidal | 0.0000 | 0.0000 | - | 182 | 0/181/0 | APICAL_MISSED, APICAL_UNDERPREDICTED |
| `neuromorpho__cIN-1360-nrn-ns-111206-transl-invX-scaled-straightened.swc` | interneuron | 0.0000 | - | - | 78 | 0/0/77 | AXON_DENDRITE_CONFUSION |
| `neuromorpho__WT2-Neuron-3.swc` | pyramidal | 0.0000 | - | - | 14 | 0/13/0 | AXON_DENDRITE_CONFUSION |
| `neuromorpho__PDE1c_vGluT2-ctrl_IN08.swc` | interneuron | 0.0674 | - | 0.0674 | 259 | 78/9/171 | BASAL_LOST |
| `neuromorpho__11220c3.swc` | pyramidal | 0.0911 | 0.0000 | 0.1822 | 879 | 0/878/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__p120107-07-AAC.swc` | interneuron | 0.0960 | - | 0.0960 | 596 | 204/30/361 | BASAL_LOST |
| `neuromorpho__Nov-9-IR-1-4d-40x-low-res.swc` | pyramidal | 0.1026 | 0.0000 | 0.2053 | 1881 | 0/1880/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__Jul26-IR2-2H.swc` | pyramidal | 0.1239 | 0.0000 | 0.2478 | 1889 | 0/1888/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST |
| `neuromorpho__m3s4s4b-med-deep.swc` | pyramidal | 0.1434 | 0.0000 | 0.2868 | 234 | 0/233/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__inv-5-7-08-s2-cell-1.swc` | pyramidal | 0.1491 | 0.0000 | 0.2983 | 2032 | 0/2031/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, BASAL_LOST |
| `neuromorpho__WT3-neuron8.swc` | pyramidal | 0.1510 | - | 0.3021 | 164 | 0/163/0 | AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__KO_HIF1a_1_3.swc` | pyramidal | 0.1518 | 0.0000 | 0.4553 | 1572 | 0/1571/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__KO3-neuron11.swc` | pyramidal | 0.1545 | - | 0.3089 | 209 | 0/208/0 | AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__389-01-01.swc` | pyramidal | 0.1652 | - | 0.3305 | 779 | 0/778/0 | AXON_DENDRITE_CONFUSION, BASAL_LOST |

### APICAL_UNDERPREDICTED  (enrichment = 8.9x, n = 276 bottom cells, 274 pyramidals)

**Rule**: Model predicted <50% as many apical nodes as GT has (and apical_F1 low).

**Examples** (worst 15 of 276):

| file | cell_type | F1 | apical_F1 | basal_F1 | n_nodes | pred_a/b/p | other categories |
|---|---|---|---|---|---|---|---|
| `neuromorpho__n423.swc` | pyramidal | 0.0000 | 0.0000 | 0.0000 | 2143 | 929/1213/0 | APICAL_MISSED, APICAL_INVERTED, BASAL_LOST |
| `neuromorpho__3PO3_5.swc` | pyramidal | 0.0000 | 0.0000 | - | 790 | 480/309/0 | APICAL_MISSED, AXON_DENDRITE_CONFUSION |
| `neuromorpho__KO3-neuron8.swc` | pyramidal | 0.0000 | 0.0000 | - | 193 | 0/192/0 | APICAL_MISSED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__110E_1_L_Ngr2KO_1.swc` | pyramidal | 0.0000 | 0.0000 | - | 182 | 0/181/0 | APICAL_MISSED, STAGE1_PROPAGATION |
| `neuromorpho__KO3-neuron3.swc` | pyramidal | 0.0000 | 0.0000 | - | 171 | 84/86/0 | APICAL_MISSED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__10month-Arctic-C5aR1-KO-52280-LR_4.swc` | pyramidal | 0.0000 | 0.0000 | - | 173 | 172/0/0 | APICAL_MISSED, APICAL_INVERTED, TRUNCATED_APICAL |
| `neuromorpho__C1042-whole-2.swc` | pyramidal | 0.0000 | 0.0000 | 0.0000 | 61 | 32/28/0 | APICAL_MISSED, BASAL_LOST |
| `neuromorpho__WT1-neuron10.swc` | pyramidal | 0.0694 | 0.0000 | 0.2083 | 79 | 35/43/0 | APICAL_MISSED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__WT_1_3.swc` | pyramidal | 0.0700 | 0.0000 | 0.2101 | 2940 | 0/2939/0 | APICAL_MISSED, AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__11220c3.swc` | pyramidal | 0.0911 | 0.0000 | 0.1822 | 879 | 0/878/0 | APICAL_MISSED, BASAL_LOST, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__Nov-9-IR-1-4d-40x-low-res.swc` | pyramidal | 0.1026 | 0.0000 | 0.2053 | 1881 | 0/1880/0 | APICAL_MISSED, APICAL_INVERTED, BASAL_LOST, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__242-1-13VN.swc` | pyramidal | 0.1164 | 0.0000 | 0.3492 | 261 | 69/191/0 | APICAL_MISSED, AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__Jul26-IR2-2H.swc` | pyramidal | 0.1239 | 0.0000 | 0.2478 | 1889 | 0/1888/0 | APICAL_MISSED, BASAL_LOST, STAGE1_PROPAGATION |
| `neuromorpho__A627_NeuronA_40x_Tile_Trace_RK.swc` | pyramidal | 0.1273 | 0.0000 | 0.2545 | 728 | 0/727/0 | APICAL_MISSED, BASAL_LOST |
| `neuromorpho__A550_NeuronB_40x_Tile_Trace_RK.swc` | pyramidal | 0.1289 | 0.0000 | 0.2578 | 447 | 102/344/0 | APICAL_MISSED, BASAL_LOST |

### APICAL_MISSED  (enrichment = 8.8x, n = 264 bottom cells, 262 pyramidals)

**Rule**: GT has apical (>1% of nodes) but model predicted ZERO apical nodes.

**Examples** (worst 15 of 264):

| file | cell_type | F1 | apical_F1 | basal_F1 | n_nodes | pred_a/b/p | other categories |
|---|---|---|---|---|---|---|---|
| `neuromorpho__n423.swc` | pyramidal | 0.0000 | 0.0000 | 0.0000 | 2143 | 929/1213/0 | APICAL_UNDERPREDICTED, APICAL_INVERTED, BASAL_LOST |
| `neuromorpho__3PO3_5.swc` | pyramidal | 0.0000 | 0.0000 | - | 790 | 480/309/0 | APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION |
| `neuromorpho__KO3-neuron8.swc` | pyramidal | 0.0000 | 0.0000 | - | 193 | 0/192/0 | APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__110E_1_L_Ngr2KO_1.swc` | pyramidal | 0.0000 | 0.0000 | - | 182 | 0/181/0 | APICAL_UNDERPREDICTED, STAGE1_PROPAGATION |
| `neuromorpho__KO3-neuron3.swc` | pyramidal | 0.0000 | 0.0000 | - | 171 | 84/86/0 | APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__10month-Arctic-C5aR1-KO-52280-LR_4.swc` | pyramidal | 0.0000 | 0.0000 | - | 173 | 172/0/0 | APICAL_UNDERPREDICTED, APICAL_INVERTED, TRUNCATED_APICAL |
| `neuromorpho__C1042-whole-2.swc` | pyramidal | 0.0000 | 0.0000 | 0.0000 | 61 | 32/28/0 | APICAL_UNDERPREDICTED, BASAL_LOST |
| `neuromorpho__WT1-neuron10.swc` | pyramidal | 0.0694 | 0.0000 | 0.2083 | 79 | 35/43/0 | APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__WT_1_3.swc` | pyramidal | 0.0700 | 0.0000 | 0.2101 | 2940 | 0/2939/0 | APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__11220c3.swc` | pyramidal | 0.0911 | 0.0000 | 0.1822 | 879 | 0/878/0 | APICAL_UNDERPREDICTED, BASAL_LOST, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__Nov-9-IR-1-4d-40x-low-res.swc` | pyramidal | 0.1026 | 0.0000 | 0.2053 | 1881 | 0/1880/0 | APICAL_UNDERPREDICTED, APICAL_INVERTED, BASAL_LOST, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__242-1-13VN.swc` | pyramidal | 0.1164 | 0.0000 | 0.3492 | 261 | 69/191/0 | APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__Jul26-IR2-2H.swc` | pyramidal | 0.1239 | 0.0000 | 0.2478 | 1889 | 0/1888/0 | APICAL_UNDERPREDICTED, BASAL_LOST, STAGE1_PROPAGATION |
| `neuromorpho__A627_NeuronA_40x_Tile_Trace_RK.swc` | pyramidal | 0.1273 | 0.0000 | 0.2545 | 728 | 0/727/0 | APICAL_UNDERPREDICTED, BASAL_LOST |
| `neuromorpho__A550_NeuronB_40x_Tile_Trace_RK.swc` | pyramidal | 0.1289 | 0.0000 | 0.2578 | 447 | 102/344/0 | APICAL_UNDERPREDICTED, BASAL_LOST |

### BASAL_LOST  (enrichment = 8.6x, n = 90 bottom cells, 72 pyramidals)

**Rule**: Basal F1 < 0.5 despite GT having basal nodes -- model labeled basal as something else.

**Examples** (worst 15 of 90):

| file | cell_type | F1 | apical_F1 | basal_F1 | n_nodes | pred_a/b/p | other categories |
|---|---|---|---|---|---|---|---|
| `neuromorpho__GGN_20170309_sc.swc` | interneuron | 0.0000 | - | 0.0000 | 36156 | 36155/0/0 | - |
| `neuromorpho__170321_1.swc` | interneuron | 0.0000 | 0.0000 | 0.0000 | 4529 | 4504/24/0 | - |
| `neuromorpho__n423.swc` | pyramidal | 0.0000 | 0.0000 | 0.0000 | 2143 | 929/1213/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED |
| `neuromorpho__C912-WT-basal-2.swc` | pyramidal | 0.0000 | - | 0.0000 | 72 | 71/0/0 | - |
| `neuromorpho__C1042-whole-2.swc` | pyramidal | 0.0000 | 0.0000 | 0.0000 | 61 | 32/28/0 | APICAL_MISSED, APICAL_UNDERPREDICTED |
| `neuromorpho__hippo-104-6.swc` | pyramidal | 0.0000 | - | 0.0000 | 70 | 0/0/69 | - |
| `neuromorpho__SW150113-01RabiesPHALPFC_x772_y876_z296_14.swc` | pyramidal | 0.0255 | - | 0.0255 | 311 | 98/4/208 | - |
| `neuromorpho__interneuronEX.swc` | interneuron | 0.0561 | - | 0.1122 | 7317 | 0/7316/0 | AXON_DENDRITE_CONFUSION |
| `neuromorpho__242-2-24VN.swc` | pyramidal | 0.0606 | 0.0000 | 0.1212 | 63 | 0/29/33 | - |
| `neuromorpho__PDE1c_vGluT2-ctrl_IN08.swc` | interneuron | 0.0674 | - | 0.0674 | 259 | 78/9/171 | STAGE1_PROPAGATION |
| `neuromorpho__WT1-neuron10.swc` | pyramidal | 0.0694 | 0.0000 | 0.2083 | 79 | 35/43/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__WT_1_3.swc` | pyramidal | 0.0700 | 0.0000 | 0.2101 | 2940 | 0/2939/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION |
| `neuromorpho__170313_2.swc` | interneuron | 0.0794 | - | 0.0794 | 29884 | 28647/1236/0 | - |
| `neuromorpho__11220c3.swc` | pyramidal | 0.0911 | 0.0000 | 0.1822 | 879 | 0/878/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__p120107-07-AAC.swc` | interneuron | 0.0960 | - | 0.0960 | 596 | 204/30/361 | STAGE1_PROPAGATION |

### AXON_DENDRITE_CONFUSION  (enrichment = 6.0x, n = 142 bottom cells, 101 pyramidals)

**Rule**: Axon F1 < 0.95 (very rare; axon usually trivially separable from dendrite).

**Examples** (worst 15 of 142):

| file | cell_type | F1 | apical_F1 | basal_F1 | n_nodes | pred_a/b/p | other categories |
|---|---|---|---|---|---|---|---|
| `neuromorpho__3PO3_5.swc` | pyramidal | 0.0000 | 0.0000 | - | 790 | 480/309/0 | APICAL_MISSED, APICAL_UNDERPREDICTED |
| `neuromorpho__KO3-neuron8.swc` | pyramidal | 0.0000 | 0.0000 | - | 193 | 0/192/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__KO3-neuron3.swc` | pyramidal | 0.0000 | 0.0000 | - | 171 | 84/86/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, TRUNCATED_APICAL |
| `neuromorpho__WT1-neuron11.swc` | pyramidal | 0.0000 | 0.0000 | - | 117 | 23/0/93 | - |
| `neuromorpho__cIN-1360-nrn-ns-111206-transl-invX-scaled-straightened.swc` | interneuron | 0.0000 | - | - | 78 | 0/0/77 | STAGE1_PROPAGATION |
| `neuromorpho__WT2-Neuron-3.swc` | pyramidal | 0.0000 | - | - | 14 | 0/13/0 | STAGE1_PROPAGATION |
| `neuromorpho__interneuronEX.swc` | interneuron | 0.0561 | - | 0.1122 | 7317 | 0/7316/0 | BASAL_LOST |
| `neuromorpho__WT1-neuron10.swc` | pyramidal | 0.0694 | 0.0000 | 0.2083 | 79 | 35/43/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__WT_1_3.swc` | pyramidal | 0.0700 | 0.0000 | 0.2101 | 2940 | 0/2939/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST |
| `neuromorpho__AINL.swc` | interneuron | 0.1111 | - | 0.2222 | 17 | 7/9/0 | BASAL_LOST |
| `neuromorpho__242-1-13VN.swc` | pyramidal | 0.1164 | 0.0000 | 0.3492 | 261 | 69/191/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST |
| `neuromorpho__WT1-neuron3.swc` | pyramidal | 0.1287 | - | 0.2573 | 150 | 0/149/0 | BASAL_LOST |
| `neuromorpho__241-1-22-AW.swc` | pyramidal | 0.1354 | 0.0000 | 0.4062 | 464 | 284/179/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, BASAL_LOST |
| `neuromorpho__WT3-neuron8.swc` | pyramidal | 0.1510 | - | 0.3021 | 164 | 0/163/0 | BASAL_LOST, STAGE1_PROPAGATION |
| `neuromorpho__KO_HIF1a_1_3.swc` | pyramidal | 0.1518 | 0.0000 | 0.4553 | 1572 | 0/1571/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, BASAL_LOST, STAGE1_PROPAGATION |

### TRUNCATED_APICAL  (enrichment = 4.0x, n = 125 bottom cells, 123 pyramidals)

**Rule**: GT has apical but very short z-extent (<50 microns).

**Examples** (worst 15 of 125):

| file | cell_type | F1 | apical_F1 | basal_F1 | n_nodes | pred_a/b/p | other categories |
|---|---|---|---|---|---|---|---|
| `neuromorpho__KO3-neuron8.swc` | pyramidal | 0.0000 | 0.0000 | - | 193 | 0/192/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, STAGE1_PROPAGATION |
| `neuromorpho__KO3-neuron3.swc` | pyramidal | 0.0000 | 0.0000 | - | 171 | 84/86/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION |
| `neuromorpho__10month-Arctic-C5aR1-KO-52280-LR_4.swc` | pyramidal | 0.0000 | 0.0000 | - | 173 | 172/0/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED |
| `neuromorpho__WT1-neuron10.swc` | pyramidal | 0.0694 | 0.0000 | 0.2083 | 79 | 35/43/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__11220c3.swc` | pyramidal | 0.0911 | 0.0000 | 0.1822 | 879 | 0/878/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST, STAGE1_PROPAGATION |
| `neuromorpho__Nov-9-IR-1-4d-40x-low-res.swc` | pyramidal | 0.1026 | 0.0000 | 0.2053 | 1881 | 0/1880/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, BASAL_LOST, STAGE1_PROPAGATION |
| `neuromorpho__m3s4s4b-med-deep.swc` | pyramidal | 0.1434 | 0.0000 | 0.2868 | 234 | 0/233/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST, STAGE1_PROPAGATION |
| `neuromorpho__DCX-257-02-A1-Cell2.swc` | pyramidal | 0.1607 | 0.0000 | 0.3214 | 95 | 44/50/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST |
| `neuromorpho__DCX-258-02-B1-Cell3.swc` | pyramidal | 0.1638 | 0.0000 | 0.4914 | 265 | 76/188/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__X041_S6_Z5_C01.swc` | pyramidal | 0.1677 | 0.0000 | 0.3354 | 661 | 0/409/251 | BASAL_LOST |
| `neuromorpho__ICR2-5-6-6.swc` | pyramidal | 0.1915 | 0.0000 | 0.5745 | 532 | 0/531/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, STAGE1_PROPAGATION |
| `neuromorpho__KO1-4-3-9.swc` | pyramidal | 0.2047 | 0.0000 | 0.6141 | 247 | 0/246/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, STAGE1_PROPAGATION |
| `neuromorpho__X046_S1_Z2_C05.swc` | pyramidal | 0.2052 | 0.0000 | 0.4104 | 660 | 47/308/304 | APICAL_OVERPREDICTED, APICAL_INVERTED, BASAL_LOST |
| `neuromorpho__10o27c3.swc` | pyramidal | 0.2114 | 0.0000 | 0.6341 | 1315 | 0/1314/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, STAGE1_PROPAGATION |
| `neuromorpho__389-02-02A.swc` | pyramidal | 0.2123 | 0.0000 | 0.6369 | 581 | 0/580/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, AXON_DENDRITE_CONFUSION, STAGE1_PROPAGATION |

### APICAL_INVERTED  (enrichment = 1.7x, n = 149 bottom cells, 149 pyramidals)

**Rule**: GT apical nodes are on average BELOW the soma (typically reversed orientation).

**Examples** (worst 15 of 149):

| file | cell_type | F1 | apical_F1 | basal_F1 | n_nodes | pred_a/b/p | other categories |
|---|---|---|---|---|---|---|---|
| `neuromorpho__n423.swc` | pyramidal | 0.0000 | 0.0000 | 0.0000 | 2143 | 929/1213/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST |
| `neuromorpho__10month-Arctic-C5aR1-KO-52280-LR_4.swc` | pyramidal | 0.0000 | 0.0000 | - | 173 | 172/0/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, TRUNCATED_APICAL |
| `neuromorpho__WT1-neuron10.swc` | pyramidal | 0.0694 | 0.0000 | 0.2083 | 79 | 35/43/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__Nov-9-IR-1-4d-40x-low-res.swc` | pyramidal | 0.1026 | 0.0000 | 0.2053 | 1881 | 0/1880/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__241-1-22-AW.swc` | pyramidal | 0.1354 | 0.0000 | 0.4062 | 464 | 284/179/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, BASAL_LOST |
| `neuromorpho__inv-5-7-08-s2-cell-1.swc` | pyramidal | 0.1491 | 0.0000 | 0.2983 | 2032 | 0/2031/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, BASAL_LOST, STAGE1_PROPAGATION |
| `neuromorpho__KO_HIF1a_1_3.swc` | pyramidal | 0.1518 | 0.0000 | 0.4553 | 1572 | 0/1571/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, BASAL_LOST, STAGE1_PROPAGATION |
| `neuromorpho__DCX-258-02-B1-Cell3.swc` | pyramidal | 0.1638 | 0.0000 | 0.4914 | 265 | 76/188/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__20171012A01.swc` | pyramidal | 0.1935 | 0.0000 | 0.5804 | 2090 | 0/2089/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, STAGE1_PROPAGATION |
| `neuromorpho__KO1-4-3-9.swc` | pyramidal | 0.2047 | 0.0000 | 0.6141 | 247 | 0/246/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__X046_S1_Z2_C05.swc` | pyramidal | 0.2052 | 0.0000 | 0.4104 | 660 | 47/308/304 | APICAL_OVERPREDICTED, BASAL_LOST, TRUNCATED_APICAL |
| `neuromorpho__389-02-02A.swc` | pyramidal | 0.2123 | 0.0000 | 0.6369 | 581 | 0/580/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL, STAGE1_PROPAGATION |
| `neuromorpho__WT1-neuron5.swc` | pyramidal | 0.2295 | 0.0000 | 0.6885 | 107 | 66/40/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION, TRUNCATED_APICAL |
| `neuromorpho__241-3-43LU.swc` | pyramidal | 0.2317 | 0.0000 | 0.6952 | 701 | 162/510/28 | APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION |
| `neuromorpho__236-3-32LS.swc` | pyramidal | 0.2356 | 0.0000 | 0.7067 | 1626 | 356/1269/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, AXON_DENDRITE_CONFUSION |

### WHOLE_CELL_MISLABEL  (enrichment = 1.3x, n = 1 bottom cells, 1 pyramidals)

**Rule**: Stage 1 confidently disagrees with GT cell type. Whole-cell label looks wrong.

**Examples** (worst 1 of 1):

| file | cell_type | F1 | apical_F1 | basal_F1 | n_nodes | pred_a/b/p | other categories |
|---|---|---|---|---|---|---|---|
| `neuromorpho__neu40.swc` | pyramidal | 0.2815 | 0.0000 | 0.5630 | 98 | 0/97/0 | APICAL_MISSED, APICAL_UNDERPREDICTED, APICAL_INVERTED, TRUNCATED_APICAL, STAGE1_PROPAGATION |

## Unmatched bottom-slice cells

**14 bottom-slice cells matched NO category** -- low F1 but no obvious failure pattern. These may need new categories or are genuinely hard cells.

Worst 10 unmatched:

| file | cell_type | F1 | apical_F1 | basal_F1 | apical_frac | apical_above_soma |
|---|---|---|---|---|---|---|
| `neuromorpho__iPRCp_05.swc` | pyramidal | 0.2562 | 0.0000 | 0.5125 | 0.281 | 2.28 |
| `neuromorpho__ofc11506-03R1.swc` | pyramidal | 0.2836 | 0.0000 | 0.5672 | 0.342 | 20.03 |
| `neuromorpho__H1-3.swc` | pyramidal | 0.3277 | 0.0000 | 0.6554 | 0.221 | 1.01 |
| `neuromorpho__het_1_51.swc` | pyramidal | 0.5050 | - | 0.5050 | 0.000 | - |
| `neuromorpho__Green_6weeks_Neuron2_addl.swc` | pyramidal | 0.5155 | - | 0.5155 | 0.000 | - |
| `neuromorpho__HIP_CA1_KO_13.swc` | pyramidal | 0.5205 | - | 0.5205 | 0.000 | - |
| `neuromorpho__Green_4weeks_Neuron1.swc` | pyramidal | 0.5376 | - | 0.5376 | 0.000 | - |
| `neuromorpho__hippo-1159-3new.swc` | pyramidal | 0.5556 | - | 0.5556 | 0.000 | - |
| `neuromorpho__NAS14.swc` | interneuron | 0.5585 | - | 0.5585 | 0.000 | - |
| `neuromorpho__Cell-16_4.swc` | pyramidal | 0.5600 | - | 0.5600 | 0.000 | - |

## Suggested actions per category

| Category | Likely action |
|---|---|
| WHOLE_CELL_MISLABEL | **CLEAN** -- drop or relabel. Already caught by T2 mostly. |
| APICAL_INVERTED | **CLEAN** -- almost certainly a coordinate/orientation GT error. |
| DISCONNECTED | **CLEAN** -- reconstruction is broken; drop. |
| TRUNCATED_APICAL | **INSPECT** -- legitimate partial reconstructions; don't drop. Maybe down-weight. |
| APICAL_MISSED | **MODEL** -- need stronger apical detector (partition asymmetry, contraction). |
| APICAL_UNDERPREDICTED | **MODEL** -- same as above. |
| APICAL_OVERPREDICTED | **MODEL** -- decision boundary is permissive; needs counter-features. |
| BASAL_LOST | **MODEL** -- usually means basal got mistaken for apical. Same model fix. |
| AXON_DENDRITE_CONFUSION | **MODEL** -- rare; usually small/atypical cells. Inspect. |
| STAGE1_PROPAGATION | **MODEL** (Stage 1) -- improve Stage 1 OR enable soft-handoff threshold tuning. |

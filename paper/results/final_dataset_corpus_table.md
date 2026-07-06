# Final dataset / QC table

| Group | Label | Total | QC pass | QC fail | Pass rate | Median nodes pass |
|---|---|---:|---:|---:|---:|---:|
| overall | all | 21054 | 11862 | 9192 | 56.3% | 1352 |
| cell_type | interneuron | 10144 | 3374 | 6770 | 33.3% | 3612 |
| cell_type | pyramidal | 10910 | 8488 | 2422 | 77.8% | 918 |
| source | allen | 299 | 203 | 96 | 67.9% | 11792 |
| source | hpf_ca1 | 1377 | 1358 | 19 | 98.6% | 31056 |
| source | neuromorpho | 19378 | 10301 | 9077 | 53.2% | 1009 |
| source_cell_type | allen:interneuron | 197 | 126 | 71 | 64.0% | 11976 |
| source_cell_type | allen:pyramidal | 102 | 77 | 25 | 75.5% | 11405 |
| source_cell_type | hpf_ca1:pyramidal | 1377 | 1358 | 19 | 98.6% | 31056 |
| source_cell_type | neuromorpho:interneuron | 9947 | 3248 | 6699 | 32.7% | 3440 |
| source_cell_type | neuromorpho:pyramidal | 9431 | 7053 | 2378 | 74.8% | 684 |

## QC fail reasons

| Reason | Count |
|---|---:|
| fail: has_soma | 5624 |
| fail: has_soma,has_neurites | 2246 |
| fail: non_empty,has_neurites,reasonable_size | 1194 |
| fail: single_root | 121 |
| fail: radius_positive | 6 |
| fail: non_empty,has_soma,reasonable_size | 1 |

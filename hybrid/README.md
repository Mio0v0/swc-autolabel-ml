# Hybrid Auto-Labeling Pipeline

ML + rule-based hybrid approach for generalizing neuron morphology auto-labeling
across different cell types (pyramidal, interneuron, Purkinje, granule, etc.).

## Architecture

```
Stage 1: Cell-Type Detection (ML)    → determines label set & weight profile
Stage 2: Branch Classification (ML + Rules)  → per-branch type assignment
Stage 3: Topology Refinement (Rules) → structural constraints & smoothing
```

## Quick Start

### 1. Download training data

**Allen Cell Types Database** (automated):
```bash
# Download all available morphologies (~367 files, ~5 min)
python -m hybrid.download_allen_data

# Or limit per type for quick testing
python -m hybrid.download_allen_data --max-per-type 20
```

**NeuroMorpho.Org** (manual — see instructions below):
Place downloaded SWC files into `data/training_morphologies/<cell_type>/`.

### 2. Train the classifier

```bash
python -m hybrid.train --data-dir data/training_morphologies
```

### 3. Test on files

```bash
python -m hybrid.test_stage1 path/to/file.swc
python -m hybrid.test_stage1 data/training_morphologies/pyramidal/*.swc
```

## Training Data Structure

```
data/training_morphologies/
├── pyramidal/          # spiny cells with apical + basal dendrites
│   ├── cell_001.swc
│   └── ...
├── interneuron/        # aspiny / sparsely spiny cells
│   ├── cell_001.swc
│   └── ...
├── purkinje/           # cerebellar Purkinje cells
│   ├── cell_001.swc
│   └── ...
├── granule/            # small granule cells
│   ├── cell_001.swc
│   └── ...
└── metadata.csv
```

## Downloading NeuroMorpho.Org Data

NeuroMorpho.Org has the largest collection of reconstructed morphologies (~190k+).
Bulk download requires their API. Here's how:

### Option A: Web interface
1. Go to https://neuromorpho.org/
2. Click "Browse" → filter by Cell Type (e.g., "Purkinje", "granule", "interneuron")
3. Select cells → click "Download selected" → choose SWC format
4. Place files in `data/training_morphologies/<cell_type>/`

### Option B: API (recommended for bulk)
```bash
# List available cell types
curl "https://neuromorpho.org/api/neuron/fields/cell_type"

# Query Purkinje cells (get metadata)
curl "https://neuromorpho.org/api/neuron/select?q=cell_type:Purkinje&size=10"

# Get download links from the neuron_name field:
# SWC URL pattern: https://neuromorpho.org/dableFiles/<archive_name>/CNG%20version/<neuron_name>.CNG.swc

# Example workflow:
# 1. Query neurons
curl -s "https://neuromorpho.org/api/neuron/select?q=cell_type:Purkinje&size=50" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for n in data['_embedded']['neuronResources']:
    name = n['neuron_name']
    archive = n['archive']
    print(f'https://neuromorpho.org/dableFiles/{archive}/CNG%20version/{name}.CNG.swc')
"
# 2. Download each URL with wget/curl
```

### Recommended cell types to download from NeuroMorpho:
- **Purkinje**: search `cell_type:Purkinje` (~1000+ cells)
- **Granule**: search `cell_type:granule` (~500+ cells)
- **Interneuron**: search `cell_type:interneuron` (~2000+ cells)
- **Pyramidal**: search `cell_type:pyramidal` (~5000+ cells, already have Allen data)

### How many files do you need?
- Minimum: ~50 per cell type for reasonable accuracy
- Recommended: ~200+ per cell type for robust generalization
- The Allen download script provides ~120 pyramidal + ~247 interneuron
- Adding ~50-100 Purkinje and ~50 granule from NeuroMorpho completes the training set

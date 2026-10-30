# AISmell: Teaching an AI to Smell

Implementation of the **Principal Odor Map (POM)** model from the Science 2023 paper:
> *"A principal odor map unifies diverse tasks in olfactory perception"*  
> (Google Research & Monell Chemical Senses Center)

**Architecture:** SMILES → Molecular Graph → 5-round MPNN → Attention Pooling → 256-dim POM → 138 Odor Labels

## Project Structure

```
AISmell/
├── setup_colab.sh          # Colab dependency installer (run first)
├── requirements.txt        # pip requirements (local dev)
├── dataset.py              # Data loading, featurization, stratified splitting
├── model.py                # MPNN-POM model (PyTorch, from scratch)
├── train.py                # Training loop with ensemble support
├── predict.py              # CLI inference tool
├── colab_notebook.ipynb    # Self-contained Colab notebook (ready to run)
├── AI_SMELL_SPEC.md        # Full technical specification
└── CLAUDE_OPUS_PROMPT.md   # Engineering prompt (reference)
```

## Quick Start (Google Colab)

1. Upload the project to Colab or open `colab_notebook.ipynb` directly
2. Set runtime to **T4 GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Run all cells — the notebook downloads the dataset automatically

## Local / CLI Usage

```bash
# Install dependencies (conda recommended)
conda create -n aismell python=3.10 -y && conda activate aismell
conda install -c conda-forge rdkit -y
pip install torch==2.4.0 dgl iterative-stratification scikit-learn tqdm

# Train a single model (5 epochs, quick test)
python train.py --epochs 5 --seed 42 --model-dir saved_models/model_1

# Train full 10-model ensemble (150 epochs each)
python train.py --ensemble 10 --epochs 150 --model-dir saved_models

# Predict what a molecule smells like
python predict.py --smiles "CCO" --model-dir saved_models --ensemble 10
python predict.py --smiles "CCO" "c1ccccc1" "CC(=O)O" --top-k 15 --output-json results.json
```

## Technical Summary

| Component | Detail |
|---|---|
| **Input** | SMILES string → RDKit molecular graph |
| **Node features** | Element (20 types), degree, hybridization, charge, H-count, aromaticity, ring (45 dims) |
| **Edge features** | Bond type, conjugation, ring, stereo (8 dims) |
| **Message passing** | 5 rounds, GRU-based node update, edge-conditioned messages |
| **Pooling** | Global attention pooling |
| **POM embedding** | 256-dimensional dense layer (dropout=0.12) |
| **Output** | 138 sigmoid scores (multi-label odor descriptors) |
| **Loss** | BCEWithLogitsLoss with per-class positive weights |
| **Data split** | Iterative stratified 80/10/10 (preserves rare labels) |
| **Evaluation** | Macro ROC-AUC across 138 descriptors |
| **Target** | ~0.89 AUC with 10-model ensemble |

## Dataset

- **4,983 curated molecules** from fragrance/flavor catalogs
- **138 olfactory semantic labels** (e.g., `fruity`, `rose`, `sweet`, `smoky`, `fishy`, `cinnamon`)
- Severe class imbalance: `fruity` (1,902 positives) vs `smoky` (72 positives)
- Auto-downloaded from the [OpenPOM repository](https://github.com/ARY2260/openpom)

# PROMPT FOR CLAUDE OPUS: Implement OpenPOM Olfactory AI & Prediction Pipeline

*Copy and paste the prompt below into Claude Opus:*

```markdown
You are an expert Machine Learning Engineer and Computational Chemist specializing in Graph Neural Networks (GNNs), Molecular Property Prediction, and PyTorch / DeepChem.

Your task is to implement an end-to-end reproducible pipeline for "Teaching an AI to Smell" based on the Science 2023 paper ("A principal odor map unifies diverse tasks in olfactory perception", Google Research / Monell) and the OpenPOM repository.

## System Specifications & Requirements

### 1. Problem Formulation
- **Input**: SMILES chemical string (e.g., `CCO` for ethanol, `CC(O)CN`, etc.).
- **Graph Representation**:
  - Atoms = Nodes (features: element type one-hot, degree, formal charge, aromaticity, hybridization).
  - Bonds = Edges (features: bond type, conjugation, ring status).
- **Core Model**: Message Passing Neural Network (MPNN) with:
  - 5 message passing rounds.
  - Molecular pooling / readout layer.
  - A 256-dimensional dense embedding layer representing the **Principal Odor Map (POM)**.
  - Dropout of 0.12.
  - Multi-label classification head outputting 138 sigmoid scores (binary probabilities for 138 odor descriptors like `fruity`, `rose`, `mint`, `smoky`, `fishy`, `cinnamon`).
- **Data & Challenges**:
  - Curated dataset with 4,983 molecules and 138 odor labels.
  - Multi-label classification with severe class imbalance (e.g. `fruity`: 1,902 positive samples vs `smoky`: 72 positive samples).
  - Iterative stratified splitting (80% train, 10% validation, 10% test).
  - Weighted Binary Cross-Entropy loss.
- **Evaluation**:
  - Macro ROC-AUC averaged across all 138 descriptors.
  - 10-model ensemble averaging for production-level inference.

---

### 2. Deliverables Required
Please write clean, well-documented, modular Python code structured as follows:

1. **`requirements.txt` / Colab Setup Script**:
   - Exact installation order for Google Colab (Linux x86_64, GPU): PyTorch 2.4.0, DGL, DeepChem (pre-release), RDKit, OpenPOM.
2. **`dataset.py`**:
   - Downloads/loads the 4,983 curated molecules dataset (`curated_datasets/*4983.csv`).
   - Featurizes molecules using RDKit / DeepChem / PyTorch Geometric graph representations.
   - Implements multi-label iterative stratified splitting (80/10/10).
   - Computes positive class weights to compensate for label imbalance in BCE loss.
3. **`model.py`**:
   - Full PyTorch implementation of `MPNNPOMModel`:
     - Embedding layers for atom and bond features.
     - 5 Message Passing layers (neighbor message aggregation and node update functions).
     - Graph-level pooling (e.g. Global Attention / Set2Set / Sum pooling).
     - Latent 256-dimensional Principal Odor Map (POM) dense layer with 0.12 dropout.
     - 138-output classification head.
     - Extraction method to return the 256-dim POM vector for any input molecule.
4. **`train.py`**:
   - Training loop with Adam optimizer, learning rate scheduler, gradient clipping.
   - Validation evaluation calculating Macro ROC-AUC across all 138 descriptors at each epoch.
   - Checkpointing logic to save model weights for ensemble members (models 1 to 10).
5. **`predict.py`**:
   - CLI tool that accepts a SMILES string (e.g. `--smiles "CCO"`).
   - Loads the trained ensemble models, runs inference, averages predictions, and outputs:
     - Top-10 predicted odor descriptors with probabilities.
     - Extracted 256-dimensional POM vector (or 2D projection/PCA).
6. **`colab_notebook.ipynb` (or complete script)**:
   - Self-contained, step-by-step cells ready to run on Google Colab with free T4 GPU.

Ensure all code handles edge cases (invalid SMILES, molecules with single isolated atoms, division by zero in rare label ROC-AUC). Provide clear instructions for running training and inference.
```

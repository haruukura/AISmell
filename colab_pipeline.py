"""
colab_pipeline.py - OpenPOM End-to-End Execution Script for Google Colab
Run on a Colab GPU instance (Runtime -> Change runtime type -> T4 GPU)
"""

# Cell 1: Dependency Installation
INSTALL_SCRIPT = """
!pip install --pre deepchem
!pip install torch==2.4.0
!pip install dgl -f https://data.dgl.ai/wheels/repo.html
!pip install openpom
"""

# Cell 2: Training & Prediction Pipeline
import os
import torch
import numpy as np

def run_openpom_training():
    """
    Demonstrates loading data, initializing MPNNPOMModel, splitting, and running training.
    Reference: openpom/examples/mpnn_pom_train.ipynb
    """
    import deepchem as dc
    from openpom.models.mpnn_pom import MPNNPOMModel
    from openpom.data.curated_datasets import load_curated_dataset
    from openpom.utils.splitters import IterativeStratifiedSplitter

    print("1. Loading curated 4,983 molecules dataset...")
    # OpenPOM includes curated_datasets with 138 smell labels
    dataset, tasks = load_curated_dataset("curated_gs_lf_4983.csv")
    print(f"Total molecules: {len(dataset)}, Total smell tasks: {len(tasks)}")

    print("2. Performing multi-label iterative stratified splitting (80/10/10)...")
    splitter = IterativeStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42
    )
    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    print("3. Instantiating MPNNPOMModel with Paper Hyperparameters...")
    # 5 message passing rounds, 256-dim POM layer, 0.12 dropout
    model = MPNNPOMModel(
        n_tasks=len(tasks),
        node_out_feats=64,
        edge_hidden_feats=64,
        num_step_message_passing=5,
        mpnn_residual=True,
        pom_latent_dim=256,
        dropout=0.12,
        mode="classification",
        learning_rate=0.001,
        model_dir="./saved_models/pom_model_1"
    )

    print("4. Training model (5 epochs for demo; 150 epochs for full benchmark)...")
    model.fit(train_dataset, nb_epoch=5)

    print("5. Evaluating on Test Set with ROC-AUC...")
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode="classification")
    test_scores = model.evaluate(test_dataset, [metric])
    print(f"Test Macro ROC-AUC: {test_scores}")

    return model, tasks

def predict_molecule(smiles_str: str, model, tasks):
    """
    Predicts top smell descriptors for a given SMILES string.
    """
    from openpom.feat.graph_featurizer import GraphFeaturizer
    featurizer = GraphFeaturizer()
    mol_features = featurizer.featurize([smiles_str])
    
    # Run inference
    predictions = model.predict_on_batch(mol_features) # shape: (1, 138)
    
    scores = predictions[0]
    top_indices = np.argsort(scores)[::-1][:10]
    
    print(f"\nTop Predicted Odors for SMILES '{smiles_str}':")
    for idx in top_indices:
        print(f"  - {tasks[idx]}: {scores[idx]:.4f}")

if __name__ == "__main__":
    print("Script prepared for Google Colab GPU environment.")

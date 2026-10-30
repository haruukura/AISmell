"""
predict.py — CLI inference tool for the MPNN-POM olfactory model.

Accepts a SMILES string (or multiple), loads a trained ensemble of models,
averages their sigmoid predictions, and prints:
  1. Top-10 predicted odor descriptors with probabilities
  2. The 256-dimensional POM embedding vector
  3. (Optional) 2D PCA projection of the POM vector

Usage:
  # Single molecule
  python predict.py --smiles "CCO" --model-dir saved_models --ensemble 10

  # Multiple molecules
  python predict.py --smiles "CCO" "c1ccccc1" "CC(=O)O" --model-dir saved_models

  # Show POM embedding as 2D PCA
  python predict.py --smiles "CCO" --model-dir saved_models --show-pca
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from dataset import MolGraph, collate_mol_graphs
from model import MPNNPOMModel

logger = logging.getLogger(__name__)


def load_model(model_path: Path, device: torch.device) -> tuple[MPNNPOMModel, dict]:
    """Load a single trained model and its metadata from *model_path*."""
    meta_file = model_path / "meta.json"
    weights_file = model_path / "best_model.pt"

    if not meta_file.exists():
        raise FileNotFoundError(f"Missing meta.json in {model_path}")
    if not weights_file.exists():
        raise FileNotFoundError(f"Missing best_model.pt in {model_path}")

    with open(meta_file) as f:
        meta = json.load(f)

    model = MPNNPOMModel(
        node_feat_dim=meta["node_feat_dim"],
        edge_feat_dim=meta["edge_feat_dim"],
        hidden_dim=meta["hidden_dim"],
        num_message_passing=meta["num_message_passing"],
        pom_dim=meta["pom_dim"],
        num_tasks=len(meta["label_cols"]),
        dropout=meta["dropout"],
    )
    model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, meta


def load_ensemble(
    base_dir: str | Path, n_models: int, device: torch.device
) -> tuple[list[MPNNPOMModel], dict]:
    """Load *n_models* ensemble members from base_dir/model_1 … model_N."""
    base = Path(base_dir)
    models = []
    meta = None

    for i in range(1, n_models + 1):
        model_path = base / f"model_{i}"
        if not model_path.exists():
            logger.warning("Ensemble member %d not found at %s, skipping.", i, model_path)
            continue
        m, meta = load_model(model_path, device)
        models.append(m)
        logger.info("Loaded model %d (val AUC: %.4f)", i, meta.get("best_val_auc", -1))

    if not models:
        # Try loading as a single model directory
        m, meta = load_model(base, device)
        models.append(m)
        logger.info("Loaded single model (val AUC: %.4f)", meta.get("best_val_auc", -1))

    return models, meta


def smiles_to_batch(smiles_list: list[str], device: torch.device) -> dict:
    """Convert SMILES strings to a batched tensor dict for the model."""
    graphs = []
    valid_smiles = []
    for s in smiles_list:
        try:
            g = MolGraph(s)
            graphs.append(g)
            valid_smiles.append(s)
        except ValueError as exc:
            logger.error("Invalid SMILES '%s': %s", s, exc)

    if not graphs:
        raise ValueError("No valid SMILES provided.")

    # Create fake labels (zeros) for collation
    dummy_labels = np.zeros((len(graphs), 138), dtype=np.float32)
    batch_items = list(zip(graphs, dummy_labels))
    batch = collate_mol_graphs(batch_items)

    # Move to device
    return {
        "node_feats": batch["node_feats"].to(device),
        "edge_index": batch["edge_index"].to(device),
        "edge_feats": batch["edge_feats"].to(device),
        "batch_idx": batch["batch_idx"].to(device),
        "num_graphs": batch["num_graphs"],
    }, valid_smiles


@torch.no_grad()
def predict_ensemble(
    models: list[MPNNPOMModel],
    batch: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ensemble prediction. Returns (probs, pom_vectors).

    probs:       (n_molecules, 138) averaged sigmoid probabilities
    pom_vectors: (n_molecules, 256) averaged POM embeddings
    """
    all_probs = []
    all_pom = []

    for model in models:
        logits = model.forward_from_batch(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        pom = model.pom_from_batch(batch).cpu().numpy()
        all_probs.append(probs)
        all_pom.append(pom)

    avg_probs = np.mean(all_probs, axis=0)
    avg_pom = np.mean(all_pom, axis=0)
    return avg_probs, avg_pom


def format_predictions(
    smiles: str,
    probs: np.ndarray,
    label_cols: list[str],
    pom_vector: np.ndarray,
    top_k: int = 10,
    show_pca: bool = False,
) -> str:
    """Format a single molecule's predictions as a readable string."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  SMILES: {smiles}")
    lines.append(f"{'='*60}")
    lines.append(f"\n  Top-{top_k} Predicted Odor Descriptors:")
    lines.append(f"  {'Rank':<6} {'Descriptor':<25} {'Probability':>12}")
    lines.append(f"  {'─'*6} {'─'*25} {'─'*12}")

    top_indices = np.argsort(probs)[::-1][:top_k]
    for rank, idx in enumerate(top_indices, 1):
        desc = label_cols[idx]
        prob = probs[idx]
        bar = "█" * int(prob * 20)
        lines.append(f"  {rank:<6} {desc:<25} {prob:>8.4f}  {bar}")

    lines.append(f"\n  POM Vector (256-dim): [{pom_vector[0]:.4f}, {pom_vector[1]:.4f}, "
                 f"{pom_vector[2]:.4f}, ... , {pom_vector[-1]:.4f}]")
    lines.append(f"  POM L2 norm: {np.linalg.norm(pom_vector):.4f}")

    if show_pca:
        # Simple inline 2-component projection (using first 2 principal directions)
        # For a proper PCA you'd fit on the training set's POM vectors
        mean = pom_vector.mean()
        std = max(pom_vector.std(), 1e-8)
        normed = (pom_vector - mean) / std
        pc1, pc2 = normed[0], normed[1]  # naive projection
        lines.append(f"  POM 2D (naive PC1, PC2): ({pc1:.4f}, {pc2:.4f})")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Predict odor descriptors from SMILES using trained MPNN-POM ensemble."
    )
    parser.add_argument(
        "--smiles", type=str, nargs="+", required=True,
        help='SMILES strings to predict (e.g., "CCO" "c1ccccc1")',
    )
    parser.add_argument(
        "--model-dir", type=str, default="saved_models",
        help="Directory containing trained model(s)",
    )
    parser.add_argument(
        "--ensemble", type=int, default=10,
        help="Number of ensemble members to load (default: 10)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top odor descriptors to display",
    )
    parser.add_argument(
        "--show-pca", action="store_true",
        help="Show a naive 2D PCA projection of the POM vector",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Optional path to save predictions as JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    models, meta = load_ensemble(args.model_dir, args.ensemble, device)
    label_cols = meta["label_cols"]
    logger.info("Loaded %d model(s) for inference on %s", len(models), device)

    # Prepare input
    batch, valid_smiles = smiles_to_batch(args.smiles, device)

    # Predict
    probs, pom_vectors = predict_ensemble(models, batch)

    # Display results
    results = []
    for i, smi in enumerate(valid_smiles):
        output = format_predictions(
            smi, probs[i], label_cols, pom_vectors[i],
            top_k=args.top_k, show_pca=args.show_pca,
        )
        print(output)

        results.append({
            "smiles": smi,
            "top_descriptors": {
                label_cols[j]: float(probs[i, j])
                for j in np.argsort(probs[i])[::-1][:args.top_k]
            },
            "pom_vector": pom_vectors[i].tolist(),
        })

    # Save JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved predictions to %s", args.output_json)


if __name__ == "__main__":
    main()

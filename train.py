"""
train.py — Training loop for the MPNN-POM olfactory model.

Features:
  - Weighted BCEWithLogitsLoss compensating for label imbalance
  - Adam optimizer with cosine-annealing LR schedule and gradient clipping
  - Macro ROC-AUC validation at each epoch
  - Checkpointing for ensemble training (10 seeds)
  - Mixed-precision (AMP) support for faster GPU training

Usage:
  # Single model (quick test, 5 epochs)
  python train.py --epochs 5 --seed 42 --model-dir saved_models/model_1

  # Full 10-model ensemble (production benchmark)
  python train.py --ensemble 10 --epochs 150 --model-dir saved_models
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataset import OdorDataset, MolGraph, Subset, collate_mol_graphs
from model import MPNNPOMModel

logger = logging.getLogger(__name__)


# ===================================================================
# Metrics
# ===================================================================

def macro_roc_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    """Compute macro-average ROC-AUC across all 138 tasks.

    Skips tasks where only one class is present (no positives or no negatives)
    to avoid undefined AUC, then averages the rest.
    """
    n_tasks = labels.shape[1]
    aucs = []
    for i in range(n_tasks):
        y_true = labels[:, i]
        y_score = preds[:, i]
        # Need both classes present for ROC-AUC
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        try:
            auc = roc_auc_score(y_true, y_score)
            aucs.append(auc)
        except ValueError:
            continue
    if not aucs:
        return 0.5  # degenerate case
    return float(np.mean(aucs))


# ===================================================================
# Training engine
# ===================================================================

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in the collated batch to *device*."""
    return {
        "node_feats": batch["node_feats"].to(device),
        "edge_index": batch["edge_index"].to(device),
        "edge_feats": batch["edge_feats"].to(device),
        "batch_idx": batch["batch_idx"].to(device),
        "num_graphs": batch["num_graphs"],
        "labels": batch["labels"].to(device),
    }


def train_one_epoch(
    model: MPNNPOMModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float = 5.0,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model.forward_from_batch(batch)
                loss = criterion(logits, batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model.forward_from_batch(batch)
            loss = criterion(logits, batch["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: MPNNPOMModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model. Returns (avg_loss, macro_roc_auc)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_labels, all_preds = [], []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model.forward_from_batch(batch)
        loss = criterion(logits, batch["labels"])
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(batch["labels"].cpu().numpy())
        all_preds.append(probs)

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    auc = macro_roc_auc(all_labels, all_preds)
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, auc


# ===================================================================
# Main training function
# ===================================================================

def train_model(
    seed: int = 42,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    pom_dim: int = 256,
    dropout: float = 0.12,
    num_message_passing: int = 5,
    max_grad_norm: float = 5.0,
    model_dir: str = "saved_models/model_1",
    data_dir: str = "data",
    use_amp: bool = True,
) -> dict:
    """Train a single MPNN-POM model and save the best checkpoint.

    Returns a dict with training history and best metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s (seed=%d)", device, seed)

    # ---- Data -----------------------------------------------------------
    dataset = OdorDataset(data_dir=data_dir)
    train_sub, val_sub, test_sub = dataset.split(seed=seed)

    pos_weights = dataset.compute_pos_weights(train_sub.indices)
    pos_weights_t = torch.tensor(pos_weights, device=device)

    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True,
        collate_fn=collate_mol_graphs, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        val_sub, batch_size=batch_size, shuffle=False,
        collate_fn=collate_mol_graphs, num_workers=0,
    )
    test_loader = DataLoader(
        test_sub, batch_size=batch_size, shuffle=False,
        collate_fn=collate_mol_graphs, num_workers=0,
    )

    # ---- Model ----------------------------------------------------------
    model = MPNNPOMModel(
        node_feat_dim=MolGraph.NODE_FEAT_DIM,
        edge_feat_dim=MolGraph.EDGE_FEAT_DIM,
        hidden_dim=hidden_dim,
        num_message_passing=num_message_passing,
        pom_dim=pom_dim,
        num_tasks=len(dataset.label_cols),
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")

    # ---- Criterion, optimizer, scheduler --------------------------------
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    # ---- Training loop --------------------------------------------------
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}

    logger.info("Starting training for %d epochs...", epochs)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, max_grad_norm
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), model_path / "best_model.pt")

        if epoch % max(1, epochs // 20) == 0 or epoch == 1 or is_best:
            marker = " ★ best" if is_best else ""
            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
                "val_AUC=%.4f  lr=%.2e  (%.1fs)%s",
                epoch, epochs, train_loss, val_loss, val_auc,
                current_lr, elapsed, marker,
            )

    # ---- Test evaluation ------------------------------------------------
    model.load_state_dict(torch.load(model_path / "best_model.pt", weights_only=True))
    test_loss, test_auc = evaluate(model, test_loader, criterion, device)
    logger.info("Test ROC-AUC: %.4f (best val epoch %d, val AUC %.4f)", test_auc, best_epoch, best_val_auc)

    # ---- Save metadata --------------------------------------------------
    meta = {
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_auc": test_auc,
        "n_params": n_params,
        "hidden_dim": hidden_dim,
        "pom_dim": pom_dim,
        "dropout": dropout,
        "num_message_passing": num_message_passing,
        "label_cols": dataset.label_cols,
        "node_feat_dim": MolGraph.NODE_FEAT_DIM,
        "edge_feat_dim": MolGraph.EDGE_FEAT_DIM,
    }
    with open(model_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save history
    with open(model_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {"meta": meta, "history": history}


# ===================================================================
# Ensemble training
# ===================================================================

def train_ensemble(
    n_models: int = 10,
    base_model_dir: str = "saved_models",
    **kwargs,
) -> list[dict]:
    """Train *n_models* independently seeded models for ensemble prediction."""
    results = []
    for i in range(1, n_models + 1):
        logger.info("=" * 60)
        logger.info("ENSEMBLE MODEL %d / %d", i, n_models)
        logger.info("=" * 60)
        res = train_model(
            seed=42 + i,
            model_dir=os.path.join(base_model_dir, f"model_{i}"),
            **kwargs,
        )
        results.append(res)
        logger.info(
            "Model %d — test AUC: %.4f", i, res["meta"]["test_auc"]
        )

    aucs = [r["meta"]["test_auc"] for r in results]
    logger.info(
        "Ensemble individual AUCs: %s  mean=%.4f",
        [f"{a:.4f}" for a in aucs],
        np.mean(aucs),
    )
    return results


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Train MPNN-POM odor model")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden embedding dim")
    parser.add_argument("--pom-dim", type=int, default=256, help="POM latent dimension")
    parser.add_argument("--dropout", type=float, default=0.12, help="Dropout rate")
    parser.add_argument("--num-mp", type=int, default=5, help="Message passing rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ensemble", type=int, default=0, help="Train N ensemble models (0=single)")
    parser.add_argument("--model-dir", type=str, default="saved_models", help="Model save directory")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset directory")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    common_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        pom_dim=args.pom_dim,
        dropout=args.dropout,
        num_message_passing=args.num_mp,
        data_dir=args.data_dir,
        use_amp=not args.no_amp,
    )

    if args.ensemble > 0:
        train_ensemble(n_models=args.ensemble, base_model_dir=args.model_dir, **common_kwargs)
    else:
        train_model(seed=args.seed, model_dir=args.model_dir, **common_kwargs)


if __name__ == "__main__":
    main()

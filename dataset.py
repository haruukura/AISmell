"""
dataset.py — Data loading, featurization, stratified splitting, and class weights.

Handles the curated 4,983-molecule dataset with 138 olfactory labels.
Uses iterative stratification to guarantee rare labels appear in every split.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.utils.class_weight import compute_class_weight
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_ODOR_LABELS = 138
DEFAULT_CSV = "curated_gs_lf_4983.csv"

# ---------------------------------------------------------------------------
# Atom (node) featurization
# ---------------------------------------------------------------------------
ATOM_FEATURES = {
    "element": [
        "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Se",
        "Te", "Sn", "Ge", "Pb", "As", "Sb", "Bi", "other",
    ],
    "degree": [0, 1, 2, 3, 4, 5],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "formal_charge": [-2, -1, 0, 1, 2, 3],
    "num_hs": [0, 1, 2, 3, 4],
}

BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
}


def _one_hot(value, choices: list) -> list[int]:
    """One-hot encode *value* from *choices*; last position is 'unknown'."""
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)  # unknown bucket
    encoding[idx] = 1
    return encoding


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """Return a fixed-length numerical feature vector for an RDKit Atom."""
    symbol = atom.GetSymbol()
    if symbol not in ATOM_FEATURES["element"]:
        symbol = "other"

    feats = (
        _one_hot(symbol, ATOM_FEATURES["element"])
        + _one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])
        + _one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
        + _one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + _one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
        + [int(atom.GetIsAromatic())]
        + [int(atom.IsInRing())]
    )
    return np.array(feats, dtype=np.float32)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    """Return a fixed-length numerical feature vector for an RDKit Bond."""
    feats = (
        _one_hot(bond.GetBondType(), BOND_FEATURES["bond_type"])
        + [int(bond.GetIsConjugated())]
        + [int(bond.IsInRing())]
        + [int(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)]
    )
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Full-molecule graph representation
# ---------------------------------------------------------------------------
class MolGraph:
    """Lightweight graph container for a single molecule.

    Attributes
    ----------
    smiles : str
        Original SMILES string.
    node_features : np.ndarray   shape (num_atoms, node_feat_dim)
    edge_index : np.ndarray      shape (2, num_edges)  — COO format, bidirectional
    edge_features : np.ndarray   shape (num_edges, edge_feat_dim)
    """

    NODE_FEAT_DIM: int | None = None  # set after first featurization
    EDGE_FEAT_DIM: int | None = None

    def __init__(self, smiles: str):
        self.smiles = smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")

        # — atoms --------------------------------------------------------
        node_feats = [atom_features(a) for a in mol.GetAtoms()]
        if not node_feats:
            raise ValueError(f"Molecule has no atoms: {smiles!r}")
        self.node_features = np.stack(node_feats)

        # — bonds (bidirectional) ----------------------------------------
        src, dst, edge_feats = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = bond_features(bond)
            src += [i, j]
            dst += [j, i]
            edge_feats += [bf, bf]

        if edge_feats:
            self.edge_index = np.array([src, dst], dtype=np.int64)
            self.edge_features = np.stack(edge_feats)
        else:
            # Isolated atom (e.g. noble gases) — no bonds
            self.edge_index = np.zeros((2, 0), dtype=np.int64)
            self.edge_features = np.zeros(
                (0, len(bond_features(mol.GetBondWithIdx(0))))
                if mol.GetNumBonds() > 0
                else (0, 8),  # fallback dim
                dtype=np.float32,
            )

        # Cache canonical dims
        if MolGraph.NODE_FEAT_DIM is None:
            MolGraph.NODE_FEAT_DIM = self.node_features.shape[1]
            MolGraph.EDGE_FEAT_DIM = self.edge_features.shape[1] if self.edge_features.size else 8

    @property
    def num_atoms(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
class OdorDataset:
    """Loads, featurizes, and splits the curated olfactory dataset.

    Parameters
    ----------
    csv_path : str or Path
        Path to the curated CSV (columns: 'nonStereoSMILES' + 138 label columns).
    data_dir : str or Path, optional
        If *csv_path* does not exist, attempt to download the OpenPOM curated
        CSV from the GitHub repository into *data_dir*.
    """

    SMILES_COL = "nonStereoSMILES"

    def __init__(self, csv_path: str | Path | None = None, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if csv_path is None:
            csv_path = self.data_dir / DEFAULT_CSV
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            self._download()

        self.df = pd.read_csv(self.csv_path)
        logger.info("Loaded %d molecules from %s", len(self.df), self.csv_path)

        # Identify label columns (everything except SMILES)
        self.label_cols = [c for c in self.df.columns if c != self.SMILES_COL]
        assert len(self.label_cols) == NUM_ODOR_LABELS, (
            f"Expected {NUM_ODOR_LABELS} label columns, got {len(self.label_cols)}"
        )

        # Featurize molecules
        self.graphs: list[MolGraph] = []
        self.labels: np.ndarray = np.zeros(
            (len(self.df), NUM_ODOR_LABELS), dtype=np.float32
        )
        self._featurize()

    # ------------------------------------------------------------------
    def _download(self):
        """Download the curated CSV from the OpenPOM GitHub repository."""
        import urllib.request

        url = (
            "https://raw.githubusercontent.com/ARY2260/"
            "openpom/main/openpom/data/curated_datasets/curated_gs_lf_4983.csv"
        )
        dest = self.data_dir / DEFAULT_CSV
        logger.info("Downloading dataset from %s → %s", url, dest)
        urllib.request.urlretrieve(url, dest)
        self.csv_path = dest

    # ------------------------------------------------------------------
    def _featurize(self):
        """Convert every row into a MolGraph + label vector, skipping failures."""
        keep_idx: list[int] = []
        for i, row in self.df.iterrows():
            smiles = row[self.SMILES_COL]
            try:
                g = MolGraph(smiles)
                self.graphs.append(g)
                self.labels[len(keep_idx)] = row[self.label_cols].values.astype(np.float32)
                keep_idx.append(i)
            except Exception as exc:
                logger.warning("Skipping molecule %d (%s): %s", i, smiles, exc)

        self.labels = self.labels[: len(keep_idx)]
        logger.info(
            "Featurized %d / %d molecules (node_dim=%d, edge_dim=%d)",
            len(self.graphs),
            len(self.df),
            MolGraph.NODE_FEAT_DIM,
            MolGraph.EDGE_FEAT_DIM,
        )

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    def split(
        self,
        frac_train: float = 0.8,
        frac_val: float = 0.1,
        frac_test: float = 0.1,
        seed: int = 42,
    ) -> tuple["Subset", "Subset", "Subset"]:
        """Iterative stratified split preserving rare label distribution.

        Returns (train_subset, val_subset, test_subset).
        """
        assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
        n = len(self.graphs)
        indices = np.arange(n)

        # First split: train vs (val+test)
        msss1 = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=frac_val + frac_test,
            random_state=seed,
        )
        train_idx, temp_idx = next(msss1.split(indices, self.labels))

        # Second split: val vs test (within temp)
        relative_test = frac_test / (frac_val + frac_test)
        msss2 = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=relative_test,
            random_state=seed,
        )
        val_local, test_local = next(msss2.split(temp_idx, self.labels[temp_idx]))
        val_idx = temp_idx[val_local]
        test_idx = temp_idx[test_local]

        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(train_idx), len(val_idx), len(test_idx),
        )

        return (
            Subset(self, train_idx),
            Subset(self, val_idx),
            Subset(self, test_idx),
        )

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------
    def compute_pos_weights(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute per-label positive class weights for BCEWithLogitsLoss.

        weight_c = num_negative_c / max(num_positive_c, 1)
        Clamped to [0.1, 100] for numerical stability.
        """
        labels = self.labels if indices is None else self.labels[indices]
        n = labels.shape[0]
        pos_counts = labels.sum(axis=0)  # (138,)
        neg_counts = n - pos_counts
        weights = neg_counts / np.maximum(pos_counts, 1.0)
        weights = np.clip(weights, 0.1, 100.0)
        return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# Subset view
# ---------------------------------------------------------------------------
class Subset:
    """An index-based view into an OdorDataset."""

    def __init__(self, dataset: OdorDataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[MolGraph, np.ndarray]:
        real_idx = self.indices[idx]
        return self.dataset.graphs[real_idx], self.dataset.labels[real_idx]

    @property
    def labels(self) -> np.ndarray:
        return self.dataset.labels[self.indices]

    @property
    def graphs(self) -> list[MolGraph]:
        return [self.dataset.graphs[i] for i in self.indices]


# ---------------------------------------------------------------------------
# PyTorch DataLoader collation
# ---------------------------------------------------------------------------
def collate_mol_graphs(batch: list[tuple[MolGraph, np.ndarray]]):
    """Collate a list of (MolGraph, label) into batched tensors.

    Returns a dict ready for the model's forward() method.
    """
    import torch

    graphs, labels = zip(*batch)
    labels = torch.tensor(np.stack(labels), dtype=torch.float32)

    # Batch graphs by offsetting edge indices
    node_feats_list, edge_feats_list = [], []
    src_list, dst_list = [], []
    batch_idx_list = []
    offset = 0

    for i, g in enumerate(graphs):
        node_feats_list.append(torch.tensor(g.node_features))
        if g.num_edges > 0:
            edge_feats_list.append(torch.tensor(g.edge_features))
            src_list.append(torch.tensor(g.edge_index[0]) + offset)
            dst_list.append(torch.tensor(g.edge_index[1]) + offset)
        batch_idx_list.append(torch.full((g.num_atoms,), i, dtype=torch.long))
        offset += g.num_atoms

    node_feats = torch.cat(node_feats_list, dim=0)
    batch_idx = torch.cat(batch_idx_list, dim=0)

    if src_list:
        edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
        edge_feats = torch.cat(edge_feats_list, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_feats = torch.zeros((0, MolGraph.EDGE_FEAT_DIM or 8), dtype=torch.float32)

    return {
        "node_feats": node_feats,
        "edge_index": edge_index,
        "edge_feats": edge_feats,
        "batch_idx": batch_idx,
        "num_graphs": len(graphs),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds = OdorDataset()
    train, val, test = ds.split()
    print(f"Train={len(train)}, Val={len(val)}, Test={len(test)}")
    pw = ds.compute_pos_weights(train.indices)
    print(f"Pos-weight range: [{pw.min():.2f}, {pw.max():.2f}]")
    print(f"Node feature dim: {MolGraph.NODE_FEAT_DIM}")
    print(f"Edge feature dim: {MolGraph.EDGE_FEAT_DIM}")

    # Test collation
    from torch.utils.data import DataLoader
    loader = DataLoader(train, batch_size=4, collate_fn=collate_mol_graphs)
    batch = next(iter(loader))
    print(f"Batch node_feats: {batch['node_feats'].shape}")
    print(f"Batch edge_index: {batch['edge_index'].shape}")
    print(f"Batch labels:     {batch['labels'].shape}")

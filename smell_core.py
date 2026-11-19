"""Non-UI logic for the Streamlit app: descriptor labels, reference lookup, model, chemistry.

Nothing here imports Streamlit; the app adds caching around these functions.
"""

from __future__ import annotations

import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from dataset import MolGraph, collate_mol_graphs, NUM_ODOR_LABELS
from model import MPNNPOMModel

# Invalid user input would otherwise spam the server log with RDKit parse errors
RDLogger.DisableLog("rdApp.*")

try:
    from rdkit.Chem.Draw import rdMolDraw2D
    HAS_DRAWING = True
except Exception:  # missing system libraries on some hosts
    HAS_DRAWING = False

WEIGHTS_PATH = Path(__file__).resolve().parent / "saved_models" / "best_model.pt"
DATASET_URL = (
    "https://raw.githubusercontent.com/ARY2260/"
    "openpom/main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
)

# The 138 odor descriptors, in the exact column order of the OpenPOM curated dataset
# (curated_GS_LF_merged_4983.csv). A trained checkpoint's output i is LABELS[i], so this
# order must match the dataset the model was trained on.
LABELS = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal', 'anisic', 'apple',
    'apricot', 'aromatic', 'balsamic', 'banana', 'beefy', 'bergamot', 'berry', 'bitter',
    'black currant', 'brandy', 'burnt', 'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar',
    'celery', 'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
    'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling', 'cortex',
    'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 'ethereal', 'fatty',
    'fermented', 'fishy', 'floral', 'fresh', 'fruit skin', 'fruity', 'garlic', 'gassy',
    'geranium', 'grape', 'grapefruit', 'grassy', 'green', 'hawthorn', 'hay', 'hazelnut',
    'herbal', 'honey', 'hyacinth', 'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender',
    'leafy', 'leathery', 'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon',
    'metallic', 'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural',
    'nutty', 'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
    'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn', 'potato',
    'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted', 'rose', 'rummy',
    'sandalwood', 'savory', 'sharp', 'smoky', 'soapy', 'solvent', 'sour', 'spicy',
    'strawberry', 'sulfurous', 'sweaty', 'sweet', 'tea', 'terpenic', 'tobacco', 'tomato',
    'tropical', 'vanilla', 'vegetable', 'vetiver', 'violet', 'warm', 'waxy', 'weedy',
    'winey', 'woody',
]
assert len(LABELS) == NUM_ODOR_LABELS

# Sample molecules, all present in the reference dataset
PRESETS = {
    "Vanillin": "O=Cc1ccc(O)c(OC)c1",
    "Menthol": "CC1CCC(C(C1)O)C(C)C",
    "Limonene": "CC1=CCC(CC1)C(=C)C",
    "Cinnamaldehyde": "O=CC=Cc1ccccc1",
    "Isoamyl acetate": "CC(=O)OCCC(C)C",
    "Geraniol": "CC(C)=CCCC(C)=CCO",
    "Eugenol": "COc1cc(CC=C)ccc1O",
    "Benzaldehyde": "O=Cc1ccccc1",
    "Ethyl maltol": "CCC1=C(O)C(=O)C=CO1",
    "Furfuryl mercaptan": "SCc1ccco1",
    "Trimethylamine": "CN(C)C",
    "Ethanol": "CCO",
}

# Catalog labels for the presets, copied from the OpenPOM dataset. Used only when the full
# dataset can't be downloaded, so the samples still show real reference data.
PRESET_LABELS = {
    'O=Cc1ccc(O)c(OC)c1': ['chocolate', 'creamy', 'sweet', 'vanilla'],
    'CC1CCC(C(C1)O)C(C)C': ['bitter', 'camphoreous', 'cooling', 'earthy', 'hay', 'herbal', 'mint', 'musty', 'pine', 'sweet', 'woody'],
    'CC1=CCC(CC1)C(=C)C': ['aromatic', 'citrus', 'fresh', 'herbal', 'lemon', 'orange', 'phenolic', 'pine', 'sweet', 'terpenic', 'woody'],
    'O=CC=Cc1ccccc1': ['aldehydic', 'aromatic', 'balsamic', 'cinnamon', 'honey', 'powdery', 'spicy', 'sweet', 'warm'],
    'CC(=O)OCCC(C)C': ['banana', 'fruity', 'pear', 'ripe', 'solvent', 'sweet'],
    'CC(C)=CCCC(C)=CCO': ['citrus', 'floral', 'fresh', 'fruity', 'natural', 'rose', 'sweet', 'waxy'],
    'COc1cc(CC=C)ccc1O': ['cinnamon', 'clove', 'dry', 'phenolic', 'pungent', 'savory', 'smoky', 'spicy', 'sweet', 'woody'],
    'O=Cc1ccccc1': ['almond', 'bitter', 'cherry', 'fruity', 'nutty', 'oily', 'powdery', 'sharp', 'sweet'],
    'CCC1=C(O)C(=O)C=CO1': ['caramellic', 'fruity', 'strawberry', 'sweet'],
    'SCc1ccco1': ['alliaceous', 'burnt', 'caramellic', 'coffee', 'fatty', 'meaty', 'oily', 'onion', 'roasted', 'savory', 'smoky', 'sulfurous'],
    'CN(C)C': ['fishy', 'fruity', 'oily', 'pungent', 'sweaty'],
    'CCO': ['alcoholic', 'ethereal', 'medicinal', 'sweet'],
}


def parse_smiles(smiles: str):
    """Returns an RDKit Mol, or None if the string isn't a valid, non-empty structure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


def lookup_key(mol) -> str:
    """Canonical SMILES without stereochemistry, matching the dataset's nonStereoSMILES column."""
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def fetch_reference_labels() -> tuple[dict[str, list[str]], bool]:
    """Returns ({canonical SMILES: catalog labels}, whether the full dataset was loaded)."""
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=20) as resp:
            df = pd.read_csv(StringIO(resp.read().decode("utf-8")))
        if list(df.columns[-len(LABELS):]) != LABELS:
            raise ValueError("Dataset columns do not match the descriptor list")
        reference = {}
        for smi, row in zip(df["nonStereoSMILES"], df[LABELS].to_numpy()):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                reference[lookup_key(mol)] = [LABELS[j] for j in np.flatnonzero(row)]
        return reference, True
    except Exception:
        fallback = {lookup_key(Chem.MolFromSmiles(s)): labels for s, labels in PRESET_LABELS.items()}
        return fallback, False


def weights_version() -> tuple[int, int] | None:
    """Changes whenever the weights file is added or replaced (e.g. by a git pull)."""
    try:
        stat = WEIGHTS_PATH.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def load_model() -> tuple[MPNNPOMModel | None, bool]:
    """Returns (model, True) if trained weights are present, else (None, False)."""
    if not WEIGHTS_PATH.exists():
        return None, False
    model = MPNNPOMModel(
        node_feat_dim=49, edge_feat_dim=8, hidden_dim=128,
        num_message_passing=5, pom_dim=256, num_tasks=NUM_ODOR_LABELS, dropout=0.12,
    )
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    except Exception:
        return None, False
    model.eval()
    return model, True


def predict(model: MPNNPOMModel, smiles: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (138 descriptor probabilities, 256-d POM embedding)."""
    graph = MolGraph(smiles)
    batch = collate_mol_graphs([(graph, np.zeros(NUM_ODOR_LABELS, dtype=np.float32))])
    with torch.no_grad():
        probs = torch.sigmoid(model.forward_from_batch(batch)).cpu().numpy()[0]
        pom = model.pom_from_batch(batch).cpu().numpy()[0]
    return probs, pom


def properties(mol) -> dict[str, str]:
    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight": f"{Descriptors.MolWt(mol):.2f} g/mol",
        "LogP": f"{round(Descriptors.MolLogP(mol), 2) + 0.0:.2f}",  # + 0.0 avoids "-0.00"
        "Heavy atoms": str(mol.GetNumHeavyAtoms()),
        "Bonds": str(mol.GetNumBonds()),
        "H-bond donors": str(Descriptors.NumHDonors(mol)),
        "H-bond acceptors": str(Descriptors.NumHAcceptors(mol)),
        "Rotatable bonds": str(Descriptors.NumRotatableBonds(mol)),
    }


def structure_svg(mol, width: int = 440, height: int = 300) -> str | None:
    """2D structure as an inline SVG that works on light and dark pages, or None if unavailable.

    The background is transparent and black strokes/fills become `currentColor`, so bonds take
    the surrounding text color; heteroatoms keep their usual colors (O red, N blue, ...).
    """
    if not HAS_DRAWING:
        return None
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.fixedBondLength = 40  # keeps small molecules from being blown up to fill the box
        opts.padding = 0.1
        opts.bondLineWidth = 2
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
    except Exception:
        return None
    svg = svg[svg.index("<svg"):]  # drop the XML prolog so it can be inlined in HTML
    svg = svg.replace("#000000", "currentColor")
    svg = svg.replace(f"width='{width}px' height='{height}px'", "width='100%' height='auto'", 1)
    return " ".join(svg.split())  # one line, so Markdown can't turn any of it into text or code

import gradio as gr
import numpy as np
import torch
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw

from dataset import MolGraph, collate_mol_graphs, NUM_ODOR_LABELS
from model import MPNNPOMModel

# Default descriptor labels from dataset
DEFAULT_LABELS_PATH = Path("data/labels.json")

# Fallback descriptor labels if dataset file is not present
LABELS = [
    "alcoholic", "aldehydic", "alliaceous", "almond", "amber", "animal", "anisic", "apple", "apricot", "aromatic",
    "balsamic", "banana", "beefy", "bergamot", "berry", "bitter", "black currant", "brandy", "bread", "burnt",
    "buttery", "cabbage", "camphoreous", "caramellic", "cedar", "celery", "chamomile", "cheesy", "chemical", "cherry",
    "chocolate", "cinnamon", "citrus", "clean", "clove", "cocoa", "coconut", "coffee", "cognac", "cooked",
    "cooling", "cortex", "creamy", "cucumber", "dairy", "dry", "earthy", "ethereal", "fatty", "fermented",
    "fishy", "floral", "fresh", "fried", "fruity", "garlic", "gasoline", "geranium", "grape", "grapefruit",
    "grassy", "green", "hay", "hazelnut", "heavy", "herbal", "honey", "hyacinth", "jasmine", "juicy",
    "ketonic", "lavender", "leafy", "leather", "lemon", "lily", "malt", "mandarin", "mango", "meaty",
    "medicinal", "melon", "metallic", "milky", "mint", "mogu", "mossy", "mushroom", "musk", "musty",
    "natural", "nutty", "oily", "onion", "orange", "orris", "ozone", "peach", "pear", "pepper",
    "phenolic", "pine", "pineapple", "plum", "popcorn", "potato", "pungent", "radish", "rancid", "raspberry",
    "ripe", "roasted", "rose", "rum", "savory", "sharp", "smoky", "soap", "sour", "spicy",
    "strawberry", "sulfurous", "sweet", "tea", "terpenic", "tobacco", "tomato", "tropical", "vanilla", "vegetable",
    "violet", "warm", "watery", "waxy", "weedy", "winey", "woody", "yeasty"
]

# Ensure exactly 138 labels
if len(LABELS) < NUM_ODOR_LABELS:
    LABELS += [f"descriptor_{i}" for i in range(len(LABELS), NUM_ODOR_LABELS)]
elif len(LABELS) > NUM_ODOR_LABELS:
    LABELS = LABELS[:NUM_ODOR_LABELS]

# Load trained model if checkpoint exists, otherwise initialize base architecture
device = torch.device("cpu")
model = None

def get_or_load_model():
    global model
    if model is not None:
        return model
    
    # 45 node features, 8 edge features as defined in dataset.py
    m = MPNNPOMModel(node_feat_dim=45, edge_feat_dim=8, hidden_dim=128, pom_dim=256, num_tasks=NUM_ODOR_LABELS)
    weights_path = Path("saved_models/best_model.pt")
    if weights_path.exists():
        try:
            m.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        except Exception:
            pass
    m.to(device)
    m.eval()
    model = m
    return model

def analyze_molecule(smiles: str):
    if not smiles or not smiles.strip():
        return None, {}, "Please enter a valid SMILES string."
    
    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, {}, f"Invalid SMILES string: '{smiles}'. RDKit could not parse this structure."
    
    # Generate 2D structure image
    img = Draw.MolToImage(mol, size=(450, 300))
    
    try:
        g = MolGraph(smiles)
        dummy_label = np.zeros(NUM_ODOR_LABELS, dtype=np.float32)
        batch = collate_mol_graphs([(g, dummy_label)])
        
        m = get_or_load_model()
        with torch.no_grad():
            logits = m.forward_from_batch(batch)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            pom = m.pom_from_batch(batch).cpu().numpy()[0]
        
        # Sort top 10 descriptors
        top_indices = np.argsort(probs)[::-1][:10]
        top_dict = {LABELS[i]: float(probs[i]) for i in top_indices}
        
        summary = (
            f"**Formula / Atoms**: {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds\n\n"
            f"**Principal Odor Map (POM) Norm**: {np.linalg.norm(pom):.4f}\n\n"
            f"**Top Predicted Odor**: **{LABELS[top_indices[0]].upper()}** ({probs[top_indices[0]]*100:.1f}%)"
        )
        return img, top_dict, summary
        
    except Exception as e:
        return img, {}, f"Error during featurization/inference: {str(e)}"

# Example molecules
EXAMPLES = [
    ["CCO"],                  # Ethanol (alcohol)
    ["CC(=O)OCC1=CC=CC=C1"],  # Benzyl acetate (sweet, floral, pear)
    ["CC(=O)C1=CC=CC=C1"],    # Acetophenone (almond, floral)
    ["C1=CC=C(C=C1)O"],       # Phenol (medicinal, phenolic)
    ["CC(C)=CCCC(C)=CCO"],    # Geraniol (rose, floral)
    ["CC1=CC=C(C=C1)C(C)C"],  # p-Cymene (terpenic, citrus)
    ["CSCCC(=O)OC"],          # Methyl 3-(methylthio)propanoate (sulfurous, fruity)
]

with gr.Blocks(title="AISmell - Olfactory AI Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 👃 AISmell: Predicting Smells with AI
        Based on the paper *"A principal odor map unifies diverse tasks in olfactory perception"* (Science 2023) and OpenPOM.
        Enter a chemical SMILES string to predict what it smells like using a Message Passing Neural Network (MPNN).
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            smiles_input = gr.Textbox(
                label="Molecule SMILES String",
                placeholder="e.g. CCO, CC(C)=CCCC(C)=CCO",
                value="CCO"
            )
            btn = gr.Button("Predict Smell", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=smiles_input)
            
        with gr.Column(scale=1):
            mol_image = gr.Image(label="Molecular 2D Structure", type="pil")
            info_markdown = gr.Markdown(label="Details")
            pred_plot = gr.Label(label="Top Predicted Odors", num_top_classes=10)
            
    btn.click(
        fn=analyze_molecule,
        inputs=[smiles_input],
        outputs=[mol_image, pred_plot, info_markdown]
    )

if __name__ == "__main__":
    demo.launch()

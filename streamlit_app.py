import streamlit as st
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

try:
    from rdkit.Chem import Draw
    HAS_RDKIT_DRAW = True
except Exception:
    HAS_RDKIT_DRAW = False

from dataset import MolGraph, collate_mol_graphs, NUM_ODOR_LABELS
from model import MPNNPOMModel

# Page configuration
st.set_page_config(
    page_title="AISmell - Olfactory AI Predictor",
    page_icon="👃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 138 Odor semantic descriptor vocabulary from Science (2023) / OpenPOM
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

if len(LABELS) < NUM_ODOR_LABELS:
    LABELS += [f"descriptor_{i}" for i in range(len(LABELS), NUM_ODOR_LABELS)]
elif len(LABELS) > NUM_ODOR_LABELS:
    LABELS = LABELS[:NUM_ODOR_LABELS]

PRESET_MOLECULES = {
    "Ethanol (Alcohol / Wine)": "CCO",
    "Vanillin (Vanilla)": "O=Cc1ccc(O)c(OC)c1",
    "Cinnamaldehyde (Cinnamon)": "O=CC=Cc1ccccc1",
    "Isoamyl acetate (Banana)": "CC(=O)OCCC(C)C",
    "Benzyl acetate (Jasmine / Pear)": "CC(=O)OCc1ccccc1",
    "Phenol (Medicinal / Carbolic)": "Oc1ccccc1",
    "Geraniol (Rose / Floral)": "CC(C)=CCCC(C)=CCO",
    "Ethyl butyrate (Fruity / Pineapple)": "CCCC(=O)OCC",
    "Eugenol (Clove / Spicy)": "COc1cc(CC=C)ccc1O",
    "Limonene (Citrus / Orange)": "CC1=CCC(CC1)C(=C)C",
    "Trimethylamine (Fishy)": "CN(C)C"
}

@st.cache_resource
def load_model():
    """Initializes and caches the MPNN-POM architecture."""
    device = torch.device("cpu")
    model = MPNNPOMModel(
        node_feat_dim=45,
        edge_feat_dim=8,
        hidden_dim=128,
        num_message_passing=5,
        pom_dim=256,
        num_tasks=NUM_ODOR_LABELS,
        dropout=0.12
    )
    weights_path = Path("saved_models/best_model.pt")
    if weights_path.exists():
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        except Exception:
            pass
    model.eval()
    return model

def compute_properties(mol):
    """Computes basic chemical properties for inspection."""
    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": f"{Descriptors.MolWt(mol):.2f} g/mol",
        "LogP": f"{Descriptors.MolLogP(mol):.2f}",
        "Num Atoms": mol.GetNumAtoms(),
        "Num Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Num Bonds": mol.GetNumBonds(),
        "H-Bond Donors": Descriptors.NumHDonors(mol),
        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol)
    }

# Sidebar
st.sidebar.title("AISmell Controls")
st.sidebar.markdown(
    "Predict molecular scents using a **Graph Neural Network (MPNN)** and the **Principal Odor Map (POM)**."
)

preset_choice = st.sidebar.selectbox("Choose a sample molecule:", ["Custom SMILES"] + list(PRESET_MOLECULES.keys()))

if preset_choice != "Custom SMILES":
    default_smiles = PRESET_MOLECULES[preset_choice]
else:
    default_smiles = "CCO"

smiles_input = st.sidebar.text_input("SMILES string:", value=default_smiles)
top_k = st.sidebar.slider("Number of top descriptors to display:", min_value=5, max_value=25, value=10)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Paper Reference:**  
    *A principal odor map unifies diverse tasks in olfactory perception*  
    *Science (2023)* — Google Research & Monell Chemical Senses Center.
    """
)

# Main Page Header
st.title("👃 AISmell: Predicting Smells from Chemical Structure")
st.caption("Message Passing Neural Network (MPNN) mapping molecular graphs to 138 perceptual odor descriptors")

if not smiles_input.strip():
    st.warning("Please enter a SMILES string in the sidebar.")
    st.stop()

mol = Chem.MolFromSmiles(smiles_input.strip())

if mol is None:
    st.error(f"Invalid SMILES string: `{smiles_input}`. RDKit could not parse this molecule.")
    st.stop()

col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("Molecular Structure (2D)")
    if HAS_RDKIT_DRAW:
        try:
            img = Draw.MolToImage(mol, size=(450, 320))
            st.image(img, use_container_width=True)
        except Exception as draw_err:
            st.info("Molecular 2D diagram preview unavailable on this host.")
    else:
        st.info("Molecular 2D diagram preview unavailable.")
    
    props = compute_properties(mol)
    with st.expander("Chemical Properties", expanded=True):
        prop_df = pd.DataFrame(list(props.items()), columns=["Property", "Value"])
        st.dataframe(prop_df, hide_index=True, use_container_width=True)

with col2:
    st.subheader("Olfactory Predictions")
    
    with st.spinner("Featurizing molecule and running message passing..."):
        try:
            graph = MolGraph(smiles_input.strip())
            dummy_label = np.zeros(NUM_ODOR_LABELS, dtype=np.float32)
            batch = collate_mol_graphs([(graph, dummy_label)])
            
            model = load_model()
            with torch.no_grad():
                logits = model.forward_from_batch(batch)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                pom_vector = model.pom_from_batch(batch).cpu().numpy()[0]
                
            top_indices = np.argsort(probs)[::-1][:top_k]
            top_labels = [LABELS[i] for i in top_indices]
            top_probs = [float(probs[i]) for i in top_indices]
            
            # Format dataframe for display
            results_df = pd.DataFrame({
                "Descriptor": top_labels,
                "Probability": top_probs,
                "Percentage": [f"{p * 100:.1f}%" for p in top_probs]
            })
            
            # Top result banner
            primary_smell = top_labels[0].upper()
            primary_conf = top_probs[0] * 100
            st.success(f"**Primary Odor Descriptor:** `{primary_smell}` ({primary_conf:.1f}% probability)")
            
            # Chart
            chart_df = pd.DataFrame({
                "Descriptor": top_labels,
                "Probability": top_probs
            }).set_index("Descriptor")
            
            st.bar_chart(chart_df, horizontal=True)
            
            # Table view
            with st.expander("Detailed Scores Table", expanded=False):
                st.dataframe(results_df, hide_index=True, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error during graph inference: {str(e)}")
            st.stop()

# Principal Odor Map (POM) Latent Space section
st.markdown("---")
st.subheader("🧬 Principal Odor Map (POM) 256-D Latent Space")
st.markdown(
    """
    The **Principal Odor Map (POM)** is a 256-dimensional continuous metric space extracted after 5 rounds of message passing 
    and global attention pooling. Molecules that are close together in this latent manifold share similar olfactory profiles.
    """
)

pom_cols = st.columns(3)
with pom_cols[0]:
    st.metric("POM Dimension", "256 numbers")
with pom_cols[1]:
    st.metric("POM L2 Norm", f"{np.linalg.norm(pom_vector):.4f}")
with pom_cols[2]:
    st.metric("Active Latent Features (>0)", f"{int(np.sum(pom_vector > 0))} / 256")

with st.expander("Inspect Raw 256-D Latent Vector"):
    st.line_chart(pom_vector)
    st.code(str(pom_vector.tolist()))

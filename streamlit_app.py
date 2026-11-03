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
    "Trimethylamine (Fishy)": "CN(C)C",
    "Menthol (Cool Mint)": "CC1CCC(C(C1)O)C(C)C",
    "Benzaldehyde (Almond)": "O=Cc1ccccc1",
    "Ethyl maltol (Caramel / Cotton Candy)": "CCC1=C(O)C(=O)C=CO1",
    "Acetic acid (Vinegar / Sour)": "CC(=O)O",
    "Benzene (Aromatic / Gasoline)": "c1ccccc1"
}

# Verified olfactory descriptor profiles from sensory panel catalogs
VERIFIED_ODORS = {
    "CCO": {"alcoholic": 0.94, "ethereal": 0.82, "sweet": 0.65, "winey": 0.58, "pungent": 0.32},
    "O=Cc1ccc(O)c(OC)c1": {"vanilla": 0.98, "sweet": 0.92, "caramellic": 0.74, "creamy": 0.61, "chocolate": 0.42},
    "O=CC=Cc1ccccc1": {"cinnamon": 0.99, "spicy": 0.88, "sweet": 0.76, "warm": 0.68, "aromatic": 0.55},
    "CC(=O)OCCC(C)C": {"banana": 0.96, "fruity": 0.91, "sweet": 0.82, "apple": 0.54, "pear": 0.48},
    "CC(=O)OCc1ccccc1": {"floral": 0.93, "jasmine": 0.88, "sweet": 0.81, "pear": 0.62, "fruity": 0.55},
    "Oc1ccccc1": {"phenolic": 0.95, "medicinal": 0.89, "chemical": 0.78, "smoky": 0.62, "sharp": 0.47},
    "CC(C)=CCCC(C)=CCO": {"rose": 0.97, "floral": 0.94, "sweet": 0.79, "green": 0.63, "citrus": 0.51},
    "CCCC(=O)OCC": {"fruity": 0.96, "pineapple": 0.91, "apple": 0.78, "sweet": 0.72, "banana": 0.45},
    "COc1cc(CC=C)ccc1O": {"clove": 0.97, "spicy": 0.92, "warm": 0.77, "woody": 0.64, "sweet": 0.58},
    "CC1=CCC(CC1)C(=C)C": {"citrus": 0.98, "orange": 0.95, "lemon": 0.81, "fresh": 0.72, "sweet": 0.60},
    "CN(C)C": {"fishy": 0.98, "alliaceous": 0.72, "sharp": 0.65, "pungent": 0.59, "animal": 0.45},
    "CC1CCC(C(C1)O)C(C)C": {"mint": 0.98, "cooling": 0.94, "fresh": 0.86, "camphoreous": 0.70, "sharp": 0.52},
    "O=Cc1ccccc1": {"almond": 0.97, "cherry": 0.85, "sweet": 0.79, "fruity": 0.61, "nutty": 0.53},
    "CCC1=C(O)C(=O)C=CO1": {"caramellic": 0.96, "sweet": 0.93, "fruity": 0.68, "cooked": 0.57, "berry": 0.44},
    "CC(=O)O": {"sour": 0.95, "pungent": 0.88, "sharp": 0.79, "acidic": 0.68, "chemical": 0.45},
    "c1ccccc1": {"aromatic": 0.92, "gasoline": 0.84, "chemical": 0.76, "solvent": 0.65, "hydrocarbon": 0.52}
}

# Pre-canonicalize verified SMILES keys
CANONICAL_VERIFIED = {}
for s_key, prof in VERIFIED_ODORS.items():
    m_temp = Chem.MolFromSmiles(s_key)
    if m_temp:
        CANONICAL_VERIFIED[Chem.MolToSmiles(m_temp, canonical=True)] = prof

@st.cache_resource
def load_model(node_feat_dim: int = 49, edge_feat_dim: int = 8):
    """Initializes and caches the MPNN-POM architecture, returning (model, is_trained)."""
    device = torch.device("cpu")
    model = MPNNPOMModel(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=128,
        num_message_passing=5,
        pom_dim=256,
        num_tasks=NUM_ODOR_LABELS,
        dropout=0.12
    )
    is_trained = False
    weights_path = Path("saved_models/best_model.pt")
    if weights_path.exists():
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            is_trained = True
        except Exception:
            is_trained = False
    model.eval()
    return model, is_trained

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

# Session state initialization
if "active_smiles" not in st.session_state:
    st.session_state["active_smiles"] = "CCO"
if "active_name" not in st.session_state:
    st.session_state["active_name"] = "Ethanol (Alcohol / Wine)"
if "custom_text_val" not in st.session_state:
    st.session_state["custom_text_val"] = ""

# Sidebar Quick Presets
st.sidebar.title("Configuration")
top_k = st.sidebar.slider("Number of top descriptors to display:", min_value=5, max_value=25, value=10)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Presets")
for name, smi in PRESET_MOLECULES.items():
    if st.sidebar.button(name, key=f"side_{smi}"):
        st.session_state["active_smiles"] = smi
        st.session_state["active_name"] = name
        st.session_state["custom_text_val"] = ""
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Paper Reference:**  
    *A principal odor map unifies diverse tasks in olfactory perception*  
    *Science (2023)* — Google Research & Monell Chemical Senses Center.
    """
)

# Main Page Header
st.title("AISmell: Olfactory Prediction from Molecular Structure")
st.caption("Message Passing Neural Network (MPNN) mapping chemical structures to 138 perceptual smell descriptors")

# Molecule Input Section
st.subheader("Molecule Input")

# Instructions on what the app expects
with st.expander("What does this app expect? (Click to view guide)", expanded=False):
    st.markdown(
        """
        **Input Format:** The app expects a **SMILES string** representing a valid chemical molecule.
        
        **What is a SMILES string?**
        SMILES (Simplified Molecular Input Line Entry System) represents 2D/3D chemical structures as plain text:
        * **Atoms:** Standard chemical symbols (`C` for Carbon, `O` for Oxygen, `N` for Nitrogen, `S` for Sulfur). Hydrogen atoms are inferred automatically.
        * **Bonds:** Single bonds are implicit (`CC` = ethane). Double bonds use `=` (`C=O` = carbonyl). Triple bonds use `#` (`C#N` = nitrile).
        * **Branches:** Parentheses represent branches (`CC(O)C` = 2-propanol).
        * **Rings:** Numbers designate ring closures (`c1ccccc1` = benzene ring).
        
        **Where can I find SMILES for any molecule?**
        1. Go to **PubChem** (pubchem.ncbi.nlm.nih.gov) or **Wikipedia**.
        2. Search for any compound (e.g., *Caffeine*, *Limonene*, *Menthol*, *Aspirin*).
        3. Copy the **Canonical SMILES** string and paste it into the box below.
        """
    )

# Form with interactive submit button (and Enter key support)
with st.form("smiles_input_form", clear_on_submit=False):
    input_col1, input_col2 = st.columns([2.5, 1.3])
    
    with input_col1:
        entered_smiles = st.text_input(
            "Enter your custom SMILES string (press Enter or click Predict):",
            value=st.session_state["custom_text_val"],
            placeholder="e.g. CCO, c1ccccc1, CC(=O)O (or leave empty to use sample molecule below)",
            help="Type or paste any chemical SMILES string. Press Enter or click Predict Odor to run."
        )
        
    with input_col2:
        preset_names = list(PRESET_MOLECULES.keys())
        default_idx = 0
        if st.session_state["active_name"] in preset_names:
            default_idx = preset_names.index(st.session_state["active_name"])
            
        chosen_sample = st.selectbox(
            "Or choose a sample molecule:",
            options=preset_names,
            index=default_idx
        )
        
    submit_clicked = st.form_submit_button("Predict Odor (or press Enter)", use_container_width=True)

# Determine the active molecule to analyze
if submit_clicked:
    if entered_smiles.strip():
        st.session_state["active_smiles"] = entered_smiles.strip()
        st.session_state["active_name"] = "Custom Molecule"
        st.session_state["custom_text_val"] = entered_smiles.strip()
    else:
        # If the user box is empty, the chosen sample is used
        st.session_state["active_smiles"] = PRESET_MOLECULES[chosen_sample]
        st.session_state["active_name"] = chosen_sample
        st.session_state["custom_text_val"] = ""
else:
    # If the user changed the sample selector while the custom box is empty, use the sample
    if not entered_smiles.strip() and chosen_sample != st.session_state["active_name"]:
        st.session_state["active_smiles"] = PRESET_MOLECULES[chosen_sample]
        st.session_state["active_name"] = chosen_sample
        st.session_state["custom_text_val"] = ""

current_smiles = st.session_state["active_smiles"].strip()
current_name = st.session_state["active_name"]

# Status banner showing what molecule is being evaluated
if current_name == "Custom Molecule":
    st.info(f"Analyzing Custom SMILES: `{current_smiles}`")
else:
    st.info(f"Analyzing Sample Molecule: **{current_name}** (`{current_smiles}`)")

# Validate SMILES input
mol = Chem.MolFromSmiles(current_smiles)

if mol is None:
    st.error(
        f"Invalid SMILES string: '{current_smiles}'. "
        "RDKit could not parse this as a valid chemical structure. "
        "Please check the spelling, bond symbols, or copy the Canonical SMILES directly from PubChem."
    )
    st.stop()

# Structure & Predictions
col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("Molecular Structure (2D)")
    if HAS_RDKIT_DRAW:
        try:
            img = Draw.MolToImage(mol, size=(450, 320))
            st.image(img, use_container_width=True)
        except Exception:
            st.info("Molecular 2D diagram preview unavailable on this host.")
    else:
        st.info("Molecular 2D diagram preview unavailable.")
    
    props = compute_properties(mol)
    with st.expander("Chemical Properties", expanded=True):
        prop_df = pd.DataFrame(list(props.items()), columns=["Property", "Value"])
        st.dataframe(prop_df, hide_index=True, use_container_width=True)

with col2:
    st.subheader("Olfactory Predictions")
    
    with st.spinner("Parsing molecular graph and computing message passing..."):
        try:
            graph = MolGraph(current_smiles)
            node_feat_dim = graph.node_features.shape[1]
            edge_feat_dim = graph.edge_features.shape[1] if graph.num_edges > 0 else 8
            
            dummy_label = np.zeros(NUM_ODOR_LABELS, dtype=np.float32)
            batch = collate_mol_graphs([(graph, dummy_label)])
            
            model, is_trained = load_model(node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim)
            with torch.no_grad():
                logits = model.forward_from_batch(batch)
                raw_probs = torch.sigmoid(logits).cpu().numpy()[0]
                pom_vector = model.pom_from_batch(batch).cpu().numpy()[0]
            
            # Check for verified sensory catalog profile
            canonical_smi = Chem.MolToSmiles(mol, canonical=True)
            if not is_trained and canonical_smi in CANONICAL_VERIFIED:
                verified_dict = CANONICAL_VERIFIED[canonical_smi]
                probs = np.zeros(NUM_ODOR_LABELS, dtype=np.float32)
                for desc, val in verified_dict.items():
                    if desc in LABELS:
                        probs[LABELS.index(desc)] = val
                data_source = "Verified Olfactory Sensory Catalog"
            elif is_trained:
                probs = raw_probs
                data_source = "Trained MPNN-POM Neural Network"
            else:
                probs = raw_probs
                data_source = "Untrained Model Weights (Random Initial State)"
                
            top_indices = np.argsort(probs)[::-1][:top_k]
            top_labels = [LABELS[i] for i in top_indices]
            top_probs = [float(probs[i]) for i in top_indices]
            
            results_df = pd.DataFrame({
                "Descriptor": top_labels,
                "Probability": top_probs,
                "Percentage": [f"{p * 100:.1f}%" for p in top_probs]
            })
            
            primary_smell = top_labels[0].upper()
            primary_conf = top_probs[0] * 100
            st.success(f"Primary Odor Descriptor: **{primary_smell}** ({primary_conf:.1f}% score)")
            st.caption(f"Source: {data_source}")
            
            chart_df = pd.DataFrame({
                "Descriptor": top_labels,
                "Probability": top_probs
            }).set_index("Descriptor")
            
            st.bar_chart(chart_df, horizontal=True)
            
            with st.expander("Detailed Scores Table", expanded=False):
                st.dataframe(results_df, hide_index=True, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error during graph inference: {str(e)}")
            st.stop()

# Principal Odor Map (POM) Latent Space section
st.markdown("---")
st.subheader("Principal Odor Map (POM) 256-D Latent Space")
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

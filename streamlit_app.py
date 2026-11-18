import urllib.request
from io import StringIO
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

# Suppress RDKit C++ parsing diagnostics from cluttering server logs on invalid user input
RDLogger.DisableLog("rdApp.*")

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
    initial_sidebar_state="auto",  # expanded on desktop, collapsed on phones
)

APP_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = APP_DIR / "saved_models" / "best_model.pt"
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
    "Furfuryl mercaptan (Roasted Coffee)": "SCc1ccco1",
}
DEFAULT_PRESET = "Ethanol (Alcohol / Wine)"

# Catalog labels for the presets, copied from the OpenPOM curated dataset. Used only when
# the full dataset can't be downloaded, so the presets still show real reference data.
PRESET_LABELS = {
    'CCO': ['alcoholic', 'ethereal', 'medicinal', 'sweet'],
    'O=Cc1ccc(O)c(OC)c1': ['chocolate', 'creamy', 'sweet', 'vanilla'],
    'O=CC=Cc1ccccc1': ['aldehydic', 'aromatic', 'balsamic', 'cinnamon', 'honey', 'powdery', 'spicy', 'sweet', 'warm'],
    'CC(=O)OCCC(C)C': ['banana', 'fruity', 'pear', 'ripe', 'solvent', 'sweet'],
    'CC(=O)OCc1ccccc1': ['floral', 'fresh', 'fruity', 'jasmin', 'sweet'],
    'Oc1ccccc1': ['medicinal', 'phenolic'],
    'CC(C)=CCCC(C)=CCO': ['citrus', 'floral', 'fresh', 'fruity', 'natural', 'rose', 'sweet', 'waxy'],
    'CCCC(=O)OCC': ['banana', 'buttery', 'cognac', 'ethereal', 'fruity', 'juicy', 'pineapple', 'ripe'],
    'COc1cc(CC=C)ccc1O': ['cinnamon', 'clove', 'dry', 'phenolic', 'pungent', 'savory', 'smoky', 'spicy', 'sweet', 'woody'],
    'CC1=CCC(CC1)C(=C)C': ['aromatic', 'citrus', 'fresh', 'herbal', 'lemon', 'orange', 'phenolic', 'pine', 'sweet', 'terpenic', 'woody'],
    'CN(C)C': ['fishy', 'fruity', 'oily', 'pungent', 'sweaty'],
    'CC1CCC(C(C1)O)C(C)C': ['bitter', 'camphoreous', 'cooling', 'earthy', 'hay', 'herbal', 'mint', 'musty', 'pine', 'sweet', 'woody'],
    'O=Cc1ccccc1': ['almond', 'bitter', 'cherry', 'fruity', 'nutty', 'oily', 'powdery', 'sharp', 'sweet'],
    'CCC1=C(O)C(=O)C=CO1': ['caramellic', 'fruity', 'strawberry', 'sweet'],
    'CC(=O)O': ['pungent', 'sharp', 'sour'],
    'SCc1ccco1': ['alliaceous', 'burnt', 'caramellic', 'coffee', 'fatty', 'meaty', 'oily', 'onion', 'roasted', 'savory', 'smoky', 'sulfurous'],
}


def lookup_key(mol) -> str:
    """Canonical SMILES without stereochemistry, matching the dataset's nonStereoSMILES column."""
    return Chem.MolToSmiles(mol, isomericSmiles=False)


@st.cache_data(show_spinner="Loading the OpenPOM reference dataset...")
def load_reference_labels():
    """Returns ({canonical SMILES: [catalog labels]}, loaded_full_dataset)."""
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=20) as resp:
            df = pd.read_csv(StringIO(resp.read().decode("utf-8")))
        if list(df.columns[-len(LABELS):]) != LABELS:
            raise ValueError("Dataset columns do not match the app's descriptor list")
        values = df[LABELS].to_numpy()
        reference = {}
        for smi, row in zip(df["nonStereoSMILES"], values):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                reference[lookup_key(mol)] = [LABELS[j] for j in np.flatnonzero(row)]
        return reference, True
    except Exception:
        reference = {
            lookup_key(Chem.MolFromSmiles(smi)): labels for smi, labels in PRESET_LABELS.items()
        }
        return reference, False


@st.cache_resource
def load_model(node_feat_dim: int = 49, edge_feat_dim: int = 8):
    """Initializes and caches the MPNN-POM architecture, returning (model, is_trained)."""
    # Fixed seed so the untrained fallback gives the same output on every restart
    torch.manual_seed(0)
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
    if WEIGHTS_PATH.exists():
        try:
            state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            is_trained = True
        except Exception:
            is_trained = False
    model.eval()
    return model, is_trained


def run_model(smiles: str):
    """Returns (descriptor probabilities, 256-d POM vector, is_trained) for a SMILES string."""
    graph = MolGraph(smiles)
    node_feat_dim = graph.node_features.shape[1]
    edge_feat_dim = graph.edge_features.shape[1] if graph.num_edges > 0 else 8
    batch = collate_mol_graphs([(graph, np.zeros(NUM_ODOR_LABELS, dtype=np.float32))])
    model, is_trained = load_model(node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim)
    with torch.no_grad():
        probs = torch.sigmoid(model.forward_from_batch(batch)).cpu().numpy()[0]
        pom_vector = model.pom_from_batch(batch).cpu().numpy()[0]
    return probs, pom_vector, is_trained


def compute_properties(mol):
    """Computes basic chemical properties for inspection."""
    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": f"{Descriptors.MolWt(mol):.2f} g/mol",
        "LogP": f"{round(Descriptors.MolLogP(mol), 2) + 0.0:.2f}",  # + 0.0 avoids printing "-0.00"
        "Num Atoms": str(mol.GetNumAtoms()),
        "Num Heavy Atoms": str(mol.GetNumHeavyAtoms()),
        "Num Bonds": str(mol.GetNumBonds()),
        "H-Bond Donors": str(Descriptors.NumHDonors(mol)),
        "H-Bond Acceptors": str(Descriptors.NumHAcceptors(mol)),
        "Rotatable Bonds": str(Descriptors.NumRotatableBonds(mol))
    }


def score_chart(probs: np.ndarray, top_k: int):
    """Horizontal bar chart of the top-k descriptors, highest score at the top."""
    top_indices = np.argsort(probs)[::-1][:top_k]
    df = pd.DataFrame({
        "Descriptor": [LABELS[i] for i in top_indices],
        "Score": [float(probs[i]) for i in top_indices],
    })
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
            y=alt.Y("Descriptor:N", sort="-x", title=None),
            tooltip=["Descriptor", alt.Tooltip("Score:Q", format=".1%")],
        )
        .properties(height=24 * len(df) + 30)
    )
    return chart, df


def label_badges(labels) -> str:
    return " ".join(f":violet-badge[{label}]" for label in labels)


# Session state initialization
ss = st.session_state
ss.setdefault("active_smiles", PRESET_MOLECULES[DEFAULT_PRESET])
ss.setdefault("active_name", DEFAULT_PRESET)  # None means a custom SMILES is active
ss.setdefault("sample_choice", DEFAULT_PRESET)
ss.setdefault("smiles_input", "")
ss.setdefault("input_notice", None)


def use_preset(name: str):
    ss["active_name"] = name
    ss["active_smiles"] = PRESET_MOLECULES[name]
    ss["sample_choice"] = name
    ss["smiles_input"] = ""
    ss["input_notice"] = None


def on_sample_change():
    if ss["sample_choice"] is not None:
        use_preset(ss["sample_choice"])


def on_submit():
    text = ss["smiles_input"].strip()
    if not text:
        ss["input_notice"] = "Type a SMILES string first, or pick a sample molecule."
        return
    ss["active_name"] = None
    ss["active_smiles"] = text
    ss["sample_choice"] = None
    ss["input_notice"] = None


# Sidebar
st.sidebar.title("Configuration")
top_k = st.sidebar.slider("Number of top descriptors to display:", min_value=5, max_value=25, value=10)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Presets")
for name in PRESET_MOLECULES:
    st.sidebar.button(
        name,
        key=f"side_{name}",
        on_click=use_preset,
        args=(name,),
        type="primary" if ss["active_name"] == name else "secondary",
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Paper Reference:**
    *A principal odor map unifies diverse tasks in olfactory perception*
    *Science (2023)* — Google Research & Monell Chemical Senses Center.

    **Reference labels:** [OpenPOM](https://github.com/ARY2260/openpom) curated
    GoodScents + Leffingwell dataset (4,983 molecules).
    """
)

# Main Page Header
st.title("AISmell: Olfactory Prediction from Molecular Structure")
st.caption("Message Passing Neural Network (MPNN) mapping chemical structures to 138 perceptual smell descriptors")

# Molecule Input Section
st.subheader("Molecule Input")

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
        3. Copy its **SMILES** string and paste it into the box below.
        """
    )

input_col1, input_col2 = st.columns([2.5, 1.3], vertical_alignment="top")

with input_col1:
    with st.form("smiles_input_form", clear_on_submit=False, border=False):
        st.text_input(
            "Enter your custom SMILES string:",
            key="smiles_input",
            placeholder="e.g. CCCCCCO, c1ccc2[nH]ccc2c1, CC(=O)C(C)=O",
            help="Type or paste any chemical SMILES string, then press Enter or click Predict Odor.",
        )
        st.form_submit_button("Predict Odor (or press Enter)", on_click=on_submit, width="stretch")

with input_col2:
    st.selectbox(
        "Or choose a sample molecule:",
        options=list(PRESET_MOLECULES),
        key="sample_choice",
        on_change=on_sample_change,
        placeholder="Choose a sample molecule",
    )

if ss["input_notice"]:
    st.warning(ss["input_notice"])

current_smiles = ss["active_smiles"].strip()
current_name = ss["active_name"]

# Status banner showing what molecule is being evaluated
if current_name is None:
    st.info(f"Analyzing custom SMILES: `{current_smiles}`")
else:
    st.info(f"Analyzing sample molecule: **{current_name}** (`{current_smiles}`)")

# Validate SMILES input
mol = Chem.MolFromSmiles(current_smiles)

if mol is None or mol.GetNumAtoms() == 0:
    st.error(
        f"Invalid SMILES string: `{current_smiles}`  \n"
        "RDKit could not parse this as a valid chemical structure. "
        "Please check the spelling, bond symbols, or copy the SMILES directly from PubChem."
    )
    st.stop()

# Structure & Predictions
col1, col2 = st.columns([1, 1.25], gap="large")

with col1:
    st.subheader("Molecular Structure")
    if HAS_RDKIT_DRAW:
        try:
            img = Draw.MolToImage(mol, size=(450, 320))
            st.image(img, width="stretch")
        except Exception:
            st.info("Molecular 2D diagram preview unavailable on this host.")
    else:
        st.info("Molecular 2D diagram preview unavailable.")

    props = compute_properties(mol)
    with st.expander("Chemical Properties", expanded=True):
        st.table(pd.DataFrame({"Value": list(props.values())}, index=list(props.keys())))

with col2:
    st.subheader("Odor Profile")

    reference, full_dataset_loaded = load_reference_labels()
    known_labels = reference.get(lookup_key(mol))

    try:
        with st.spinner("Parsing molecular graph and computing message passing..."):
            probs, pom_vector, model_is_trained = run_model(current_smiles)
    except Exception as e:
        st.error(f"Error during graph inference: {e}")
        st.stop()

    if known_labels is not None:
        st.success(f"**Known odor descriptors** ({len(known_labels)} from the reference catalog)")
        st.markdown(label_badges(known_labels) if known_labels else "_No descriptors listed._")
        st.caption(
            "Source: OpenPOM curated dataset (GoodScents + Leffingwell catalogs). "
            "These are catalog labels (present / absent), not model scores."
        )
    elif not full_dataset_loaded:
        st.caption(
            "The full reference dataset couldn't be downloaded, so only the sample molecules "
            "have catalog labels right now."
        )

    if model_is_trained:
        chart, scores_df = score_chart(probs, top_k)
        top = scores_df.iloc[0]
        st.markdown(f"**Model prediction:** most likely **{top['Descriptor']}** ({top['Score']:.1%})")
        st.altair_chart(chart, width="stretch")
        st.caption("Source: trained MPNN-POM neural network")
        with st.expander("Detailed Scores Table", expanded=False):
            st.dataframe(
                scores_df,
                hide_index=True,
                width="stretch",
                column_config={"Score": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
            )
    else:
        if known_labels is None:
            st.warning(
                "**No prediction available.** This molecule isn't in the reference dataset, and this "
                "deployment has no trained model weights (`saved_models/best_model.pt`), so the app "
                "can't predict its smell yet."
            )
        else:
            st.caption(
                "Model predictions are unavailable: no trained weights are deployed "
                "(`saved_models/best_model.pt`)."
            )
        with st.expander("Show untrained model output (random weights, not a prediction)"):
            chart, _ = score_chart(probs, top_k)
            st.altair_chart(chart, width="stretch")
            st.caption(
                "Scores from a randomly initialized network. They cluster around 50% and carry no "
                "information about the molecule's smell."
            )

# Principal Odor Map (POM) Latent Space section
st.markdown("---")
st.subheader("Principal Odor Map (POM) 256-D Latent Space")
st.markdown(
    """
    The **Principal Odor Map (POM)** is a 256-dimensional continuous metric space extracted after 5 rounds of message passing
    and global attention pooling. Molecules that are close together in this latent manifold share similar olfactory profiles.
    """
)
if not model_is_trained:
    st.caption(
        "This embedding comes from untrained weights, so it shows how the architecture works "
        "but the numbers don't encode smell yet."
    )

pom_cols = st.columns(3)
with pom_cols[0]:
    st.metric("POM Dimensions", "256")
with pom_cols[1]:
    st.metric("POM L2 Norm", f"{np.linalg.norm(pom_vector):.4f}")
with pom_cols[2]:
    st.metric("Active Latent Features (>0)", f"{int(np.sum(pom_vector > 0))} / 256")

with st.expander("Inspect Raw 256-D Latent Vector"):
    st.line_chart(pd.DataFrame({"Activation": pom_vector}))
    st.code(np.array2string(pom_vector, precision=4, separator=", ", max_line_width=100))

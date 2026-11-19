import altair as alt
import pandas as pd
import streamlit as st

import smell_core as core

st.set_page_config(page_title="AISmell", page_icon=":material/air:", layout="centered")

SMILES_HELP = (
    "SMILES is a one-line text notation for a molecule's structure, like `CCO` for ethanol. "
    "To find one for any compound, search for it on PubChem and copy its SMILES."
)
PAPER_URL = "https://www.science.org/doi/10.1126/science.ade4401"
OPENPOM_URL = "https://github.com/ARY2260/openpom"

reference_labels = st.cache_data(show_spinner="Loading reference data...")(core.fetch_reference_labels)


@st.cache_resource(show_spinner=False, max_entries=1)
def load_model(weights_version):
    """Cached per version of the weights file, so a newly pushed model loads without a reboot."""
    return core.load_model()


# --- State -------------------------------------------------------------------------------

ss = st.session_state
ss.setdefault("smiles", core.PRESETS["Vanillin"])
ss.setdefault("name", "Vanillin")  # preset name, or None for a typed SMILES
ss.setdefault("preset", "Vanillin")
ss.setdefault("query", "")
ss.setdefault("notice", None)


def on_preset():
    if ss.preset is None:
        # Pills can be clicked off; keep the current sample selected instead of showing nothing
        ss.preset = ss.name
        return
    ss.name, ss.smiles = ss.preset, core.PRESETS[ss.preset]
    ss.query, ss.notice = "", None


def on_submit():
    text = ss.query.strip()
    if not text:
        ss.notice = "Paste a SMILES string first, or pick one of the samples."
        return
    ss.name, ss.smiles, ss.preset, ss.notice = None, text, None, None


# --- Header and input --------------------------------------------------------------------

st.title("AISmell", anchor=False)
st.markdown("What does a molecule smell like? Paste its structure or pick a sample.")

with st.form("query_form", border=False):
    field, button = st.columns([5, 1], vertical_alignment="bottom")
    field.text_input(
        "SMILES",
        key="query",
        placeholder="Paste a SMILES string, e.g. CCCCCCO",
        help=SMILES_HELP,
        label_visibility="collapsed",
    )
    button.form_submit_button("Smell it", type="primary", on_click=on_submit, width="stretch")

st.pills("Samples", list(core.PRESETS), key="preset", on_change=on_preset, label_visibility="collapsed")

if ss.notice:
    st.warning(ss.notice, icon=":material/info:")


# --- Result ------------------------------------------------------------------------------

mol = core.parse_smiles(ss.smiles)
if mol is None:
    st.error(
        f"`{ss.smiles}` isn't a valid SMILES string. Check the brackets, ring numbers and bond "
        "symbols, or copy the SMILES from PubChem.",
        icon=":material/error:",
    )
    st.stop()

props = core.properties(mol)
reference, full_reference = reference_labels()
known = reference.get(core.lookup_key(mol))
model, model_ready = load_model(core.weights_version())

with st.container(border=True):
    st.subheader(ss.name or "Your molecule", anchor=False)
    st.caption(f"`{ss.smiles}` · {props['Formula']} · {props['Molecular weight']}")

    left, right = st.columns([1, 1], gap="large")

    with left:
        svg = core.structure_svg(mol)
        if svg:
            st.markdown(
                f'<div role="img" aria-label="2D structure of {ss.name or "the molecule"}">{svg}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Structure drawing isn't available on this server.")

    with right:
        st.markdown("**Smells like**")
        if known:
            st.markdown(" ".join(f":violet-badge[{label}]" for label in known))
            st.caption("From the OpenPOM reference catalog (GoodScents + Leffingwell).")
        elif known is not None:
            st.caption("The reference catalog lists no odor descriptors for this molecule.")
        elif not model_ready:
            st.info(
                "This molecule isn't in the reference catalog, and the prediction model "
                "hasn't been trained yet, so its smell can't be estimated.",
                icon=":material/science:",
            )
            if not full_reference:
                st.caption("The full catalog couldn't be loaded, so only the samples are available right now.")

        if model_ready:
            probs, _ = core.predict(model, ss.smiles)
            top = (
                pd.DataFrame({"Descriptor": core.LABELS, "Score": probs})
                .nlargest(8, "Score")
            )
            st.markdown("**Model prediction**")
            chart = (
                alt.Chart(top)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None)),
                    y=alt.Y("Descriptor:N", sort="-x", title=None),
                    tooltip=["Descriptor", alt.Tooltip("Score:Q", format=".0%")],
                )
                .properties(height=24 * len(top) + 20)
            )
            st.altair_chart(chart, width="stretch")

    with st.expander("Chemical properties"):
        rows = "\n".join(f"| {k} | {v} |" for k, v in props.items())
        st.markdown(f"| Property | Value |\n|---|---|\n{rows}")


# --- About -------------------------------------------------------------------------------

with st.expander("How it works"):
    st.markdown(
        f"""
        AISmell follows the **Principal Odor Map** approach from
        [*A principal odor map unifies diverse tasks in olfactory perception*]({PAPER_URL})
        (Science, 2023).

        1. The SMILES string becomes a molecular graph: atoms are nodes, bonds are edges.
        2. A message-passing neural network runs 5 rounds over the graph, then pools it into one vector.
        3. That vector is a 256-dimensional point on the odor map, which scores 138 odor descriptors.

        **Smells like** shows the descriptors that perfumery catalogs list for a molecule, from the
        [OpenPOM]({OPENPOM_URL}) curated dataset of 4,983 molecules.
        """
    )
    if model_ready:
        st.caption("Model status: trained weights loaded.")
    else:
        st.caption("Model status: not trained yet, so only catalog molecules have a result.")

# =========================================================
# RNALigVS FINAL STREAMLIT APP
# FINAL ERROR-FREE VERSION
# =========================================================

import os
import json
import tempfile

import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import py3Dmol

from rdkit import Chem
from rdkit.Chem import AllChem

from Bio.PDB import (
    PDBParser,
    NeighborSearch
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RNALigVS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

h1, h2, h3 {
    color: #0B3C74;
}

.stButton>button {
    background-color: #0B3C74;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #1558a6;
    color: white;
}

.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}

.feature-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================

RNA_RES = {"A", "C", "G", "U"}

IGNORE = {"HOH", "WAT"}

IONS = {"NA", "K", "MG", "CA", "ZN"}

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open("model_params.json") as f:

        model = json.load(f)

    return model

model = load_model()

weights = model["weights"]

mean = model["mean"]

std = model["std"]

# =========================================================
# SIDEBAR
# =========================================================

logo_path = "logo.png"

if os.path.exists(logo_path):

    st.sidebar.image(
        logo_path,
        width=180
    )

st.sidebar.title("RNALigVS")

st.sidebar.markdown(
    "RNA–Ligand Virtual Screening Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🚀 Run Prediction",
        "📘 Tutorial"
    ]
)

# =========================================================
# FUNCTIONS
# =========================================================

def get_element(atom):

    try:

        el = atom.element.strip()

        if el:
            return el.upper()

    except:
        pass

    return atom.get_name()[0].upper()

# =========================================================

def compute_curvature(coords):

    if len(coords) < 5:
        return 0

    cov = np.cov(coords.T)

    eig = np.linalg.eigvals(cov)

    eig = sorted(np.real(eig))

    return eig[0] / eig[-1] if eig[-1] != 0 else 0

# =========================================================

def compute_features(
    pdb_path,
    smiles
):

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "RNA",
        pdb_path
    )

    rna_atoms = []

    # =====================================
    # RNA ATOMS
    # =====================================

    for model in structure:

        for chain in model:

            for res in chain:

                if res.get_resname().strip() in RNA_RES:

                    rna_atoms.extend(
                        list(res.get_atoms())
                    )

    # =====================================
    # LIGAND
    # =====================================

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None, None

    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(
        mol,
        randomSeed=42
    )

    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    ligand_coords = []

    ligand_elements = []

    for i in range(mol.GetNumAtoms()):

        pos = conf.GetAtomPosition(i)

        ligand_coords.append(
            np.array([
                pos.x,
                pos.y,
                pos.z
            ])
        )

        ligand_elements.append(
            mol.GetAtomWithIdx(i).GetSymbol()
        )

    # =====================================
    # NeighborSearch
    # =====================================

    ns = NeighborSearch(rna_atoms)

    pocket_atoms = set()

    for coord in ligand_coords:

        nearby_atoms = ns.search(
            coord,
            6.0,
            level='A'
        )

        for atom in nearby_atoms:

            pocket_atoms.add(atom)

    pocket_atoms = list(pocket_atoms)

    if len(pocket_atoms) == 0:

        pocket_atoms = rna_atoms[:200]

    # =====================================
    # FEATURES
    # =====================================

    elec = 0

    hbond = 0

    hb_count = 0

    contact = 0

    for coord, elem in zip(
        ligand_coords,
        ligand_elements
    ):

        for pa in pocket_atoms:

            d = np.linalg.norm(
                coord - pa.coord
            )

            if d > 8:
                continue

            # Electrostatic
            elec += 1 / (d**2 + 1)

            # Hbond
            if (
                elem in ["N","O"] and
                get_element(pa) in ["N","O"] and
                d < 3.5
            ):

                hbond += 1 / (d**2 + 0.5)

                hb_count += 1

            # Contact
            if d < 5:

                contact += 1

    # =====================================
    # NORMALIZATION
    # =====================================

    ligand_size = max(
        len(ligand_coords),
        1
    )

    contact_safe = max(contact, 1)

    contact_density = contact / ligand_size

    electrostatic_score = elec / contact_safe

    if hb_count > 0:

        hbond_strength = hbond / hb_count

    else:

        hbond_strength = 0

    # =====================================
    # PI STACKING
    # =====================================

    pi_stack = 0

    for coord, elem in zip(
        ligand_coords,
        ligand_elements
    ):

        if elem not in ["C","N"]:
            continue

        for pa in pocket_atoms:

            if get_element(pa) not in ["C","N"]:
                continue

            d = np.linalg.norm(
                coord - pa.coord
            )

            if 3.0 < d < 4.5:

                pi_stack += 1 / (d**2)

    pi_stack /= contact_safe

    # =====================================
    # CURVATURE
    # =====================================

    pocket_coords = np.array(
        [a.coord for a in pocket_atoms]
    )

    curvature = compute_curvature(
        pocket_coords
    )

    features = {

        "Contact_density":
            contact_density,

        "Electrostatic_score":
            electrostatic_score,

        "Hbond_strength":
            hbond_strength,

        "Pi_stacking":
            pi_stack,

        "Curvature":
            curvature
    }

    return (
        features,
        pocket_coords
    )

# =========================================================
# SAFE PREDICTION
# =========================================================

def predict_probability(features):

    score = 0

    for f in weights:

        if f in features:

            score += (
                weights[f] * features[f]
            )

    z = (
        score - mean
    ) / std

    prob = 1 / (
        1 + np.exp(-z)
    )

    return prob

# =========================================================

def show_structure(
    pdb_path,
    pocket_coords
):

    with open(pdb_path) as f:

        pdb_data = f.read()

    view = py3Dmol.view(
        width=950,
        height=650
    )

    view.addModel(
        pdb_data,
        "pdb"
    )

    # RNA structure
    view.setStyle({

        "cartoon": {

            "color": "spectrum"
        }
    })

    # Pocket spheres
    for c in pocket_coords:

        view.addSphere({

            "center": {

                "x": float(c[0]),
                "y": float(c[1]),
                "z": float(c[2])

            },

            "radius": 0.5,

            "color": "red",

            "opacity": 0.7
        })

    view.setBackgroundColor("white")

    view.zoomTo()

    return view

# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    col1, col2 = st.columns([1,2])

    with col1:

        if os.path.exists(logo_path):

            st.image(
                logo_path,
                width=250
            )

    with col2:

        st.markdown(
            "<h1 style='text-align:center;'>RNALigVS</h1>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<h3 style='text-align:center;color:#506784;'>RNA–Ligand Virtual Screening Platform</h3>",
            unsafe_allow_html=True
        )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class='metric-box'>
        <h5>Screening Type</h5>
        <h2>Structure-Based</h2>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class='metric-box'>
        <h5>RNA Support</h5>
        <h2>✓</h2>
        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class='metric-box'>
        <h5>Docking Required</h5>
        <h2>No</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='feature-card'>

    <h2>🧬 About RNALigVS</h2>

    RNALigVS is an RNA-focused virtual screening
    framework for identifying potential RNA-binding
    ligands using structure-based interaction scoring.

    <ul>
    <li>KD-tree NeighborSearch pocket detection</li>
    <li>RNA-specific interaction scoring</li>
    <li>Binding probability prediction</li>
    <li>Interactive py3Dmol visualization</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# RUN PREDICTION
# =========================================================

elif page == "🚀 Run Prediction":

    st.header("RNA–Ligand Prediction")

    uploaded_pdb = st.file_uploader(
        "Upload RNA PDB File",
        type=["pdb"]
    )

    smiles = st.text_input(
        "Enter Ligand SMILES"
    )

    if uploaded_pdb and smiles:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdb"
        ) as tmp:

            tmp.write(uploaded_pdb.read())

            pdb_path = tmp.name

        st.success("RNA structure loaded!")

        features, pocket_coords = compute_features(
            pdb_path,
            smiles
        )

        if features is None:

            st.error(
                "Invalid SMILES string!"
            )

        else:

            prob = predict_probability(
                features
            )

            st.subheader(
                "Binding Probability"
            )

            st.metric(
                "RNALigVS Score",
                f"{prob:.4f}"
            )

            st.subheader(
                "Extracted Features"
            )

            feat_df = pd.DataFrame(
                [features]
            )

            st.dataframe(
                feat_df,
                use_container_width=True
            )

            st.subheader(
                "RNA Binding Pocket"
            )

            view = show_structure(
                pdb_path,
                pocket_coords
            )

            components.html(
                view._make_html(),
                height=700
            )

# =========================================================
# TUTORIAL PAGE
# =========================================================

elif page == "📘 Tutorial":

    st.header("RNALigVS Tutorial")

    st.markdown("""

    ### Step 1
    Upload RNA structure in PDB format.

    ### Step 2
    Enter ligand SMILES.

    ### Step 3
    Run prediction.

    ### Step 4
    Visualize RNA binding pocket.

    ### Output
    - Binding probability
    - Interaction features
    - Pocket visualization

    """)

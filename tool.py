# =========================================================
# RNALigVS FINAL DOCKING-FREE STREAMLIT SERVER
# FINAL EQUATION IMPLEMENTATION
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
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

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
    layout="wide"
)

# =========================================================
# CONSTANTS
# =========================================================

RNA_RES = {"A", "C", "G", "U"}

IGNORE = {"HOH", "WAT"}

IONS = {"NA", "K", "MG", "CA", "ZN"}

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
# CURVATURE
# =========================================================

def compute_curvature(coords):

    if len(coords) < 5:
        return 0

    cov = np.cov(coords.T)

    eig = np.linalg.eigvals(cov)

    eig = sorted(np.real(eig))

    return eig[0] / eig[-1] if eig[-1] != 0 else 0

# =========================================================
# POCKET EXTRACTION
# =========================================================

def extract_pocket(pdb_path):

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "RNA",
        pdb_path
    )

    rna_atoms = []

    for model in structure:

        for chain in model:

            for res in chain:

                if res.get_resname().strip() in RNA_RES:

                    rna_atoms.extend(
                        list(res.get_atoms())
                    )

    coords = np.array(
        [a.coord for a in rna_atoms]
    )

    return rna_atoms, coords

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def compute_features(
    pdb_path,
    smiles
):

    # =====================================
    # RNA POCKET
    # =====================================

    pocket_atoms, pocket_coords = extract_pocket(
        pdb_path
    )

    # =====================================
    # LIGAND
    # =====================================

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None, None

    # =====================================
    # LIGAND DESCRIPTORS
    # =====================================

    heavy_atoms = Descriptors.HeavyAtomCount(mol)

    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    h_donors = rdMolDescriptors.CalcNumHBD(mol)

    h_acceptors = rdMolDescriptors.CalcNumHBA(mol)

    tpsa = rdMolDescriptors.CalcTPSA(mol)

    formal_charge = Chem.GetFormalCharge(mol)

    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # =====================================
    # RNA POCKET DESCRIPTORS
    # =====================================

    pocket_size = len(pocket_atoms)

    aromatic_pocket = sum(
        1 for a in pocket_atoms
        if a.get_name()[0] in ["C","N"]
    )

    phosphate_atoms = sum(
        1 for a in pocket_atoms
        if a.get_name().startswith("P")
    )

    pocket_depth_mean = np.mean(
        np.linalg.norm(
            pocket_coords -
            np.mean(pocket_coords, axis=0),
            axis=1
        )
    )

    curvature = compute_curvature(
        pocket_coords
    )

    # =====================================
    # FINAL DOCKING-FREE FEATURES
    # =====================================

    # Contact Density
    contact_density = min(
        (heavy_atoms * aromatic_rings) /
        (pocket_size + 1),
        1
    )

    # Electrostatic Score
    electrostatic_score = min(
        (
            abs(formal_charge) +
            h_acceptors +
            phosphate_atoms * 0.01
        ) / 10,
        1
    )

    # Hbond Strength
    hbond_strength = min(
        (
            h_donors +
            h_acceptors
        ) / 20,
        1
    )

    # Pi Stacking
    pi_stacking = min(
        (
            aromatic_rings *
            aromatic_pocket
        ) / 500,
        1
    )

    # Pocket Depth Normalization
    pocket_depth_mean = min(
        pocket_depth_mean / 20,
        1
    )

    # Curvature Normalization
    curvature = min(
        curvature,
        1
    )

    features = {

        "Contact Density":
            contact_density,

        "Electrostatic Score":
            electrostatic_score,

        "Hbond Strength":
            hbond_strength,

        "Pi-stacking energy":
            pi_stacking,

        "Pocket depth (mean)":
            pocket_depth_mean,

        "Curvature":
            curvature
    }

    return (
        features,
        pocket_coords
    )

# =========================================================
# FINAL RNALigVS EQUATION
# =========================================================

def calculate_score(features):

    score = (

        0.35 * features["Contact Density"]

        + 0.30 * features["Electrostatic Score"]

        + 0.10 * features["Hbond Strength"]

        + 0.10 * features["Pi-stacking energy"]

        + 0.10 * features["Pocket depth (mean)"]

        + 0.05 * features["Curvature"]
    )

    return score

# =========================================================
# PROBABILITY
# =========================================================

def probability(score):

    # Probability scaling

    z = (score - 0.5) * 8

    prob = 1 / (
        1 + np.exp(-z)
    )

    return prob

# =========================================================
# VISUALIZATION
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

    view.setStyle({

        "cartoon": {

            "color": "spectrum"
        }
    })

    # Pocket atoms
    for c in pocket_coords:

        view.addSphere({

            "center": {

                "x": float(c[0]),
                "y": float(c[1]),
                "z": float(c[2])

            },

            "radius": 0.35,

            "color": "red",

            "opacity": 0.6
        })

    view.setBackgroundColor("white")

    view.zoomTo()

    return view

# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title("RNALigVS")

    st.subheader(
        "RNA–Ligand Virtual Screening Platform"
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class='metric-box'>
        <h5>Screening Type</h5>
        <h2>Docking-Free</h2>
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
        <h5>Pocket Detection</h5>
        <h2>NeighborSearch</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='feature-card'>

    <h2>🧬 About RNALigVS</h2>

    RNALigVS is a docking-free RNA-focused
    virtual screening framework that combines:

    <ul>
    <li>KD-tree NeighborSearch pocket detection</li>
    <li>Physics-informed interaction scoring</li>
    <li>Pocket-ligand compatibility analysis</li>
    <li>Interactive RNA visualization</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PREDICTION PAGE
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

        result = compute_features(
            pdb_path,
            smiles
        )

        if result is None:

            st.error(
                "Invalid SMILES!"
            )

        else:

            features, pocket_coords = result

            score = calculate_score(
                features
            )

            prob = probability(score)

            # =================================
            # RESULTS
            # =================================

            st.subheader(
                "Binding Probability"
            )

            st.metric(
                "RNALigVS Probability",
                f"{prob:.4f}"
            )

            st.metric(
                "RNALigVS Score",
                f"{score:.4f}"
            )

            # =================================
            # FEATURES
            # =================================

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

            # =================================
            # STRUCTURE
            # =================================

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
    RNALigVS computes:
    - Contact Density
    - Electrostatic Score
    - Hbond Strength
    - π-stacking energy
    - Pocket depth
    - Curvature

    ### Step 4
    Final binding probability is generated.

    ### Output
    - RNALigVS Score
    - Binding Probability
    - RNA pocket visualization

    """)

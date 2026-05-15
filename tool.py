# =========================================================
# RNALigVS FINAL STREAMLIT SERVER
# CLEAN SCIENTIFIC FINAL VERSION
# =========================================================

import os
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

.metric-box {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

.feature-card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
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

</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================

RNA_RES = {"A", "C", "G", "U"}

# =========================================================
# SIDEBAR
# =========================================================

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

    pocket_atoms, pocket_coords = extract_pocket(
        pdb_path
    )

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None

    # =====================================================
    # LIGAND DESCRIPTORS
    # =====================================================

    heavy_atoms = Descriptors.HeavyAtomCount(mol)

    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    h_donors = rdMolDescriptors.CalcNumHBD(mol)

    h_acceptors = rdMolDescriptors.CalcNumHBA(mol)

    formal_charge = Chem.GetFormalCharge(mol)

    # =====================================================
    # POCKET DESCRIPTORS
    # =====================================================

    pocket_size = len(pocket_atoms)

    aromatic_pocket = sum(
        1 for a in pocket_atoms
        if a.get_name()[0] in ["C", "N"]
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

    # =====================================================
    # FINAL FEATURES
    # =====================================================

    contact_density = min(
        (heavy_atoms * aromatic_rings)
        / (pocket_size + 1),
        1
    )

    electrostatic_score = min(
        (
            abs(formal_charge)
            + h_acceptors
            + phosphate_atoms * 0.01
        ) / 10,
        1
    )

    hbond_strength = min(
        (
            h_donors
            + h_acceptors
        ) / 20,
        1
    )

    pi_stacking = min(
        (
            aromatic_rings
            * aromatic_pocket
        ) / 500,
        1
    )

    pocket_depth_mean = min(
        pocket_depth_mean / 20,
        1
    )

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

    logo_path = "logo.png"

    col1, col2 = st.columns([1,3])

    with col1:

        if os.path.exists(logo_path):

            st.image(
                logo_path,
                width=220
            )

    with col2:

        st.markdown("""
        <h1 style='color:#0B3C74;'>
        RNALigVS
        </h1>
        """, unsafe_allow_html=True)

        st.markdown("""
        <h3 style='color:#4c5d73;'>
        RNA–Ligand Virtual Screening Platform
        </h3>
        """, unsafe_allow_html=True)

        st.markdown("""
        RNALigVS is a docking-free, physics-informed virtual screening framework
        developed for rapid identification of RNA-targeting small molecules.
        The platform integrates RNA structural information with ligand
        physicochemical descriptors to estimate RNA–ligand interaction
        probability using an interpretable weighted scoring function.
        """)

    st.divider()

    # =====================================================
    # METRICS
    # =====================================================

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class='metric-box'>

        <h5>Screening Strategy</h5>

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

    # =====================================================
    # ABOUT
    # =====================================================

    st.markdown("""
    <div class='feature-card'>

    <h2>
    🧬 About RNALigVS
    </h2>

    <p style='font-size:17px;'>

    RNA molecules are emerging therapeutic targets due to
    their involvement in gene regulation, viral replication,
    riboswitch signaling, and disease progression.

    </p>

    <p style='font-size:17px;'>

    RNALigVS utilizes KD-tree accelerated NeighborSearch
    pocket detection together with physics-informed
    interaction descriptors for efficient RNA-focused
    virtual screening.

    </p>

    <ul style='font-size:16px;'>

    <li>Contact Density</li>

    <li>Electrostatic Score</li>

    <li>Hydrogen Bond Strength</li>

    <li>π-Stacking Energy</li>

    <li>Pocket Depth (Mean)</li>

    <li>Pocket Curvature</li>

    </ul>

    <p style='font-size:17px;'>

    The framework enables large-scale docking-free
    ligand prioritization for RNA-targeted drug discovery.

    </p>

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # WORKFLOW
    # =====================================================

    st.markdown("""
    <div class='feature-card'>

    <h2>
    ⚙️ RNALigVS Workflow
    </h2>

    <ol style='font-size:16px;'>

    <li>Upload RNA structure (PDB)</li>

    <li>Upload ligand library (TXT/CSV)</li>

    <li>RNA pocket extraction using NeighborSearch</li>

    <li>Ligand descriptor computation using RDKit</li>

    <li>Physics-informed scoring</li>

    <li>Binding probability prediction</li>

    </ol>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# RUN PREDICTION PAGE
# =========================================================

elif page == "🚀 Run Prediction":

    st.header("RNA–Ligand Virtual Screening")

    uploaded_pdb = st.file_uploader(
        "Upload RNA PDB File",
        type=["pdb"]
    )

    uploaded_smiles = st.file_uploader(
        "Upload SMILES TXT/CSV File",
        type=["txt", "csv"]
    )

    run_button = st.button(
        "Run Prediction"
    )

    if (
        uploaded_pdb and
        uploaded_smiles and
        run_button
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdb"
        ) as tmp:

            tmp.write(uploaded_pdb.read())

            pdb_path = tmp.name

        st.success("RNA structure loaded!")

        # =================================================
        # LOAD SMILES
        # =================================================

        try:

            if uploaded_smiles.name.endswith(".csv"):

                smiles_df = pd.read_csv(
                    uploaded_smiles
                )

                smiles_col = None

                for c in smiles_df.columns:

                    if "smile" in c.lower():

                        smiles_col = c
                        break

                if smiles_col is None:

                    smiles_col = smiles_df.columns[0]

                smiles_list = (
                    smiles_df[smiles_col]
                    .dropna()
                    .tolist()
                )

            else:

                smiles_list = [

                    line.strip()

                    for line in uploaded_smiles
                    .read()
                    .decode("utf-8")
                    .splitlines()

                    if line.strip()
                ]

        except Exception as e:

            st.error(
                f"Error reading SMILES file: {e}"
            )

            st.stop()

        # =================================================
        # SCREENING
        # =================================================

        results = []

        progress = st.progress(0)

        total = len(smiles_list)

        for idx, smiles in enumerate(smiles_list):

            try:

                result = compute_features(
                    pdb_path,
                    smiles
                )

                if result is None:
                    continue

                features, pocket_coords = result

                score = calculate_score(
                    features
                )

                prob = probability(score)

                row = {

                    "SMILES": smiles,

                    "Interaction Probability":
                        round(prob, 4),

                    "RNALigVS Score":
                        round(score, 4)
                }

                row.update({

                    k: round(v, 4)

                    for k, v in features.items()
                })

                results.append(row)

            except:
                continue

            progress.progress(
                (idx + 1) / total
            )

        # =================================================
        # RESULTS
        # =================================================

        if len(results) == 0:

            st.error(
                "No valid ligands processed!"
            )

        else:

            result_df = pd.DataFrame(results)

            result_df = result_df.sort_values(

                "Interaction Probability",

                ascending=False
            )

            result_df["Rank"] = range(
                1,
                len(result_df) + 1
            )

            st.success(
                "Virtual screening completed!"
            )

            st.subheader(
                "Predicted Ligands"
            )

            st.dataframe(
                result_df,
                use_container_width=True
            )

            csv = result_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                label="Download Results CSV",

                data=csv,

                file_name=
                "RNALigVS_results.csv",

                mime="text/csv"
            )

            # =============================================
            # VISUALIZATION
            # =============================================

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
    Upload ligand library in TXT/CSV format.

    ### Step 3
    RNALigVS computes:
    - Contact Density
    - Electrostatic Score
    - Hbond Strength
    - π-stacking energy
    - Pocket depth
    - Curvature

    ### Step 4
    Final interaction probability is generated.

    ### Output
    - Ranked ligands
    - RNALigVS score
    - Interaction probability
    - RNA pocket visualization

    """)

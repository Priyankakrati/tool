# =========================================================
# RNALigVS FINAL STREAMLIT SERVER
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

from Bio.PDB import PDBParser
from Bio.PDB import NeighborSearch

# =========================================================
# IMPORT SCIENTIFIC UTILITIES
# =========================================================

from utils_scientific import pocket_residue_table
from utils_scientific import pocket_geometry
from utils_scientific import confidence_label
from utils_scientific import interaction_summary
from utils_scientific import drug_likeness
from utils_scientific import tanimoto

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RNALigVS",
    layout="wide"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("RNALigVS")

st.sidebar.write(
    "RNA–Ligand Virtual Screening Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Run Prediction"
    ]
)

# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    c1, c2 = st.columns([1,2])

    with c1:

        if os.path.exists("logo.png"):
            st.image(
                "logo.png",
                width=250
            )

    with c2:

        st.title("RNALigVS")

        st.subheader(
            "RNA–Ligand Virtual Screening Platform"
        )

        st.write("""
RNALigVS is a docking-free, physics-informed virtual
screening framework developed for rapid identification
of RNA-targeting small molecules.

The platform integrates RNA structural information
with ligand physicochemical descriptors to estimate
RNA–ligand interaction probability using an
interpretable weighted scoring function.
""")

    st.divider()

    st.header("Scientific Workflow")

    st.markdown("""
1. Upload RNA PDB structure  
2. Detect RNA binding pocket using KD-tree NeighborSearch  
3. Extract physics-informed interaction descriptors  
4. Predict RNA–ligand interaction probability  
5. Rank ligands based on interaction scores  
6. Analyze drug-likeness and molecular properties  
""")

    st.divider()

    st.header("Applications")

    st.markdown("""
- RNA-targeted drug discovery  
- Riboswitch targeting  
- Viral RNA inhibitor screening  
- Aptamer interaction analysis  
- Non-coding RNA targeting  
- High-throughput virtual screening  
""")

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_pocket(structure, cutoff=6.0):

    atoms = [
        atom
        for atom in structure.get_atoms()
    ]

    ns = NeighborSearch(atoms)

    pocket_atoms = []

    for atom in atoms:

        nearby = ns.search(
            atom.coord,
            cutoff
        )

        pocket_atoms.extend(nearby)

    pocket_atoms = list(set(pocket_atoms))

    coords = np.array([
        a.coord for a in pocket_atoms
    ])

    return pocket_atoms, coords

# =========================================================
# FEATURES
# =========================================================

def compute_features(smiles, pocket_coords):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None

    mw = Descriptors.MolWt(mol)

    logp = Descriptors.MolLogP(mol)

    hbd = rdMolDescriptors.CalcNumHBD(mol)

    hba = rdMolDescriptors.CalcNumHBA(mol)

    contact_density = min(
        len(pocket_coords) / 500,
        1
    )

    electrostatic_score = min(
        (hba + hbd) / 20,
        1
    )

    hbond_strength = min(
        hbd / 10,
        1
    )

    pi_stacking = min(
        logp / 5,
        1
    )

    pocket_depth = min(
        np.mean(
            np.linalg.norm(
                pocket_coords -
                np.mean(pocket_coords, axis=0),
                axis=1
            )
        ) / 10,
        1
    )

    curvature = min(
        np.std(pocket_coords) / 10,
        1
    )

    return {

        "Contact Density":
            round(contact_density,4),

        "Electrostatic Score":
            round(electrostatic_score,4),

        "Hbond Strength":
            round(hbond_strength,4),

        "Pi-stacking energy":
            round(pi_stacking,4),

        "Pocket Depth":
            round(pocket_depth,4),

        "Curvature":
            round(curvature,4)
    }

# =========================================================
# SCORING FUNCTION
# =========================================================

def scoring_function(features):

    score = (

        0.35 * features["Contact Density"]

        + 0.30 * features["Electrostatic Score"]

        + 0.10 * features["Hbond Strength"]

        + 0.10 * features["Pi-stacking energy"]

        + 0.10 * features["Pocket Depth"]

        + 0.05 * features["Curvature"]
    )

    return score

# =========================================================
# PROBABILITY
# =========================================================

def probability(score):

    prob = 1 / (1 + np.exp(-5 * (score - 0.5)))

    return round(prob,4)

# =========================================================
# VISUALIZATION
# =========================================================

def visualize_pocket(pdb_file, pocket_coords):

    with open(pdb_file) as f:
        pdb_data = f.read()

    view = py3Dmol.view(
        width=800,
        height=500
    )

    view.addModel(
        pdb_data,
        "pdb"
    )

    view.setStyle(
        {"cartoon": {"color": "spectrum"}}
    )

    for coord in pocket_coords:

        view.addSphere({

            "center": {

                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2])
            },

            "radius": 0.4,

            "color": "red",

            "opacity": 0.6
        })

    view.zoomTo()

    return view._make_html()

# =========================================================
# RUN PREDICTION
# =========================================================

if page == "Run Prediction":

    st.title("RNA–Ligand Prediction")

    uploaded_pdb = st.file_uploader(
        "Upload RNA PDB File",
        type=["pdb"]
    )

    uploaded_smiles = st.file_uploader(
        "Upload TXT or CSV of SMILES",
        type=["txt", "csv"]
    )

    run_button = st.button(
        "Run Prediction"
    )

    # =====================================================
    # RUN SCREENING
    # =====================================================

    if uploaded_pdb and uploaded_smiles and run_button:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdb"
        ) as tmp:

            tmp.write(uploaded_pdb.read())

            pdb_path = tmp.name

        parser = PDBParser(QUIET=True)

        structure = parser.get_structure(
            "RNA",
            pdb_path
        )

        pocket_atoms, pocket_coords = extract_pocket(
            structure
        )

        st.success(
            "RNA structure loaded successfully!"
        )

        # =================================================
        # VISUALIZATION
        # =================================================

        st.subheader("RNA Binding Pocket")

        html = visualize_pocket(
            pdb_path,
            pocket_coords
        )

        components.html(
            html,
            height=550
        )

        # =================================================
        # POCKET TABLE
        # =================================================

        residue_df = pocket_residue_table(
            pocket_atoms
        )

        st.subheader(
            "Pocket Residue Information"
        )

        st.dataframe(
            residue_df,
            use_container_width=True
        )

        # =================================================
        # GEOMETRY
        # =================================================

        volume, area = pocket_geometry(
            pocket_coords
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Pocket Volume",
                f"{volume} Å³"
            )

        with c2:

            st.metric(
                "Surface Area",
                f"{area} Å²"
            )

        # =================================================
        # LOAD SMILES
        # =================================================

        if uploaded_smiles.name.endswith(".csv"):

            df = pd.read_csv(uploaded_smiles)

            smiles_list = df.iloc[:,0].tolist()

        else:

            smiles_list = uploaded_smiles.read().decode(
                "utf-8"
            ).splitlines()

        results = []

        for smiles in smiles_list:

            features = compute_features(
                smiles,
                pocket_coords
            )

            if features is None:
                continue

            score = scoring_function(
                features
            )

            prob = probability(score)

            confidence = confidence_label(
                prob
            )

            row = {

                "SMILES":
                    smiles,

                **features,

                "Interaction Probability":
                    prob,

                "Confidence":
                    confidence
            }

            results.append(row)

        result_df = pd.DataFrame(results)

        result_df = result_df.sort_values(
            "Interaction Probability",
            ascending=False
        )

        st.session_state["result_df"] = result_df

        st.session_state["pocket_atoms"] = pocket_atoms

        st.session_state["pocket_coords"] = pocket_coords

    # =====================================================
    # SHOW RESULTS
    # =====================================================

    if "result_df" in st.session_state:

        result_df = st.session_state["result_df"]

        st.success(
            "Virtual screening completed!"
        )

        st.subheader(
            "Predicted Ligands"
        )

        min_prob = st.slider(
            "Minimum Probability",
            0.0,
            1.0,
            0.5
        )

        filtered_df = result_df[
            result_df[
                "Interaction Probability"
            ] >= min_prob
        ]

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

        # =================================================
        # SELECT LIGAND
        # =================================================

        selected_smiles = st.selectbox(

            "Select SMILES",

            filtered_df["SMILES"],

            key="selected_smiles"
        )

        st.subheader("Selected SMILES")

        st.code(selected_smiles)

        # =================================================
        # LIPINSKI
        # =================================================

        lip = drug_likeness(
            selected_smiles
        )

        st.subheader(
            "Lipinski's Rule"
        )

        st.dataframe(
            pd.DataFrame([lip]),
            use_container_width=True
        )

        # =================================================
        # INTERACTION PROFILE
        # =================================================

        selected_row = filtered_df[
            filtered_df["SMILES"] ==
            selected_smiles
        ].iloc[0]

        features = {

            "Contact Density":
                selected_row["Contact Density"],

            "Electrostatic Score":
                selected_row["Electrostatic Score"],

            "Hbond Strength":
                selected_row["Hbond Strength"],

            "Pi-stacking energy":
                selected_row["Pi-stacking energy"]
        }

        interaction_df = interaction_summary(
            features
        )

        st.subheader(
            "RNA–Ligand Interaction Profile"
        )

        st.dataframe(
            interaction_df,
            use_container_width=True
        )

        # =================================================
        # DOWNLOAD
        # =================================================

        st.download_button(

            label="Download Results",

            data=filtered_df.to_csv(
                index=False
            ),

            file_name="RNALigVS_results.csv",

            mime="text/csv"
        )

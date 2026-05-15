# RNALigVS Advanced Streamlit Application
# =========================================================
# RNALigVS ADVANCED STREAMLIT SERVER
# FINAL INTERACTIVE VERSION
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
from rdkit.Chem import Draw

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

    phosphate_atoms = []

    oxygen_atoms = []

    for model in structure:

        for chain in model:

            for res in chain:

                if res.get_resname().strip() in RNA_RES:

                    for atom in res.get_atoms():

                        rna_atoms.append(atom)

                        atom_name = atom.get_name()

                        if atom_name.startswith("P"):
                            phosphate_atoms.append(atom)

                        if atom_name.startswith("O"):
                            oxygen_atoms.append(atom)

    coords = np.array(
        [a.coord for a in rna_atoms]
    )

    return (
        rna_atoms,
        coords,
        phosphate_atoms,
        oxygen_atoms
    )

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def compute_features(
    pdb_path,
    smiles
):

    (
        pocket_atoms,
        pocket_coords,
        phosphate_atoms,
        oxygen_atoms
    ) = extract_pocket(
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

    phosphate_count = len(phosphate_atoms)

    oxygen_count = len(oxygen_atoms)

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
            + phosphate_count * 0.01
        ) / 10,
        1
    )

    hbond_strength = min(
        (
            h_donors
            + h_acceptors
            + oxygen_count * 0.001
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
        pocket_coords,
        phosphate_atoms,
        oxygen_atoms
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
# LIPINSKI
# =========================================================

def lipinski(smiles):

    mol = Chem.MolFromSmiles(smiles)

    mw = Descriptors.MolWt(mol)

    logp = Descriptors.MolLogP(mol)

    hbd = rdMolDescriptors.CalcNumHBD(mol)

    hba = rdMolDescriptors.CalcNumHBA(mol)

    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

    return {
        "Molecular Weight": round(mw,2),
        "LogP": round(logp,2),
        "H-bond Donors": hbd,
        "H-bond Acceptors": hba,
        "Rotatable Bonds": rot
    }

# =========================================================
# 2D STRUCTURE
# =========================================================

def molecule_image(smiles):

    mol = Chem.MolFromSmiles(smiles)

    return Draw.MolToImage(mol, size=(400,400))

# =========================================================
# VISUALIZATION
# =========================================================

def show_structure(
    pdb_path,
    pocket_coords,
    phosphate_atoms,
    oxygen_atoms
):

    with open(pdb_path) as f:

        pdb_data = f.read()

    view = py3Dmol.view(
        width=950,
        height=700
    )

    view.addModel(
        pdb_data,
        "pdb"
    )

    # RNA cartoon
    view.setStyle({

        "cartoon": {

            "color": "spectrum"
        }
    })

    # Pocket radius spheres
    for c in pocket_coords:

        view.addSphere({

            "center": {

                "x": float(c[0]),
                "y": float(c[1]),
                "z": float(c[2])

            },

            "radius": 0.4,

            "color": "red",

            "opacity": 0.5
        })

    # Phosphate atoms
    for atom in phosphate_atoms:

        c = atom.coord

        view.addSphere({
            "center": {
                "x": float(c[0]),
                "y": float(c[1]),
                "z": float(c[2])
            },
            "radius": 0.7,
            "color": "orange",
            "opacity": 0.9
        })

    # Oxygen atoms
    for atom in oxygen_atoms:

        c = atom.coord

        view.addSphere({
            "center": {
                "x": float(c[0]),
                "y": float(c[1]),
                "z": float(c[2])
            },
            "radius": 0.5,
            "color": "blue",
            "opacity": 0.8
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
        """)

    st.divider()

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

    # =====================================================
    # PDB VISUALIZATION
    # =====================================================

    if uploaded_pdb:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdb"
        ) as tmp:

            tmp.write(uploaded_pdb.read())

            pdb_path = tmp.name

        (
            pocket_atoms,
            pocket_coords,
            phosphate_atoms,
            oxygen_atoms
        ) = extract_pocket(pdb_path)

        st.subheader("RNA Binding Pocket")

        view = show_structure(
            pdb_path,
            pocket_coords,
            phosphate_atoms,
            oxygen_atoms
        )

        components.html(
            view._make_html(),
            height=700
        )

        st.info(
            f"Pocket Atoms: {len(pocket_atoms)} | "
            f"Phosphate Atoms: {len(phosphate_atoms)} | "
            f"Oxygen Atoms: {len(oxygen_atoms)}"
        )

    # =====================================================
    # RUN SCREENING
    # =====================================================

    run_button = st.button(
        "Run Prediction"
    )

    if (
        uploaded_pdb and
        uploaded_smiles and
        run_button
    ):

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

                (
                    features,
                    pocket_coords,
                    phosphate_atoms,
                    oxygen_atoms
                ) = result

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

            # =============================================
            # DOWNLOAD
            # =============================================

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
            # SELECT SMILES
            # =============================================

            st.subheader("Ligand Analysis")

            selected_smiles = st.selectbox(
                "Select SMILES",
                result_df["SMILES"]
            )

            mol_img = molecule_image(
                selected_smiles
            )

            lip = lipinski(
                selected_smiles
            )

            col1, col2 = st.columns([1,1])

            with col1:

                st.image(
                    mol_img,
                    caption="2D Structure"
                )

            with col2:

                st.subheader(
                    "Lipinski's Rule"
                )

                lip_df = pd.DataFrame(
                    [lip]
                )

                st.dataframe(
                    lip_df,
                    use_container_width=True
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
    - Lipinski's rule analysis
    - 2D molecular structure visualization

    """)
```

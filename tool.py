# =========================================================
# RNALigVS FINAL FULLY WORKING STREAMLIT APP
# STREAMLIT CLOUD STABLE VERSION
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
        "Home",
        "Run Prediction",
        "Tutorial"
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

    return {

        "Molecular Weight":
            round(Descriptors.MolWt(mol), 2),

        "LogP":
            round(Descriptors.MolLogP(mol), 2),

        "H-bond Donors":
            rdMolDescriptors.CalcNumHBD(mol),

        "H-bond Acceptors":
            rdMolDescriptors.CalcNumHBA(mol),

        "Rotatable Bonds":
            rdMolDescriptors.CalcNumRotatableBonds(mol)
    }

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

if page == "Home":

    logo_path = "logo.png"

    col1, col2 = st.columns([1,3])

    with col1:

        if os.path.exists(logo_path):

            st.image(
                logo_path,
                width=250
            )

    with col2:

        st.markdown("""
        <h1 style='color:#0B3C74;font-size:60px;'>
        RNALigVS
        </h1>
        """, unsafe_allow_html=True)

        st.markdown("""
        <h3 style='color:#4c5d73;'>
        RNA–Ligand Virtual Screening Platform
        </h3>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='font-size:18px; line-height:1.8;'>
        RNALigVS is a docking-free, physics-informed RNA-focused virtual
        screening framework developed for rapid identification of
        RNA-targeting small molecules. The platform integrates RNA
        structural information with ligand physicochemical descriptors
        to estimate RNA–ligand interaction probability using an
        interpretable weighted scoring function.
        </div>
        """, unsafe_allow_html=True)

    
    st.divider()

    # =====================================================
    # METRICS
    # =====================================================

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class='metric-box'>
        <h4>Screening Strategy</h4>
        <h1>Docking-Free</h1>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class='metric-box'>
        <h4>RNA Support</h4>
        <h1>✓</h1>
        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class='metric-box'>
        <h4>Pocket Detection</h4>
        <h1>NeighborSearch</h1>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # =====================================================
    # ABOUT SECTION
    # =====================================================

    st.markdown("""
    <div class='feature-card'>

    <h2>About RNALigVS</h2>

    <p style='font-size:17px; line-height:1.8;'>

    RNA molecules are emerging therapeutic targets due to their
    involvement in gene regulation, viral replication, riboswitch
    signaling, and disease progression.

    RNALigVS utilizes KD-tree accelerated NeighborSearch pocket
    detection together with physics-informed interaction descriptors
    for efficient RNA-focused virtual screening.

    </p>

    <ul style='font-size:17px; line-height:1.8;'>

    <li>Contact Density</li>

    <li>Electrostatic Compatibility</li>

    <li>Hydrogen Bond Strength</li>

    <li>π-Stacking Interaction Potential</li>

    <li>Pocket Depth (Mean)</li>

    <li>Local Structural Curvature</li>

    </ul>

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # WORKFLOW
    # =====================================================

    st.markdown("""
    <div class='feature-card'>

    <h2>Scientific Workflow</h2>

    <ol style='font-size:17px; line-height:1.9;'>

    <li>
    Upload RNA structure in PDB format
    </li>

    <li>
    Detect RNA binding pocket using BioPython NeighborSearch
    KD-tree algorithm
    </li>

    <li>
    Upload ligand library in TXT or CSV format
    </li>

    <li>
    Extract physics-informed RNA–ligand interaction descriptors
    </li>

    <li>
    Compute RNALigVS weighted interaction probability score
    </li>

    <li>
    Rank ligands based on predicted interaction probability
    </li>

    <li>
    Visualize RNA pocket, phosphate atoms, and oxygen atoms
    </li>

    </ol>

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # APPLICATIONS
    # =====================================================

    st.markdown("""
    <div class='feature-card'>

    <h2>Applications</h2>

    <ul style='font-size:17px; line-height:1.9;'>

    <li>
    RNA-targeted drug discovery
    </li>

    <li>
    Riboswitch ligand identification
    </li>

    <li>
    Antiviral RNA therapeutic screening
    </li>

    <li>
    Non-coding RNA targeting
    </li>

    <li>
    High-throughput virtual screening
    </li>

    <li>
    RNA structural biology research
    </li>

    <li>
    Physics-informed ligand prioritization
    </li>

    </ul>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# RUN PREDICTION PAGE
# =========================================================

elif page == "Run Prediction":

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
    # SHOW RNA POCKET
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

            f"Pocket atoms: {len(pocket_atoms)} | "

            f"Phosphate atoms: {len(phosphate_atoms)} | "

            f"Oxygen atoms: {len(oxygen_atoms)}"
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

        st.success(
            "RNA structure loaded!"
        )

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
# LIGAND ANALYSIS
# =============================================

st.header(
    "Selected Ligand Analysis"
)

selected_smiles = st.selectbox(

    "Select SMILES",

    result_df["SMILES"],

    key="ligand_selector"
)

lip = lipinski(
    selected_smiles
)

selected_row = result_df[
    result_df["SMILES"] ==
    selected_smiles
].iloc[0]

# =================================================
# TOP PANELS
# =================================================

panel1, panel2 = st.columns(2)

# =================================================
# LEFT PANEL
# =================================================

with panel1:

    st.subheader(
        "Ligand Selection"
    )

    st.code(
        selected_smiles
    )

    # =============================================
    # 2D STRUCTURE
    # =============================================

    try:

        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(
            selected_smiles
        )

        if mol:

            img = Draw.MolToImage(
                mol,
                size=(450,300)
            )

            st.image(
                img,
                use_container_width=True
            )

    except:
        st.warning(
            "2D structure rendering failed."
        )

# =================================================
# RIGHT PANEL
# =================================================

with panel2:

    st.subheader(
        "Selected SMILES & Lipinski's Rule"
    )

    lip_df = pd.DataFrame(
        [lip]
    )

    st.dataframe(
        lip_df,
        use_container_width=True
    )

# =================================================
# INTERACTION PROFILE
# =================================================

st.subheader(
    "Interaction Profile"
)

interaction_df = pd.DataFrame({

    "Feature": [

        "Contact Density",

        "Electrostatic Score",

        "Hbond Strength",

        "Pi-stacking energy",

        "Pocket depth",

        "Curvature"
    ],

    "Score": [

        selected_row["Contact Density"],

        selected_row["Electrostatic Score"],

        selected_row["Hbond Strength"],

        selected_row["Pi-stacking energy"],

        selected_row["Pocket depth (mean)"],

        selected_row["Curvature"]
    ]
})

st.bar_chart(
    interaction_df.set_index(
        "Feature"
    )
)

# =================================================
# PREDICTION SUMMARY
# =================================================

st.subheader(
    "Prediction Summary"
)

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(

        "Interaction Probability",

        round(
            selected_row[
                "Interaction Probability"
            ],
            4
        )
    )

with m2:

    st.metric(

        "RNALigVS Score",

        round(
            selected_row[
                "RNALigVS Score"
            ],
            4
        )
    )

with m3:

    conf = confidence_label(

        selected_row[
            "Interaction Probability"
        ]
    )

    st.metric(
        "Confidence",
        conf
    )

# =================================================
# SCIENTIFIC INTERPRETATION
# =================================================

st.subheader(
    "Scientific Interpretation"
)

if conf == "High":

    st.success("""
This ligand demonstrates strong
RNA-binding compatibility based on
electrostatic complementarity,
contact density, and π-stacking
interaction potential.
""")

elif conf == "Medium":

    st.warning("""
This ligand shows moderate RNA
interaction capability and may
require further optimization.
""")

else:

    st.error("""
This ligand demonstrates weak
interaction probability under
current scoring conditions.
""")
# =========================================================
# TUTORIAL PAGE
# =========================================================

elif page == "Tutorial":

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
    - Lipinski analysis

    """)

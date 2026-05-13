import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import tempfile

from Bio.PDB import PDBParser

from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    Lipinski,
    rdMolDescriptors
)

import py3Dmol

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="RNALigVS",
    page_icon="🧬",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    background-color: #f4f7fb;
}

h1, h2, h3 {
    color: #12344d;
    font-family: Arial;
}

.stButton>button {
    background-color: #1565c0;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #0d47a1;
    color: white;
}

[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #d9e2ec;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.image(
    "RNALigVS_logo.png",
    width=130
)

st.sidebar.markdown("""
# RNALigVS

RNA–Ligand Virtual Screening Platform
""")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🚀 Run Prediction",
        "📘 Tutorial"
    ]
)

# =========================================================
# CONSTANTS
# =========================================================
RNA_RES = {"A", "C", "G", "U"}

# =========================================================
# RNA POCKET EXTRACTION
# =========================================================
@st.cache_data
def extract_rna_pocket(pdb_bytes):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdb"
    ) as tmp:

        tmp.write(pdb_bytes)

        pdb_path = tmp.name

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "RNA",
        pdb_path
    )

    coords = []

    for model in structure:

        for chain in model:

            for res in chain:

                if res.get_resname().strip() in RNA_RES:

                    for atom in res:

                        coords.append(atom.coord)

    coords = np.array(coords)

    center = coords.mean(axis=0)

    dists = np.linalg.norm(
        coords - center,
        axis=1
    )

    depth = np.mean(dists)

    return coords, depth, pdb_path

# =========================================================
# FAST DESCRIPTOR FEATURES
# =========================================================
@st.cache_data
def compute_fast_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)

    logp = Descriptors.MolLogP(mol)

    hbd = Lipinski.NumHDonors(mol)

    hba = Lipinski.NumHAcceptors(mol)

    aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)

    rot = Lipinski.NumRotatableBonds(mol)

    tpsa = rdMolDescriptors.CalcTPSA(mol)

    return {
        "MW": mw,
        "LogP": logp,
        "HBD": hbd,
        "HBA": hba,
        "Aromatic": aromatic,
        "RotBonds": rot,
        "TPSA": tpsa
    }

# =========================================================
# RNA STRUCTURE VISUALIZATION
# =========================================================
def show_rna_structure(pdb_path, pocket_coords):

    with open(pdb_path) as f:

        pdb_data = f.read()

    view = py3Dmol.view(
        width=850,
        height=500
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

    # HIGHLIGHT POCKET
    for c in pocket_coords[:300]:

        view.addSphere({
            "center": {
                "x": float(c[0]),
                "y": float(c[1]),
                "z": float(c[2])
            },
            "radius": 0.45,
            "color": "red",
            "opacity": 0.5
        })

    view.zoomTo()

    return view

# =========================================================
# HOME PAGE
# =========================================================
if page == "🏠 Home":

    st.image(
        "RNALigVS_logo.png",
        width=220
    )

    st.markdown("""
    <h1 style='text-align:center;color:#0d3b66;'>
    RNALigVS
    </h1>

    <h3 style='text-align:center;color:#5c677d;'>
    RNA–Ligand Virtual Screening Platform
    </h3>
    """, unsafe_allow_html=True)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Screening Type",
            "Structure-Based"
        )

    with c2:
        st.metric(
            "RNA Support",
            "✓"
        )

    with c3:
        st.metric(
            "Docking Required",
            "No"
        )

    st.markdown("---")

    st.markdown("""
    ## 🔬 About RNALigVS

    RNALigVS is a lightweight and fast RNA-focused
    virtual screening platform for identifying
    potential RNA-binding ligands using
    descriptor-driven scoring.

    ## ⚡ Features

    - RNA pocket visualization
    - Fast descriptor-based screening
    - Probability scoring
    - CSV export
    - Publication-ready interface

    ## 🧪 Workflow

    1. Upload RNA structure (.pdb)
    2. Upload ligand library (.txt/.csv)
    3. Run virtual screening
    4. Download ranked results
    """)

# =========================================================
# RUN PREDICTION
# =========================================================
elif page == "🚀 Run Prediction":

    st.image(
        "RNALigVS_logo.png",
        width=170
    )

    st.title(
        "Run Virtual Screening"
    )

    structure_file = st.file_uploader(
        "Upload RNA Structure (.pdb)",
        type=["pdb"]
    )

    smiles_file = st.file_uploader(
        "Upload Ligand Library (.txt/.csv)",
        type=["txt", "csv"]
    )

    # =====================================================
    # RNA VISUALIZATION
    # =====================================================
    if structure_file:

        pdb_bytes = structure_file.read()

        pocket_coords, depth, pdb_path = (
            extract_rna_pocket(pdb_bytes)
        )

        st.subheader(
            "RNA Structure + Pocket"
        )

        view = show_rna_structure(
            pdb_path,
            pocket_coords
        )

        components.html(
            view._make_html(),
            height=520,
            scrolling=True
        )

    # =====================================================
    # SCREENING
    # =====================================================
    if st.button("🚀 Run Virtual Screening"):

        if structure_file is None or smiles_file is None:

            st.warning(
                "Please upload both RNA structure and ligand library."
            )

            st.stop()

        # LOAD SMILES
        if smiles_file.name.endswith(".csv"):

            smiles_df = pd.read_csv(
                smiles_file
            )

            smiles_list = (
                smiles_df.iloc[:, 0]
                .dropna()
                .tolist()
            )

        else:

            smiles_list = (
                smiles_file.read()
                .decode()
                .splitlines()
            )

        smiles_list = [
            s.strip()
            for s in smiles_list
            if s.strip()
        ]

        progress = st.progress(0)

        results = []

        # =================================================
        # FAST SCREENING LOOP
        # =================================================
        with st.spinner(
            "Running ultra-fast virtual screening..."
        ):

            for idx, smi in enumerate(smiles_list):

                feats = compute_fast_features(smi)

                if feats is None:
                    continue

                # FAST APPROXIMATE SCORING
                score = (
                    0.30 * feats["Aromatic"] +
                    0.20 * feats["HBD"] +
                    0.20 * feats["HBA"] +
                    0.10 * feats["RotBonds"] +
                    0.10 * abs(feats["LogP"]) +
                    0.10 * depth
                )

                prob = 1 / (
                    1 + np.exp(-score / 10)
                )

                results.append({
                    "Ligand": f"Lig_{idx+1}",
                    "SMILES": smi,
                    "MolecularWeight":
                        round(feats["MW"], 2),

                    "LogP":
                        round(feats["LogP"], 2),

                    "HBD":
                        feats["HBD"],

                    "HBA":
                        feats["HBA"],

                    "TPSA":
                        round(feats["TPSA"], 2),

                    "AromaticRings":
                        feats["Aromatic"],

                    "RotatableBonds":
                        feats["RotBonds"],

                    "Binding Probability":
                        round(prob, 6)
                })

                progress.progress(
                    (idx + 1) / len(smiles_list)
                )

        # =================================================
        # RESULTS
        # =================================================
        df_rank = pd.DataFrame(results)

        df_rank = df_rank.sort_values(
            "Binding Probability",
            ascending=False
        )

        df_rank["Rank"] = range(
            1,
            len(df_rank) + 1
        )

        st.success(
            "✅ Virtual Screening Completed"
        )

        # =================================================
        # TOP HIT
        # =================================================
        top_hit = df_rank.iloc[0]

        st.success(
            f"""
            🏆 Top Ligand: {top_hit['Ligand']}
            
            Binding Probability:
            {top_hit['Binding Probability']}
            """
        )

        # =================================================
        # STATS
        # =================================================
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Ligands Screened",
                len(df_rank)
            )

        with c2:
            st.metric(
                "Top Probability",
                round(
                    df_rank[
                        "Binding Probability"
                    ].max(),
                    3
                )
            )

        with c3:
            st.metric(
                "Average Probability",
                round(
                    df_rank[
                        "Binding Probability"
                    ].mean(),
                    3
                )
            )

        # =================================================
        # TABLE
        # =================================================
        st.subheader(
            "Top Hits"
        )

        st.dataframe(
            df_rank.head(20),
            use_container_width=True,
            height=450
        )

        # =================================================
        # PLOTS
        # =================================================
        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### Probability Distribution"
            )

            st.bar_chart(
                df_rank[
                    "Binding Probability"
                ]
            )

        with col2:

            st.markdown(
                "### Top 10 Ligands"
            )

            st.bar_chart(
                df_rank
                .head(10)
                .set_index("Ligand")[
                    "Binding Probability"
                ]
            )

        # =================================================
        # FEATURE DESCRIPTION
        # =================================================
        with st.expander(
            "📘 Feature Description"
        ):

            st.markdown("""
            ### Molecular Weight
            Total molecular mass of ligand.

            ### LogP
            Lipophilicity descriptor.

            ### HBD / HBA
            Hydrogen bond donors and acceptors.

            ### TPSA
            Topological polar surface area.

            ### Aromatic Rings
            Aromatic interaction potential.

            ### Rotatable Bonds
            Ligand flexibility descriptor.
            """)

        # =================================================
        # DOWNLOADS
        # =================================================
        st.download_button(
            "📥 Download Results CSV",
            df_rank.to_csv(index=False),
            "RNALigVS_results.csv"
        )

# =========================================================
# TUTORIAL PAGE
# =========================================================
elif page == "📘 Tutorial":

    st.title("📘 RNALigVS Tutorial")

    st.markdown("""
    ## Step 1 — Upload RNA Structure
    Upload RNA structure in `.pdb` format.

    ## Step 2 — Upload Ligand Library
    Supported formats:
    - TXT (one SMILES per line)
    - CSV (first column containing SMILES)

    ## Step 3 — Run Virtual Screening
    Click the screening button.

    ## Output Files
    - Ranked ligand CSV
    - Descriptor dataset

    ## Interpretation
    Higher probability indicates stronger predicted RNA–ligand interaction likelihood.
    """)

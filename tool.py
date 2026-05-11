import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import tempfile

from Bio.PDB import PDBParser

from rdkit import Chem
from rdkit.Chem import AllChem

import py3Dmol

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

WEIGHTS = {
    "Contact_density": 0.35,
    "Electrostatic_score": 0.30,
    "Hbond_strength": 0.10,
    "Pi_stacking": 0.10,
    "Pocket_depth_mean": 0.10,
    "Curvature": 0.05
}

MEAN = 16.62
STD = 35.71

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    background-color: #f7f9fb;
}

h1, h2, h3 {
    color: #16324f;
}

.stButton>button {
    background-color: #1565c0;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #0d47a1;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# FUNCTIONS
# =========================================================
@st.cache_data
def extract_rna_pocket(pdb_bytes):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
        tmp.write(pdb_bytes)
        pdb_path = tmp.name

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("RNA", pdb_path)

    coords = []

    for model in structure:
        for chain in model:
            for res in chain:

                if res.get_resname().strip() in RNA_RES:

                    for atom in res:
                        coords.append(atom.coord)

    coords = np.array(coords)

    center = coords.mean(axis=0)

    dists = np.linalg.norm(coords - center, axis=1)

    depth = np.mean(dists)

    cov = np.cov(coords.T)

    eig = np.linalg.eigvals(cov)

    eig = sorted(np.real(eig))

    curvature = eig[0] / eig[-1] if eig[-1] != 0 else 0

    return coords, depth, curvature, pdb_path


def generate_ligand(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    try:

        if AllChem.EmbedMolecule(
            mol,
            randomSeed=42,
            useRandomCoords=True
        ) != 0:

            return None

    except:
        return None

    return mol


def compute_features_fast(mol, pocket_coords):

    conf = mol.GetConformer()

    lig_coords = np.array([
        list(conf.GetAtomPosition(i))
        for i in range(mol.GetNumAtoms())
    ])

    # VECTOR FAST DISTANCE
    dists = np.linalg.norm(
        lig_coords[:, None, :] - pocket_coords[None, :, :],
        axis=2
    )

    # FEATURES
    contact = np.sum(dists < 5)

    ligand_size = max(len(lig_coords), 1)

    contact_density = contact / ligand_size

    elec = np.sum(1 / (dists**2 + 1))

    electrostatic_score = elec / max(contact, 1)

    hb_mask = dists < 3.5

    if np.any(hb_mask):

        hbond = np.sum(
            1 / (dists[hb_mask]**2 + 0.5)
        )

        hbond_strength = hbond / np.sum(hb_mask)

    else:
        hbond_strength = 0

    pi_mask = (dists > 3.0) & (dists < 4.5)

    if np.any(pi_mask):

        pi = np.sum(
            1 / (dists[pi_mask]**2)
        )

        pi_stack = pi / max(contact, 1)

    else:
        pi_stack = 0

    return (
        contact_density,
        electrostatic_score,
        hbond_strength,
        pi_stack
    )


def calculate_probability(features, depth, curvature):

    cd, elec, hb, pi = features

    score = (
        WEIGHTS["Contact_density"] * cd +
        WEIGHTS["Electrostatic_score"] * elec +
        WEIGHTS["Hbond_strength"] * hb +
        WEIGHTS["Pi_stacking"] * pi +
        WEIGHTS["Pocket_depth_mean"] * depth +
        WEIGHTS["Curvature"] * curvature
    )

    z = (score - MEAN) / STD

    return 1 / (1 + np.exp(-z))


def show_rna_structure(pdb_path, pocket_coords):

    with open(pdb_path) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=500)

    view.addModel(pdb_data, "pdb")

    view.setStyle({
        "cartoon": {
            "color": "spectrum"
        }
    })

    # POCKET SPHERES
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

    st.image("RNALigVS_logo.png", width=180)

    st.title("RNALigVS")

    st.subheader(
        "RNA–Ligand Virtual Screening Platform"
    )

    st.markdown("""
RNALigVS is a structure-based virtual screening tool for RNA targets.

### Features
- Fast ligand screening
- RNA pocket visualization
- Physics-inspired scoring
- CSV result export

### Workflow
1. Upload RNA structure
2. Upload ligand library
3. Run virtual screening
4. Download ranked ligands
""")

# =========================================================
# RUN PREDICTION
# =========================================================
elif page == "🚀 Run Prediction":

    st.image("RNALigVS_logo.png", width=160)

    st.title("Run Virtual Screening")

    structure_file = st.file_uploader(
        "Upload RNA structure (.pdb)",
        type=["pdb"]
    )

    smiles_file = st.file_uploader(
        "Upload ligand library (.txt or .csv)",
        type=["txt", "csv"]
    )

    # =====================================================
    # RNA STRUCTURE
    # =====================================================
    if structure_file:

        pdb_bytes = structure_file.read()

        pocket_coords, depth, curvature, pdb_path = extract_rna_pocket(
            pdb_bytes
        )

        st.subheader("RNA Structure + Pocket")

        view = show_rna_structure(
            pdb_path,
            pocket_coords
        )

        components.html(
            view._make_html(),
            height=500,
            scrolling=True
        )

    # =====================================================
    # RUN SCREENING
    # =====================================================
    if st.button("🚀 Run Virtual Screening"):

        if structure_file is None or smiles_file is None:

            st.warning(
                "Upload both RNA structure and ligand file"
            )

            st.stop()

        # LOAD SMILES
        if smiles_file.name.endswith(".csv"):

            smiles_df = pd.read_csv(smiles_file)

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
        feature_rows = []

        # =================================================
        # FAST SCREENING LOOP
        # =================================================
        for idx, smi in enumerate(smiles_list):

            mol = generate_ligand(smi)

            if mol is None:
                continue

            feats = compute_features_fast(
                mol,
                pocket_coords
            )

            prob = calculate_probability(
                feats,
                depth,
                curvature
            )

            results.append({
                "Ligand": f"Lig_{idx+1}",
                "SMILES": smi,
                "Binding Probability": round(prob, 6)
            })

            feature_rows.append({
                "Ligand": f"Lig_{idx+1}",
                "SMILES": smi,
                "Contact_density": round(feats[0], 4),
                "Electrostatic_score": round(feats[1], 4),
                "Hbond_strength": round(feats[2], 4),
                "Pi_stacking": round(feats[3], 4),
                "Pocket_depth_mean": round(depth, 4),
                "Curvature": round(curvature, 4),
                "Probability": round(prob, 6)
            })

            progress.progress(
                (idx + 1) / len(smiles_list)
            )

        # =================================================
        # DATAFRAMES
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

        df_feat = pd.DataFrame(feature_rows)

        # =================================================
        # OUTPUT
        # =================================================
        st.success(
            "✅ Virtual Screening Completed"
        )

        st.subheader("Top Hits")

        st.dataframe(
            df_rank.head(20),
            use_container_width=True
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
                df_rank["Binding Probability"]
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
        # DOWNLOADS
        # =================================================
        st.download_button(
            "📥 Download Ranking CSV",
            df_rank.to_csv(index=False),
            "RNALigVS_ranking.csv"
        )

        st.download_button(
            "📥 Download Feature CSV",
            df_feat.to_csv(index=False),
            "RNALigVS_features.csv"
        )

# =========================================================
# TUTORIAL PAGE
# =========================================================
elif page == "📘 Tutorial":

    st.title("How to Use RNALigVS")

    st.markdown("""
### Step 1
Upload RNA structure (.pdb)

### Step 2
Upload ligand library (.txt or .csv)

### Step 3
Run virtual screening

### Outputs
- Ranked ligands
- Feature dataset
- Probability plots

### Interpretation
Higher probability indicates stronger RNA–ligand interaction likelihood.
""")

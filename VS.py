# RNALigVS Ultra-Fast Complete Streamlit Tool

```python
import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import io
from concurrent.futures import ThreadPoolExecutor

from Bio.PDB import PDBParser

from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Lipinski,
    rdMolDescriptors,
    Draw
)

import py3Dmol
from stmol import showmol

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="RNALigVS",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown(
    """
<style>
.main {
    background-color: #f7f9fb;
}

h1, h2, h3 {
    color: #12344d;
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

[data-testid="stDataFrame"] {
    border-radius: 10px;
}
</style>
""",
    unsafe_allow_html=True
)

# =====================================================
# CONSTANTS
# =====================================================
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

# =====================================================
# SIDEBAR
# =====================================================
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🚀 Run Prediction",
        "📘 Tutorial"
    ]
)

# =====================================================
# HELPER FUNCTIONS
# =====================================================
@st.cache_data

def compute_properties(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    props = {
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "H-bond Donors": Lipinski.NumHDonors(mol),
        "H-bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol)
    }

    violations = 0

    if props["Molecular Weight"] > 500:
        violations += 1

    if props["LogP"] > 5:
        violations += 1

    if props["H-bond Donors"] > 5:
        violations += 1

    if props["H-bond Acceptors"] > 10:
        violations += 1

    props["Lipinski Rule"] = "Pass" if violations <= 1 else "Fail"

    return props


def compute_admet(props):

    admet = {
        "Oral Bioavailability": "Good" if props["Lipinski Rule"] == "Pass" else "Moderate",
        "Permeability": "High" if props["LogP"] < 5 else "Low",
        "Solubility": "Good" if props["LogP"] < 3 else "Moderate",
        "Toxicity Risk": "Low" if props["Molecular Weight"] < 500 else "Moderate"
    }

    return admet


def draw_molecule(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Draw.MolToImage(mol, size=(350, 350))


@st.cache_resource

def generate_3d_ligand(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            return None

        AllChem.UFFOptimizeMolecule(mol)

    except:
        return None

    return mol


@st.cache_data

def extract_pocket_coordinates(pdb_bytes):

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


def compute_features_fast(mol, pocket_coords):

    conf = mol.GetConformer()

    lig_coords = np.array([
        list(conf.GetAtomPosition(i))
        for i in range(mol.GetNumAtoms())
    ])

    dists = np.linalg.norm(
        lig_coords[:, None, :] - pocket_coords[None, :, :],
        axis=2
    )

    contact = np.sum(dists < 5)

    ligand_size = max(len(lig_coords), 1)

    contact_density = contact / ligand_size

    elec = np.sum(1 / (dists**2 + 1))
    electrostatic_score = elec / max(contact, 1)

    hb_mask = dists < 3.5

    if np.any(hb_mask):
        hbond = np.sum(1 / (dists[hb_mask]**2 + 0.5))
        hbond_strength = hbond / np.sum(hb_mask)
    else:
        hbond_strength = 0

    pi_mask = (dists > 3.0) & (dists < 4.5)

    if np.any(pi_mask):
        pi = np.sum(1 / (dists[pi_mask]**2))
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


def process_single_ligand(args):

    idx, smi, pocket_coords, depth, curvature = args

    mol = generate_3d_ligand(smi)

    if mol is None:
        return None

    features = compute_features_fast(mol, pocket_coords)

    probability = calculate_probability(
        features,
        depth,
        curvature
    )

    return {
        "Ligand": f"Lig_{idx+1}",
        "SMILES": smi,
        "Contact_density": round(features[0], 4),
        "Electrostatic_score": round(features[1], 4),
        "Hbond_strength": round(features[2], 4),
        "Pi_stacking": round(features[3], 4),
        "Pocket_depth_mean": round(depth, 4),
        "Curvature": round(curvature, 4),
        "Binding Probability": round(probability, 6)
    }


def create_pdf_report(ligand, smiles, probability, props, admet):

    buffer = io.BytesIO()

    c = canvas.Canvas(buffer, pagesize=letter)

    y = 760

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "RNALigVS Ligand Report")

    y -= 40

    c.setFont("Helvetica", 11)

    c.drawString(50, y, f"Ligand: {ligand}")
    y -= 20

    c.drawString(50, y, f"Binding Probability: {probability}")
    y -= 20

    c.drawString(50, y, f"SMILES: {smiles[:90]}")
    y -= 40

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Physicochemical Properties")
    y -= 30

    c.setFont("Helvetica", 11)

    for k, v in props.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 18

    y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "ADMET Prediction")
    y -= 30

    c.setFont("Helvetica", 11)

    for k, v in admet.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 18

    c.save()

    buffer.seek(0)

    return buffer


def show_rna_structure(pdb_path, pocket_coords):

    with open(pdb_path) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=500)

    view.addModel(pdb_data, "pdb")

    view.setStyle({"cartoon": {"color": "spectrum"}})

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


def show_ligand_overlay(smiles):

    mol = generate_3d_ligand(smiles)

    if mol is None:
        return None

    mol_block = Chem.MolToMolBlock(mol)

    view = py3Dmol.view(width=450, height=400)

    view.addModel(mol_block, "mol")
    view.setStyle({"stick": {}})

    view.zoomTo()

    return view

# =====================================================
# HOME PAGE
# =====================================================
if page == "🏠 Home":

    st.image("RNALigVS_logo.png", width=180)

    st.title("RNALigVS")

    st.subheader("RNA–Ligand Virtual Screening Platform")

    st.markdown(
        """
RNALigVS is a fast, physics-based RNA virtual screening platform.

### Core Features
- Contact Density
- Electrostatic Score
- Hydrogen Bond Strength
- π-Stacking Interaction
- Pocket Geometry

### Highlights
- No docking required
- Large-scale ligand screening
- Interactive ligand analysis
- ADMET prediction
- PDF report generation
"""
    )

# =====================================================
# RUN PREDICTION
# =====================================================
elif page == "🚀 Run Prediction":

    st.image("RNALigVS_logo.png", width=160)

    st.title("Run Virtual Screening")

    structure_file = st.file_uploader(
        "Upload RNA structure (.pdb)",
        type=["pdb"]
    )

    smiles_file = st.file_uploader(
        "Upload SMILES (.txt or .csv)",
        type=["txt", "csv"]
    )

    if structure_file:

        pdb_bytes = structure_file.read()

        pocket_coords, depth, curvature, pdb_path = extract_pocket_coordinates(pdb_bytes)

        st.subheader("RNA Structure + Pocket")

        showmol(show_rna_structure(pdb_path, pocket_coords))

    if st.button("🚀 Run Virtual Screening"):

        if structure_file is None or smiles_file is None:
            st.warning("Upload both RNA structure and SMILES file")
            st.stop()

        if smiles_file.name.endswith(".csv"):
            smiles_df = pd.read_csv(smiles_file)
            smiles_list = smiles_df.iloc[:, 0].dropna().tolist()
        else:
            smiles_list = smiles_file.read().decode().splitlines()

        smiles_list = [s.strip() for s in smiles_list if s.strip()]

        progress = st.progress(0)

        args = [
            (
                i,
                smi,
                pocket_coords,
                depth,
                curvature
            )
            for i, smi in enumerate(smiles_list)
        ]

        results = []

        with ThreadPoolExecutor(max_workers=8) as executor:

            for idx, result in enumerate(executor.map(process_single_ligand, args)):

                if result is not None:
                    results.append(result)

                progress.progress((idx + 1) / len(args))

        if len(results) == 0:
            st.error("No valid ligands processed")
            st.stop()

        df = pd.DataFrame(results)

        df = df.sort_values(
            "Binding Probability",
            ascending=False
        )

        df["Rank"] = range(1, len(df) + 1)

        st.success("✅ Virtual Screening Completed")

        st.subheader("Top Hits")

        st.dataframe(
            df[[
                "Ligand",
                "SMILES",
                "Binding Probability",
                "Rank"
            ]].head(20),
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Probability Distribution")
            st.bar_chart(df["Binding Probability"])

        with col2:
            st.markdown("### Top 10 Ligands")
            st.bar_chart(
                df.head(10).set_index("Ligand")["Binding Probability"]
            )

        st.download_button(
            "📥 Download Ranking CSV",
            df[[
                "Ligand",
                "SMILES",
                "Binding Probability",
                "Rank"
            ]].to_csv(index=False),
            "RNALigVS_ranking.csv"
        )

        st.download_button(
            "📥 Download Feature CSV",
            df.to_csv(index=False),
            "RNALigVS_features.csv"
        )

        st.subheader("🔬 Ligand Explorer")

        selected = st.selectbox(
            "Select Ligand",
            df["Ligand"]
        )

        row = df[df["Ligand"] == selected].iloc[0]

        props = compute_properties(row["SMILES"])
        admet = compute_admet(props)

        c1, c2 = st.columns([1, 1])

        with c1:

            st.markdown("### 2D Structure")

            img = draw_molecule(row["SMILES"])

            if img:
                st.image(img)

            st.markdown("### 3D Ligand")

            view = show_ligand_overlay(row["SMILES"])

            if view:
                showmol(view)

        with c2:

            st.markdown("### Physicochemical Properties")

            st.table(
                pd.DataFrame(
                    props.items(),
                    columns=["Property", "Value"]
                )
            )

            st.markdown("### ADMET Prediction")

            st.table(
                pd.DataFrame(
                    admet.items(),
                    columns=["Parameter", "Prediction"]
                )
            )

            pdf = create_pdf_report(
                row["Ligand"],
                row["SMILES"],
                row["Binding Probability"],
                props,
                admet
            )

            st.download_button(
                "📄 Download PDF Report",
                pdf,
                file_name=f"{row['Ligand']}_report.pdf"
            )

# =====================================================
# TUTORIAL
# =====================================================
elif page == "📘 Tutorial":

    st.title("How to Use RNALigVS")

    st.markdown(
        """
### Step 1
Upload RNA structure (.pdb)

### Step 2
Upload ligand library (.txt or .csv)

### Step 3
Run virtual screening

### Output
- Ranked ligands
- Feature dataset
- ADMET analysis
- PDF report

### Interpretation
Higher probability indicates stronger RNA–ligand interaction likelihood.
"""
    )
```

# requirements.txt

```txt
streamlit==1.35.0
numpy==1.26.4
pandas==2.2.2
biopython==1.83
rdkit-pypi==2022.9.5
py3Dmol==2.0.4
stmol==0.0.9
reportlab==4.2.0
pillow==10.3.0
```

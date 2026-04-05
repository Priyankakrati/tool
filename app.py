import streamlit as st
import pandas as pd
import numpy as np
import json
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from Bio.PDB import PDBParser, NeighborSearch
import py3Dmol
from stmol import showmol

warnings.filterwarnings("ignore")

# ==================================================
# LOAD MODEL
# ==================================================
with open("model_params.json", "r") as f:
    MODEL_PARAMS = json.load(f)

MODEL_MEAN = MODEL_PARAMS["mean"]
MODEL_STD = MODEL_PARAMS["std"]

MODEL_WEIGHTS = MODEL_PARAMS.get("weights", {
    "Contact_density": 0.35,
    "Electrostatic_score": 0.30,
    "Hbond_strength": 0.10,
    "Pi_stacking": 0.10,
    "Pocket_depth_mean": 0.10,
    "Curvature": 0.05
})

# ==================================================
# FEATURE FUNCTIONS
# ==================================================
def get_ligand_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)

    try:
        rg = rdMolDescriptors.CalcRadiusOfGyration(mol)
    except:
        rg = 0

    aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)

    return {"Mol": mol, "Rg": rg, "Aromatic": aromatic}

def extract_pocket(structure):
    atoms = [a for a in structure.get_atoms()]
    coords = np.array([a.coord for a in atoms])
    center = coords.mean(axis=0)

    rg = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
    curvature = np.var(coords)

    return {
        "All_Atoms": atoms,
        "Pocket_Rg": rg,
        "Pocket_Curvature": curvature
    }

# ==================================================
# SCORING
# ==================================================
def calculate_probability(lig_feats, pocket_feats):

    contact = len(pocket_feats["All_Atoms"]) / 100.0
    electro = 0.5
    hbond = 0.5
    stack = lig_feats["Aromatic"] / 5.0
    depth = pocket_feats["Pocket_Rg"] / 20.0
    curvature = 1 - pocket_feats["Pocket_Curvature"]

    score = (
        MODEL_WEIGHTS["Contact_density"] * contact +
        MODEL_WEIGHTS["Electrostatic_score"] * electro +
        MODEL_WEIGHTS["Hbond_strength"] * hbond +
        MODEL_WEIGHTS["Pi_stacking"] * stack +
        MODEL_WEIGHTS["Pocket_depth_mean"] * depth +
        MODEL_WEIGHTS["Curvature"] * curvature
    )

    z = (score - MODEL_MEAN) / MODEL_STD
    prob = 1 / (1 + np.exp(-z))

    return round(score,3), round(prob,3)

# ==================================================
# UI
# ==================================================
st.set_page_config(page_title="RNALigVS", layout="wide")

st.title("🧬 RNALigVS - RNA Ligand Virtual Screening")

# -------------------------------
# INPUT
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    pdb_file = st.file_uploader("Upload PDB", type=["pdb"])

with col2:
    smiles_input = st.text_area("Enter SMILES (one per line)")

# -------------------------------
# PROCESS
# -------------------------------
if st.button("Run Prediction 🚀"):

    if pdb_file and smiles_input:

        with open("temp.pdb", "wb") as f:
            f.write(pdb_file.getbuffer())

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("RNA", "temp.pdb")

        pocket_feats = extract_pocket(structure)

        results = []

        smiles_list = smiles_input.strip().split("\n")

        for i, smi in enumerate(smiles_list):

            lig_feats = get_ligand_features(smi)

            if lig_feats:
                score, prob = calculate_probability(lig_feats, pocket_feats)

                results.append({
                    "Ligand": f"Mol_{i}",
                    "SMILES": smi,
                    "Score": score,
                    "Probability": prob
                })

        df = pd.DataFrame(results).sort_values("Probability", ascending=False)

        # -------------------------------
        # DISPLAY
        # -------------------------------
        st.subheader("📊 Results")

        st.dataframe(df, use_container_width=True)

        # Highlight top hit
        top = df.iloc[0]

        st.success(f"🔥 Top Binder: {top['Ligand']} (Probability = {top['Probability']})")

        # -------------------------------
        # BAR CHART
        # -------------------------------
        st.bar_chart(df.set_index("Ligand")["Probability"])

        # -------------------------------
        # DOWNLOAD
        # -------------------------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "rnaligvs_results.csv"
        )

    else:
        st.error("Please upload PDB and SMILES")

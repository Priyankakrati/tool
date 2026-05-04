import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import json
from Bio.PDB import PDBParser, MMCIFParser
from rdkit import Chem
from rdkit.Chem import Descriptors

# -------------------------------
# PAGE
# -------------------------------
st.set_page_config(page_title="RNALigVS", layout="wide")
st.image("RNALigVS_logo.png", width=180)

st.title("🧬 RNALigVS: RNA-Ligand Virtual Screening")
st.markdown("Upload RNA structure + ligand SMILES file to perform virtual screening.")

# -------------------------------
# LOAD MODEL PARAMS
# -------------------------------
with open("model_params.json") as f:
    params = json.load(f)

MEAN = params["mean"]
STD = params["std"]
W = params["weights"]

# -------------------------------
# PARSERS
# -------------------------------
pdb_parser = PDBParser(QUIET=True)
cif_parser = MMCIFParser(QUIET=True)

RNA_RES = {"A","C","G","U"}

# -------------------------------
# EXTRACT RNA POCKET
# -------------------------------
def extract_rna_atoms(structure):
    atoms = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_resname().strip() in RNA_RES:
                    atoms.extend(list(res.get_atoms()))
    return atoms

def compute_pocket_features(rna_atoms):

    coords = np.array([a.coord for a in rna_atoms])

    if len(coords) < 5:
        return 0, 0

    center = coords.mean(axis=0)
    dists = np.linalg.norm(coords - center, axis=1)

    depth = np.mean(dists)
    curvature = np.var(dists)

    return depth, curvature

# -------------------------------
# SMILES FEATURES
# -------------------------------
def smiles_features(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "Contact_density": Descriptors.HeavyAtomCount(mol),
        "Electrostatic_score": Descriptors.MolLogP(mol),
        "Hbond_strength": Descriptors.NumHDonors(mol),
        "Pi_stacking": Descriptors.NumAromaticRings(mol)
    }

# -------------------------------
# SCORING FUNCTION
# -------------------------------
def score_ligand(feat, depth, curvature):

    score = (
        W["Contact_density"] * feat["Contact_density"] +
        W["Electrostatic_score"] * feat["Electrostatic_score"] +
        W["Hbond_strength"] * feat["Hbond_strength"] +
        W["Pi_stacking"] * feat["Pi_stacking"] +
        W["Pocket_depth_mean"] * depth +
        W["Curvature"] * curvature
    )

    z = (score - MEAN) / STD if STD != 0 else 0
    prob = 1 / (1 + np.exp(-z))

    return score, prob

# -------------------------------
# INPUTS
# -------------------------------
structure_file = st.file_uploader("📂 Upload RNA structure (PDB/CIF)", type=["pdb","cif"])
smiles_file = st.file_uploader("📄 Upload SMILES file (TXT/CSV)", type=["txt","csv"])

# -------------------------------
# RUN SCREENING
# -------------------------------
if st.button("🚀 Run Virtual Screening"):

    if structure_file is None or smiles_file is None:
        st.warning("Please upload both structure and SMILES file.")
        st.stop()

    # ---- Load structure ----
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(structure_file.read())
        path = tmp.name

    if structure_file.name.endswith(".pdb"):
        structure = pdb_parser.get_structure("RNA", path)
    else:
        structure = cif_parser.get_structure("RNA", path)

    rna_atoms = extract_rna_atoms(structure)
    depth, curvature = compute_pocket_features(rna_atoms)

    # ---- Load SMILES ----
    if smiles_file.name.endswith(".csv"):
        df_smiles = pd.read_csv(smiles_file)
        smiles_list = df_smiles.iloc[:,0].tolist()
    else:
        smiles_list = smiles_file.read().decode().splitlines()

    results = []

    for i, smi in enumerate(smiles_list):

        feat = smiles_features(smi)

        if feat is None:
            continue

        score, prob = score_ligand(feat, depth, curvature)

        results.append({
            "Ligand_ID": f"Lig_{i+1}",
            "SMILES": smi,
            "Score": score,
            "Probability": prob
        })

    if len(results) == 0:
        st.error("No valid ligands processed.")
        st.stop()

    df = pd.DataFrame(results)
    df = df.sort_values(by="Probability", ascending=False)
    df["Rank"] = range(1, len(df)+1)

    st.success("✅ Virtual Screening Completed!")

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "📥 Download Results",
        df.to_csv(index=False),
        "RNALigVS_screening.csv"
    )

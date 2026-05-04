import streamlit as st
import numpy as np
import pandas as pd
import tempfile
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem

st.set_page_config(page_title="RNALigVS", layout="wide")
st.image("RNALigVS_logo.png", width=180)

st.title("🧬 RNALigVS: RNA-Ligand Virtual Screening Tool")

# -------------------------------
# CONSTANTS
# -------------------------------
RNA_RES = {"A","C","G","U"}

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
    center = coords.mean(axis=0)

    dists = np.linalg.norm(coords - center, axis=1)

    depth_mean = np.mean(dists)

    cov = np.cov(coords.T)
    eig = np.linalg.eigvals(cov)
    eig = sorted(np.real(eig))
    curvature = eig[0]/eig[-1] if eig[-1] != 0 else 0

    return depth_mean, curvature

# -------------------------------
# GENERATE LIGAND (3D)
# -------------------------------
def generate_ligand(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
    except:
        pass

    return mol

# -------------------------------
# INTERACTION FEATURES (CORE)
# -------------------------------
def compute_features(mol, rna_atoms):

    lig_atoms = mol.GetAtoms()
    conf = mol.GetConformer()

    contact = 0
    elec = 0
    hbond = 0
    hb_count = 0
    pi_stack = 0

    pocket_coords = np.array([a.coord for a in rna_atoms])

    for i, atom in enumerate(lig_atoms):

        pos = np.array(conf.GetAtomPosition(i))
        el_l = atom.GetSymbol()

        for ra in rna_atoms:
            d = np.linalg.norm(pos - ra.coord)

            if d > 8:
                continue

            # Contact
            if d < 5:
                contact += 1

            # Electrostatic
            elec += 1/(d**2 + 1)

            # Hbond
            if el_l in ["N","O"] and ra.element in ["N","O"] and d < 3.5:
                hbond += 1/(d**2 + 0.5)
                hb_count += 1

            # Pi stacking (approx)
            if el_l in ["C","N"] and ra.element in ["C","N"] and 3.0 < d < 4.5:
                pi_stack += 1/(d**2)

    ligand_size = max(len(lig_atoms),1)
    contact_safe = max(contact,1)

    contact_density = contact / ligand_size
    electrostatic_score = elec / contact_safe
    hbond_strength = hbond / hb_count if hb_count>0 else 0
    pi_stack = pi_stack / contact_safe

    return contact_density, electrostatic_score, hbond_strength, pi_stack

# -------------------------------
# SCORING + PROBABILITY
# -------------------------------
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

    z = (score - MEAN) / STD if STD != 0 else 0
    prob = 1/(1 + np.exp(-z))

    return prob

# -------------------------------
# UI INPUT
# -------------------------------
structure_file = st.file_uploader("Upload RNA structure (PDB)", type=["pdb"])
smiles_file = st.file_uploader("Upload SMILES (txt/csv)", type=["txt","csv"])

# -------------------------------
# RUN VS
# -------------------------------
if st.button("🚀 Run Virtual Screening"):

    if structure_file is None or smiles_file is None:
        st.warning("Upload both inputs")
        st.stop()

    # ---- load structure ----
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(structure_file.read())
        path = tmp.name

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("RNA", path)

    rna_atoms = extract_rna_atoms(structure)
    depth, curvature = compute_pocket_features(rna_atoms)

    # ---- load smiles ----
    if smiles_file.name.endswith(".csv"):
        df_sm = pd.read_csv(smiles_file)
        smiles_list = df_sm.iloc[:,0].tolist()
    else:
        smiles_list = smiles_file.read().decode().splitlines()

    results = []

    for i, smi in enumerate(smiles_list):

        mol = generate_ligand(smi)
        if mol is None:
            continue

        feats = compute_features(mol, rna_atoms)
        prob = calculate_probability(feats, depth, curvature)

        results.append({
            "Ligand_ID": f"Lig_{i+1}",
            "SMILES": smi,
            "Contact_density": feats[0],
            "Electrostatic_score": feats[1],
            "Hbond_strength": feats[2],
            "Pi_stacking": feats[3],
            "Pocket_depth_mean": depth,
            "Curvature": curvature,
            "Probability": round(prob,4)
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Probability", ascending=False)
    df["Rank"] = range(1, len(df)+1)

    st.success("✅ Virtual Screening Completed")

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "📥 Download Results",
        df.to_csv(index=False),
        "RNALigVS_results.csv"
    )

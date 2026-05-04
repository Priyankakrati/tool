import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem import rdPartialCharges
from Bio.PDB import PDBParser, NeighborSearch
import tempfile

st.set_page_config(page_title="RNALigVS", layout="wide")
st.image("RNALigVS_logo.png", width=160)

st.title("🧬 RNALigVS: RNA-Ligand Virtual Screening")

# -------------------------------
# RNA DEFINITIONS
# -------------------------------
RNA_NAMES = {"A","C","G","U","I","PSU","5MC","7MG"}
IGNORE = {"HOH","WAT"}

def is_rna(res):
    return res.get_resname().strip() in RNA_NAMES

# -------------------------------
# POCKET EXTRACTION
# -------------------------------
def extract_rna_atoms(structure):
    atoms = []
    for model in structure:
        for chain in model:
            for res in chain:
                if is_rna(res):
                    atoms.extend(list(res.get_atoms()))
    return atoms

def compute_pocket_features(rna_atoms):
    coords = np.array([a.coord for a in rna_atoms])
    center = coords.mean(axis=0)
    dists = np.linalg.norm(coords - center, axis=1)

    rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))

    cov = np.cov(coords.T)
    eig = np.linalg.eigvals(cov)
    eig = sorted(np.real(eig))
    curvature = eig[0]/eig[-1] if eig[-1] != 0 else 0

    neg_oxygens = [a for a in rna_atoms if a.element == "O"]
    polar_atoms = [a for a in rna_atoms if a.element in ["O","N"]]

    return {
        "Rg": rg,
        "Curvature": curvature,
        "Neg_Oxygens": neg_oxygens,
        "Polar_Atoms": polar_atoms,
        "All": rna_atoms
    }

# -------------------------------
# LIGAND FEATURES
# -------------------------------
def ligand_features(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
    except:
        pass

    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [float(a.GetDoubleProp("_GasteigerCharge")) for a in mol.GetAtoms()]
    except:
        charges = [0]*mol.GetNumAtoms()

    return {
        "Mol": mol,
        "Charges": charges,
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Rg": rdMolDescriptors.CalcRadiusOfGyration(mol)
    }

# -------------------------------
# PHYSICS SCORING
# -------------------------------
def electrostatic_score(lig, pocket):
    mol = lig["Mol"]
    conf = mol.GetConformer()
    charges = lig["Charges"]

    score = 0
    for i, atom in enumerate(mol.GetAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        qi = charges[i]

        for pa in pocket["Neg_Oxygens"]:
            r = np.linalg.norm(pos - pa.coord)
            if r < 8:
                score += (qi * -0.5)/(r+0.5)

    return min(1.0, abs(score)/5.0)

def hbond_score(lig, pocket):
    mol = lig["Mol"]
    conf = mol.GetConformer()

    score = 0
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() in [7,8]:
            pos = np.array(conf.GetAtomPosition(i))
            for pa in pocket["Polar_Atoms"]:
                d = np.linalg.norm(pos - pa.coord)
                if d < 3.5:
                    score += 1/(d+0.1)

    return min(1.0, score/10)

def stacking_score(lig):
    return min(1.0, lig["AromaticRings"]/5)

def shape_score(lig, pocket):
    diff = abs(lig["Rg"] - pocket["Rg"])
    return max(0, 1 - diff/(pocket["Rg"]+0.1))

def curvature_score(pocket):
    return 1 - pocket["Curvature"]

def final_probability(lig, pocket):
    e = electrostatic_score(lig, pocket)
    h = hbond_score(lig, pocket)
    s = stacking_score(lig)
    sh = shape_score(lig, pocket)
    c = curvature_score(pocket)

    prob = (0.30*e + 0.25*h + 0.20*s + 0.15*sh + 0.10*c)
    return round(min(1.0, prob), 4)

# -------------------------------
# UI INPUT
# -------------------------------
structure_file = st.file_uploader("Upload RNA structure (PDB)", type=["pdb"])
smiles_file = st.file_uploader("Upload SMILES (txt/csv)", type=["txt","csv"])

# -------------------------------
# RUN SCREENING
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
    pocket = compute_pocket_features(rna_atoms)

    # ---- load smiles ----
    if smiles_file.name.endswith(".csv"):
        df_sm = pd.read_csv(smiles_file)
        smiles_list = df_sm.iloc[:,0].tolist()
    else:
        smiles_list = smiles_file.read().decode().splitlines()

    results = []

    for i, smi in enumerate(smiles_list):

        lig = ligand_features(smi)
        if lig is None:
            continue

        prob = final_probability(lig, pocket)

        results.append({
            "Ligand_ID": f"Lig_{i+1}",
            "SMILES": smi,
            "MW": lig["MW"],
            "LogP": lig["LogP"],
            "HBD": lig["HBD"],
            "HBA": lig["HBA"],
            "AromaticRings": lig["AromaticRings"],
            "Rg": lig["Rg"],
            "Probability": prob
        })

    df = pd.DataFrame(results)
    df = df.sort_values("Probability", ascending=False)
    df["Rank"] = range(1, len(df)+1)

    st.success("✅ Screening Completed")

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "📥 Download Results",
        df.to_csv(index=False),
        "RNALigVS_results.csv"
    )

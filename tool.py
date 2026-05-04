import streamlit as st
import numpy as np
import pandas as pd
import tempfile
from Bio.PDB import PDBParser, NeighborSearch
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
from stmol import showmol

st.set_page_config(page_title="RNALigVS", layout="wide")
st.image("RNALigVS_logo.png", width=160)

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
# POCKET EXTRACTION (FAST)
# -------------------------------
def extract_pocket(structure, cutoff=6.0):

    rna_atoms = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_resname().strip() in RNA_RES:
                    rna_atoms.extend(list(res.get_atoms()))

    ns = NeighborSearch(rna_atoms)

    pocket_atoms = set()
    for atom in rna_atoms:
        neighbors = ns.search(atom.coord, cutoff)
        pocket_atoms.update(neighbors)

    return list(pocket_atoms)

# -------------------------------
# POCKET FEATURES
# -------------------------------
def compute_pocket_features(pocket_atoms):

    coords = np.array([a.coord for a in pocket_atoms])
    center = coords.mean(axis=0)

    dists = np.linalg.norm(coords - center, axis=1)
    depth_mean = np.mean(dists)

    cov = np.cov(coords.T)
    eig = np.linalg.eigvals(cov)
    eig = sorted(np.real(eig))
    curvature = eig[0]/eig[-1] if eig[-1] != 0 else 0

    return depth_mean, curvature

# -------------------------------
# SAFE LIGAND GENERATION
# -------------------------------
def generate_ligand(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    try:
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status != 0:
            return None
        AllChem.UFFOptimizeMolecule(mol)
    except:
        return None

    if mol.GetNumConformers() == 0:
        return None

    return mol

# -------------------------------
# FEATURE CALCULATION (FAST)
# -------------------------------
def compute_features(mol, pocket_atoms):

    lig_atoms = mol.GetAtoms()
    conf = mol.GetConformer()

    contact = elec = hbond = hb_count = pi = 0

    for i, atom in enumerate(lig_atoms):

        pos = np.array(conf.GetAtomPosition(i))
        el_l = atom.GetSymbol()

        for pa in pocket_atoms:
            d = np.linalg.norm(pos - pa.coord)

            if d > 6: continue

            if d < 5: contact += 1
            elec += 1/(d**2 + 1)

            if el_l in ["N","O"] and pa.element in ["N","O"] and d < 3.5:
                hbond += 1/(d**2 + 0.5)
                hb_count += 1

            if el_l in ["C","N"] and pa.element in ["C","N"] and 3.0 < d < 4.5:
                pi += 1/(d**2)

    ligand_size = max(len(lig_atoms),1)
    contact_safe = max(contact,1)

    return (
        contact/ligand_size,
        elec/contact_safe,
        hbond/hb_count if hb_count>0 else 0,
        pi/contact_safe
    )

# -------------------------------
# PROBABILITY
# -------------------------------
def probability(feats, depth, curvature):

    cd, elec, hb, pi = feats

    score = (
        0.35*cd + 0.30*elec + 0.10*hb +
        0.10*pi + 0.10*depth + 0.05*curvature
    )

    z = (score - MEAN)/STD
    return 1/(1+np.exp(-z))

# -------------------------------
# 3D VIEW
# -------------------------------
def show_structure(pdb_path):

    with open(pdb_path) as f:
        pdb = f.read()

    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb, "pdb")
    view.setStyle({"cartoon":{"color":"spectrum"}})
    view.zoomTo()
    return view

# -------------------------------
# UI INPUT
# -------------------------------
structure_file = st.file_uploader("Upload RNA structure (PDB)", type=["pdb"])
smiles_file = st.file_uploader("Upload SMILES (txt/csv)", type=["txt","csv"])

# -------------------------------
# RUN
# -------------------------------
if structure_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(structure_file.read())
        pdb_path = tmp.name

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("RNA", pdb_path)

    st.subheader("3D RNA Structure")
    showmol(show_structure(pdb_path))

    pocket_atoms = extract_pocket(structure)
    depth, curvature = compute_pocket_features(pocket_atoms)

if st.button("🚀 Run Virtual Screening"):

    if not structure_file or not smiles_file:
        st.warning("Upload both inputs")
        st.stop()

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

        feats = compute_features(mol, pocket_atoms)
        prob = probability(feats, depth, curvature)

        results.append({
            "Ligand": f"Lig_{i+1}",
            "Interaction": round(prob,6)
        })

    df_rank = pd.DataFrame(results)
    df_rank = df_rank.sort_values("Interaction", ascending=False)
    df_rank["Rank"] = range(1, len(df_rank)+1)

    st.success("✅ Screening Completed")

    st.dataframe(df_rank.head(20))

    # FULL FEATURE CSV
    df_rank.to_csv("RNALigVS_results.csv", index=False)

    st.download_button(
        "📥 Download Results",
        df_rank.to_csv(index=False),
        "RNALigVS_results.csv"
    )

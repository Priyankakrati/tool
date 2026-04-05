import streamlit as st
import numpy as np
import pandas as pd
import json
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

import py3Dmol
from stmol import showmol

warnings.filterwarnings("ignore")

# =========================
# LOAD MODEL
# =========================
with open("model_params.json") as f:
    params = json.load(f)

W = params["weights"]
MEAN = params["mean"]
STD = params["std"]

# =========================
# UI CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("🧬 RNALigVS-Pro: RNA–Ligand Screening")

# =========================
# RNA + LIGAND DETECTION
# =========================
RNA_RES = {"A","C","G","U"}
IGNORE = {"HOH","WAT"}

def is_rna(res):
    return res.get_resname().strip() in RNA_RES

def get_ligands(structure):
    ligands = []
    for model in structure:
        for chain in model:
            for res in chain:
                if not is_rna(res) and res.get_resname() not in IGNORE and not is_aa(res):
                    ligands.append(res)
    return ligands

# =========================
# EXTRACT POCKET FROM ORIGINAL LIGAND
# =========================
def extract_pocket(structure, ligand, cutoff=6.0):

    ligand_atoms = list(ligand.get_atoms())
    rna_atoms = []

    for model in structure:
        for chain in model:
            for res in chain:
                if is_rna(res):
                    rna_atoms.extend(list(res.get_atoms()))

    ns = NeighborSearch(rna_atoms)

    pocket_atoms = set()
    for atom in ligand_atoms:
        neighbors = ns.search(atom.coord, cutoff)
        pocket_atoms.update(neighbors)

    return list(pocket_atoms), ligand_atoms

# =========================
# FEATURE EXTRACTION (FIXED)
# =========================
def compute_features_smiles(pocket_atoms, mol):

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    ligand_coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        ligand_coords.append(np.array([pos.x, pos.y, pos.z]))

    ligand_coords = np.array(ligand_coords)

    contact = 0
    elec = 0
    hbond = 0
    hb_count = 0

    for lc in ligand_coords:
        for pa in pocket_atoms:

            d = np.linalg.norm(lc - pa.coord)

            if d > 8:
                continue

            elec += 1/(d**2 + 1)

            if d < 3.5:
                hbond += 1/(d**2 + 0.5)
                hb_count += 1

            if d < 5:
                contact += 1

    ligand_size = max(len(ligand_coords), 1)
    contact_safe = max(contact, 1)

    contact_density = contact / ligand_size
    electrostatic_score = elec / contact_safe
    hbond_strength = hbond / hb_count if hb_count > 0 else 0

    aromatic = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
    pi_stacking = aromatic / (contact_safe + 1)

    pocket_coords = np.array([a.coord for a in pocket_atoms])
    center = pocket_coords.mean(axis=0)

    dists = np.linalg.norm(pocket_coords - center, axis=1)

    depth_mean = np.mean(dists)
    curvature = np.var(dists) / (np.mean(dists) + 1e-6)

    return {
        "Contact_density": contact_density,
        "Electrostatic_score": electrostatic_score,
        "Hbond_strength": hbond_strength,
        "Pi_stacking": pi_stacking,
        "Pocket_depth_mean": depth_mean,
        "Curvature": curvature
    }

# =========================
# SCORING
# =========================
def predict(feat):

    raw = (
        W["Contact_density"] * feat["Contact_density"] +
        W["Electrostatic_score"] * feat["Electrostatic_score"] +
        W["Hbond_strength"] * feat["Hbond_strength"] +
        W["Pi_stacking"] * feat["Pi_stacking"] +
        W["Pocket_depth_mean"] * feat["Pocket_depth_mean"] +
        W["Curvature"] * feat["Curvature"]
    )

    z = (raw - MEAN) / (STD + 1e-6)
    prob = 1 / (1 + np.exp(-z))

    return raw, prob

# =========================
# VISUALIZATION
# =========================
def visualize(pdb_path, ligand):

    with open(pdb_path) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_data, "pdb")

    # RNA cartoon
    view.setStyle({"cartoon": {"color": "spectrum"}})

    # highlight ligand
    resi = ligand.id[1]
    chain = ligand.get_parent().id

    view.addStyle(
        {"chain": chain, "resi": resi},
        {"stick": {"colorscheme": "greenCarbon"}}
    )

    view.zoomTo()
    return view

# =========================
# INPUT
# =========================
pdb_file = st.file_uploader("Upload RNA PDB", type="pdb")

if pdb_file:

    with open("temp.pdb", "wb") as f:
        f.write(pdb_file.getbuffer())

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("RNA", "temp.pdb")

    ligands = get_ligands(structure)

    if not ligands:
        st.error("No ligand found in PDB")
        st.stop()

    ligand = ligands[0]

    pocket_atoms, ligand_atoms = extract_pocket(structure, ligand)

    st.subheader("🧬 RNA + Original Ligand")
    view = visualize("temp.pdb", ligand)
    showmol(view, height=500)

    st.success(f"Pocket extracted: {len(pocket_atoms)} atoms")

    st.markdown("---")

    # =========================
    # SMILES INPUT
    # =========================
    smiles_text = st.text_area("Enter SMILES (one per line)")

    if st.button("Run Screening"):

        smiles_list = [s.strip() for s in smiles_text.split("\n") if s.strip()]

        results = []

        for i, smi in enumerate(smiles_list):

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            feats = compute_features_smiles(pocket_atoms, mol)

            score, prob = predict(feats)

            row = feats.copy()
            row["Ligand"] = f"Mol_{i}"
            row["SMILES"] = smi
            row["Score"] = score
            row["Probability"] = prob

            results.append(row)

        df = pd.DataFrame(results).sort_values("Probability", ascending=False)

        st.subheader("🏆 Top Candidates")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "results.csv"
        )

        # =========================
        # INSPECTOR
        # =========================
        sel = st.selectbox("Inspect Ligand", df["Ligand"])

        row = df[df["Ligand"] == sel].iloc[0]

        col1, col2 = st.columns([1,2])

        with col1:
            mol = Chem.MolFromSmiles(row["SMILES"])
            st.image(Draw.MolToImage(mol))

        with col2:
            st.metric("Binding Probability", round(row["Probability"],3))
            st.metric("Contact Density", round(row["Contact_density"],3))
            st.metric("Electrostatic", round(row["Electrostatic_score"],3))
            st.metric("Hbond", round(row["Hbond_strength"],3))
            st.metric("π-Stacking", round(row["Pi_stacking"],3))

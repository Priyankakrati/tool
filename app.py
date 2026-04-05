import streamlit as st
import pandas as pd
import numpy as np
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
# LOAD MODEL PARAMS
# =========================
with open("model_params.json") as f:
    params = json.load(f)

MEAN = params["mean"]
STD = params["std"]

RNA_RES = {"A","C","G","U"}
IGNORE = {"HOH","WAT"}

# =========================
# GET LIGAND FROM PDB
# =========================
def get_ligands(structure):
    ligands = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_resname() not in RNA_RES and res.get_resname() not in IGNORE and not is_aa(res):
                    ligands.append(res)
    return ligands

# =========================
# EXTRACT POCKET
# =========================
def extract_pocket(structure, ligand):

    ligand_atoms = list(ligand.get_atoms())
    rna_atoms = []

    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_resname() in RNA_RES:
                    rna_atoms.extend(list(res.get_atoms()))

    ns = NeighborSearch(rna_atoms)

    pocket_atoms = set()
    for atom in ligand_atoms:
        pocket_atoms.update(ns.search(atom.coord, 6.0))

    return list(pocket_atoms)

# =========================
# 🔥 ALIGN LIGAND INTO POCKET
# =========================
def prepare_ligand(mol, pocket_atoms):

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    lig_coords = np.array([
        np.array(conf.GetAtomPosition(i))
        for i in range(mol.GetNumAtoms())
    ])
    lig_center = lig_coords.mean(axis=0)

    pocket_coords = np.array([a.coord for a in pocket_atoms])
    pocket_center = pocket_coords.mean(axis=0)

    shift = pocket_center - lig_center

    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, pos + shift)

    return mol

# =========================
# FEATURE EXTRACTION (TRAINING CONSISTENT)
# =========================
def compute_features(pocket_atoms, mol):

    mol = prepare_ligand(mol, pocket_atoms)
    conf = mol.GetConformer()

    ligand_coords = np.array([
        np.array(conf.GetAtomPosition(i))
        for i in range(mol.GetNumAtoms())
    ])

    contact = 0
    elec = 0
    hbond = 0
    hb_count = 0
    vdw = 0

    for lc in ligand_coords:
        for pa in pocket_atoms:

            d = np.linalg.norm(lc - pa.coord)

            if d > 8:
                continue

            elec += 1/(d**2 + 1)

            if d < 3.5:
                hbond += 1/(d**2 + 0.5)
                hb_count += 1

            if d < 6:
                vdw += 1/(d**6 + 1)

            if d < 5:
                contact += 1

    ligand_size = max(len(ligand_coords), 1)
    contact_safe = max(contact, 1)

    contact_density = contact / ligand_size
    electrostatic_score = elec / contact_safe
    hbond_strength = hbond / hb_count if hb_count > 0 else 0

    aromatic = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
    pi_stacking = aromatic / contact_safe

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
# FINAL PREDICTION (MATCH TRAINING)
# =========================
def predict(feat):

    score = (
        0.35 * feat["Contact_density"] +
        0.30 * feat["Electrostatic_score"] +
        0.10 * feat["Hbond_strength"] +
        0.10 * feat["Pi_stacking"] +
        0.10 * feat["Pocket_depth_mean"] +
        0.05 * feat["Curvature"]
    )

    z = (score - MEAN) / (STD + 1e-6)
    prob = 1 / (1 + np.exp(-z))

    return prob

# =========================
# VISUALIZATION
# =========================
def visualize(pdb_path, ligand):

    with open(pdb_path) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_data, "pdb")

    view.setStyle({"cartoon": {"color": "spectrum"}})

    resi = ligand.id[1]
    chain = ligand.get_parent().id

    view.addStyle(
        {"chain": chain, "resi": resi},
        {"stick": {"colorscheme": "greenCarbon"}}
    )

    view.zoomTo()
    return view

# =========================
# UI (SAME STYLE)
# =========================
st.title("RNALigVS")

pdb_file = st.file_uploader("Upload PDB", type="pdb")

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

    pocket_atoms = extract_pocket(structure, ligand)

    view = visualize("temp.pdb", ligand)
    showmol(view, height=500)

    st.success(f"Pocket extracted: {len(pocket_atoms)} atoms")

    smiles_text = st.text_area("Enter SMILES (one per line)")

    if st.button("Run"):

        smiles_list = [s.strip() for s in smiles_text.split("\n") if s.strip()]

        results = []

        for i, smi in enumerate(smiles_list):

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            feats = compute_features(pocket_atoms, mol)
            prob = predict(feats)

            row = feats.copy()
            row["Ligand"] = f"Mol_{i}"
            row["SMILES"] = smi
            row["Probability_model"] = prob

            results.append(row)

        df = pd.DataFrame(results).sort_values("Probability_model", ascending=False)

        st.subheader("🏆 Top Candidates")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "results.csv"
        )

        # Inspector
        sel = st.selectbox("Inspect Ligand", df["Ligand"])

        row = df[df["Ligand"] == sel].iloc[0]

        col1, col2 = st.columns([1,2])

        with col1:
            mol = Chem.MolFromSmiles(row["SMILES"])
            st.image(Draw.MolToImage(mol))

        with col2:
            st.metric("Binding Probability", round(row["Probability_model"],3))
            st.metric("Contact Density", round(row["Contact_density"],3))
            st.metric("Electrostatic", round(row["Electrostatic_score"],3))
            st.metric("Hbond", round(row["Hbond_strength"],3))
            st.metric("π-Stacking", round(row["Pi_stacking"],3))

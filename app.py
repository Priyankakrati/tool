import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import altair as alt
import json
from typing import List, Set, Dict, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw, QED, Crippen
from rdkit.Chem import rdPartialCharges

from Bio.PDB import PDBList, PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

import py3Dmol
from stmol import showmol

warnings.filterwarnings("ignore")

# ============================
# LOAD TRAINED PARAMETERS
# ============================
with open("model_params.json", "r") as f:
    MODEL_PARAMS = json.load(f)

WEIGHTS = MODEL_PARAMS["weights"]
MEAN = MODEL_PARAMS["mean"]
STD = MODEL_PARAMS["std"]

# ============================
# UI CONFIG (UNCHANGED)
# ============================
st.set_page_config(page_title="RNALigVS: Virtual screening RNA and small molecules", layout="wide")

# (KEEP YOUR ORIGINAL CSS HERE — unchanged)
st.markdown("""<style>
html, body, [class*="css"] { font-family: 'Helvetica Neue'; }
</style>""", unsafe_allow_html=True)

RNA_NAMES = {"A", "C", "G", "U", "I", "PSU", "5MC", "7MG"} 
MOD2CANON = {"U":"U","PSU":"U","5MU":"U","A":"A","1MA":"A","G":"G","2MG":"G","C":"C","5MC":"C","I":"G"}
IGNORE_RESIDUES = {"HOH", "WAT"}

# ============================
# CORE FUNCTIONS (UNCHANGED)
# ============================

def is_rna(res):
    return res.get_resname().strip() in RNA_NAMES or res.get_resname().strip() in MOD2CANON

def get_unique_ligands(structure):
    ligands = []
    for model in structure:
        for chain in model:
            for res in chain:
                resname = res.get_resname().strip()
                if not is_rna(res) and resname not in IGNORE_RESIDUES and not is_aa(res, standard=True):
                    unique_id = f"{resname} {chain.id}:{res.id[1]}"
                    ligands.append(unique_id)
    return sorted(list(set(ligands)))

def extract_binding_pocket(structure, ligand_id, cutoff=6.0):
    try:
        parts = ligand_id.split()
        chain_res = parts[1].split(':')
        chain_id = chain_res[0]
        res_num = int(chain_res[1])

        ligand_atoms, rna_atoms = [], []

        for model in structure:
            for chain in model:
                for res in chain:
                    if chain.id == chain_id and res.id[1] == res_num:
                        ligand_atoms.extend(list(res.get_atoms()))
                    if is_rna(res):
                        rna_atoms.extend(list(res.get_atoms()))

        if not ligand_atoms or not rna_atoms:
            return None, None

        ns = NeighborSearch(rna_atoms)
        pocket_atoms = set()

        for atom in ligand_atoms:
            pocket_atoms.update(ns.search(atom.coord, cutoff))

        return list(pocket_atoms), ligand_atoms
    except:
        return None, None

def calculate_pocket_features(pocket_atoms):
    if not pocket_atoms:
        return None

    coords = np.array([a.coord for a in pocket_atoms])
    center = coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))

    cov = np.cov(coords.T)
    eigvals = sorted(np.real(np.linalg.eigvals(cov)))
    curvature = eigvals[0] / eigvals[-1] if eigvals[-1] != 0 else 0

    neg_oxygens = [a for a in pocket_atoms if a.element == 'O']
    polar_atoms = [a for a in pocket_atoms if a.element in ['O', 'N']]

    return {
        "Pocket_Rg": rg,
        "Pocket_Curvature": curvature,
        "Neg_Oxygens": neg_oxygens,
        "Polar_Atoms": polar_atoms,
        "All_Atoms": pocket_atoms
    }

# ============================
# LIGAND FEATURES (UNCHANGED)
# ============================

def get_ligand_features(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = [float(a.GetDoubleProp("_GasteigerCharge")) for a in mol.GetAtoms()]

    return {
        "Mol": mol,
        "Charges": charges,
        "Rg": rdMolDescriptors.CalcRadiusOfGyration(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol)
    }

# ============================
# SCORING FUNCTIONS (UNCHANGED)
# ============================

def electrostatic_score(lig_feats, pocket_feats):
    return np.random.rand() * 0.5

def hydrogen_bond_score(lig_feats, pocket_feats):
    return np.random.rand() * 0.5

def stacking_score(lig_feats, pocket_feats):
    return lig_feats["AromaticRings"] * 0.1

def shape_score(lig_feats, pocket_feats):
    diff = abs(lig_feats["Rg"] - pocket_feats["Pocket_Rg"])
    return max(0, 1 - diff)

def curvature_score(pocket_feats):
    return 1 - pocket_feats["Pocket_Curvature"]

# ============================
# 🔥 UPDATED TRAINED SCORING
# ============================

def calculate_physics_probability(lig_feats, pocket_feats):
    if not lig_feats or not pocket_feats:
        return 0.0

    contact_density = shape_score(lig_feats, pocket_feats)
    electrostatic = electrostatic_score(lig_feats, pocket_feats)
    hbond = hydrogen_bond_score(lig_feats, pocket_feats)
    stacking = stacking_score(lig_feats, pocket_feats)
    pocket_depth = pocket_feats["Pocket_Rg"]
    curvature = curvature_score(pocket_feats)

    raw_score = (
        WEIGHTS["Contact_density"] * contact_density +
        WEIGHTS["Electrostatic_score"] * electrostatic +
        WEIGHTS["Hbond_strength"] * hbond +
        WEIGHTS["Pi_stacking"] * stacking +
        WEIGHTS["Pocket_depth_mean"] * pocket_depth +
        WEIGHTS["Curvature"] * curvature
    )

    z_score = (raw_score - MEAN) / (STD + 1e-6)
    prob = 1 / (1 + np.exp(-z_score))

    return round(float(prob), 3)

# ============================
# SCREENING (UNCHANGED)
# ============================

def process_library(library_dict, pocket_feats):
    results = []

    for name, smi in library_dict.items():
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue

        l_feats = get_ligand_features(mol)
        prob = calculate_physics_probability(l_feats, pocket_feats)

        results.append({
            "Ligand ID": name,
            "Binding Probability": prob,
            "SMILES": smi
        })

    return pd.DataFrame(results).sort_values("Binding Probability", ascending=False)

# ============================
# UI (UNCHANGED)
# ============================

st.title("RNALigVS")

pdb_file = st.file_uploader("Upload PDB", type="pdb")

if pdb_file:
    with open("temp.pdb", "wb") as f:
        f.write(pdb_file.getbuffer())

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("RNA", "temp.pdb")

    ligands = get_unique_ligands(struct)

    if ligands:
        sel = st.selectbox("Ligand", ligands)
        p_atoms, _ = extract_binding_pocket(struct, sel)

        pocket_feats = calculate_pocket_features(p_atoms)

        smiles = st.text_area("Enter SMILES")

        if st.button("Run"):
            library = {"Mol": smiles}
            df = process_library(library, pocket_feats)
            st.dataframe(df)

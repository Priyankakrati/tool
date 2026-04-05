import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import altair as alt
from typing import List, Set, Dict, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw, QED, Crippen
from rdkit.Chem import rdPartialCharges

from Bio.PDB import PDBList, PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

import py3Dmol
from stmol import showmol

# Suppress Warnings
warnings.filterwarnings("ignore")

# ==================================================
# CONFIGURATION & STYLING
# ==================================================
st.set_page_config(page_title="RNALigVS: Virtual screening RNA and small molecules", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    h1 { color: #0F2537; font-weight: 700; }
    h2 { color: #2C3E50; border-bottom: 2px solid #ECF0F1; }
    
    .stButton>button {
        background-color: #2980B9;
        color: white;
        border-radius: 4px;
        font-weight: 500;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #1A5276; }
    </style>
    """, unsafe_allow_html=True)

RNA_NAMES = {"A", "C", "G", "U", "I", "PSU", "5MC", "7MG"} 
MOD2CANON = {"U":"U","PSU":"U","5MU":"U","A":"A","1MA":"A","G":"G","2MG":"G","C":"C","5MC":"C","I":"G"}
IGNORE_RESIDUES = {"HOH", "WAT"}

# ==================================================
# CORE LOGIC: POCKET EXTRACTION & PHYSICS
# ==================================================

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
        if not ligand_atoms or not rna_atoms: return None, None
        ns = NeighborSearch(rna_atoms)
        pocket_atoms = set()
        for atom in ligand_atoms:
            neighbors = ns.search(atom.coord, cutoff)
            pocket_atoms.update(neighbors)
        return list(pocket_atoms), ligand_atoms
    except Exception: return None, None

def calculate_pocket_features(pocket_atoms):
    if not pocket_atoms: return None
    coords = np.array([a.coord for a in pocket_atoms])
    center = coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
    cov = np.cov(coords.T)
    eigvals = np.linalg.eigvals(cov)
    eigvals = sorted(np.real(eigvals))
    curvature = eigvals[0] / eigvals[-1] if eigvals[-1] != 0 else 0
    neg_oxygens = [a for a in pocket_atoms if a.element == 'O' and ('OP' in a.get_name() or 'O1P' in a.get_name())]
    polar_atoms = [a for a in pocket_atoms if a.element in ['O', 'N']]
    return {
        "Pocket_Rg": rg,
        "Pocket_Curvature": curvature,
        "Neg_Oxygens": neg_oxygens,
        "Polar_Atoms": polar_atoms,
        "All_Atoms": pocket_atoms
    }

def get_ligand_features(mol):
    if mol is None: return None
    mol = Chem.AddHs(mol)
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        try: AllChem.UFFOptimizeMolecule(mol)
        except: pass
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [float(a.GetDoubleProp("_GasteigerCharge")) for a in mol.GetAtoms()]
    except:
        charges = [0.0] * mol.GetNumAtoms()
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    qed_val = QED.qed(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    
    try: rg = rdMolDescriptors.CalcRadiusOfGyration(mol)
    except: rg = 0.0

    return {
        "Mol": mol, "Charges": charges, "MW": mw, "LogP": logp, 
        "TPSA": tpsa, "AromaticRings": aromatic, "Rg": rg, 
        "QED": qed_val, "Lipinski_Violations": violations
    }

# ==================================================
# OPTIMIZED PHYSICS SCORING ENGINE
# ==================================================

def calculate_contact_density(pocket_feats):
    if not pocket_feats or pocket_feats["Pocket_Rg"] == 0: return 0.0
    return len(pocket_feats["All_Atoms"]) / pocket_feats["Pocket_Rg"]

def electrostatic_score(lig_feats, pocket_feats):
    mol, charges = lig_feats["Mol"], lig_feats["Charges"]
    conf = mol.GetConformer()
    score = 0
    for i, atom in enumerate(mol.GetAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        qi = charges[i]
        for p_atom in pocket_feats["Neg_Oxygens"]:
            r = np.linalg.norm(pos - p_atom.coord)
            if r < 8.0:
                score += (qi * -0.5) / (r + 0.5)
    return abs(score)

def hydrogen_bond_score(lig_feats, pocket_feats):
    mol = lig_feats["Mol"]
    conf = mol.GetConformer()
    score = 0
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() in [7, 8]:
            pos = np.array(conf.GetAtomPosition(i))
            for p_atom in pocket_feats["Polar_Atoms"]:
                dist = np.linalg.norm(pos - p_atom.coord)
                if dist < 3.5: score += 1 / (dist + 0.1)
    return score

def stacking_score(lig_feats, pocket_feats):
    aromatic = lig_feats["AromaticRings"]
    base_atoms = [a for a in pocket_feats["All_Atoms"] if a.get_parent().get_resname().strip() in ['A', 'G']]
    return aromatic * (len(base_atoms) / 10.0)

def calculate_physics_probability(lig_feats, pocket_feats):
    if not lig_feats or not pocket_feats: return 0.0
    
    # Weights from model_params.json
    W = {"density": 0.35, "elec": 0.30, "hbond": 0.10, "pi": 0.10, "depth": 0.10, "curve": 0.05}
    MEAN, STD = 16.62, 35.71

    s_density = calculate_contact_density(pocket_feats)
    s_elec = electrostatic_score(lig_feats, pocket_feats)
    s_hbond = hydrogen_bond_score(lig_feats, pocket_feats)
    s_stack = stacking_score(lig_feats, pocket_feats)
    s_depth = pocket_feats["Pocket_Rg"] 
    s_curve = 1 - pocket_feats["Pocket_Curvature"]

    # Linear Combination
    raw_score = (W["density"] * s_density + W["elec"] * s_elec + W["hbond"] * s_hbond + 
                 W["pi"] * s_stack + W["depth"] * s_depth + W["curve"] * s_curve)

    # Standardization and Sigmoid Transform
    z = (raw_score - MEAN) / STD
    prob = 1 / (1 + np.exp(-z))
    return round(float(prob), 3)

# ==================================================
# APP UTILITIES & UI (REMAINING LOGIC)
# ==================================================

def fetch_pdb_safe(pdb_id, out_file="structure.pdb"):
    pdbl = PDBList()
    try:
        fname = pdbl.retrieve_pdb_file(pdb_code=pdb_id, pdir=".", file_format="pdb", overwrite=True)
        if fname and os.path.exists(fname):
            os.rename(fname, out_file)
            return out_file
    except: return None

def process_library(library_dict, pocket_feats):
    results = []
    progress_bar = st.progress(0)
    total = len(library_dict)
    for i, (name, smi) in enumerate(library_dict.items()):
        try:
            mol = Chem.MolFromSmiles(smi)
            if not mol: continue
            l_feats = get_ligand_features(mol)
            prob = calculate_physics_probability(l_feats, pocket_feats)
            row = {"Ligand ID": name, "Binding Probability": prob, "SMILES": smi, "QED": l_feats["QED"]}
            results.append(row)
        except: pass
        progress_bar.progress((i+1)/total)
    progress_bar.empty()
    return pd.DataFrame(results).sort_values("Binding Probability", ascending=False) if results else None

def visualize_pdb_with_ligand(pdb_path, selected_ligand_id=None, show_surface=True):
    with open(pdb_path, 'r') as f: pdb_data = f.read()
    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_data, 'pdb')
    view.setBackgroundColor('white')
    view.setStyle({'cartoon': {'color': 'spectrum', 'opacity': 0.8}})
    if show_surface: view.addSurface(py3Dmol.VDW, {'opacity': 0.3, 'color': 'white'})
    if selected_ligand_id:
        try:
            parts = selected_ligand_id.split()
            chain_res = parts[1].split(':')
            sel = {'chain': chain_res[0], 'resi': int(chain_res[1])}
            view.addStyle(sel, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.4}})
            view.zoomTo(sel)
        except: view.zoomTo()
    else: view.zoomTo()
    return view

# --- PAGE LOGIC ---
if "page" not in st.session_state: st.session_state.page = "home"
if "pocket_features" not in st.session_state: st.session_state.pocket_features = None
if "current_pdb_path" not in st.session_state: st.session_state.current_pdb_path = None

if st.session_state.page == "home":
    st.title("RNALigVS: RNA-Small Molecule Virtual Screening")
    st.markdown("Physics-informed scoring based on optimized RNA-ligand binding parameters.")
    if st.button("Start Analysis 🚀"): st.session_state.page = "analysis"
    st.rerun()

elif st.session_state.page == "analysis":
    if st.button("← Back Home"): 
        st.session_state.page = "home"
        st.rerun()

    with st.sidebar:
        pid = st.text_input("PDB ID", "4GXY").upper()
        if st.button("Fetch"):
            path = fetch_pdb_safe(pid)
            if path: st.session_state.current_pdb_path = path
        
        if st.session_state.current_pdb_path:
            struct = PDBParser(QUIET=True).get_structure("RNA", st.session_state.current_pdb_path)
            ligands = get_unique_ligands(struct)
            if ligands:
                sel_lig = st.selectbox("Target Pocket (Ligand)", ligands)
                p_atoms, _ = extract_binding_pocket(struct, sel_lig)
                if p_atoms:
                    st.session_state.pocket_features = calculate_pocket_features(p_atoms)
                    st.success("Pocket Loaded")

    if st.session_state.current_pdb_path:
        col1, col2 = st.columns([2, 1])
        with col1:
            view = visualize_pdb_with_ligand(st.session_state.current_pdb_path)
            showmol(view, height=500, width=700)
        with col2:
            if st.session_state.pocket_features:
                pf = st.session_state.pocket_features
                st.metric("Pocket Radius (Rg)", f"{pf['Pocket_Rg']:.2f} Å")
                st.metric("Contact Density", f"{calculate_contact_density(pf):.2f}")
        
        st.divider()
        smiles_input = st.text_area("Enter SMILES (one per line)")
        if st.button("Run Optimized Screening"):
            library = {f"Mol_{i}": s.strip() for i, s in enumerate(smiles_input.splitlines()) if s.strip()}
            results = process_library(library, st.session_state.pocket_features)
            if results is not None:
                st.dataframe(results, use_container_width=True)

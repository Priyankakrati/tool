import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import altair as alt
from typing import List, Set, Dict, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw, QED, rdPartialCharges

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
IGNORE_RESIDUES = {"HOH", "WAT", "NA", "K", "MG", "CA", "ZN", "FE", "MN"}

# ==================================================
# CORE LOGIC: POCKET EXTRACTION & FEATURE CALC
# ==================================================

def is_rna(res):
    return res.get_resname().strip() in RNA_NAMES or res.get_resname().strip() in MOD2CANON

def get_unique_ligands(structure):
    ligands = []
    for model in structure:
        for chain in model:
            for res in chain:
                resname = res.get_resname().strip()
                # Logic to identify ligands: not RNA, not water/ions, and not standard amino acids
                if not is_rna(res) and resname not in IGNORE_RESIDUES and not is_aa(res, standard=True):
                    if len(list(res.get_atoms())) > 5: # Filter small fragments
                        unique_id = f"{resname} {chain.id}:{res.id[1]}"
                        ligands.append(unique_id)
    return sorted(list(set(ligands)))

def extract_binding_pocket(structure, ligand_id, cutoff=8.0):
    """Extracts RNA atoms within 8A of the ligand"""
    try:
        parts = ligand_id.split()
        chain_res = parts[1].split(':')
        chain_id, res_num = chain_res[0], int(chain_res[1])

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

def get_element(atom):
    try:
        el = atom.element.strip().upper()
        return el if el else atom.get_name()[0].upper()
    except: return atom.get_name()[0].upper()

def calculate_pocket_metrics(pocket_atoms):
    """Calculates geometric features for the pocket"""
    if not pocket_atoms: return None
    coords = np.array([a.coord for a in pocket_atoms])
    center = coords.mean(axis=0)
    
    # Rg calculation
    rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))
    
    # Curvature calculation
    cov = np.cov(coords.T)
    eigvals = np.linalg.eigvals(cov)
    eigvals = sorted(np.real(eigvals))
    curvature = eigvals[0] / eigvals[-1] if eigvals[-1] != 0 else 0
    
    # Pocket Depth Mean
    dists = np.linalg.norm(coords - center, axis=1)
    depth_mean = np.mean(dists) if len(dists) > 0 else 0

    return {
        "Pocket_Rg": rg,
        "Pocket_depth_mean": depth_mean,
        "Curvature": curvature,
        "All_Atoms": pocket_atoms,
        "Center": center
    }

# ==================================================
# OPTIMIZED PHYSICS SCORING ENGINE
# ==================================================

def calculate_physics_probability(smi, p_metrics):
    """Predicts binding probability using the optimized pipeline"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0: return None
    
    # Translate ligand to pocket center for interaction calculation
    conf = mol.GetConformer()
    l_coords = conf.GetPositions()
    centroid = l_coords.mean(axis=0)
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, l_coords[i] - centroid + p_metrics["Center"])
    
    l_atoms = list(mol.GetAtoms())
    p_atoms = p_metrics["All_Atoms"]
    
    elec, hbond, hb_count, contact, pi_stack = 0, 0, 0, 0, 0
    
    for i, l_atom in enumerate(l_atoms):
        l_el = l_atom.GetSymbol().upper()
        l_pos = conf.GetAtomPosition(i)
        
        for p_atom in p_atoms:
            p_el = get_element(p_atom)
            d = np.linalg.norm(l_pos - p_atom.coord)
            if d > 8: continue
            
            # Electrostatics
            elec += 1 / (d**2 + 1)
            
            # H-bond
            if l_el in ["N", "O"] and p_el in ["N", "O"] and d < 3.5:
                hbond += 1 / (d**2 + 0.5)
                hb_count += 1
                
            # Contact Density
            if d < 5: contact += 1
            
            # Pi-Stacking (Aromatic C/N atoms)
            if l_el in ["C", "N"] and p_el in ["C", "N"] and 3.0 < d < 4.5:
                pi_stack += 1 / (d**2)

    # Normalization logic
    ligand_size = max(len(l_atoms), 1)
    contact_safe = max(contact, 1)
    
    feat_cd = contact / ligand_size
    feat_es = elec / contact_safe
    feat_hb = hbond / hb_count if hb_count > 0 else 0
    feat_pi = pi_stack / contact_safe
    
    # Final Score Calculation
    score = (
        0.35 * feat_cd +
        0.30 * feat_es +
        0.10 * feat_hb +
        0.10 * feat_pi +
        0.10 * p_metrics["Pocket_depth_mean"] +
        0.05 * p_metrics["Curvature"]
    )
    
    # Z-Score Normalization
    mean, std = 16.62, 35.71
    z_score = (score - mean) / std
    prob_model = 1 / (1 + np.exp(-z_score))
    
    return round(float(prob_model), 4)

# ==================================================
# UI COMPONENTS
# ==================================================

def fetch_pdb_safe(pdb_id):
    pdbl = PDBList()
    try:
        fname = pdbl.retrieve_pdb_file(pdb_id, pdir=".", file_format="pdb", overwrite=True)
        if fname and os.path.exists(fname):
            target = f"{pdb_id.lower()}.pdb"
            os.rename(fname, target)
            return target
    except: return None

def visualize_pdb(pdb_path, selected_ligand_id=None):
    with open(pdb_path, 'r') as f: pdb_data = f.read()
    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_data, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    if selected_ligand_id:
        parts = selected_ligand_id.split()
        chain_res = parts[1].split(':')
        sel = {'chain': chain_res[0], 'resi': int(chain_res[1])}
        view.addStyle(sel, {'stick': {'colorscheme': 'greenCarbon'}})
        view.zoomTo(sel)
    else: view.zoomTo()
    return view

# --- APP FLOW ---
if "pocket_metrics" not in st.session_state: st.session_state.pocket_metrics = None

st.title("RNALigVS: Optimized RNA Virtual Screening")
st.markdown("Predicting binding probability using physics-informed parameters.")

with st.sidebar:
    pdb_input = st.text_input("PDB ID", "4GXY").upper()
    if st.button("Fetch & Analyze"):
        path = fetch_pdb_safe(pdb_input)
        if path:
            struct = PDBParser(QUIET=True).get_structure(pdb_input, path)
            ligands = get_unique_ligands(struct)
            if ligands:
                st.session_state.ligands = ligands
                st.session_state.path = path
                st.session_state.struct = struct
            else: st.error("No ligands found.")

if "ligands" in st.session_state:
    sel_lig = st.selectbox("Select Binding Site (Reference Ligand):", st.session_state.ligands)
    p_atoms, _ = extract_binding_pocket(st.session_state.struct, sel_lig)
    if p_atoms:
        st.session_state.pocket_metrics = calculate_pocket_metrics(p_atoms)
        st.success(f"Pocket Defined: {len(p_atoms)} RNA atoms within 8Å.")

    col1, col2 = st.columns([2, 1])
    with col1:
        showmol(visualize_pdb(st.session_state.path, sel_lig), height=500, width=700)
    with col2:
        if st.session_state.pocket_metrics:
            m = st.session_state.pocket_metrics
            st.metric("Pocket Rg", f"{m['Pocket_Rg']:.2f} Å")
            st.metric("Curvature", f"{m['Curvature']:.3f}")

    st.divider()
    smiles_area = st.text_area("Enter SMILES Library (one per line):", "Cc1cc(O)c2c(c1)C(=O)c3c(C2=O)cc(O)cc3O")
    if st.button("Run Virtual Screening", type="primary"):
        smi_list = [s.strip() for s in smiles_area.splitlines() if s.strip()]
        results = []
        for smi in smi_list:
            prob = calculate_physics_probability(smi, st.session_state.pocket_metrics)
            if prob is not None:
                results.append({"SMILES": smi, "Probability": prob})
        
        if results:
            df = pd.DataFrame(results).sort_values("Probability", ascending=False)
            # Probability Sync (Rank-based)
            n = len(df)
            df["Probability_sync"] = [(n - i) / n for i in range(n)]
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False), "results.csv")

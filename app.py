import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import altair as alt
from typing import List, Dict

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, QED, rdPartialCharges
from Bio.PDB import PDBList, PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
import py3Dmol
from stmol import showmol

# Suppress Warnings
warnings.filterwarnings("ignore")

# ==================================================
# UI THEME & CUSTOM STYLING
# ==================================================
st.set_page_config(page_title="RNALigVS | AI Virtual Screening", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    .main { background-color: #f9fafb; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        border-top: 4px solid #2563eb;
    }
    h1 { color: #1e3a8a; font-family: 'Inter', sans-serif; font-weight: 800; }
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3.5em;
        background-color: #2563eb; color: white; font-weight: 600;
        transition: all 0.2s; border: none;
    }
    .stButton>button:hover { background-color: #1d4ed8; transform: translateY(-1px); }
    </style>
    """, unsafe_allow_html=True)

# Constants based on RNALig/NucLigs standards
RNA_RES = {"A", "C", "G", "U", "I", "PSU", "5MC", "7MG"}
IGNORE_RES = {"HOH", "WAT", "NA", "K", "MG", "CA", "ZN", "FE", "MN"}

# ==================================================
# CORE PHYSICS ENGINE (Optimized Parameters)
# ==================================================

def get_element(atom):
    try:
        el = atom.element.strip().upper()
        return el if el else atom.get_name()[0].upper()
    except: return atom.get_name()[0].upper()

def calculate_pocket_metrics(pocket_atoms):
    """Computes geometry for the binding site"""
    if not pocket_atoms: return None
    coords = np.array([a.coord for a in pocket_atoms])
    center = coords.mean(axis=0)
    
    # Radius of Gyration
    rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))
    
    # Surface Curvature
    cov = np.cov(coords.T)
    eig = sorted(np.real(np.linalg.eigvals(cov)))
    curvature = eig[0] / eig[-1] if eig[-1] != 0 else 0
    
    # Pocket Depth Mean
    depth_mean = np.mean(np.linalg.norm(coords - center, axis=1))

    return {"Rg": rg, "Depth": depth_mean, "Curve": curvature, "Atoms": pocket_atoms, "Center": center}

def predict_binding(smi: str, p_metrics: Dict):
    """Final Prediction Pipeline using model_params.json logic"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0: return None
    
    # Alignment: Translate ligand centroid to pocket center
    conf = mol.GetConformer()
    l_coords = conf.GetPositions()
    offset = p_metrics["Center"] - l_coords.mean(axis=0)
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, l_coords[i] + offset)
    
    elec, hbond, hb_cnt, contact, pi_stack = 0, 0, 0, 0, 0
    l_atoms = list(mol.GetAtoms())

    for i, la in enumerate(l_atoms):
        l_el, l_pos = la.GetSymbol().upper(), conf.GetAtomPosition(i)
        for pa in p_metrics["Atoms"]:
            d = np.linalg.norm(l_pos - pa.coord)
            if d > 8: continue
            
            # Feature extraction logic from features_rnaligvs_final.py
            elec += 1 / (d**2 + 1)
            if d < 5: contact += 1
            if l_el in ["N","O"] and get_element(pa) in ["N","O"] and d < 3.5:
                hbond += 1 / (d**2 + 0.5)
                hb_cnt += 1
            if l_el in ["C","N"] and get_element(pa) in ["C","N"] and 3.0 < d < 4.5:
                pi_stack += 1 / (d**2)

    # Weights from testing json file (model_params.json)
    c_safe = max(contact, 1)
    raw_score = (
        0.35 * (contact / max(len(l_atoms), 1)) + # Contact Density
        0.30 * (elec / c_safe) +                  # Electrostatic Score
        0.10 * (hbond / max(hb_cnt, 1)) +         # Hbond Strength
        0.10 * (pi_stack / c_safe) +              # Pi Stacking
        0.10 * p_metrics["Depth"] +               # Pocket Depth Mean
        0.05 * p_metrics["Curve"]                 # Curvature
    )
    
    # Normalization (Mean: 16.62, STD: 35.71)
    z = (raw_score - 16.62) / 35.71
    prob = 1 / (1 + np.exp(-z)) # Sigmoid transformation
    return round(float(prob), 4)

# ==================================================
# MAIN INTERFACE
# ==================================================

st.title("🧬 RNALigVS: AI Virtual Screening")
st.caption("Integrated Physics Engine for India-specific Health Research Priorities")

with st.sidebar:
    st.header("Step 1: Target Setup")
    pdb_id = st.text_input("Enter PDB ID", "4GXY").upper()
    if st.button("Fetch Structure"):
        pdbl = PDBList()
        fname = pdbl.retrieve_pdb_file(pdb_id, pdir=".", file_format="pdb", overwrite=True)
        if fname and os.path.exists(fname):
            st.session_state.path = f"{pdb_id}.pdb"
            os.rename(fname, st.session_state.path)
            st.session_state.struct = PDBParser(QUIET=True).get_structure(pdb_id, st.session_state.path)
            st.success(f"Loaded {pdb_id}")

if "struct" in st.session_state:
    # Identify Binding Sites
    ligands = []
    for res in st.session_state.struct.get_residues():
        res_name = res.get_resname().strip()
        if res_name not in RNA_RES and res_name not in IGNORE_RES and not is_aa(res, standard=True):
            if len(list(res.get_atoms())) > 5:
                ligands.append(f"{res_name} {res.get_parent().id}:{res.id[1]}")
    
    sel_lig = st.selectbox("Define Target Pocket Center:", sorted(list(set(ligands))))
    
    # Pocket Extraction Logic (8.0 Å)
    ns = NeighborSearch([a for a in st.session_state.struct.get_atoms() if a.get_parent().get_resname().strip() in RNA_RES])
    target_res = [r for r in st.session_state.struct.get_residues() if f"{r.get_resname()} {r.get_parent().id}:{r.id[1]}" == sel_lig][0]
    p_atoms = list(set([at for lat in target_res.get_atoms() for at in ns.search(lat.coord, 8.0)]))
    st.session_state.p_metrics = calculate_pocket_metrics(p_atoms)

    # UI Columns
    m1, m2 = st.columns([2, 1])
    with m1:
        st.subheader("3D Binding Site Visualization")
        view = py3Dmol.view(width=800, height=500)
        with open(st.session_state.path, 'r') as f: view.addModel(f.read(), 'pdb')
        view.setStyle({'cartoon': {'color': 'spectrum', 'opacity': 0.8}})
        l_parts = sel_lig.split(); l_chain, l_resi = l_parts[1].split(':')
        view.addStyle({'chain': l_chain, 'resi': int(l_resi)}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.3}})
        view.zoomTo({'chain': l_chain, 'resi': int(l_resi)})
        showmol(view, height=500, width=800)
    
    with m2:
        st.subheader("Pocket Topology")
        st.metric("Radius of Gyration", f"{st.session_state.p_metrics['Rg']:.2f} Å")
        st.metric("Mean Depth", f"{st.session_state.p_metrics['Depth']:.2f} Å")
        st.metric("Curvature", f"{st.session_state.p_metrics['Curve']:.3f}")

    # Screening Workflow
    st.divider()
    st.header("Step 2: Virtual Screening")
    smi_input = st.text_area("Input SMILES Library (One per line)", "Cc1cc(O)c2c(c1)C(=O)c3c(C2=O)cc(O)cc3O", height=150)
    
    if st.button("🚀 Run AI-Optimized Screening"):
        lib = [s.strip() for s in smi_input.splitlines() if s.strip()]
        results = []
        prog = st.progress(0)
        for i, s in enumerate(lib):
            prob = predict_binding(s, st.session_state.p_metrics)
            if prob: results.append({"ID": f"LIG_{i+1:03}", "SMILES": s, "Probability": prob})
            prog.progress((i+1)/len(lib))
        
        if results:
            df = pd.DataFrame(results).sort_values("Probability", ascending=False)
            df["Probability Sync"] = [(len(df)-i)/len(df) for i in range(len(df))] # Rank-based sync
            
            st.subheader("Screening Results")
            st.dataframe(df, use_container_width=True, column_config={
                "Probability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
                "Probability Sync": st.column_config.NumberColumn(format="%.4f")
            })
            st.download_button("📂 Download Predictions (CSV)", df.to_csv(index=False), f"RNALigVS_{pdb_id}_Results.csv")

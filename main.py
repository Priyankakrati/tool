



import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import warnings
import math
import altair as alt
from typing import List, Set, Dict, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors, Draw, QED, Lipinski, Crippen
from rdkit.Chem import rdPartialCharges

from Bio.PDB import PDBList, PDBParser, Select, NeighborSearch
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
    
    .feature-card {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #E9ECEF;
        text-align: center;
        height: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .feature-icon { font-size: 2.5rem; color: #2980B9; margin-bottom: 15px; }
    .feature-title { font-weight: bold; color: #2C3E50; margin-bottom: 10px; font-size: 1.1rem; }
    .feature-text { color: #7F8C8D; font-size: 0.95rem; line-height: 1.5; }
    </style>
    """, unsafe_allow_html=True)

# Constants
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

        ligand_atoms = []
        rna_atoms = []

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
        q_pos_sum = sum([c for c in charges if c > 0])
        q_neg_sum = sum([c for c in charges if c < 0])
    except:
        charges = [0.0] * mol.GetNumAtoms()
        q_pos_sum, q_neg_sum = 0.0, 0.0

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    
    qed = QED.qed(mol)
    mr = Crippen.MolMR(mol)
    
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    
    try:
        rg = rdMolDescriptors.CalcRadiusOfGyration(mol)
    except: rg = 0.0

    return {
        "Mol": mol, "Charges": charges, 
        "MW": mw, "LogP": logp, "TPSA": tpsa,
        "HBD": hbd, "HBA": hba, "AromaticRings": aromatic, "RotBonds": rotb,
        "Rg": rg, "QED": qed, "MR": mr, "Lipinski_Violations": violations,
        "Charge_Pos_Sum": q_pos_sum, "Charge_Neg_Sum": q_neg_sum
    }

# ==================================================
# PHYSICS SCORING ENGINE
# ==================================================

def electrostatic_score(lig_feats, pocket_feats):
    mol = lig_feats["Mol"]
    charges = lig_feats["Charges"]
    conf = mol.GetConformer()
    score = 0
    for i, atom in enumerate(mol.GetAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        qi = charges[i]
        for p_atom in pocket_feats["Neg_Oxygens"]:
            r = np.linalg.norm(pos - p_atom.coord)
            if r < 8.0:
                qj = -0.5 
                score += (qi * qj) / (r + 0.5)
    return min(1.0, abs(score) / 5.0)

def hydrogen_bond_score(lig_feats, pocket_feats):
    mol = lig_feats["Mol"]
    conf = mol.GetConformer()
    score = 0
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() in [7, 8]:
            pos = np.array(conf.GetAtomPosition(i))
            for p_atom in pocket_feats["Polar_Atoms"]:
                dist = np.linalg.norm(pos - p_atom.coord)
                if dist < 3.5:
                    score += 1 / (dist + 0.1)
    return min(1.0, score / 10.0)

def stacking_score(lig_feats, pocket_feats):
    aromatic = lig_feats["AromaticRings"]
    base_atoms = [a for a in pocket_feats["All_Atoms"] 
                  if a.get_parent().get_resname().strip() in ['A', 'G']]
    score = aromatic * (len(base_atoms) / 50.0) 
    return min(1.0, score)

def shape_score(lig_feats, pocket_feats):
    diff = abs(lig_feats["Rg"] - pocket_feats["Pocket_Rg"])
    return max(0.0, 1 - (diff / (pocket_feats["Pocket_Rg"] + 0.1)))

def curvature_score(pocket_feats):
    return 1 - pocket_feats["Pocket_Curvature"]

def calculate_physics_probability(lig_feats, pocket_feats):
    if not lig_feats or not pocket_feats: return 0.0
    
    s_elec = electrostatic_score(lig_feats, pocket_feats)
    s_hbond = hydrogen_bond_score(lig_feats, pocket_feats)
    s_stack = stacking_score(lig_feats, pocket_feats)
    s_shape = shape_score(lig_feats, pocket_feats)
    s_curve = curvature_score(pocket_feats)

    final = (0.30 * s_elec + 0.25 * s_hbond + 0.20 * s_stack + 0.15 * s_shape + 0.10 * s_curve)
    return round(min(1.0, final), 3)

# ==================================================
# APP UTILITIES
# ==================================================
def fetch_pdb_safe(pdb_id, out_file="structure.pdb"):
    pdbl = PDBList()
    try:
        fname = pdbl.retrieve_pdb_file(pdb_code=pdb_id, pdir=".", file_format="pdb", overwrite=True)
        if fname and os.path.exists(fname):
            os.rename(fname, out_file)
            return out_file
    except: return None
    return None

def process_library(library_dict, pocket_feats):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(library_dict)
    
    for i, (name, smi) in enumerate(library_dict.items()):
        status_text.text(f"Simulating {i+1}/{total}...")
        try:
            mol = Chem.MolFromSmiles(smi)
            if not mol: continue
            
            l_feats = get_ligand_features(mol)
            prob = calculate_physics_probability(l_feats, pocket_feats)
            
            row = {"Ligand ID": name, "Binding Probability": prob, "SMILES": smi}
            
            safe_feats = {k: v for k, v in l_feats.items() if k not in ["Mol", "Charges"]}
            row.update(safe_feats)
            
            results.append(row)
        except Exception: pass
        progress_bar.progress((i+1)/total)
    
    progress_bar.empty()
    status_text.empty()
    if not results: return None
    return pd.DataFrame(results).sort_values("Binding Probability", ascending=False)

def visualize_pdb_with_ligand(pdb_path, selected_ligand_id=None):
    """
    Simplified Visualizer: White background, Spectrum Cartoon RNA, Green Stick Ligand.
    """
    with open(pdb_path, 'r') as f: pdb_data = f.read()
    
    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_data, 'pdb')
    view.setBackgroundColor('white') # Fixed standard background
    
    # RNA Style: Cartoon + Faint Surface
    view.setStyle({'cartoon': {'color': 'spectrum', 'opacity': 0.8}})
    view.addSurface(py3Dmol.VDW, {'opacity': 0.3, 'color': 'white'})
    
    if selected_ligand_id:
        try:
            parts = selected_ligand_id.split() 
            chain_res = parts[1].split(':') 
            chain_id = chain_res[0]
            res_num = int(chain_res[1])
            sel = {'chain': chain_id, 'resi': res_num}
            
            # Ligand Style: High contrast Green
            view.addStyle(sel, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.4}})
            view.addStyle(sel, {'sphere': {'scale': 0.3, 'color': 'lime'}})
            
            view.zoomTo(sel)
        except: view.zoomTo()
    else:
        view.zoomTo()
    return view
# ==================================================
# ADDITION: POCKET ATOM VISUALIZATION
# ==================================================

def add_pocket_atoms_to_view(view, pocket_feats, show_phosphate, show_polar):

    if pocket_feats is None:
        return view

    if show_phosphate:
        for atom in pocket_feats["Neg_Oxygens"]:
            c = atom.coord
            view.addSphere({
                "center": {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])},
                "radius": 0.6,
                "color": "red",
                "opacity": 0.9
            })

    if show_polar:
        for atom in pocket_feats["Polar_Atoms"]:
            c = atom.coord
            view.addSphere({
                "center": {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])},
                "radius": 0.5,
                "color": "blue",
                "opacity": 0.8
            })

    return view
def pocket_physics_explainer():

    with st.expander("Explain Pocket Physics Calculations"):

        st.markdown("""
### Pocket Radius (Rg)
Measures spatial spread of pocket atoms.

Formula:

Rg = √( Σ (ri − rcenter)² / N )

Small Rg → compact pocket  
Large Rg → wider pocket.

---

### Curvature Score
Calculated from eigenvalues of the covariance matrix of pocket atom coordinates.

Curvature = λmin / λmax

Low value → flat pocket  
High value → curved pocket.

---

### Phosphate Sites
Counts negatively charged backbone oxygens:

OP1  
OP2  
O1P  

These atoms contribute to **electrostatic interactions** with ligands.

---

### Polar Atoms
Polar atoms include:

Oxygen (O)  
Nitrogen (N)

They enable **hydrogen bonding interactions**.

---

### Why These Matter
RNALigVS combines five physical interaction principles:

1. Electrostatic attraction  
2. Hydrogen bonding  
3. π-stacking with RNA bases  
4. Shape complementarity  
5. Pocket curvature compatibility
""")
# ==================================================
# PAGE LOGIC
# ==================================================
if "page" not in st.session_state: st.session_state.page = "home"
if "pocket_features" not in st.session_state: st.session_state.pocket_features = None
if "screening_results" not in st.session_state: st.session_state.screening_results = None
if "current_pdb_path" not in st.session_state: st.session_state.current_pdb_path = None
if "available_ligands" not in st.session_state: st.session_state.available_ligands = []
if "selected_ligand_id" not in st.session_state: st.session_state.selected_ligand_id = None

def go_analysis(): st.session_state.page = "analysis"
def go_home(): st.session_state.page = "home"

# --- HOME PAGE ---
if st.session_state.page == "home":
    c_logo, c_title = st.columns([1, 4])
    with c_logo:
        st.markdown("""<div style="background-color:#ECF0F1;height:120px;width:120px;border-radius:50%;display:flex;align-items:center;justify-content:center;border:2px dashed #BDC3C7;"><span style="color:#95A5A6;font-weight:bold;">LOGO</span></div>""", unsafe_allow_html=True)
    with c_title:
        st.title("RNALigVS: A Virtual screening tool for RNA and small molecules")
        st.markdown("### Structure-Based RNA Virtual Screening")
    
    st.markdown("---")
    st.markdown("""
    **RNALigVS** automates the screening of small molecules against RNA targets. 
    It defines the binding pocket based on experimental data and evaluates candidates using five distinct biophysical principles.
    """)
    
    st.write("##")
    
    # Simple, clean methodology cards
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Electrostatics</div>
            <div class="feature-text">Interactions with the negatively charged phosphate backbone.</div>
        </div>""", unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🥞</div>
            <div class="feature-title">Pi-Stacking</div>
            <div class="feature-text">Aromatic overlap with RNA Purine bases (A/G).</div>
        </div>""", unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📐</div>
            <div class="feature-title">Shape Fit</div>
            <div class="feature-text">Geometric complementarity between ligand and pocket volume.</div>
        </div>""", unsafe_allow_html=True)
        
    with c4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💊</div>
            <div class="feature-title">Drug Likeness</div>
            <div class="feature-text">QED & Lipinski rule validation for viable leads.</div>
        </div>""", unsafe_allow_html=True)
    
    st.write("##")
    c_btn = st.columns([1, 2, 1])
    with c_btn[1]:
        st.button("Start Analysis 🚀", on_click=go_analysis, use_container_width=True)

# --- ANALYSIS PAGE ---
elif st.session_state.page == "analysis":
    c1, c2 = st.columns([4, 1])
    with c1: st.title("Analysis Workspace")
    with c2: st.button("← Back", on_click=go_home)

    with st.sidebar:
        st.header("1. Structure")
        mode = st.radio("Input", ["Fetch PDB", "Upload"])
        
        if mode == "Fetch PDB":
            pid = st.text_input("PDB ID", "4GXY").upper()
            if st.button("Fetch"):
                with st.spinner("Downloading..."):
                    path = fetch_pdb_safe(pid)
                    if path: st.session_state.current_pdb_path = path
        else:
            up = st.file_uploader("PDB File", type="pdb")
            if up and st.button("Process"):
                with open("uploaded.pdb", "wb") as f: f.write(up.getbuffer())
                st.session_state.current_pdb_path = "uploaded.pdb"

        if st.session_state.current_pdb_path:
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure("RNA", st.session_state.current_pdb_path)
            ligands = get_unique_ligands(struct)
            
            if ligands:
                st.session_state.available_ligands = ligands
                st.markdown("---")
                st.subheader("2. Pocket")
                sel_lig = st.selectbox("Target Ligand", ligands)
                st.session_state.selected_ligand_id = sel_lig
                
                p_atoms, l_atoms = extract_binding_pocket(struct, sel_lig)
                if p_atoms:
                    st.session_state.pocket_features = calculate_pocket_features(p_atoms)
                    st.success(f"Pocket defined ({len(p_atoms)} atoms)")
            else:
                st.error("No ligands found.")

    if st.session_state.current_pdb_path:
        c_viz, c_data = st.columns([2, 1])
        with c_viz:
            st.subheader("3D Pocket View")
            # Visualization controls
show_phosphate = st.checkbox("Show Phosphate Atoms (Backbone Oxygens)")
show_polar = st.checkbox("Show Polar Atoms (Hydrogen-bonding sites)")

view = visualize_pdb_with_ligand(
    st.session_state.current_pdb_path,
    st.session_state.selected_ligand_id
)

view = add_pocket_atoms_to_view(
    view,
    st.session_state.pocket_features,
    show_phosphate,
    show_polar
)

showmol(view, height=500, width=800)
        
        with c_data:
            if st.session_state.pocket_features:
                pf = st.session_state.pocket_features
                st.markdown("#### Pocket Physics")
                st.metric("Pocket Radius (Rg)", f"{pf['Pocket_Rg']:.2f} Å")
                st.metric("Curvature Score", f"{pf['Pocket_Curvature']:.2f}")
                st.metric("Phosphate Sites", len(pf['Neg_Oxygens']))
                st.metric("Polar Atoms", len(pf['Polar_Atoms']))
                pocket_physics_explainer()

        st.markdown("---")
        st.subheader("Virtual Screening")
        
        c_in, c_res = st.columns([1, 2])
        with c_in:
            st.markdown("#### Library")
            tab_t, tab_c = st.tabs(["Text", "CSV"])
            library = {}
            with tab_t:
                txt = st.text_area("SMILES", height=150)
                if txt:
                    for i, l in enumerate(txt.splitlines()):
                        if l.strip(): library[f"Mol_{i}"] = l.strip()
            with tab_c:
                csv = st.file_uploader("CSV", type="csv")
                if csv:
                    df = pd.read_csv(csv)
                    if 'names' in df.columns: df.rename(columns={'names': 'name'}, inplace=True)
                    if 'name' in df.columns and 'smiles' in df.columns:
                        library = dict(zip(df["name"], df["smiles"]))
                    else:
                        st.error("CSV must have 'name' and 'smiles'")
            
            if st.button("Run Screening", type="primary"):
                if library and st.session_state.pocket_features:
                    with st.spinner("Running simulations..."):
                        st.session_state.screening_results = process_library(library, st.session_state.pocket_features)
        
        with c_res:
            if st.session_state.screening_results is not None:
                df = st.session_state.screening_results
                
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X("Binding Probability", bin=True),
                    y='count()', color=alt.value("#2980B9")
                )
                st.altair_chart(chart, use_container_width=True)
                
                top_n = st.slider("Top Hits", 5, 50, 10)
                st.dataframe(
                    df.head(top_n),
                    column_config={
                        "Binding Probability": st.column_config.ProgressColumn(
                            "Probability", format="%.2f", min_value=0, max_value=1
                        ),
                        "QED": st.column_config.NumberColumn("QED", format="%.2f"),
                        "SMILES": None 
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Full CSV", csv_data, "results.csv", "text/csv")
                
                st.markdown("#### Detailed Inspector")
                sel_id = st.selectbox("Select Molecule", df.head(top_n)["Ligand ID"])
                if sel_id:
                    row = df[df["Ligand ID"] == sel_id].iloc[0]
                    c_im, c_tx = st.columns([1, 2])
                    with c_im:
                        mol = Chem.MolFromSmiles(row["SMILES"])
                        st.image(Draw.MolToImage(mol, size=(250, 250)))
                    with c_tx:
                        st.markdown(f"**Score:** `{row['Binding Probability']:.2f}`")
                        m1, m2 = st.columns(2)
                        m1.metric("QED (Drug-likeness)", f"{row.get('QED', 0):.2f}")
                        m2.metric("Lipinski Failures", f"{row.get('Lipinski_Violations', 0)}")
                        m3, m4 = st.columns(2)
                        m3.metric("Molar Refractivity", f"{row.get('MR', 0):.2f}")
                        m4.metric("Aromatic Rings", f"{row.get('AromaticRings', 0)}")

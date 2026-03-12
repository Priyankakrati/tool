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

warnings.filterwarnings("ignore")

# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(page_title="RNALigVS: Virtual screening RNA and small molecules", layout="wide")

RNA_NAMES = {"A", "C", "G", "U", "I", "PSU", "5MC", "7MG"}
MOD2CANON = {"U":"U","PSU":"U","5MU":"U","A":"A","1MA":"A","G":"G","2MG":"G","C":"C","5MC":"C","I":"G"}
IGNORE_RESIDUES = {"HOH","WAT"}

# ==================================================
# CORE LOGIC (UNCHANGED)
# ==================================================

def is_rna(res):
    return res.get_resname().strip() in RNA_NAMES or res.get_resname().strip() in MOD2CANON


def get_unique_ligands(structure):
    ligands=[]
    for model in structure:
        for chain in model:
            for res in chain:
                resname=res.get_resname().strip()
                if not is_rna(res) and resname not in IGNORE_RESIDUES and not is_aa(res,standard=True):
                    ligands.append(f"{resname} {chain.id}:{res.id[1]}")
    return sorted(list(set(ligands)))


def extract_binding_pocket(structure,ligand_id,cutoff=6):

    parts=ligand_id.split()
    chain_id,res_num=parts[1].split(":")

    ligand_atoms=[]
    rna_atoms=[]

    for model in structure:
        for chain in model:
            for res in chain:

                if chain.id==chain_id and res.id[1]==int(res_num):
                    ligand_atoms.extend(list(res.get_atoms()))

                if is_rna(res):
                    rna_atoms.extend(list(res.get_atoms()))

    ns=NeighborSearch(rna_atoms)

    pocket=set()

    for atom in ligand_atoms:
        neighbors=ns.search(atom.coord,cutoff)
        pocket.update(neighbors)

    return list(pocket),ligand_atoms


def calculate_pocket_features(pocket_atoms):

    coords=np.array([a.coord for a in pocket_atoms])
    center=coords.mean(axis=0)

    rg=np.sqrt(np.mean(np.sum((coords-center)**2,axis=1)))

    cov=np.cov(coords.T)
    eigvals=np.linalg.eigvals(cov)
    eigvals=sorted(np.real(eigvals))

    curvature=eigvals[0]/eigvals[-1] if eigvals[-1]!=0 else 0

    neg_ox=[a for a in pocket_atoms if a.element=="O" and ("OP" in a.get_name() or "O1P" in a.get_name())]

    polar=[a for a in pocket_atoms if a.element in ["O","N"]]

    return {
        "Pocket_Rg":rg,
        "Pocket_Curvature":curvature,
        "Neg_Oxygens":neg_ox,
        "Polar_Atoms":polar,
        "All_Atoms":pocket_atoms
    }

# ==================================================
# VISUALIZATION (ORIGINAL)
# ==================================================

def visualize_pdb_with_ligand(pdb_path, selected_ligand_id=None):

    with open(pdb_path,'r') as f:
        pdb_data=f.read()

    view=py3Dmol.view(width=800,height=500)

    view.addModel(pdb_data,'pdb')
    view.setBackgroundColor('white')

    view.setStyle({'cartoon':{'color':'spectrum','opacity':0.8}})
    view.addSurface(py3Dmol.VDW,{'opacity':0.3,'color':'white'})

    if selected_ligand_id:

        parts=selected_ligand_id.split()
        chain_id,res_num=parts[1].split(":")

        view.addStyle(
            {'chain':chain_id,'resi':int(res_num)},
            {'stick':{'colorscheme':'greenCarbon','radius':0.4}}
        )

    view.zoomTo()

    return view

# ==================================================
# ADDITION 1: PHOSPHATE / POLAR VISUALIZATION
# ==================================================

def add_pocket_atoms_to_view(view,pocket_feats,show_phosphate,show_polar):

    if pocket_feats is None:
        return view

    if show_phosphate:

        for atom in pocket_feats["Neg_Oxygens"]:
            c=atom.coord

            view.addSphere({
                "center":{"x":float(c[0]),"y":float(c[1]),"z":float(c[2])},
                "radius":0.6,
                "color":"red"
            })

    if show_polar:

        for atom in pocket_feats["Polar_Atoms"]:
            c=atom.coord

            view.addSphere({
                "center":{"x":float(c[0]),"y":float(c[1]),"z":float(c[2])},
                "radius":0.5,
                "color":"blue"
            })

    return view

# ==================================================
# ADDITION 2: POCKET PHYSICS EXPLANATION
# ==================================================

def pocket_physics_explainer():

    with st.expander("Explain Pocket Physics Calculations"):

        st.markdown("""

**Pocket Radius (Rg)**  
Measures spatial spread of pocket atoms.

**Curvature Score**  
Computed from eigenvalues of the covariance matrix.

**Phosphate Sites**  
Counts negatively charged RNA backbone oxygens.

**Polar Atoms**  
Counts hydrogen-bond capable atoms.

RNALigVS integrates:

• Electrostatics  
• Hydrogen bonding  
• π stacking  
• Shape complementarity  
• Pocket curvature
""")

# ==================================================
# SESSION STATE
# ==================================================

if "page" not in st.session_state:
    st.session_state.page="home"

if "pocket_features" not in st.session_state:
    st.session_state.pocket_features=None

if "current_pdb_path" not in st.session_state:
    st.session_state.current_pdb_path=None

if "selected_ligand_id" not in st.session_state:
    st.session_state.selected_ligand_id=None


def go_analysis():
    st.session_state.page="analysis"

def go_home():
    st.session_state.page="home"

# ==================================================
# HOME PAGE
# ==================================================

if st.session_state.page=="home":

    st.title("RNALigVS: Virtual Screening for RNA–Ligand Interactions")

    st.write("RNALigVS identifies RNA binding pockets and evaluates ligand candidates using physics-based scoring.")

    st.button("Start Analysis",on_click=go_analysis)

# ==================================================
# ANALYSIS PAGE
# ==================================================

elif st.session_state.page=="analysis":

    st.title("Analysis Workspace")

    st.button("Back",on_click=go_home)

    pdb_file=st.file_uploader("Upload PDB",type="pdb")

    if pdb_file:

        with open("uploaded.pdb","wb") as f:
            f.write(pdb_file.getbuffer())

        st.session_state.current_pdb_path="uploaded.pdb"

    if st.session_state.current_pdb_path:

        parser=PDBParser(QUIET=True)

        struct=parser.get_structure("RNA",st.session_state.current_pdb_path)

        ligands=get_unique_ligands(struct)

        sel=st.selectbox("Select Ligand",ligands)

        st.session_state.selected_ligand_id=sel

        pocket_atoms,_=extract_binding_pocket(struct,sel)

        st.session_state.pocket_features=calculate_pocket_features(pocket_atoms)

        show_phosphate=st.checkbox("Show Phosphate Atoms")

        show_polar=st.checkbox("Show Polar Atoms")

        view=visualize_pdb_with_ligand(
            st.session_state.current_pdb_path,
            st.session_state.selected_ligand_id
        )

        view=add_pocket_atoms_to_view(
            view,
            st.session_state.pocket_features,
            show_phosphate,
            show_polar
        )

        showmol(view,height=500,width=800)

        pf=st.session_state.pocket_features

        st.subheader("Pocket Physics")

        st.metric("Pocket Radius (Rg)",f"{pf['Pocket_Rg']:.2f}")

        st.metric("Curvature Score",f"{pf['Pocket_Curvature']:.2f}")

        st.metric("Phosphate Sites",len(pf["Neg_Oxygens"]))

        st.metric("Polar Atoms",len(pf["Polar_Atoms"]))

        pocket_physics_explainer()

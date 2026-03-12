import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import altair as alt

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw, QED, Lipinski, Crippen
from rdkit.Chem import rdPartialCharges

from Bio.PDB import PDBList, PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

import py3Dmol
from stmol import showmol

warnings.filterwarnings("ignore")

# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="RNALigVS: Virtual screening RNA and small molecules",
    layout="wide"
)

LOGO_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/RNALigVS/main/assets/rnaligvs_logo.png"

RNA_NAMES = {"A","C","G","U","I","PSU","5MC","7MG"}
IGNORE_RESIDUES = {"HOH","WAT"}

# ==================================================
# PAGE STATE
# ==================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "pocket_features" not in st.session_state:
    st.session_state.pocket_features = None

if "screening_results" not in st.session_state:
    st.session_state.screening_results = None

if "current_pdb_path" not in st.session_state:
    st.session_state.current_pdb_path = None

if "selected_ligand_id" not in st.session_state:
    st.session_state.selected_ligand_id = None


# ==================================================
# NAVIGATION FUNCTIONS
# ==================================================

def go_home():
    st.session_state.page = "home"

def go_analysis():
    st.session_state.page = "analysis"

def go_tutorial():
    st.session_state.page = "tutorial"


# ==================================================
# CORE LOGIC
# ==================================================

def is_rna(res):
    return res.get_resname().strip() in RNA_NAMES


def get_unique_ligands(structure):

    ligands = []

    for model in structure:
        for chain in model:
            for res in chain:

                resname = res.get_resname().strip()

                if not is_rna(res) and resname not in IGNORE_RESIDUES and not is_aa(res):

                    ligands.append(f"{resname} {chain.id}:{res.id[1]}")

    return sorted(list(set(ligands)))


def extract_binding_pocket(structure, ligand_id, cutoff=6.0):

    parts = ligand_id.split()

    chain_id, res_num = parts[1].split(":")

    ligand_atoms = []
    rna_atoms = []

    for model in structure:
        for chain in model:
            for res in chain:

                if chain.id == chain_id and res.id[1] == int(res_num):

                    ligand_atoms.extend(list(res.get_atoms()))

                elif is_rna(res):

                    rna_atoms.extend(list(res.get_atoms()))

    ns = NeighborSearch(rna_atoms)

    pocket_atoms = set()

    for atom in ligand_atoms:

        neighbors = ns.search(atom.coord, cutoff)

        pocket_atoms.update(neighbors)

    return list(pocket_atoms), ligand_atoms


def calculate_pocket_features(pocket_atoms):

    coords = np.array([a.coord for a in pocket_atoms])

    center = coords.mean(axis=0)

    rg = np.sqrt(np.mean(np.sum((coords-center)**2,axis=1)))

    cov = np.cov(coords.T)

    eigvals = np.linalg.eigvals(cov)

    eigvals = sorted(np.real(eigvals))

    curvature = eigvals[0]/eigvals[-1] if eigvals[-1] != 0 else 0

    neg_oxygens = [a for a in pocket_atoms if a.element=="O" and ("OP" in a.get_name() or "O1P" in a.get_name())]

    polar_atoms = [a for a in pocket_atoms if a.element in ["O","N"]]

    return {
        "Pocket_Rg":rg,
        "Pocket_Curvature":curvature,
        "Neg_Oxygens":neg_oxygens,
        "Polar_Atoms":polar_atoms
    }


# ==================================================
# VISUALIZATION
# ==================================================

def visualize_pdb_with_ligand(pdb_path, selected_ligand_id=None, pocket_feats=None, show_phosphate=False, show_polar=False):

    with open(pdb_path,"r") as f:
        pdb_data=f.read()

    view = py3Dmol.view(width=800,height=500)

    view.addModel(pdb_data,"pdb")

    view.setBackgroundColor("white")

    view.setStyle({"cartoon":{"color":"spectrum","opacity":0.8}})

    view.addSurface(py3Dmol.VDW,{"opacity":0.3,"color":"white"})

    if selected_ligand_id:

        parts = selected_ligand_id.split()

        chain_id,res_num = parts[1].split(":")

        view.addStyle(
            {"chain":chain_id,"resi":int(res_num)},
            {"stick":{"colorscheme":"greenCarbon","radius":0.4}}
        )

    if pocket_feats:

        if show_phosphate:

            for atom in pocket_feats["Neg_Oxygens"]:

                c = atom.coord

                view.addSphere({
                    "center":{"x":float(c[0]),"y":float(c[1]),"z":float(c[2])},
                    "radius":0.6,
                    "color":"red"
                })

        if show_polar:

            for atom in pocket_feats["Polar_Atoms"]:

                c = atom.coord

                view.addSphere({
                    "center":{"x":float(c[0]),"y":float(c[1]),"z":float(c[2])},
                    "radius":0.5,
                    "color":"blue"
                })

    view.zoomTo()

    return view


# ==================================================
# HOME PAGE
# ==================================================

if st.session_state.page=="home":

    col1,col2 = st.columns([1,4])

    with col1:
        st.image(LOGO_URL,width=130)

    with col2:
        st.title("RNALigVS")
        st.markdown("### RNA–Ligand Virtual Screening Platform")

    st.markdown("---")

    st.markdown("""
RNALigVS automates RNA pocket analysis and ligand screening using physics-based interaction scoring.
""")

    st.button("Start Analysis",on_click=go_analysis)


# ==================================================
# TUTORIAL PAGE
# ==================================================

elif st.session_state.page=="tutorial":

    st.title("RNALigVS Tutorial")

    st.markdown("""
1. Upload RNA–ligand complex
2. Select reference ligand
3. Pocket detected automatically
4. Visualize phosphate and polar atoms
5. Run ligand screening
""")

    st.button("Back",on_click=go_home)


# ==================================================
# ANALYSIS PAGE
# ==================================================

elif st.session_state.page=="analysis":

    st.title("Analysis Workspace")

    with st.sidebar:

        st.header("1. Structure")

        up = st.file_uploader("Upload PDB",type="pdb")

        if up and st.button("Process"):

            with open("uploaded.pdb","wb") as f:
                f.write(up.getbuffer())

            st.session_state.current_pdb_path="uploaded.pdb"

        if st.session_state.current_pdb_path:

            parser=PDBParser(QUIET=True)

            struct=parser.get_structure("RNA",st.session_state.current_pdb_path)

            ligands=get_unique_ligands(struct)

            if ligands:

                st.subheader("2. Pocket")

                sel=st.selectbox("Target Ligand",ligands)

                st.session_state.selected_ligand_id=sel

                p_atoms,l_atoms=extract_binding_pocket(struct,sel)

                st.session_state.pocket_features=calculate_pocket_features(p_atoms)

    if st.session_state.current_pdb_path:

        show_phosphate = st.checkbox("Show Phosphate Atoms")

        show_polar = st.checkbox("Show Polar Atoms")

        view = visualize_pdb_with_ligand(
            st.session_state.current_pdb_path,
            st.session_state.selected_ligand_id,
            st.session_state.pocket_features,
            show_phosphate,
            show_polar
        )

        showmol(view,height=500,width=800)


    st.markdown("---")

    st.subheader("Virtual Screening")

    smiles_input = st.text_area("SMILES")

    if st.button("Run Screening"):

        lines=[l.strip() for l in smiles_input.splitlines() if l.strip()]

        results=[]

        for i,smi in enumerate(lines):

            mol=Chem.MolFromSmiles(smi)

            if mol:

                mw=Descriptors.MolWt(mol)

                logp=Descriptors.MolLogP(mol)

                qed_val=QED.qed(mol)

                results.append({
                    "Ligand":f"Mol_{i+1}",
                    "SMILES":smi,
                    "MW":mw,
                    "LogP":logp,
                    "QED":qed_val
                })

        df=pd.DataFrame(results)

        st.dataframe(df)

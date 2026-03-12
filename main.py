import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

import py3Dmol
from stmol import showmol


st.set_page_config(page_title="RNALigVS", layout="wide")


RNA_NAMES={"A","C","G","U","I","PSU","5MC","7MG"}
IGNORE_RESIDUES={"HOH","WAT"}


if "page" not in st.session_state:
    st.session_state.page="home"

if "pocket_features" not in st.session_state:
    st.session_state.pocket_features=None


# -----------------------------
# HOME PAGE
# -----------------------------

if st.session_state.page=="home":

    st.title("RNALigVS")
    st.subheader("RNA–Ligand Virtual Screening Platform")

    st.markdown("""
RNALigVS performs **structure-based virtual screening against RNA targets**.

The tool detects RNA binding pockets using structural proximity
and evaluates ligands based on pocket compatibility.
""")

    c1,c2,c3,c4=st.columns(4)

    with c1:
        st.markdown("### ⚡ Electrostatics")

    with c2:
        st.markdown("### 🧬 Pi stacking")

    with c3:
        st.markdown("### 📐 Shape fitting")

    with c4:
        st.markdown("### 💊 Drug likeness")

    st.button("Start Analysis",on_click=lambda:st.session_state.update(page="analysis"))


# -----------------------------
# ANALYSIS PAGE
# -----------------------------

if st.session_state.page=="analysis":

    col1,col2=st.columns([4,1])

    with col1:
        st.title("Analysis Workspace")

    with col2:
        st.button("Back",on_click=lambda:st.session_state.update(page="home"))


# -----------------------------
# FUNCTIONS
# -----------------------------

def is_rna(res):
    return res.get_resname().strip() in RNA_NAMES


def get_unique_ligands(structure):

    ligands=[]

    for model in structure:
        for chain in model:
            for res in chain:

                resname=res.get_resname().strip()

                if not is_rna(res) and resname not in IGNORE_RESIDUES and not is_aa(res):

                    ligands.append(f"{resname} {chain.id}:{res.id[1]}")

    return sorted(list(set(ligands)))


def extract_binding_pocket(structure,ligand_id,cutoff=6):

    parts=ligand_id.split()
    chain,resnum=parts[1].split(":")

    ligand_atoms=[]
    rna_atoms=[]

    for model in structure:
        for chain_obj in model:
            for res in chain_obj:

                if chain_obj.id==chain and res.id[1]==int(resnum):

                    ligand_atoms.extend(list(res.get_atoms()))

                elif is_rna(res):

                    rna_atoms.extend(list(res.get_atoms()))

    ns=NeighborSearch(rna_atoms)

    pocket_atoms=set()

    for atom in ligand_atoms:

        neighbors=ns.search(atom.coord,cutoff)

        pocket_atoms.update(neighbors)

    return list(pocket_atoms)


def calculate_pocket_features(atoms):

    coords=np.array([a.coord for a in atoms])

    center=coords.mean(axis=0)

    rg=np.sqrt(np.mean(np.sum((coords-center)**2,axis=1)))

    phosphate=[]
    polar=[]
    residues=set()

    for a in atoms:

        name=a.get_name().strip()
        element=a.element

        if element=="O" and ("OP" in name or "O1P" in name or "O2P" in name):
            phosphate.append(a)

        if element in ["O","N"]:
            polar.append(a)
            residues.add(a.get_parent().get_resname())

    return{

        "Pocket_Rg":rg,
        "Center":center.tolist(),
        "Phosphate":phosphate,
        "Polar":polar,
        "Residues":list(residues)
    }


def main_viewer(pdb_path):

    with open(pdb_path) as f:
        pdb=f.read()

    view=py3Dmol.view(width=700,height=500)

    view.addModel(pdb,"pdb")

    view.setStyle({"cartoon":{"color":"spectrum"}})

    view.addSurface(py3Dmol.VDW,{"opacity":0.3,"color":"white"})

    view.setStyle({"hetflag":True},{"stick":{"colorscheme":"greenCarbon"}})

    view.zoomTo()

    return view


def pocket_viewer(pdb_path,phos=None,polar=None):

    with open(pdb_path) as f:
        pdb=f.read()

    view=py3Dmol.view(width=350,height=350)

    view.addModel(pdb,"pdb")

    view.setStyle({"cartoon":{"color":"lightgrey"}})

    if phos:

        for a in phos:

            c=a.coord

            view.addSphere({"center":{"x":float(c[0]),"y":float(c[1]),"z":float(c[2])},
                            "radius":0.6,
                            "color":"red"})

    if polar:

        for a in polar:

            c=a.coord

            view.addSphere({"center":{"x":float(c[0]),"y":float(c[1]),"z":float(c[2])},
                            "radius":0.6,
                            "color":"blue"})

    view.zoomTo()

    return view


# -----------------------------
# SIDEBAR
# -----------------------------

pdb_file=st.sidebar.file_uploader("Upload PDB",type="pdb")

if pdb_file:

    with open("temp.pdb","wb") as f:
        f.write(pdb_file.getbuffer())

    parser=PDBParser(QUIET=True)

    struct=parser.get_structure("RNA","temp.pdb")

    ligands=get_unique_ligands(struct)

    selected=st.sidebar.selectbox("Target Ligand",ligands)

    atoms=extract_binding_pocket(struct,selected)

    st.session_state.pocket_features=calculate_pocket_features(atoms)


# -----------------------------
# MAIN LAYOUT
# -----------------------------

if pdb_file:

    left,center,right=st.columns([1,2,1])

    with center:

        st.subheader("3D Pocket View")

        showmol(main_viewer("temp.pdb"),height=500,width=700)

    with right:

        st.subheader("Pocket Interaction Viewer")

        show_phos=st.checkbox("Show Phosphate Sites")
        show_polar=st.checkbox("Show Polar Atoms")

        phos=None
        polar=None

        if show_phos:
            phos=st.session_state.pocket_features["Phosphate"]

        if show_polar:
            polar=st.session_state.pocket_features["Polar"]

        showmol(pocket_viewer("temp.pdb",phos,polar),height=350,width=350)

        st.markdown("**Interacting Residues:**")

        st.write(st.session_state.pocket_features["Residues"])

import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw, QED, Crippen
from rdkit.Chem import rdPartialCharges

from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

import py3Dmol
from stmol import showmol

# -----------------------------
# CONFIG
# -----------------------------

st.set_page_config(
    page_title="RNALigVS",
    layout="wide"
)

LOGO_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/RNALigVS/main/assets/rnaligvs_logo.png"

RNA_NAMES = {"A","C","G","U","I","PSU","5MC","7MG"}
IGNORE_RESIDUES = {"HOH","WAT"}

# -----------------------------
# HEADER
# -----------------------------

col1,col2 = st.columns([1,4])

with col1:
    st.image(LOGO_URL,width=130)

with col2:
    st.title("RNALigVS")
    st.markdown("### RNA–Ligand Virtual Screening Platform")

st.markdown("---")

# -----------------------------
# HELPER FUNCTIONS
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
    chain,res=parts[1].split(":")
    res=int(res)

    ligand_atoms=[]
    rna_atoms=[]

    for model in structure:
        for chain_obj in model:
            for res_obj in chain_obj:

                if chain_obj.id==chain and res_obj.id[1]==res:
                    ligand_atoms.extend(list(res_obj.get_atoms()))

                if is_rna(res_obj):
                    rna_atoms.extend(list(res_obj.get_atoms()))

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

    return {"Pocket_Rg":rg,"All_Atoms":atoms}


# -----------------------------
# VISUALIZATION
# -----------------------------

def visualize(pdb_path,center=None):

    with open(pdb_path) as f:
        pdb=f.read()

    view=py3Dmol.view(width=700,height=500)

    view.addModel(pdb,"pdb")

    view.setStyle({"cartoon":{"color":"spectrum"}})

    if center:

        x,y,z=center

        view.addSphere({
            "center":{"x":x,"y":y,"z":z},
            "radius":1.5,
            "color":"green"
        })

    view.zoomTo()

    return view


# -----------------------------
# SIDEBAR INPUT
# -----------------------------

st.sidebar.header("1️⃣ Upload RNA Structure")

pdb_file=st.sidebar.file_uploader("Upload PDB",type="pdb")

if pdb_file:

    with open("structure.pdb","wb") as f:
        f.write(pdb_file.getbuffer())

    parser=PDBParser(QUIET=True)

    struct=parser.get_structure("RNA","structure.pdb")

    ligands=get_unique_ligands(struct)

    st.sidebar.header("2️⃣ Pocket Definition")

    if ligands:

        ligand=st.sidebar.selectbox("Ligand",ligands)

        pocket_atoms=extract_binding_pocket(struct,ligand)

        pocket_features=calculate_pocket_features(pocket_atoms)

        st.sidebar.success("Pocket detected automatically")

        pocket_center=None

    else:

        st.sidebar.warning("No ligand found")

        x=st.sidebar.number_input("Pocket X",0.0)
        y=st.sidebar.number_input("Pocket Y",0.0)
        z=st.sidebar.number_input("Pocket Z",0.0)

        radius=st.sidebar.slider("Radius",3,10,6)

        pocket_center=(x,y,z)

        atoms=[a for a in struct.get_atoms()]

        ns=NeighborSearch(atoms)

        pocket_atoms=ns.search(np.array(pocket_center),radius)

        pocket_features=calculate_pocket_features(pocket_atoms)

# -----------------------------
# VISUALIZATION
# -----------------------------

if pdb_file:

    st.subheader("3D RNA Structure")

    view=visualize("structure.pdb",pocket_center)

    showmol(view,height=500,width=700)

    st.markdown("---")

# -----------------------------
# VIRTUAL SCREENING
# -----------------------------

if pdb_file:

    st.subheader("Virtual Screening")

    smiles_text=st.text_area("Enter SMILES library")

    library={}

    if smiles_text:

        for i,line in enumerate(smiles_text.splitlines()):

            library[f"Mol_{i}"]=line.strip()

    if st.button("Run Screening"):

        results=[]

        for name,smi in library.items():

            mol=Chem.MolFromSmiles(smi)

            if mol:

                mw=Descriptors.MolWt(mol)

                logp=Descriptors.MolLogP(mol)

                qed=QED.qed(mol)

                rg=rdMolDescriptors.CalcRadiusOfGyration(mol)

                shape=1-abs(rg-pocket_features["Pocket_Rg"])/(pocket_features["Pocket_Rg"]+0.1)

                prob=max(0,min(1,shape*qed))

                results.append({
                    "Ligand":name,
                    "Probability":prob,
                    "MW":mw,
                    "LogP":logp,
                    "QED":qed
                })

        df=pd.DataFrame(results).sort_values("Probability",ascending=False)

        st.dataframe(df)

        chart=alt.Chart(df).mark_bar().encode(
            x="Probability",
            y="Ligand"
        )

        st.altair_chart(chart,use_container_width=True)

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "results.csv"
        )
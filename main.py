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


# ------------------------------------------------
# CONFIG
# ------------------------------------------------
st.set_page_config(page_title="RNALigVS", layout="wide")

LOGO_URL = "https://raw.githubusercontent.com/Priyankakrati/tool/main/RNALigVS_logo.png"

RNA_NAMES = {"A","C","G","U","I","PSU","5MC","7MG"}
IGNORE_RESIDUES = {"HOH","WAT"}


# ------------------------------------------------
# PAGE STATE
# ------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page="home"

if "pocket_features" not in st.session_state:
    st.session_state.pocket_features=None


# ------------------------------------------------
# NAVIGATION
# ------------------------------------------------
c1,c2,c3 = st.columns(3)

with c1:
    if st.button("🏠 Home"):
        st.session_state.page="home"

with c2:
    if st.button("🧬 Analysis"):
        st.session_state.page="analysis"

with c3:
    if st.button("📘 Tutorial"):
        st.session_state.page="tutorial"


# ------------------------------------------------
# FUNCTIONS
# ------------------------------------------------
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


def extract_binding_pocket(structure, ligand_id, cutoff=6):

    parts = ligand_id.split()

    chain_id,res_num = parts[1].split(":")

    ligand_atoms=[]
    rna_atoms=[]

    for model in structure:
        for chain in model:
            for res in chain:

                if chain.id==chain_id and res.id[1]==int(res_num):

                    ligand_atoms.extend(list(res.get_atoms()))

                elif is_rna(res):

                    rna_atoms.extend(list(res.get_atoms()))

    ns=NeighborSearch(rna_atoms)

    pocket=set()

    for atom in ligand_atoms:

        neighbors=ns.search(atom.coord,cutoff)

        pocket.update(neighbors)

    return list(pocket)


def calculate_pocket_features(atoms):

    coords=np.array([a.coord for a in atoms])

    center=coords.mean(axis=0)

    rg=np.sqrt(np.mean(np.sum((coords-center)**2,axis=1)))

    neg_oxygens=[a for a in atoms if a.element=="O" and ("OP" in a.get_name() or "O1P" in a.get_name())]

    polar_atoms=[a for a in atoms if a.element in ["O","N"]]

    return {
        "Pocket_Rg":rg,
        "Neg_Oxygens":neg_oxygens,
        "Polar_Atoms":polar_atoms
    }


def visualize_structure(pdb_path):

    with open(pdb_path) as f:
        pdb_data=f.read()

    view=py3Dmol.view(width=700,height=500)

    view.addModel(pdb_data,"pdb")

    view.setStyle({"cartoon":{"color":"lightblue"}})

    view.addSurface(py3Dmol.VDW,{"opacity":0.3})

    view.zoomTo()

    return view


# ------------------------------------------------
# HOME PAGE
# ------------------------------------------------
if st.session_state.page=="home":

    col1,col2 = st.columns([1,4])

    with col1:
        st.image(LOGO_URL,width=130)

    with col2:
        st.title("RNALigVS")
        st.markdown("### RNA–Ligand Virtual Screening Platform")

    st.markdown("---")

    st.markdown("""
RNALigVS is an interactive platform for RNA binding pocket detection and ligand screening.

### Features

• RNA pocket detection  
• Pocket physics calculation  
• Phosphate interaction analysis  
• Polar atom identification  
• Virtual screening

Click **Analysis** to start.
""")


# ------------------------------------------------
# TUTORIAL PAGE
# ------------------------------------------------
elif st.session_state.page=="tutorial":

    st.title("RNALigVS Tutorial")

    st.markdown("""
### Step 1
Upload an RNA–ligand PDB structure.

### Step 2
Select the ligand used to define the pocket.

### Step 3
RNALigVS detects RNA atoms within **6 Å**.

### Step 4
Run virtual screening with SMILES molecules.

### Visualization

Red spheres → phosphate atoms  
Blue spheres → polar atoms
""")


# ------------------------------------------------
# ANALYSIS PAGE
# ------------------------------------------------
elif st.session_state.page=="analysis":

    st.title("RNALigVS Analysis")

    pdb_file = st.sidebar.file_uploader("Upload RNA Structure",type="pdb")

    if pdb_file:

        with open("temp_struct.pdb","wb") as f:
            f.write(pdb_file.getbuffer())

        parser=PDBParser(QUIET=True)

        struct=parser.get_structure("RNA","temp_struct.pdb")

        ligands=get_unique_ligands(struct)

        selected=st.sidebar.selectbox("Select ligand",ligands)

        pocket_atoms=extract_binding_pocket(struct,selected)

        st.session_state.pocket_features=calculate_pocket_features(pocket_atoms)


        # Structure viewer
        view=visualize_structure("temp_struct.pdb")

        showmol(view,height=500,width=700)


        # Pocket physics
        st.markdown("### Pocket Physics")

        col1,col2,col3 = st.columns(3)

        col1.metric("Pocket Radius (Rg)",f"{st.session_state.pocket_features['Pocket_Rg']:.2f} Å")

        col2.metric("Phosphate Sites",len(st.session_state.pocket_features["Neg_Oxygens"]))

        col3.metric("Polar Atoms",len(st.session_state.pocket_features["Polar_Atoms"]))


        # ------------------------------------------------
        # VIRTUAL SCREENING
        # ------------------------------------------------
        st.markdown("---")

        st.subheader("Virtual Screening")

        smiles_input=st.text_area(
            "Enter SMILES library (one per line)",
            "c1ccccc1\nCC(=O)Oc1ccccc1C(=O)O",
            height=150
        )

        if st.button("Run Screening"):

            results=[]

            lines=[l.strip() for l in smiles_input.splitlines() if l.strip()]

            progress=st.progress(0)

            for i,smi in enumerate(lines):

                mol=Chem.MolFromSmiles(smi)

                if mol:

                    mw=Descriptors.MolWt(mol)

                    logp=Descriptors.MolLogP(mol)

                    qed_val=QED.qed(mol)

                    results.append({
                        "Ligand":f"Mol_{i+1}",
                        "SMILES":smi,
                        "MW":round(mw,2),
                        "LogP":round(logp,2),
                        "QED":round(qed_val,2)
                    })

                progress.progress((i+1)/len(lines))

            if results:

                df=pd.DataFrame(results)

                st.dataframe(df)

                chart=alt.Chart(df).mark_bar().encode(
                    x="QED",
                    y="Ligand"
                )

                st.altair_chart(chart)

                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False),
                    "results.csv",
                    "text/csv"
                )

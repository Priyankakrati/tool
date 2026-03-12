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


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="RNALigVS", layout="wide")

RNA_NAMES = {"A","C","G","U","I","PSU","5MC","7MG"}
IGNORE_RESIDUES = {"HOH","WAT"}

if "page" not in st.session_state:
    st.session_state.page="home"

if "pocket_features" not in st.session_state:
    st.session_state.pocket_features=None


# -----------------------------
# INTRODUCTION PAGE
# -----------------------------
if st.session_state.page=="home":

    st.title("RNALigVS")

    st.markdown("""
### RNA–Ligand Virtual Screening Platform

RNALigVS provides an interactive environment for:

• RNA binding pocket detection  
• Pocket geometry analysis  
• Phosphate interaction detection  
• Polar atom mapping  
• Virtual screening of small molecules

Click **Start Analysis** to begin.
""")

    if st.button("Start Analysis"):
        st.session_state.page="analysis"

    st.stop()


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

    chain_id,res_seq=parts[1].split(":")

    ligand_atoms=[]
    rna_atoms=[]

    for model in structure:
        for chain in model:
            for res in chain:

                if chain.id==chain_id and res.id[1]==int(res_seq):
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

    phosphate=[]
    polar=[]

    for a in atoms:

        name=a.get_name().strip()

        if a.element=="O" and ("OP" in name or "O1P" in name or "O2P" in name):
            phosphate.append(a)

        if a.element in ["O","N"]:
            polar.append(a)

    return {
        "Pocket_Rg":rg,
        "Center":center.tolist(),
        "Phosphate":phosphate,
        "Polar":polar
    }


# -----------------------------
# MAIN RNA VIEWER
# -----------------------------
def visualize_main(pdb_path,show_phosphate=False,show_polar=False,features=None):

    with open(pdb_path) as f:
        pdb_data=f.read()

    view=py3Dmol.view(width=700,height=500)

    view.addModel(pdb_data,"pdb")

    view.setStyle({"cartoon":{"color":"lightblue"}})

    view.addSurface(py3Dmol.VDW,{"opacity":0.3,"color":"white"})

    if show_phosphate and features:

        for a in features["Phosphate"]:

            coord=a.coord

            view.addSphere({
                "center":{"x":float(coord[0]),"y":float(coord[1]),"z":float(coord[2])},
                "radius":0.6,
                "color":"red"
            })

    if show_polar and features:

        for a in features["Polar"]:

            coord=a.coord

            view.addSphere({
                "center":{"x":float(coord[0]),"y":float(coord[1]),"z":float(coord[2])},
                "radius":0.5,
                "color":"blue"
            })

    view.zoomTo()

    return view


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Upload RNA Structure")

pdb_file=st.sidebar.file_uploader("Upload PDB",type="pdb")


if pdb_file:

    with open("temp_struct.pdb","wb") as f:
        f.write(pdb_file.getbuffer())

    parser=PDBParser(QUIET=True)

    struct=parser.get_structure("RNA","temp_struct.pdb")

    ligands=get_unique_ligands(struct)

    if ligands:

        selected=st.sidebar.selectbox("Reference ligand",ligands)

        pocket_atoms=extract_binding_pocket(struct,selected)

        st.session_state.pocket_features=calculate_pocket_features(pocket_atoms)


# -----------------------------
# MAIN DISPLAY
# -----------------------------
if pdb_file:

    left,right=st.columns([3,1])

    with left:

        st.subheader("RNA–Ligand Structure")

        show_phosphate=st.checkbox("Show Phosphate Atoms")

        show_polar=st.checkbox("Show Polar Atoms")

        view=visualize_main(
            "temp_struct.pdb",
            show_phosphate,
            show_polar,
            st.session_state.pocket_features
        )

        showmol(view,height=500,width=700)


    with right:

        st.subheader("Pocket Physics")

        pf=st.session_state.pocket_features

        st.metric("Pocket Radius (Rg)",f"{pf['Pocket_Rg']:.2f} Å")

        st.metric("Phosphate Sites",len(pf["Phosphate"]))

        st.metric("Polar Atoms",len(pf["Polar"]))


# -----------------------------
# VIRTUAL SCREENING
# -----------------------------
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

            mol_rg=rdMolDescriptors.CalcRadiusOfGyration(mol)

            pocket_rg=st.session_state.pocket_features["Pocket_Rg"]

            shape_score=1-(abs(mol_rg-pocket_rg)/(pocket_rg+0.1))

            prob=max(0.01,min(0.99,shape_score*qed_val))

            results.append({
                "Ligand":f"Mol_{i+1}",
                "SMILES":smi,
                "Prob_Score":round(prob,3),
                "MW":round(mw,2),
                "LogP":round(logp,2),
                "QED":round(qed_val,2)
            })

        progress.progress((i+1)/len(lines))

    if results:

        df=pd.DataFrame(results).sort_values("Prob_Score",ascending=False)

        c1,c2=st.columns([2,1])

        with c1:
            st.dataframe(df,use_container_width=True)

        with c2:

            chart=alt.Chart(df).mark_bar().encode(
                x="Prob_Score",
                y=alt.Y("Ligand",sort="-x"),
                color="Prob_Score"
            )

            st.altair_chart(chart,use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "results.csv",
            "text/csv"
        )

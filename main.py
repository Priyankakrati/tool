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

LOGO_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/RNALigVS/main/assets/rnaligvs_logo.png"

RNA_NAMES = {"A","C","G","U","I","PSU","5MC","7MG"}
IGNORE_RESIDUES = {"HOH","WAT"}


# -----------------------------
# SESSION STATE
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "pocket_features" not in st.session_state:
    st.session_state.pocket_features = None


# -----------------------------
# NAVIGATION
# -----------------------------
col_nav1,col_nav2,col_nav3 = st.columns(3)

with col_nav1:
    if st.button("🏠 Home"):
        st.session_state.page="home"

with col_nav2:
    if st.button("🧬 Analysis"):
        st.session_state.page="analysis"

with col_nav3:
    if st.button("📘 Tutorial"):
        st.session_state.page="tutorial"



# =====================================================
# HOME PAGE
# =====================================================
if st.session_state.page == "home":

    col1,col2 = st.columns([1,4])

    with col1:
        st.image(LOGO_URL,width=150)

    with col2:
        st.title("RNALigVS")
        st.markdown("### RNA–Ligand Virtual Screening Platform")

    st.markdown("---")

    st.markdown("""
RNALigVS is an interactive platform for **RNA binding pocket analysis and ligand screening**.

### Features

• Automatic RNA pocket detection  
• Phosphate interaction detection  
• Polar atom identification  
• Interactive 3D RNA visualization  
• AI-assisted ligand screening  

### Workflow

1️⃣ Upload RNA–ligand PDB  
2️⃣ Define pocket automatically  
3️⃣ Visualize pocket physics  
4️⃣ Screen ligand library
""")



# =====================================================
# TUTORIAL PAGE
# =====================================================
elif st.session_state.page == "tutorial":

    st.title("RNALigVS Tutorial")

    st.markdown("""
### Step 1 — Upload Structure
Upload an RNA–ligand complex PDB file.

### Step 2 — Select Reference Ligand
Choose the ligand to define the binding pocket.

### Step 3 — Visualize Pocket
RNALigVS automatically detects RNA atoms within **6 Å**.

### Step 4 — Run Virtual Screening
Provide SMILES library to evaluate ligand binding probability.

### Interpretation

• **Red spheres** → phosphate atoms  
• **Blue spheres** → polar atoms  
• **Surface** → RNA binding pocket
""")



# =====================================================
# ANALYSIS PAGE
# =====================================================
elif st.session_state.page == "analysis":

    st.title("RNALigVS Analysis")

    st.sidebar.header("Upload RNA Structure")

    pdb_file = st.sidebar.file_uploader("Upload PDB",type="pdb")

    if pdb_file:

        with open("temp_struct.pdb","wb") as f:
            f.write(pdb_file.getbuffer())

        parser=PDBParser(QUIET=True)

        struct=parser.get_structure("RNA","temp_struct.pdb")


        # -----------------------------
        # FIND LIGANDS
        # -----------------------------
        ligands=[]

        for model in struct:
            for chain in model:
                for res in chain:

                    resname=res.get_resname().strip()

                    if resname not in RNA_NAMES and resname not in IGNORE_RESIDUES and not is_aa(res):

                        ligands.append(f"{resname} {chain.id}:{res.id[1]}")


        ligands=sorted(list(set(ligands)))

        if ligands:

            selected=st.sidebar.selectbox("Reference ligand",ligands)

            chain_id,res_seq=selected.split()[1].split(":")

            ligand_atoms=[]
            rna_atoms=[]

            for model in struct:
                for chain in model:
                    for res in chain:

                        if chain.id==chain_id and res.id[1]==int(res_seq):

                            ligand_atoms.extend(list(res.get_atoms()))

                        elif res.get_resname().strip() in RNA_NAMES:

                            rna_atoms.extend(list(res.get_atoms()))


            ns=NeighborSearch(rna_atoms)

            pocket=set()

            for atom in ligand_atoms:

                neighbors=ns.search(atom.coord,6)

                pocket.update(neighbors)

            pocket=list(pocket)


            coords=np.array([a.coord for a in pocket])

            center=coords.mean(axis=0)

            rg=np.sqrt(np.mean(np.sum((coords-center)**2,axis=1)))

            st.session_state.pocket_features={"Pocket_Rg":rg}


        # -----------------------------
        # MAIN VIEWER
        # -----------------------------
        st.subheader("RNA Structure")

        with open("temp_struct.pdb") as f:
            pdb_data=f.read()

        view=py3Dmol.view(width=700,height=500)

        view.addModel(pdb_data,"pdb")

        view.setStyle({"cartoon":{"color":"lightblue"}})

        view.addSurface(py3Dmol.VDW,{"opacity":0.3})

        view.zoomTo()

        showmol(view,height=500,width=700)


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

                st.dataframe(df,use_container_width=True)

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

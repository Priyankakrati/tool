import streamlit as st
import numpy as np
import pandas as pd
import tempfile
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, Draw
import py3Dmol
from stmol import showmol
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

st.set_page_config(page_title="RNALigVS", layout="wide")

# -------------------------------
# NAVIGATION
# -------------------------------
page = st.sidebar.radio("Navigation", ["🏠 Home", "🚀 Run Prediction", "📘 Tutorial"])

RNA_RES = {"A","C","G","U"}

# ===============================
# 🧪 PROPERTY FUNCTIONS
# ===============================
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    props = {
        "Molecular Weight": round(Descriptors.MolWt(mol),2),
        "LogP": round(Descriptors.MolLogP(mol),2),
        "H-bond Donors": Lipinski.NumHDonors(mol),
        "H-bond Acceptors": Lipinski.NumHAcceptors(mol),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol),2),
    }

    violations = 0
    if props["Molecular Weight"] > 500: violations += 1
    if props["LogP"] > 5: violations += 1
    if props["H-bond Donors"] > 5: violations += 1
    if props["H-bond Acceptors"] > 10: violations += 1

    props["Lipinski"] = "Pass" if violations <= 1 else "Fail"
    return props

# -------------------------------
# ADMET (simple proxy rules)
# -------------------------------
def compute_admet(props):

    admet = {}

    admet["Oral Bioavailability"] = "Good" if props["Lipinski"]=="Pass" else "Low"
    admet["Permeability"] = "High" if props["LogP"] < 5 else "Low"
    admet["Solubility"] = "Good" if props["LogP"] < 3 else "Moderate"
    admet["Toxicity Risk"] = "Low" if props["Molecular Weight"] < 500 else "Moderate"

    return admet

# -------------------------------
# DRAW 2D
# -------------------------------
def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(300,300))

# -------------------------------
# 3D LIGAND
# -------------------------------
def show_ligand_3d(smiles):

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    if AllChem.EmbedMolecule(mol) != 0:
        return None

    AllChem.UFFOptimizeMolecule(mol)
    mb = Chem.MolToMolBlock(mol)

    view = py3Dmol.view(width=400, height=400)
    view.addModel(mb, "mol")
    view.setStyle({"stick":{}})
    view.zoomTo()
    return view

# -------------------------------
# PDF REPORT
# -------------------------------
def generate_pdf(ligand, smiles, prob, props, admet):

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.drawString(50, 750, f"Ligand: {ligand}")
    c.drawString(50, 730, f"SMILES: {smiles}")
    c.drawString(50, 710, f"Binding Probability: {prob}")

    y = 680
    c.drawString(50, y, "Physicochemical Properties:")
    y -= 20

    for k,v in props.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 20

    y -= 10
    c.drawString(50, y, "ADMET Prediction:")
    y -= 20

    for k,v in admet.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 20

    c.save()
    buffer.seek(0)
    return buffer

# ===============================
# 🏠 HOME
# ===============================
if page == "🏠 Home":
    st.title("RNALigVS")
    st.write("RNA–Ligand Virtual Screening Tool")

# ===============================
# 🚀 RUN
# ===============================
elif page == "🚀 Run Prediction":

    st.title("Run Screening")

    structure_file = st.file_uploader("Upload PDB", type=["pdb"])
    smiles_file = st.file_uploader("Upload SMILES", type=["txt","csv"])

    def extract_coords(structure):
        coords = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.get_resname().strip() in RNA_RES:
                        for atom in res:
                            coords.append(atom.coord)
        return np.array(coords)

    def compute_features(mol, pocket):
        conf = mol.GetConformer()
        lig = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        d = np.linalg.norm(lig[:,None,:]-pocket[None,:,:], axis=2)
        return np.sum(d<5)/len(lig)

    if st.button("Run"):

        parser = PDBParser(QUIET=True)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(structure_file.read())
            pdb_path = tmp.name

        structure = parser.get_structure("RNA", pdb_path)
        pocket = extract_coords(structure)

        if smiles_file.name.endswith(".csv"):
            df = pd.read_csv(smiles_file)
            smiles_list = df.iloc[:,0].tolist()
        else:
            smiles_list = smiles_file.read().decode().splitlines()

        results = []

        for i,smi in enumerate(smiles_list):

            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)

            if AllChem.EmbedMolecule(mol) != 0:
                continue

            AllChem.UFFOptimizeMolecule(mol)

            score = compute_features(mol, pocket)
            prob = 1/(1+np.exp(-score))

            results.append({
                "Ligand": f"Lig_{i+1}",
                "SMILES": smi,
                "Binding Probability": round(prob,4)
            })

        df_rank = pd.DataFrame(results).sort_values("Binding Probability", ascending=False)

        st.dataframe(df_rank)

        # ---------------------------
        # LIGAND EXPLORER
        # ---------------------------
        st.subheader("Ligand Explorer")

        selected = st.selectbox("Select Ligand", df_rank["Ligand"])

        smi = df_rank[df_rank["Ligand"]==selected]["SMILES"].values[0]
        prob = df_rank[df_rank["Ligand"]==selected]["Binding Probability"].values[0]

        props = compute_properties(smi)
        admet = compute_admet(props)

        col1, col2 = st.columns(2)

        with col1:
            st.image(draw_molecule(smi))
            v = show_ligand_3d(smi)
            if v:
                showmol(v)

        with col2:
            st.write("### Properties")
            st.write(props)

            st.write("### ADMET")
            st.write(admet)

            pdf = generate_pdf(selected, smi, prob, props, admet)

            st.download_button(
                "Download PDF Report",
                pdf,
                file_name=f"{selected}_report.pdf"
            )

# ===============================
# 📘 TUTORIAL
# ===============================
elif page == "📘 Tutorial":
    st.write("Upload PDB + SMILES and run screening.")

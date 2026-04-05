import streamlit as st
import pandas as pd
import numpy as np
import json
import warnings
import requests
from io import BytesIO
from PIL import Image

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

import py3Dmol
from stmol import showmol
import plotly.express as px

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="RNALigVS", layout="wide")

# =========================
# LOGO
# =========================
logo_url = "https://raw.githubusercontent.com/YOUR_REPO/main/logo.png"

try:
    response = requests.get(logo_url)
    logo = Image.open(BytesIO(response.content))
except:
    logo = None

col1, col2 = st.columns([1,6])
with col1:
    if logo:
        st.image(logo, width=80)

with col2:
    st.markdown("""
    <h1 style='margin-bottom:0;'>RNALigVS</h1>
    <p style='margin-top:0;'>RNA–Ligand Virtual Screening Platform</p>
    """, unsafe_allow_html=True)

st.markdown("---")

# =========================
# LOAD MODEL
# =========================
with open("model_params.json") as f:
    params = json.load(f)

MEAN = params["mean"]
STD = params["std"]

RNA_RES = {"A","C","G","U"}
IGNORE = {"HOH","WAT"}

# =========================
# STRUCTURE FUNCTIONS
# =========================
def is_rna(res):
    return res.get_resname().strip() in RNA_RES

def get_ligands(structure):
    ligands = []
    for model in structure:
        for chain in model:
            for res in chain:
                if not is_rna(res) and res.get_resname() not in IGNORE and not is_aa(res):
                    ligands.append(res)
    return ligands

def extract_pocket(structure, ligand):
    ligand_atoms = list(ligand.get_atoms())
    rna_atoms = []

    for model in structure:
        for chain in model:
            for res in chain:
                if is_rna(res):
                    rna_atoms.extend(list(res.get_atoms()))

    pocket_atoms = []
    for ra in rna_atoms:
        for la in ligand_atoms:
            if np.linalg.norm(ra.coord - la.coord) < 8:
                pocket_atoms.append(ra)
                break

    return pocket_atoms if len(pocket_atoms) > 10 else rna_atoms[:200]

# =========================
# FEATURE EXTRACTION
# =========================
def compute_features(pocket_atoms, mol):

    mol = Chem.AddHs(mol)

    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    coords = np.array([
        np.array(conf.GetAtomPosition(i))
        for i in range(mol.GetNumAtoms())
    ])

    # ALIGN
    shift = np.mean([a.coord for a in pocket_atoms], axis=0) - coords.mean(axis=0)
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, pos + shift)

    coords = np.array([
        np.array(conf.GetAtomPosition(i))
        for i in range(mol.GetNumAtoms())
    ])

    elec = hbond = contact = hb_count = 0

    for lc in coords:
        for pa in pocket_atoms:
            d = np.linalg.norm(lc - pa.coord)

            if d > 8:
                continue

            elec += 1/(d**2 + 1)

            if pa.element in ["O","N"] and d < 3.5:
                hbond += 1/(d**2 + 0.5)
                hb_count += 1

            if d < 5:
                contact += 1

    contact_density = contact / len(coords)
    electrostatic = elec / max(contact,1)
    hbond_strength = hbond / hb_count if hb_count else 0
    pi = Chem.rdMolDescriptors.CalcNumAromaticRings(mol) / max(contact,1)

    pocket_coords = np.array([a.coord for a in pocket_atoms])
    center = pocket_coords.mean(axis=0)
    dists = np.linalg.norm(pocket_coords - center, axis=1)

    depth = np.mean(dists)
    curvature = np.var(dists)/(np.mean(dists)+1e-6)

    return {
        "Contact_density": contact_density,
        "Electrostatic_score": electrostatic,
        "Hbond_strength": hbond_strength,
        "Pi_stacking": pi,
        "Pocket_depth_mean": depth,
        "Curvature": curvature,
        "coords": coords,
        "mol": mol
    }

# =========================
# PREDICTION
# =========================
def predict(feat):

    score = (
        0.35*feat["Contact_density"] +
        0.30*feat["Electrostatic_score"] +
        0.10*feat["Hbond_strength"] +
        0.10*feat["Pi_stacking"] +
        0.10*feat["Pocket_depth_mean"] +
        0.05*feat["Curvature"]
    )

    z = (score - MEAN) / (STD + 1e-6)
    return 1/(1+np.exp(-z))

# =========================
# INTERACTIONS
# =========================
def add_interactions(view, coords, pocket_atoms):
    for lc in coords:
        for pa in pocket_atoms:
            d = np.linalg.norm(lc - pa.coord)

            if pa.element in ["O","N"] and d < 3.5:
                view.addLine({"start": {"x":lc[0],"y":lc[1],"z":lc[2]},
                              "end": {"x":pa.coord[0],"y":pa.coord[1],"z":pa.coord[2]},
                              "color":"blue","dashed":True})
            elif d < 5:
                view.addLine({"start": {"x":lc[0],"y":lc[1],"z":lc[2]},
                              "end": {"x":pa.coord[0],"y":pa.coord[1],"z":pa.coord[2]},
                              "color":"orange"})

# =========================
# VISUALIZATION
# =========================
def visualize(pdb_path, ligand, feat=None, pocket_atoms=None):

    with open(pdb_path) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_data,"pdb")

    view.setStyle({"cartoon":{"color":"spectrum"}})

    view.addStyle({"chain": ligand.get_parent().id,"resi":ligand.id[1]},
                  {"stick":{"colorscheme":"greenCarbon"}})

    if feat:
        for c in feat["coords"]:
            view.addSphere({"center":{"x":c[0],"y":c[1],"z":c[2]},
                            "radius":0.3,"color":"red"})
        add_interactions(view, feat["coords"], pocket_atoms)

    view.zoomTo()
    return view

# =========================
# FEATURE PLOT
# =========================
def plot_features(feat):
    weights = {
        "Contact_density":0.35,
        "Electrostatic_score":0.30,
        "Hbond_strength":0.10,
        "Pi_stacking":0.10,
        "Pocket_depth_mean":0.10,
        "Curvature":0.05
    }

    df = pd.DataFrame({
        "Feature": list(weights.keys()),
        "Contribution": [feat[k]*weights[k] for k in weights]
    })

    return px.bar(df, x="Feature", y="Contribution", title="Feature Contribution")

# =========================
# UI
# =========================
st.title("Analysis Workspace")

pdb_file = st.file_uploader("Upload PDB", type="pdb")

if pdb_file:

    with open("temp.pdb","wb") as f:
        f.write(pdb_file.getbuffer())

    structure = PDBParser(QUIET=True).get_structure("RNA","temp.pdb")
    ligand = get_ligands(structure)[0]
    pocket_atoms = extract_pocket(structure, ligand)

    col1,col2 = st.columns([2,1])

    with col1:
        showmol(visualize("temp.pdb", ligand), height=500)

    with col2:
        coords = np.array([a.coord for a in pocket_atoms])
        rg = np.sqrt(np.mean(np.sum((coords-coords.mean(0))**2,1)))

        st.metric("Pocket Radius (Rg)", f"{rg:.2f} Å")
        st.metric("Atoms", len(pocket_atoms))

    st.markdown("---")

    tab1,tab2,tab3 = st.tabs(["Text","CSV","SDF"])

    library={}
    sdf_mols=[]

    with tab1:
        txt = st.text_area("Enter SMILES")
        if txt:
            for i,s in enumerate(txt.splitlines()):
                library[f"Mol_{i}"]=s

    with tab2:
        f = st.file_uploader("Upload CSV")
        if f:
            df=pd.read_csv(f)
            df.columns=[c.lower() for c in df.columns]
            library=dict(zip(df.iloc[:,0],df.iloc[:,1]))

    with tab3:
        f = st.file_uploader("Upload SDF")
        if f:
            open("temp.sdf","wb").write(f.getbuffer())
            for i,m in enumerate(Chem.SDMolSupplier("temp.sdf")):
                if m:
                    sdf_mols.append((f"Mol_{i}",m))

    if st.button("Run Screening"):

        results=[]

        items = library.items() if library else sdf_mols

        for name, mol in items:

            if isinstance(mol,str):
                mol = Chem.MolFromSmiles(mol)

            feat = compute_features(pocket_atoms, mol)
            prob = predict(feat)

            feat["Ligand"]=name
            feat["Probability"]=prob

            results.append(feat)

        df=pd.DataFrame(results).sort_values("Probability",ascending=False)
        st.dataframe(df)

        sel=st.selectbox("Inspect",df["Ligand"])
        row=df[df["Ligand"]==sel].iloc[0]

        showmol(visualize("temp.pdb", ligand, row, pocket_atoms), height=400)

        st.plotly_chart(plot_features(row))

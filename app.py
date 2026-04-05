import streamlit as st
import pandas as pd
import numpy as np
import json
import warnings
from concurrent.futures import ThreadPoolExecutor

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

import py3Dmol
from stmol import showmol

warnings.filterwarnings("ignore")

# =========================
# LOAD MODEL PARAMS
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
            if np.linalg.norm(ra.coord - la.coord) < 6:  # reduced cutoff (faster)
                pocket_atoms.append(ra)
                break

    if len(pocket_atoms) < 10:
        pocket_atoms = rna_atoms[:200]

    return pocket_atoms

# =========================
# FAST FEATURE EXTRACTION
# =========================
def compute_features_fast(pocket_atoms, smi):

    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)

        if mol.GetNumConformers() == 0:
            status = AllChem.EmbedMolecule(mol, randomSeed=42)

            if status != 0:
                status = AllChem.EmbedMolecule(mol, useRandomCoords=True)

            if status != 0:
                return None

            # optional speed boost → comment if needed
            # AllChem.UFFOptimizeMolecule(mol)

        conf = mol.GetConformer()

        coords = np.array([
            np.array(conf.GetAtomPosition(i))
            for i in range(mol.GetNumAtoms())
        ])

        # ALIGN
        shift = np.mean([a.coord for a in pocket_atoms], axis=0) - coords.mean(axis=0)
        coords += shift

        pocket_coords = np.array([a.coord for a in pocket_atoms])

        # 🔥 VECTORIZED DISTANCE
        dists = np.linalg.norm(coords[:, None, :] - pocket_coords[None, :, :], axis=2)

        mask8 = dists < 8
        mask5 = dists < 5
        mask35 = dists < 3.5

        elec = np.sum(1/(dists[mask8]**2 + 1))
        contact = np.sum(mask5)

        hbond = np.sum(1/(dists[mask35]**2 + 0.5))
        hb_count = np.sum(mask35)

        ligand_size = max(len(coords), 1)
        contact_safe = max(contact, 1)

        feat = {
            "Contact_density": contact / ligand_size,
            "Electrostatic_score": elec / contact_safe,
            "Hbond_strength": hbond / hb_count if hb_count else 0,
            "Pi_stacking": Chem.rdMolDescriptors.CalcNumAromaticRings(mol) / contact_safe,
        }

        # geometry
        center = pocket_coords.mean(axis=0)
        dists_p = np.linalg.norm(pocket_coords - center, axis=1)

        feat["Pocket_depth_mean"] = np.mean(dists_p)
        feat["Curvature"] = np.var(dists_p) / (np.mean(dists_p) + 1e-6)

        return feat

    except:
        return None

# =========================
# PREDICTION
# =========================
def predict(feat):

    score = (
        0.35 * feat["Contact_density"] +
        0.30 * feat["Electrostatic_score"] +
        0.10 * feat["Hbond_strength"] +
        0.10 * feat["Pi_stacking"] +
        0.10 * feat["Pocket_depth_mean"] +
        0.05 * feat["Curvature"]
    )

    z = (score - MEAN) / (STD + 1e-6)
    return 1 / (1 + np.exp(-z))

# =========================
# VISUALIZATION
# =========================
def visualize(pdb_path, ligand):

    with open(pdb_path) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})

    view.addStyle(
        {"chain": ligand.get_parent().id, "resi": ligand.id[1]},
        {"stick": {"colorscheme": "greenCarbon"}}
    )

    view.zoomTo()
    return view

# =========================
# UI
# =========================
st.title("Analysis Workspace")

pdb_file = st.file_uploader("Upload PDB", type="pdb")

if pdb_file:

    with open("temp.pdb", "wb") as f:
        f.write(pdb_file.getbuffer())

    structure = PDBParser(QUIET=True).get_structure("RNA", "temp.pdb")
    ligand = get_ligands(structure)[0]
    pocket_atoms = extract_pocket(structure, ligand)

    col1, col2 = st.columns([2,1])

    with col1:
        showmol(visualize("temp.pdb", ligand), height=500)

    with col2:
        coords = np.array([a.coord for a in pocket_atoms])
        rg = np.sqrt(np.mean(np.sum((coords - coords.mean(axis=0))**2, axis=1)))

        st.metric("Pocket Radius (Rg)", f"{rg:.2f} Å")
        st.metric("Atoms", len(pocket_atoms))

    st.markdown("---")

    # INPUT
    tab1, tab2 = st.tabs(["Text", "CSV"])
    library = {}

    with tab1:
        txt = st.text_area("Enter SMILES")
        if txt:
            for i, s in enumerate(txt.splitlines()):
                library[f"Mol_{i}"] = s

    with tab2:
        file = st.file_uploader("Upload CSV")
        if file:
            df = pd.read_csv(file)
            df.columns = [c.lower() for c in df.columns]
            library = dict(zip(df.iloc[:,0], df.iloc[:,1]))
            st.success(f"{len(library)} molecules loaded")

    # RUN
    if st.button("Run Screening"):

        progress = st.progress(0)
        results = []

        items = list(library.items())

        # 🔥 PARALLEL EXECUTION
        def process(item):
            name, smi = item
            feat = compute_features_fast(pocket_atoms, smi)
            if feat is None:
                return None
            feat["Ligand"] = name
            feat["SMILES"] = smi
            feat["Probability_model"] = predict(feat)
            return feat

        with ThreadPoolExecutor(max_workers=8) as executor:
            for i, res in enumerate(executor.map(process, items)):
                progress.progress((i+1)/len(items))
                if res:
                    results.append(res)

        df = pd.DataFrame(results).sort_values("Probability_model", ascending=False)

        st.dataframe(df)

        sel = st.selectbox("Inspect", df["Ligand"])
        row = df[df["Ligand"]==sel].iloc[0]

        st.image(Draw.MolToImage(Chem.MolFromSmiles(row["SMILES"])))
        st.metric("Binding Probability", round(row["Probability_model"],3))

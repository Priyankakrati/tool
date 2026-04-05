import streamlit as st
import pandas as pd
import numpy as np
import json
import warnings

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
            if np.linalg.norm(ra.coord - la.coord) < 8:
                pocket_atoms.append(ra)
                break

    if len(pocket_atoms) < 10:
        pocket_atoms = rna_atoms[:200]

    return pocket_atoms

# =========================
# FEATURE EXTRACTION (TRAINING CONSISTENT)
# =========================
def compute_features_smiles(pocket_atoms, mol):

    mol = Chem.AddHs(mol)

    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    ligand_coords = np.array([
        np.array(conf.GetAtomPosition(i))
        for i in range(mol.GetNumAtoms())
    ])

    # ALIGN TO POCKET
    lig_center = ligand_coords.mean(axis=0)
    pocket_center = np.mean([a.coord for a in pocket_atoms], axis=0)

    shift = pocket_center - lig_center

    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, pos + shift)

    ligand_coords = np.array([
        np.array(conf.GetAtomPosition(i))
        for i in range(mol.GetNumAtoms())
    ])

    # FEATURES
    elec = hbond = vdw = contact = hb_count = 0

    for lc in ligand_coords:
        for pa in pocket_atoms:

            d = np.linalg.norm(lc - pa.coord)

            if d > 8:
                continue

            elec += 1/(d**2 + 1)

            if pa.element in ["N","O"] and d < 3.5:
                hbond += 1/(d**2 + 0.5)
                hb_count += 1

            if d < 6:
                vdw += 1/(d**6 + 1)

            if d < 5:
                contact += 1

    ligand_size = max(len(ligand_coords), 1)
    contact_safe = max(contact, 1)

    contact_density = contact / ligand_size
    electrostatic_score = elec / contact_safe
    hbond_strength = hbond / hb_count if hb_count > 0 else 0

    aromatic = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
    pi_stacking = aromatic / contact_safe

    pocket_coords = np.array([a.coord for a in pocket_atoms])
    center = pocket_coords.mean(axis=0)
    dists = np.linalg.norm(pocket_coords - center, axis=1)

    depth_mean = np.mean(dists)
    curvature = np.var(dists) / (np.mean(dists) + 1e-6)

    return {
        "Contact_density": contact_density,
        "Electrostatic_score": electrostatic_score,
        "Hbond_strength": hbond_strength,
        "Pi_stacking": pi_stacking,
        "Pocket_depth_mean": depth_mean,
        "Curvature": curvature
    }

# =========================
# FINAL PREDICTION
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
    prob = 1 / (1 + np.exp(-z))

    return prob

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

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("RNA", "temp.pdb")

    ligands = get_ligands(structure)

    if not ligands:
        st.error("No ligand found")
        st.stop()

    ligand = ligands[0]
    pocket_atoms = extract_pocket(structure, ligand)

    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("3D Pocket View")
        showmol(visualize("temp.pdb", ligand), height=500)

    with col2:
        st.subheader("Pocket Physics")

        coords = np.array([a.coord for a in pocket_atoms])
        rg = np.sqrt(np.mean(np.sum((coords - coords.mean(axis=0))**2, axis=1)))

        st.metric("Pocket Radius (Rg)", f"{rg:.2f} Å")
        st.metric("Atoms", len(pocket_atoms))

    st.markdown("---")

    st.subheader("Virtual Screening")

    # =========================
    # INPUT OPTIONS
    # =========================
    tab1, tab2, tab3 = st.tabs(["Text", "CSV", "SDF"])

    library = {}
    sdf_molecules = []

    with tab1:
        smiles_text = st.text_area("Enter SMILES")

        if smiles_text:
            for i, smi in enumerate(smiles_text.splitlines()):
                smi = smi.strip()
                if smi:
                    library[f"Mol_{i}"] = smi

    with tab2:
        csv_file = st.file_uploader("Upload CSV", type=["csv"])

        if csv_file:
            df_csv = pd.read_csv(csv_file)
            df_csv.columns = [c.lower() for c in df_csv.columns]

            if "name" in df_csv.columns and "smiles" in df_csv.columns:
                library = dict(zip(df_csv["name"], df_csv["smiles"]))

            elif "ligand_id" in df_csv.columns and "smiles" in df_csv.columns:
                library = dict(zip(df_csv["ligand_id"], df_csv["smiles"]))

            st.success(f"{len(library)} molecules loaded")

    with tab3:
        sdf_file = st.file_uploader("Upload SDF", type=["sdf"])

        if sdf_file:
            with open("temp.sdf", "wb") as f:
                f.write(sdf_file.getbuffer())

            supplier = Chem.SDMolSupplier("temp.sdf")

            for i, mol in enumerate(supplier):
                if mol is None:
                    continue
                name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"Mol_{i}"
                sdf_molecules.append((name, mol))

            st.success(f"{len(sdf_molecules)} molecules loaded")

    # =========================
    # RUN SCREENING
    # =========================
    if st.button("Run Screening"):

        results = []

        if library:
            for name, smi in library.items():

                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                feats = compute_features_smiles(pocket_atoms, mol)
                prob = predict(feats)

                row = feats.copy()
                row["Ligand"] = name
                row["SMILES"] = smi
                row["Probability_model"] = prob

                results.append(row)

        elif sdf_molecules:
            for name, mol in sdf_molecules:

                feats = compute_features_smiles(pocket_atoms, mol)
                prob = predict(feats)

                row = feats.copy()
                row["Ligand"] = name
                row["SMILES"] = Chem.MolToSmiles(mol)
                row["Probability_model"] = prob

                results.append(row)

        df = pd.DataFrame(results).sort_values("Probability_model", ascending=False)

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "results.csv"
        )

        sel = st.selectbox("Inspect Ligand", df["Ligand"])

        row = df[df["Ligand"] == sel].iloc[0]

        col1, col2 = st.columns([1,2])

        with col1:
            mol = Chem.MolFromSmiles(row["SMILES"])
            st.image(Draw.MolToImage(mol))

        with col2:
            st.metric("Binding Probability", round(row["Probability_model"],3))
            st.metric("Contact Density", round(row["Contact_density"],3))
            st.metric("Electrostatic", round(row["Electrostatic_score"],3))
            st.metric("Hbond", round(row["Hbond_strength"],3))

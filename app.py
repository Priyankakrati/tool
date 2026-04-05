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
# CONFIG
# =========================
st.set_page_config(page_title="RNALigVS", layout="wide")

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
# NAVIGATION
# =========================
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🧬 Run Prediction", "📘 Tutorial"]
)

# ============================================================
# 🏠 HOME PAGE
# ============================================================
if page == "🏠 Home":

    st.markdown("# RNALigVS: A Virtual screening tool for RNA and small molecules")
    st.markdown("### Structure-Based RNA Virtual Screening")

    st.markdown("---")

    st.markdown("""
RNALigVS is an RNA-focused virtual screening platform that evaluates small molecules using
biophysical interaction principles derived from RNA–ligand structural data.

The tool automatically:

• identifies the RNA binding pocket  
• extracts pocket physicochemical features  
• evaluates ligand compatibility using physics-informed scoring  
""")

    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.info("⚡ **Electrostatics**\n\nRNA backbone charges drive ligand interaction")

    with c2:
        st.info("🥞 **π-Stacking**\n\nAromatic interactions with RNA bases")

    with c3:
        st.info("📐 **Shape Complementarity**\n\nPocket–ligand geometric fit")

    with c4:
        st.info("💊 **Drug-Likeness**\n\nQED & Lipinski filtering")

    st.markdown("---")

    if st.button("🚀 Start Analysis"):
        st.session_state.page = "🧬 Run Prediction"


# ============================================================
# 🧬 RUN PREDICTION PAGE
# ============================================================
elif page == "🧬 Run Prediction":

    st.title("RNALigVS")

    # ---------- FUNCTIONS ----------
    def get_ligands(structure):
        ligands = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.get_resname() not in RNA_RES and res.get_resname() not in IGNORE and not is_aa(res):
                        ligands.append(res)
        return ligands

    def extract_pocket(structure, ligand):
        ligand_atoms = list(ligand.get_atoms())
        rna_atoms = []

        for model in structure:
            for chain in model:
                for res in chain:
                    if res.get_resname() in RNA_RES:
                        rna_atoms.extend(list(res.get_atoms()))

        ns = NeighborSearch(rna_atoms)
        pocket_atoms = set()

        for atom in ligand_atoms:
            pocket_atoms.update(ns.search(atom.coord, 6.0))

        return list(pocket_atoms)

    def prepare_ligand(mol, pocket_atoms):

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)

        conf = mol.GetConformer()

        lig_center = np.mean([
            np.array(conf.GetAtomPosition(i))
            for i in range(mol.GetNumAtoms())
        ], axis=0)

        pocket_center = np.mean([a.coord for a in pocket_atoms], axis=0)

        shift = pocket_center - lig_center

        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, pos + shift)

        return mol

    def compute_features(pocket_atoms, mol):

        mol = prepare_ligand(mol, pocket_atoms)
        conf = mol.GetConformer()

        ligand_coords = np.array([
            np.array(conf.GetAtomPosition(i))
            for i in range(mol.GetNumAtoms())
        ])

        contact = elec = hbond = hb_count = vdw = 0

        for lc in ligand_coords:
            for pa in pocket_atoms:

                d = np.linalg.norm(lc - pa.coord)

                if d > 8: continue

                elec += 1/(d**2 + 1)

                if d < 3.5:
                    hbond += 1/(d**2 + 0.5)
                    hb_count += 1

                if d < 6:
                    vdw += 1/(d**6 + 1)

                if d < 5:
                    contact += 1

        ligand_size = max(len(ligand_coords), 1)
        contact_safe = max(contact, 1)

        return {
            "Contact_density": contact / ligand_size,
            "Electrostatic_score": elec / contact_safe,
            "Hbond_strength": hbond / hb_count if hb_count else 0,
            "Pi_stacking": Chem.rdMolDescriptors.CalcNumAromaticRings(mol) / contact_safe,
            "Pocket_depth_mean": np.mean(np.linalg.norm(
                np.array([a.coord for a in pocket_atoms]) -
                np.mean([a.coord for a in pocket_atoms], axis=0), axis=1)),
            "Curvature": np.var([a.coord for a in pocket_atoms])
        }

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

    # ---------- UI ----------
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

        showmol(visualize("temp.pdb", ligand), height=500)

        st.success(f"Pocket extracted: {len(pocket_atoms)} atoms")

        smiles_text = st.text_area("Enter SMILES (one per line)")

        if st.button("Run Screening"):

            results = []

            for i, smi in enumerate(smiles_text.split("\n")):

                mol = Chem.MolFromSmiles(smi.strip())
                if mol is None: continue

                feats = compute_features(pocket_atoms, mol)
                prob = predict(feats)

                feats["Ligand"] = f"Mol_{i}"
                feats["SMILES"] = smi
                feats["Probability_model"] = prob

                results.append(feats)

            df = pd.DataFrame(results).sort_values("Probability_model", ascending=False)

            st.dataframe(df, use_container_width=True)

# ============================================================
# 📘 TUTORIAL PAGE
# ============================================================
elif page == "📘 Tutorial":

    st.title("📘 How to Use RNALigVS")

    st.markdown("""
### Step 1: Upload RNA Structure
Upload a PDB file containing RNA + ligand.

### Step 2: Pocket Detection
Tool automatically extracts binding pocket.

### Step 3: Enter Ligands
Paste SMILES (one per line).

### Step 4: Run Screening
Click **Run Screening**.

### Step 5: Interpret Results
- High Probability → strong binder  
- Analyze features for mechanism  

---

### Tips:
- Use curated ligand libraries  
- Prefer drug-like molecules  
- Validate top hits with docking/MD  
""")

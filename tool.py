# =========================================================
# RNALigVS FINAL STREAMLIT APP
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import tempfile
import py3Dmol

from rdkit import Chem
from rdkit.Chem import AllChem

from Bio.PDB import (
    PDBParser,
    NeighborSearch
)

import streamlit.components.v1 as components

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RNALigVS",
    layout="wide"
)

st.title("🧬 RNALigVS")
st.markdown(
    "### RNA–Ligand Virtual Screening Platform"
)

# =========================================================
# CONSTANTS
# =========================================================

RNA_RES = {"A", "C", "G", "U"}

IGNORE = {"HOH", "WAT"}

IONS = {"NA", "K", "MG", "CA", "ZN"}

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open("model_params.json") as f:

        model = json.load(f)

    return model

model = load_model()

weights = model["weights"]

mean = model["mean"]

std = model["std"]

# =========================================================
# SAFE ELEMENT
# =========================================================

def get_element(atom):

    try:

        el = atom.element.strip()

        if el:
            return el.upper()

    except:
        pass

    return atom.get_name()[0].upper()

# =========================================================
# CURVATURE
# =========================================================

def compute_curvature(coords):

    if len(coords) < 5:
        return 0

    cov = np.cov(coords.T)

    eig = np.linalg.eigvals(cov)

    eig = sorted(np.real(eig))

    return eig[0] / eig[-1] if eig[-1] != 0 else 0

# =========================================================
# PI STACKING
# =========================================================

def is_aromatic_atom(atom):

    return atom.get_name()[0] in ["C", "N"]

# =========================================================

def compute_pi_stacking(
    ligand_atoms,
    pocket_atoms
):

    score = 0

    for la in ligand_atoms:

        if not is_aromatic_atom(la):
            continue

        for pa in pocket_atoms:

            if not is_aromatic_atom(pa):
                continue

            d = np.linalg.norm(
                la.coord - pa.coord
            )

            if 3.0 < d < 4.5:

                score += 1 / (d**2)

    return score

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def compute_features(
    pdb_file,
    smiles
):

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "RNA",
        pdb_file
    )

    # =====================================
    # RNA ATOMS
    # =====================================

    rna_atoms = []

    for model in structure:

        for chain in model:

            for res in chain:

                name = res.get_resname().strip()

                if name in RNA_RES:

                    rna_atoms.extend(
                        list(res.get_atoms())
                    )

    # =====================================
    # LIGAND GENERATION
    # =====================================

    mol = Chem.MolFromSmiles(smiles)

    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(
        mol,
        randomSeed=42
    )

    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    ligand_coords = []

    for i in range(mol.GetNumAtoms()):

        pos = conf.GetAtomPosition(i)

        ligand_coords.append(
            np.array([
                pos.x,
                pos.y,
                pos.z
            ])
        )

    # =====================================
    # NeighborSearch
    # =====================================

    ns = NeighborSearch(rna_atoms)

    pocket_atoms = set()

    for coord in ligand_coords:

        nearby_atoms = ns.search(
            coord,
            6.0,
            level='A'
        )

        for atom in nearby_atoms:

            pocket_atoms.add(atom)

    pocket_atoms = list(pocket_atoms)

    if len(pocket_atoms) < 10:

        pocket_atoms = rna_atoms[:200]

    # =====================================
    # INTERACTION FEATURES
    # =====================================

    elec = 0
    hbond = 0
    contact = 0
    hb_count = 0

    ligand_atoms = []

    for i in range(mol.GetNumAtoms()):

        atom = mol.GetAtomWithIdx(i)

        pos = conf.GetAtomPosition(i)

        ligand_atoms.append({

            "coord": np.array([
                pos.x,
                pos.y,
                pos.z
            ]),

            "element": atom.GetSymbol()
        })

    for la in ligand_atoms:

        for pa in pocket_atoms:

            d = np.linalg.norm(
                la["coord"] - pa.coord
            )

            if d > 8:
                continue

            elec += 1 / (d**2 + 1)

            if (
                la["element"] in ["N","O"] and
                get_element(pa) in ["N","O"] and
                d < 3.5
            ):

                hbond += 1 / (d**2 + 0.5)

                hb_count += 1

            if d < 5:

                contact += 1

    # =====================================
    # NORMALIZATION
    # =====================================

    ligand_size = max(len(ligand_atoms), 1)

    contact_safe = max(contact, 1)

    contact_density = contact / ligand_size

    electrostatic_score = elec / contact_safe

    if hb_count > 0:

        hbond_strength = hbond / hb_count

    else:

        hbond_strength = 0

    # =====================================
    # PI STACKING
    # =====================================

    pi_stack = 0

    for la in ligand_atoms:

        if la["element"] not in ["C","N"]:
            continue

        for pa in pocket_atoms:

            if get_element(pa) not in ["C","N"]:
                continue

            d = np.linalg.norm(
                la["coord"] - pa.coord
            )

            if 3.0 < d < 4.5:

                pi_stack += 1 / (d**2)

    pi_stack /= contact_safe

    # =====================================
    # CURVATURE
    # =====================================

    pocket_coords = np.array(
        [a.coord for a in pocket_atoms]
    )

    curvature = compute_curvature(
        pocket_coords
    )

    # =====================================
    # FINAL FEATURES
    # =====================================

    features = {

        "Contact_density":
            contact_density,

        "Electrostatic_score":
            electrostatic_score,

        "Hbond_strength":
            hbond_strength,

        "Pi_stacking":
            pi_stack,

        "Curvature":
            curvature
    }

    return (
        features,
        pocket_coords
    )

# =========================================================
# PREDICTION
# =========================================================

def predict_probability(features):

    score = sum(
        weights[f] * features[f]
        for f in weights
    )

    z = (
        score - mean
    ) / std

    # Pure NumPy sigmoid
    prob = 1 / (
        1 + np.exp(-z)
    )

    return prob

# =========================================================
# VISUALIZATION
# =========================================================

def show_structure(
    pdb_file,
    pocket_coords
):

    with open(pdb_file) as f:

        pdb_data = f.read()

    view = py3Dmol.view(
        width=900,
        height=600
    )

    view.addModel(
        pdb_data,
        "pdb"
    )

    view.setStyle({
        "cartoon": {
            "color": "spectrum"
        }
    })

    # =====================================
    # POCKET VISUALIZATION
    # =====================================

    for c in pocket_coords:

        view.addSphere({

            "center": {

                "x": float(c[0]),
                "y": float(c[1]),
                "z": float(c[2])

            },

            "radius": 0.5,

            "color": "red",

            "opacity": 0.7
        })

    view.setBackgroundColor("white")

    view.zoomTo()

    return view

# =========================================================
# USER INPUT
# =========================================================

uploaded_pdb = st.file_uploader(
    "Upload RNA PDB File",
    type=["pdb"]
)

smiles = st.text_input(
    "Enter Ligand SMILES"
)

# =========================================================
# RUN PREDICTION
# =========================================================

if uploaded_pdb and smiles:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdb"
    ) as tmp:

        tmp.write(uploaded_pdb.read())

        pdb_path = tmp.name

    st.success("RNA structure loaded!")

    # =====================================
    # FEATURES
    # =====================================

    features, pocket_coords = compute_features(
        pdb_path,
        smiles
    )

    # =====================================
    # PROBABILITY
    # =====================================

    prob = predict_probability(
        features
    )

    # =====================================
    # DISPLAY RESULTS
    # =====================================

    st.subheader(
        "Binding Probability"
    )

    st.metric(
        "RNALigVS Score",
        f"{prob:.4f}"
    )

    # =====================================
    # FEATURES TABLE
    # =====================================

    st.subheader(
        "Extracted Features"
    )

    feat_df = pd.DataFrame(
        [features]
    )

    st.dataframe(feat_df)

    # =====================================
    # VISUALIZATION
    # =====================================

    st.subheader(
        "RNA Binding Pocket"
    )

    view = show_structure(
        pdb_path,
        pocket_coords
    )

    components.html(
        view._make_html(),
        height=650
    )

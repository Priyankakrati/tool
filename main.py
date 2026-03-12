import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

import py3Dmol
from stmol import showmol


# -----------------------------
# CONFIG & STATE
# -----------------------------
st.set_page_config(page_title="RNALigVS", layout="wide")

if 'pocket_features' not in st.session_state:
    st.session_state.pocket_features = None

LOGO_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/RNALigVS/main/assets/rnaligvs_logo.png"

RNA_NAMES = {"A", "C", "G", "U", "I", "PSU", "5MC", "7MG"}
IGNORE_RESIDUES = {"HOH", "WAT"}


# -----------------------------
# HEADER
# -----------------------------
col1, col2 = st.columns([1, 4])

with col1:
    try:
        st.image(LOGO_URL, width=130)
    except:
        st.title("🧬")

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
    ligands = []

    for model in structure:
        for chain in model:
            for res in chain:

                resname = res.get_resname().strip()

                if not is_rna(res) and resname not in IGNORE_RESIDUES and not is_aa(res):

                    ligands.append(f"{resname} {chain.id}:{res.id[1]}")

    return sorted(list(set(ligands)))


def extract_binding_pocket(structure, ligand_id, cutoff=6.0):

    parts = ligand_id.split()
    chain_id, res_seq = parts[1].split(":")

    ligand_atoms = []
    rna_atoms = []

    for model in structure:
        for chain in model:
            for res in chain:

                if chain.id == chain_id and res.id[1] == int(res_seq):
                    ligand_atoms.extend(list(res.get_atoms()))

                elif is_rna(res):
                    rna_atoms.extend(list(res.get_atoms()))

    if not ligand_atoms:
        return []

    ns = NeighborSearch(rna_atoms)

    pocket_atoms = set()

    for atom in ligand_atoms:

        neighbors = ns.search(atom.coord, cutoff)

        pocket_atoms.update(neighbors)

    return list(pocket_atoms)


def calculate_pocket_features(atoms):

    if not atoms:
        return {"Pocket_Rg": 1.0}

    coords = np.array([a.coord for a in atoms])

    center = coords.mean(axis=0)

    rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))

    phosphate_atoms = []
    polar_atoms = []

    for a in atoms:

        name = a.get_name().strip()
        element = a.element

        if element == "O" and ("OP" in name or "O1P" in name or "O2P" in name):
            phosphate_atoms.append(a)

        if element in ["O", "N"]:
            polar_atoms.append(a)

    return {
        "Pocket_Rg": rg,
        "Center": center.tolist(),
        "Phosphate": phosphate_atoms,
        "Polar": polar_atoms
    }


def visualize_mol(pdb_path, center=None, phosphate_atoms=None):

    with open(pdb_path, "r") as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=700, height=500)

    view.addModel(pdb_data, "pdb")

    view.setStyle({"cartoon": {"color": "spectrum"}})

    view.setStyle({'resn': list(RNA_NAMES)}, {"cartoon": {"color": "orange"}})

    if center:

        view.addSphere({
            "center": {"x": center[0], "y": center[1], "z": center[2]},
            "radius": 2.0,
            "color": "yellow",
            "opacity": 0.7
        })

    # visualize phosphate atoms
    if phosphate_atoms:

        for atom in phosphate_atoms:

            coord = atom.coord

            view.addSphere({
                "center": {"x": float(coord[0]), "y": float(coord[1]), "z": float(coord[2])},
                "radius": 0.6,
                "color": "red"
            })

    view.zoomTo()

    return view


# -----------------------------
# SIDEBAR & LOGIC
# -----------------------------

st.sidebar.header("1️⃣ Upload RNA Structure")

pdb_file = st.sidebar.file_uploader("Upload PDB", type="pdb")

if pdb_file:

    with open("temp_struct.pdb", "wb") as f:
        f.write(pdb_file.getbuffer())

    parser = PDBParser(QUIET=True)

    struct = parser.get_structure("RNA", "temp_struct.pdb")

    ligands = get_unique_ligands(struct)

    st.sidebar.header("2️⃣ Pocket Definition")

    if ligands:

        selected_ligand = st.sidebar.selectbox("Choose reference ligand", ligands)

        p_atoms = extract_binding_pocket(struct, selected_ligand)

        st.session_state.pocket_features = calculate_pocket_features(p_atoms)

        st.sidebar.success(f"Pocket atoms detected: {len(p_atoms)}")

    else:

        st.sidebar.warning("No ligand found. Define center manually.")

        px = st.sidebar.number_input("X", 0.0)
        py = st.sidebar.number_input("Y", 0.0)
        pz = st.sidebar.number_input("Z", 0.0)

        rad = st.sidebar.slider("Radius", 3, 12, 6)

        all_atoms = list(struct.get_atoms())

        ns = NeighborSearch(all_atoms)

        p_atoms = ns.search(np.array([px, py, pz]), rad)

        st.session_state.pocket_features = calculate_pocket_features(p_atoms)

        st.session_state.pocket_features["Center"] = [px, py, pz]


# -----------------------------
# MAIN DISPLAY
# -----------------------------

if pdb_file:

    st.subheader("3D RNA Structure Visualization")

    show_phosphate = st.checkbox("Show Phosphate Sites")

    ctr = st.session_state.pocket_features.get("Center") if st.session_state.pocket_features else None

    phos_atoms = None

    if show_phosphate and st.session_state.pocket_features:

        phos_atoms = st.session_state.pocket_features["Phosphate"]

    view = visualize_mol("temp_struct.pdb", center=ctr, phosphate_atoms=phos_atoms)

    showmol(view, height=500, width=700)


    st.markdown("---")

    st.subheader("Pocket Physics")

    col1, col2, col3 = st.columns(3)

    if st.session_state.pocket_features:

        with col1:
            st.metric(
                "Pocket Radius (Rg)",
                f"{st.session_state.pocket_features['Pocket_Rg']:.2f} Å",
                help="""
Radius of gyration measures the spatial spread of pocket atoms.

Rg = sqrt( Σ (ri - center)^2 / N )

Small Rg → tight pocket  
Large Rg → large pocket
"""
            )

        with col2:
            st.metric(
                "Phosphate Sites",
                len(st.session_state.pocket_features["Phosphate"]),
                help="""
Counts phosphate oxygen atoms (OP1 / OP2 / O1P / O2P)
within 6 Å of the ligand.

These atoms represent negatively charged RNA backbone
sites that contribute to electrostatic interactions.
"""
            )

        with col3:
            st.metric(
                "Polar Atoms",
                len(st.session_state.pocket_features["Polar"]),
                help="""
Polar atoms (O and N) in the pocket.

These atoms are potential hydrogen bond donors or acceptors
that contribute to ligand binding.
"""
            )


    st.markdown("---")

    st.subheader("Virtual Screening")

    smiles_input = st.text_area(
        "Enter SMILES library (one per line)",
        "c1ccccc1\nCC(=O)Oc1ccccc1C(=O)O",
        height=150
    )

    if st.button("🚀 Run Screening"):

        if not st.session_state.pocket_features:
            st.error("Please define a pocket first!")

        else:

            results = []

            lines = [l.strip() for l in smiles_input.splitlines() if l.strip()]

            progress_bar = st.progress(0)

            for i, smi in enumerate(lines):

                mol = Chem.MolFromSmiles(smi)

                if mol:

                    mw = Descriptors.MolWt(mol)

                    logp = Descriptors.MolLogP(mol)

                    qed_val = QED.qed(mol)

                    mol_rg = rdMolDescriptors.CalcRadiusOfGyration(mol)

                    pocket_rg = st.session_state.pocket_features["Pocket_Rg"]

                    shape_score = 1 - (abs(mol_rg - pocket_rg) / (pocket_rg + 0.1))

                    prob = max(0.01, min(0.99, shape_score * qed_val))

                    results.append({
                        "Ligand": f"Mol_{i+1}",
                        "SMILES": smi,
                        "Prob_Score": round(prob, 3),
                        "MW": round(mw, 2),
                        "LogP": round(logp, 2),
                        "QED": round(qed_val, 2)
                    })

                progress_bar.progress((i + 1) / len(lines))

            if results:

                df = pd.DataFrame(results).sort_values("Prob_Score", ascending=False)

                c1, c2 = st.columns([2, 1])

                with c1:
                    st.dataframe(df, use_container_width=True)

                with c2:

                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X("Prob_Score", title="Binding Probability"),
                        y=alt.Y("Ligand", sort='-x'),
                        color=alt.Color("Prob_Score", scale=alt.Scale(scheme='viridis'))
                    ).properties(height=300)

                    st.altair_chart(chart, use_container_width=True)

                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False),
                    "results.csv",
                    "text/csv"
                )

            else:

                st.error("No valid SMILES strings provided.")

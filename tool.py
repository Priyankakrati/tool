import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser
import json

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="RNALigVS",
    layout="wide"
)

# -------------------------------
# LOAD LOGO
# -------------------------------
st.image("RNALigVS_logo.png", width=200)
st.title("🧬 RNALigVS: RNA-Ligand Virtual Screening Tool")
st.markdown("Upload RNA–ligand complex structures (PDB/CIF) to predict binding affinity & rank ligands.")

# -------------------------------
# LOAD MODEL PARAMS
# -------------------------------
with open("model_params.json") as f:
    params = json.load(f)

MEAN = params["mean"]
STD = params["std"]
WEIGHTS = params["weights"]

# -------------------------------
# PARSERS
# -------------------------------
pdb_parser = PDBParser(QUIET=True)
cif_parser = MMCIFParser(QUIET=True)

RNA_RES = {"A","C","G","U"}
IGNORE = {"HOH","WAT"}
IONS = {"NA","K","MG","CA","ZN","FE","MN"}

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def get_element(atom):
    try:
        el = atom.element.strip()
        if el:
            return el.upper()
    except:
        pass
    return atom.get_name()[0].upper()

def compute_rg(coords):
    if len(coords) < 2:
        return 0
    center = coords.mean(axis=0)
    return np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))

def compute_curvature(coords):
    if len(coords) < 5:
        return 0
    cov = np.cov(coords.T)
    eig = np.linalg.eigvals(cov)
    eig = sorted(np.real(eig))
    return eig[0]/eig[-1] if eig[-1] != 0 else 0

def is_aromatic_atom(atom):
    return atom.get_name()[0] in ["C","N"]

def compute_pi_stacking(ligand_atoms, pocket_atoms):
    score = 0
    for la in ligand_atoms:
        if not is_aromatic_atom(la):
            continue
        for pa in pocket_atoms:
            if not is_aromatic_atom(pa):
                continue
            d = np.linalg.norm(la.coord - pa.coord)
            if 3.0 < d < 4.5:
                score += 1/(d**2)
    return score

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def compute_features(structure, pdb_id):

    rna_atoms = []
    ligand_atoms = []

    for model in structure:
        for chain in model:
            for res in chain:
                name = res.get_resname().strip()

                if name in RNA_RES:
                    rna_atoms.extend(list(res.get_atoms()))

                elif res.id[0] != " " and name not in IGNORE and name not in IONS:
                    atoms = list(res.get_atoms())
                    if len(atoms) > 5:
                        ligand_atoms.extend(atoms)

    if len(ligand_atoms) == 0:
        return None

    pocket_atoms = []
    for ra in rna_atoms:
        for la in ligand_atoms:
            if np.linalg.norm(ra.coord - la.coord) < 8:
                pocket_atoms.append(ra)
                break

    if len(pocket_atoms) < 10:
        pocket_atoms = rna_atoms[:200]

    elec = hbond = vdw = contact = hb_count = 0

    for la in ligand_atoms:
        el_l = get_element(la)

        for pa in pocket_atoms:
            el_p = get_element(pa)
            d = np.linalg.norm(la.coord - pa.coord)

            if d > 8:
                continue

            elec += 1/(d**2 + 1)

            if el_l in ["N","O"] and el_p in ["N","O"] and d < 3.5:
                hbond += 1/(d**2 + 0.5)
                hb_count += 1

            if d < 6:
                vdw += 1/(d**6 + 1)

            if d < 5:
                contact += 1

    ligand_size = max(len(ligand_atoms), 1)
    contact_safe = max(contact, 1)

    contact_density = contact / ligand_size
    electrostatic_score = elec / contact_safe
    vdw_score = vdw / contact_safe
    hbond_strength = hbond / hb_count if hb_count > 0 else 0

    pi_stack = compute_pi_stacking(ligand_atoms, pocket_atoms) / contact_safe

    pocket_coords = np.array([a.coord for a in pocket_atoms])
    lig_coords = np.array([a.coord for a in ligand_atoms])

    rg_lig = compute_rg(lig_coords)
    rg_pocket = compute_rg(pocket_coords)

    center = pocket_coords.mean(axis=0)
    dists = np.linalg.norm(pocket_coords - center, axis=1)

    depth_mean = np.mean(dists) if len(dists) > 0 else 0
    curvature = compute_curvature(pocket_coords)

    return {
        "PDB_ID": pdb_id,
        "Contact_density": contact_density,
        "Electrostatic_score": electrostatic_score,
        "Hbond_strength": hbond_strength,
        "Pi_stacking": pi_stack,
        "Pocket_depth_mean": depth_mean,
        "Curvature": curvature
    }

# -------------------------------
# PREDICTION
# -------------------------------
def predict(df):

    df["Score"] = (
        WEIGHTS["Contact_density"] * df["Contact_density"] +
        WEIGHTS["Electrostatic_score"] * df["Electrostatic_score"] +
        WEIGHTS["Hbond_strength"] * df["Hbond_strength"] +
        WEIGHTS["Pi_stacking"] * df["Pi_stacking"] +
        WEIGHTS["Pocket_depth_mean"] * df["Pocket_depth_mean"] +
        WEIGHTS["Curvature"] * df["Curvature"]
    )

    df["Z_score"] = (df["Score"] - MEAN) / STD if STD != 0 else 0
    df["Probability"] = 1 / (1 + np.exp(-df["Z_score"]))

    df = df.sort_values(by="Probability", ascending=False)
    df["Rank"] = range(1, len(df)+1)

    return df

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload PDB/CIF files",
    type=["pdb", "cif"],
    accept_multiple_files=True
)

if uploaded_files:

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:

        for file in uploaded_files:

            file_path = os.path.join(tmpdir, file.name)

            with open(file_path, "wb") as f:
                f.write(file.read())

            pdb_id = file.name.split(".")[0]

            try:
                if file.name.endswith(".pdb"):
                    structure = pdb_parser.get_structure(pdb_id, file_path)
                else:
                    structure = cif_parser.get_structure(pdb_id, file_path)

                feats = compute_features(structure, pdb_id)

                if feats:
                    results.append(feats)

            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

    if len(results) == 0:
        st.warning("No valid complexes found.")
    else:
        df = pd.DataFrame(results)
        df = predict(df)

        st.success("✅ Prediction Completed!")

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="RNALigVS_results.csv",
            mime="text/csv"
        )

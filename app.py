import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import json
import altair as alt

from rdkit import Chem
from rdkit.Chem import Draw

from Bio.PDB import PDBParser

import py3Dmol
from stmol import showmol

# Import your feature engine
from features_rnaligvs_final import compute_features

warnings.filterwarnings("ignore")

# =========================
# LOAD MODEL
# =========================
with open("model_params.json") as f:
    params = json.load(f)

W = params["weights"]
MEAN = params["mean"]
STD = params["std"]

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="RNALigVS-Pro",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("🧬 RNALigVS-Pro: RNA–Ligand Screening & Biological Interpretation")

st.markdown("""
### 🔬 What this tool does:

This platform performs **structure-based RNA–ligand screening** using:

✔ Physics-driven interaction modeling  
✔ Data-driven optimized scoring  
✔ Biologically interpretable features  

---

### 🧠 Biological Meaning of Features:

| Feature | Biological Role |
|--------|----------------|
| Contact Density | Binding compactness |
| Electrostatic Score | RNA backbone attraction |
| Hbond Strength | Specific recognition |
| π-Stacking | Base stacking stabilization |
| Pocket Depth | Ligand burial |
| Curvature | Pocket geometry |
""")

st.markdown("---")

# =========================
# INPUT SECTION
# =========================
st.sidebar.header("🔹 Input")

pdb_file = st.sidebar.file_uploader("Upload RNA Structure (PDB)", type=["pdb"])

# =========================
# VISUALIZATION
# =========================
def show_structure(pdb_path):
    with open(pdb_path) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view

# =========================
# SCORING FUNCTION
# =========================
def predict_score(feat):

    raw = (
        W["Contact_density"] * feat["Contact_density"] +
        W["Electrostatic_score"] * feat["Electrostatic_score"] +
        W["Hbond_strength"] * feat["Hbond_strength"] +
        W["Pi_stacking"] * feat["Pi_stacking"] +
        W["Pocket_depth_mean"] * feat["Pocket_depth_mean"] +
        W["Curvature"] * feat["Curvature"]
    )

    z = (raw - MEAN) / (STD + 1e-6)
    prob = 1 / (1 + np.exp(-z))

    return raw, prob

# =========================
# MAIN WORKFLOW
# =========================
if pdb_file:

    with open("temp.pdb", "wb") as f:
        f.write(pdb_file.getbuffer())

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("RNA", "temp.pdb")

    st.subheader("🧬 RNA Structure Visualization")
    view = show_structure("temp.pdb")
    showmol(view, height=500, width=800)

    st.markdown("---")

    st.subheader("🧪 Ligand Library")

    smiles_input = st.text_area("Enter SMILES (one per line)")

    if st.button("🚀 Run Screening"):

        smiles_list = [s.strip() for s in smiles_input.split("\n") if s.strip()]

        results = []

        for i, smi in enumerate(smiles_list):

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            feats = compute_features(structure, f"Mol_{i}")

            if feats is None:
                continue

            score, prob = predict_score(feats)

            row = feats.copy()
            row["Ligand"] = f"Mol_{i}"
            row["SMILES"] = smi
            row["Score"] = score
            row["Probability"] = prob

            results.append(row)

        if len(results) == 0:
            st.error("❌ No valid results")
        else:

            df = pd.DataFrame(results).sort_values("Probability", ascending=False)

            st.success("✅ Screening Completed")

            # =========================
            # HISTOGRAM
            # =========================
            st.subheader("📊 Probability Distribution")

            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Probability", bin=True),
                y="count()"
            )

            st.altair_chart(chart, use_container_width=True)

            # =========================
            # TOP HITS
            # =========================
            st.subheader("🏆 Top Ligand Candidates")

            top_n = st.slider("Select Top Hits", 5, 20, 10)

            st.dataframe(df.head(top_n), use_container_width=True)

            # =========================
            # BIOLOGICAL INTERPRETATION
            # =========================
            st.subheader("🧠 Biological Interpretation")

            sel = st.selectbox("Select Ligand", df["Ligand"])

            row = df[df["Ligand"] == sel].iloc[0]

            col1, col2 = st.columns([1,2])

            with col1:
                mol = Chem.MolFromSmiles(row["SMILES"])
                st.image(Draw.MolToImage(mol))

            with col2:
                st.markdown(f"### 🔬 Binding Probability: `{row['Probability']:.3f}`")

                st.markdown("""
#### Interpretation:
- High probability → strong RNA binding candidate  
- Driven by multi-feature synergy  
""")

                st.metric("Contact Density", round(row["Contact_density"],3))
                st.metric("Electrostatics", round(row["Electrostatic_score"],3))
                st.metric("Hbond Strength", round(row["Hbond_strength"],3))
                st.metric("π-Stacking", round(row["Pi_stacking"],3))
                st.metric("Pocket Depth", round(row["Pocket_depth_mean"],3))
                st.metric("Curvature", round(row["Curvature"],3))

            # =========================
            # DOWNLOAD
            # =========================
            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇ Download Results",
                csv,
                "rnaligvs_results.csv",
                "text/csv"
            )

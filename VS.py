import streamlit as st
import numpy as np
import pandas as pd
import tempfile
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
from stmol import showmol

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="RNALigVS", layout="wide")

# -------------------------------
# STYLING
# -------------------------------
st.markdown("""
<style>
body { background-color: #f7f9fb; }
h1, h2, h3 { color: #1f3c5b; }
.stButton>button {
    background-color: #2c7fb8;
    color: white;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #1a5a85;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# NAVIGATION
# -------------------------------
page = st.sidebar.radio("Navigation", ["🏠 Home", "🚀 Run Prediction", "📘 Tutorial"])

# -------------------------------
# CONSTANTS
# -------------------------------
RNA_RES = {"A","C","G","U"}

WEIGHTS = {
    "Contact_density": 0.35,
    "Electrostatic_score": 0.30,
    "Hbond_strength": 0.10,
    "Pi_stacking": 0.10,
    "Pocket_depth_mean": 0.10,
    "Curvature": 0.05
}

MEAN = 16.62
STD = 35.71

# ===============================
# 🏠 HOME
# ===============================
if page == "🏠 Home":

    st.image("RNALigVS_logo.png", width=160)
    st.title("RNALigVS")
    st.subheader("RNA–Ligand Virtual Screening Platform")

    st.markdown("""
RNALigVS is a **physics-based virtual screening tool** for RNA–ligand interactions.

### 🔬 Features
- Contact Density  
- Electrostatic Score  
- Hydrogen Bond Strength  
- π–Stacking  
- Pocket Geometry  

### ⚡ Advantages
- No docking required  
- Fast screening  
- Interpretable features  

👉 Go to **Run Prediction** to start.
""")

# ===============================
# 🚀 RUN PREDICTION
# ===============================
elif page == "🚀 Run Prediction":

    st.image("RNALigVS_logo.png", width=140)
    st.title("Run Virtual Screening")

    # -------------------------------
    # FUNCTIONS
    # -------------------------------
    def extract_pocket_coords(structure):
        coords = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.get_resname().strip() in RNA_RES:
                        for atom in res:
                            coords.append(atom.coord)
        return np.array(coords)

    def compute_pocket_features(coords):
        center = coords.mean(axis=0)
        dists = np.linalg.norm(coords - center, axis=1)

        depth = np.mean(dists)

        cov = np.cov(coords.T)
        eig = np.linalg.eigvals(cov)
        eig = sorted(np.real(eig))
        curvature = eig[0]/eig[-1] if eig[-1] != 0 else 0

        return depth, curvature

    def generate_ligand(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)

        try:
            if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
                return None
            AllChem.UFFOptimizeMolecule(mol)
        except:
            return None

        if mol.GetNumConformers() == 0:
            return None

        return mol

    def compute_features_fast(mol, pocket_coords):

        conf = mol.GetConformer()

        lig_coords = np.array([
            list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())
        ])

        dists = np.linalg.norm(
            lig_coords[:, None, :] - pocket_coords[None, :, :],
            axis=2
        )

        contact = np.sum(dists < 5)
        ligand_size = max(len(lig_coords), 1)
        contact_density = contact / ligand_size

        elec = np.sum(1/(dists**2 + 1))
        electrostatic_score = elec / max(contact, 1)

        hb_mask = dists < 3.5
        if np.any(hb_mask):
            hbond = np.sum(1/(dists[hb_mask]**2 + 0.5))
            hbond_strength = hbond / np.sum(hb_mask)
        else:
            hbond_strength = 0

        pi_mask = (dists > 3.0) & (dists < 4.5)
        if np.any(pi_mask):
            pi = np.sum(1/(dists[pi_mask]**2))
            pi_stack = pi / max(contact, 1)
        else:
            pi_stack = 0

        return contact_density, electrostatic_score, hbond_strength, pi_stack

    def calculate_probability(feats, depth, curvature):

        cd, elec, hb, pi = feats

        score = (
            0.35*cd +
            0.30*elec +
            0.10*hb +
            0.10*pi +
            0.10*depth +
            0.05*curvature
        )

        z = (score - MEAN)/STD
        return 1/(1+np.exp(-z))

    def show_structure(pdb_path, pocket_coords):

        with open(pdb_path) as f:
            pdb = f.read()

        view = py3Dmol.view(width=700, height=450)
        view.addModel(pdb, "pdb")
        view.setStyle({"cartoon": {"color": "spectrum"}})

        for c in pocket_coords[:300]:
            view.addSphere({
                "center": {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])},
                "radius": 0.5,
                "color": "red",
                "opacity": 0.6
            })

        view.zoomTo()
        return view

    # -------------------------------
    # INPUT
    # -------------------------------
    structure_file = st.file_uploader("Upload RNA structure (PDB)", type=["pdb"])
    smiles_file = st.file_uploader("Upload SMILES (txt/csv)", type=["txt","csv"])

    if structure_file:

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(structure_file.read())
            pdb_path = tmp.name

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("RNA", pdb_path)

        pocket_coords = extract_pocket_coords(structure)
        depth, curvature = compute_pocket_features(pocket_coords)

        st.subheader("RNA Structure + Pocket")
        showmol(show_structure(pdb_path, pocket_coords))

    # -------------------------------
    # RUN
    # -------------------------------
    if st.button("🚀 Run Virtual Screening"):

        if not structure_file or not smiles_file:
            st.warning("Upload both inputs")
            st.stop()

        if smiles_file.name.endswith(".csv"):
            df_sm = pd.read_csv(smiles_file)
            smiles_list = df_sm.iloc[:,0].tolist()
        else:
            smiles_list = smiles_file.read().decode().splitlines()

        results = []
        feature_rows = []

        for i, smi in enumerate(smiles_list):

            mol = generate_ligand(smi)
            if mol is None:
                continue

            feats = compute_features_fast(mol, pocket_coords)
            prob = calculate_probability(feats, depth, curvature)

            results.append({
                "Ligand": f"Lig_{i+1}",
                "SMILES": smi,
                "Binding Probability": round(prob,6)
            })

            feature_rows.append({
                "Ligand": f"Lig_{i+1}",
                "SMILES": smi,
                "Contact_density": feats[0],
                "Electrostatic_score": feats[1],
                "Hbond_strength": feats[2],
                "Pi_stacking": feats[3],
                "Pocket_depth_mean": depth,
                "Curvature": curvature,
                "Probability": prob
            })

        df_rank = pd.DataFrame(results).sort_values("Binding Probability", ascending=False)
        df_rank["Rank"] = range(1, len(df_rank)+1)

        df_feat = pd.DataFrame(feature_rows)

        st.success("✅ Screening Completed")

        # Top ligand
        st.success(f"🏆 Top Ligand: {df_rank.iloc[0]['Ligand']} "
                   f"(Probability = {df_rank.iloc[0]['Binding Probability']:.3f})")

        # Table
        st.subheader("Top Hits")
        st.dataframe(df_rank.head(20), use_container_width=True)

        # Charts
        st.subheader("Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Probability Distribution")
            st.bar_chart(df_rank["Binding Probability"])

        with col2:
            st.markdown("#### Top 10 Ligands")
            st.bar_chart(df_rank.head(10).set_index("Ligand")["Binding Probability"])

        # Feature explanation
        with st.expander("📘 Feature Description"):
            st.markdown("""
- Contact Density: ligand–pocket contacts  
- Electrostatic Score: distance-based interactions  
- H-bond Strength: interactions ≤ 3.5 Å  
- π–Stacking: aromatic interactions  
- Pocket Depth: spatial distribution  
- Curvature: pocket shape  
""")

        # Downloads
        st.download_button(
            "📥 Download Ranking CSV",
            df_rank.to_csv(index=False),
            "RNALigVS_ranking.csv"
        )

        st.download_button(
            "📥 Download Feature CSV",
            df_feat.to_csv(index=False),
            "RNALigVS_features.csv"
        )

# ===============================
# 📘 TUTORIAL
# ===============================
elif page == "📘 Tutorial":

    st.title("How to Use RNALigVS")

    st.markdown("""
### Step 1: Upload RNA Structure
Upload `.pdb` file.

### Step 2: Upload SMILES
TXT or CSV file.

### Step 3: Run Screening

### Output
- Ranked ligands  
- Feature dataset  

### Interpretation
Higher probability → stronger binding
""")

import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser

# -------------------------------
# PARSERS
# -------------------------------
pdb_parser = PDBParser(QUIET=True)
cif_parser = MMCIFParser(QUIET=True)

RNA_RES = {"A","C","G","U"}
IGNORE = {"HOH","WAT"}
IONS = {"NA","K","MG","CA","ZN","FE","MN"}

# -------------------------------
# SAFE ELEMENT
# -------------------------------
def get_element(atom):
    try:
        el = atom.element.strip()
        if el:
            return el.upper()
    except:
        pass
    return atom.get_name()[0].upper()

# -------------------------------
# GEOMETRY
# -------------------------------
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

# -------------------------------
# π-STACKING DETECTION (approx)
# -------------------------------
def is_aromatic_atom(atom):
    return atom.get_name()[0] in ["C","N"]

def compute_pi_stacking(ligand_atoms, pocket_atoms):
    stack_score = 0

    for la in ligand_atoms:
        if not is_aromatic_atom(la):
            continue

        for pa in pocket_atoms:
            if not is_aromatic_atom(pa):
                continue

            d = np.linalg.norm(la.coord - pa.coord)

            if 3.0 < d < 4.5:
                stack_score += 1/(d**2)

    return stack_score

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def compute_features(structure, pdb_id):

    rna_atoms = []
    ligand_atoms = []

    # ---- Extract atoms ----
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
        print(f"❌ No ligand {pdb_id}")
        return None

    # -------------------------------
    # POCKET (8 Å)
    # -------------------------------
    pocket_atoms = []

    for ra in rna_atoms:
        for la in ligand_atoms:
            if np.linalg.norm(ra.coord - la.coord) < 8:
                pocket_atoms.append(ra)
                break

    if len(pocket_atoms) < 10:
        pocket_atoms = rna_atoms[:200]

    # -------------------------------
    # INTERACTION FEATURES
    # -------------------------------
    elec = 0
    hbond = 0
    vdw = 0
    contact = 0
    hb_count = 0

    for la in ligand_atoms:
        el_l = get_element(la)

        for pa in pocket_atoms:
            el_p = get_element(pa)

            d = np.linalg.norm(la.coord - pa.coord)

            if d > 8:
                continue

            # Electrostatic
            elec += 1/(d**2 + 1)

            # Hbond
            if el_l in ["N","O"] and el_p in ["N","O"] and d < 3.5:
                hbond += 1/(d**2 + 0.5)
                hb_count += 1

            # vdW
            if d < 6:
                vdw += 1/(d**6 + 1)

            # Contact
            if d < 5:
                contact += 1

    # -------------------------------
    # NORMALIZATION (CRITICAL)
    # -------------------------------
    ligand_size = max(len(ligand_atoms), 1)
    contact_safe = max(contact, 1)

    contact_density = contact / ligand_size
    electrostatic_score = elec / contact_safe
    vdw_score = vdw / contact_safe

    if hb_count > 0:
        hbond_strength = hbond / hb_count
    else:
        hbond_strength = 0

    # -------------------------------
    # π-STACKING
    # -------------------------------
    pi_stack = compute_pi_stacking(ligand_atoms, pocket_atoms)
    pi_stack /= contact_safe

    # -------------------------------
    # GEOMETRY
    # -------------------------------
    pocket_coords = np.array([a.coord for a in pocket_atoms])
    lig_coords = np.array([a.coord for a in ligand_atoms])

    rg_lig = compute_rg(lig_coords)
    rg_pocket = compute_rg(pocket_coords)

    center = pocket_coords.mean(axis=0)
    dists = np.linalg.norm(pocket_coords - center, axis=1)

    depth_mean = np.mean(dists) if len(dists) > 0 else 0
    depth_max = np.max(dists) if len(dists) > 0 else 0
    curvature = compute_curvature(pocket_coords)

    return {
        "PDB_ID": pdb_id,
        "Contact_density": contact_density,
        "Electrostatic_score": electrostatic_score,
        "Hbond_strength": hbond_strength,
        "Pi_stacking": pi_stack,
        "Ligand_Rg": rg_lig,
        "Pocket_Rg": rg_pocket,
        "Pocket_depth_mean": depth_mean,
        "Pocket_depth_max": depth_max,
        "Curvature": curvature
    }

# -------------------------------
# MAIN
# -------------------------------
def main():

    folder = "PDB_files"
    results = []

    for file in os.listdir(folder):

        if not (file.endswith(".pdb") or file.endswith(".cif")):
            continue

        pdb_id = file.split(".")[0]
        path = os.path.join(folder, file)

        print(f"Processing {file}")

        try:
            if file.endswith(".pdb"):
                structure = pdb_parser.get_structure(pdb_id, path)
            else:
                structure = cif_parser.get_structure(pdb_id, path)
        except Exception as e:
            print(f"❌ Error {file}: {e}")
            continue

        feats = compute_features(structure, pdb_id)

        if feats:
            results.append(feats)

    if len(results) == 0:
        print("❌ No features extracted!")
        return

    df = pd.DataFrame(results)
    df.to_csv("RNALigVS_final_features.csv", index=False)

    print("\n🔥 FINAL FEATURES GENERATED!")

if __name__ == "__main__":
    main()

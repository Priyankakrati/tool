import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs


# =========================================================
# POCKET RESIDUE TABLE
# =========================================================

def pocket_residue_table(pocket_atoms):

    rows = []

    coords = np.array(
        [a.coord for a in pocket_atoms]
    )

    center = np.mean(coords, axis=0)

    for atom in pocket_atoms:

        residue = atom.get_parent()

        distance = np.linalg.norm(
            atom.coord - center
        )

        rows.append({

            "Residue":
                residue.get_resname(),

            "Residue ID":
                residue.id[1],

            "Atom":
                atom.get_name(),

            "Distance (Å)":
                round(distance, 2)
        })

    return pd.DataFrame(rows)


# =========================================================
# POCKET GEOMETRY
# =========================================================

def pocket_geometry(coords):

    if len(coords) < 4:
        return 0, 0

    hull = ConvexHull(coords)

    return (
        round(hull.volume, 2),
        round(hull.area, 2)
    )


# =========================================================
# CONFIDENCE
# =========================================================

def confidence_label(prob):

    if prob >= 0.8:
        return "High"

    elif prob >= 0.5:
        return "Medium"

    return "Low"


# =========================================================
# INTERACTION SUMMARY
# =========================================================

def interaction_summary(features):

    interactions = {

        "Hydrogen Bond Potential":

            round(
                features["Hbond Strength"] * 100,
                2
            ),

        "Electrostatic Compatibility":

            round(
                features["Electrostatic Score"] * 100,
                2
            ),

        "π-Stacking Potential":

            round(
                features["Pi-stacking energy"] * 100,
                2
            ),

        "Contact Density":

            round(
                features["Contact Density"] * 100,
                2
            )
    }

    return pd.DataFrame(

        interactions.items(),

        columns=[
            "Interaction Type",
            "Score"
        ]
    )


# =========================================================
# DRUG LIKENESS
# =========================================================

def drug_likeness(smiles):

    mol = Chem.MolFromSmiles(smiles)

    mw = Descriptors.MolWt(mol)

    logp = Descriptors.MolLogP(mol)

    hbd = rdMolDescriptors.CalcNumHBD(mol)

    hba = rdMolDescriptors.CalcNumHBA(mol)

    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

    lipinski_pass = (

        mw < 500 and
        logp < 5 and
        hbd <= 5 and
        hba <= 10
    )

    veber_pass = rot <= 10

    return {

        "Molecular Weight":
            round(mw, 2),

        "LogP":
            round(logp, 2),

        "HBD":
            hbd,

        "HBA":
            hba,

        "Rotatable Bonds":
            rot,

        "Lipinski":
            lipinski_pass,

        "Veber":
            veber_pass
    }


# =========================================================
# TANIMOTO SIMILARITY
# =========================================================

def tanimoto(sm1, sm2):

    m1 = Chem.MolFromSmiles(sm1)

    m2 = Chem.MolFromSmiles(sm2)

    fp1 = AllChem.GetMorganFingerprintAsBitVect(
        m1,
        2
    )

    fp2 = AllChem.GetMorganFingerprintAsBitVect(
        m2,
        2
    )

    return round(

        DataStructs.TanimotoSimilarity(
            fp1,
            fp2
        ),

        3
    )

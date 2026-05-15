# =========================================================
# RNALigVS SCIENTIFIC UTILITIES
# =========================================================

import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

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
# POCKET RESIDUE TABLE
# =========================================================

def pocket_residue_table(pocket_atoms):

    rows = []

    coords = np.array([
        a.coord for a in pocket_atoms
    ])

    center = np.mean(coords, axis=0)

    for atom in pocket_atoms:

        residue = atom.get_parent()

        distance = np.linalg.norm(
            atom.coord - center
        )

        rows.append({

            "Residue": residue.get_resname(),

            "Residue ID": residue.id[1],

            "Atom": atom.get_name(),

            "Distance (Å)": round(distance, 2)
        })

    return pd.DataFrame(rows)

# =========================================================
# POCKET GEOMETRY
# =========================================================

def pocket_geometry(coords):

    if len(coords) < 4:
        return 0, 0

    hull = ConvexHull(coords)

    volume = hull.volume

    area = hull.area

    return round(volume,2), round(area,2)

# =========================================================
# CONFIDENCE SCORE
    )

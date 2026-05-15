# =========================================================
# FILE 1: features_rnaligvs_final.py
# =========================================================

import os
import numpy as np
import pandas as pd

from Bio.PDB import (
    PDBParser,
    MMCIFParser,
    NeighborSearch
)

# =========================================================
# PARSERS
# =========================================================

pdb_parser = PDBParser(QUIET=True)
cif_parser = MMCIFParser(QUIET=True)

RNA_RES = {"A", "C", "G", "U"}

IGNORE = {"HOH", "WAT"}

IONS = {"NA", "K", "MG", "CA", "ZN"}

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
# AROMATIC DETECTION
# =========================================================

def is_aromatic_atom(atom):

    return atom.get_name()[0] in ["C", "N"]

# =========================================================
# PI STACKING
# =========================================================

def compute_pi_stacking(
    ligand_atoms,
    pocket_atoms
):

    stack_score = 0

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

                stack_score += 1 / (d**2)

    return stack_score

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def compute_features(structure, pdb_id):

    rna_atoms = []
    ligand_atoms = []

    # =====================================
    # EXTRACT RNA + LIGAND
    # =====================================

    for model in structure:

        for chain in model:

            for res in chain:

                name = res.get_resname().strip()

                # RNA
                if name in RNA_RES:

                    rna_atoms.extend(
                        list(res.get_atoms())
                    )

                # Ligand
                elif (
                    res.id[0] != " " and
                    name not in IGNORE and
                    name not in IONS
                ):

                    atoms = list(res.get_atoms())

                    if len(atoms) > 5:

                        ligand_atoms.extend(atoms)

    if len(ligand_atoms) == 0:

        print(f"❌ No ligand found: {pdb_id}")

        return None

    # =====================================
    # NeighborSearch KD-tree
    # =====================================

    ns = NeighborSearch(rna_atoms)

    pocket_atoms = set()

    for la in ligand_atoms:

        nearby_atoms = ns.search(
            la.coord,
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

    for la in ligand_atoms:

        el_l = get_element(la)

        for pa in pocket_atoms:

            el_p = get_element(pa)

            d = np.linalg.norm(
                la.coord - pa.coord
            )

            if d > 8:
                continue

            # Electrostatic
            elec += 1 / (d**2 + 1)

            # Hydrogen bond
            if (
                el_l in ["N", "O"] and
                el_p in ["N", "O"] and
                d < 3.5
            ):

                hbond += 1 / (d**2 + 0.5)

                hb_count += 1

            # Contact
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

    pi_stack = compute_pi_stacking(
        ligand_atoms,
        pocket_atoms
    )

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

    return {

        "PDB_ID": pdb_id,

        "Contact_density": contact_density,

        "Electrostatic_score": electrostatic_score,

        "Hbond_strength": hbond_strength,

        "Pi_stacking": pi_stack,

        "Curvature": curvature
    }

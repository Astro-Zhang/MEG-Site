"""
Extracts residue centroids and ligand heavy atom coordinates from PDB files.
"""

import sys
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm


AMINO_ACID_MAP = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def load_pdb_protein(pdb_path: str) -> np.ndarray:
    """Parse residue centroids from a protein PDB file."""
    residue_coords = defaultdict(list)
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    residue_name = line[17:20].strip()
                    residue_id = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    if residue_name in AMINO_ACID_MAP:
                        residue_coords[residue_id].append([x, y, z])
                except ValueError:
                    continue

    centers = [np.mean(np.array(coords), axis=0) for coords in residue_coords.values()]
    return np.array(centers)


def process_protein(protein_folder: str):
    """Process protein subfolders to generate and save coordinates."""
    process_count = 0
    for folder_name in tqdm(os.listdir(protein_folder)):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        protein_pdb = os.path.join(protein_subfolder, "protein.pdb")
        output_file = os.path.join(protein_subfolder, "coords.npy")
        coords = load_pdb_protein(protein_pdb)
        np.save(output_file, coords)
        process_count += 1
    return process_count


def main():
    protein_directory = "./protad"
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    processed_count = process_protein(protein_directory)
    print(f"Processed {processed_count} protein folders.")


if __name__ == "__main__":
    main()

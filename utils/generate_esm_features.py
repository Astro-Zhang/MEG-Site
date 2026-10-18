"""
Generates ESM-2 embeddings for protein sequences from PDB files.
"""

import sys
import os
import numpy as np
import torch
from collections import OrderedDict
import esm
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device).eval()


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


def extract_sequence_from_pdb(pdb_path: str) -> str:
    """Extract amino acid sequence from a PDB file."""
    with open(pdb_path, "r") as f:
        lines = f.readlines()

    residues = OrderedDict()
    for line in lines:
        if line.startswith("ATOM"):
            try:
                residue_name = line[17:20].strip()
                residue_id = int(line[22:26].strip())
                if residue_name in AMINO_ACID_MAP:
                    residues[residue_id] = residue_name
            except ValueError:
                continue

    return "".join(AMINO_ACID_MAP[residue] for residue in residues.values())


@torch.no_grad()
def get_esm2_embeddings(sequence: str, max_len: int = 1024) -> np.ndarray:
    """Generate ESM-2 embeddings for a protein sequence in chunks."""
    embeddings = []
    for i in range(0, len(sequence), max_len):
        chunk_sequence = sequence[i : i + max_len]
        data = [("protein", chunk_sequence)]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        assert batch_lens[0] - 1 == len(chunk_sequence) + 1

        batch_tokens = batch_tokens.to(device)
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        embeddings.append(
            results["representations"][33][0, 1 : len(chunk_sequence) + 1].cpu().numpy()
        )
        torch.cuda.empty_cache()

    return np.concatenate(embeddings, axis=0)


def process_protein(protein_folder: str):
    """Process protein subfolders to generate and save ESM-2 embeddings."""
    processed_count = 0
    for folder_name in tqdm(os.listdir(protein_folder)):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        esm2_file = os.path.join(protein_subfolder, "esm.npy")
        pdb_file = os.path.join(protein_subfolder, "protein.pdb")
        sequence = extract_sequence_from_pdb(pdb_file)
        esm2_features = get_esm2_embeddings(sequence)
        np.save(esm2_file, esm2_features)
        processed_count += 1
    return processed_count


def main():
    protein_directory = "./protad"
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    processed_count = process_protein(protein_directory)
    print(f"Processed {processed_count} proteins and generated ESM-2 embeddings.")


if __name__ == "__main__":
    main()

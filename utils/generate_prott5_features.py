"""
Generates ProtT5 embeddings for protein sequences from PDB files.
"""

import re
import sys
import os
import numpy as np
import torch
from collections import OrderedDict
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(
    "Rostlab/prot_t5_xl_uniref50", do_lower_case=False, legacy=True
)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device).eval()


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


@torch.no_grad()
def generate_embeddings(sequence: str) -> np.ndarray:
    """Generate ProtT5 embeddings for a protein sequence."""
    processed_sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    ids = tokenizer([processed_sequence], add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids["input_ids"]).to(device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(device)
    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    return embedding_repr.last_hidden_state[0, : len(sequence)].cpu().numpy()


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


def process_protein(protein_folder: str):
    """Process protein subfolders to generate and save ProtT5 embeddings."""
    processed_count = 0
    for folder_name in tqdm(os.listdir(protein_folder)):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        prott5_file = os.path.join(protein_subfolder, "prott5.npy")
        pdb_file = os.path.join(protein_subfolder, "protein.pdb")
        sequence = extract_sequence_from_pdb(pdb_file)
        prott5_features = generate_embeddings(sequence)
        np.save(prott5_file, prott5_features)
        processed_count += 1
    return processed_count


def main():
    protein_directory = "./protad"
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    processed_count = process_protein(protein_directory)
    print(f"Processed {processed_count} proteins and generated ProtT5 embeddings.")


if __name__ == "__main__":
    main()

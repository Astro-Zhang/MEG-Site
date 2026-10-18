"""
Combines ESM-2 and ProtT5 features into a single feature file for each protein.
"""

import sys
import os
import numpy as np
from tqdm import tqdm


def process_protein(protein_folder: str):
    """
    Process protein subfolders to combine ESM-2 and ProtT5 features.

    Args:
        protein_folder (str): Directory containing protein subfolders.
    """
    process_count = 0
    for folder_name in tqdm(os.listdir(protein_folder)):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        esm_file = os.path.join(protein_subfolder, "esm.npy")
        prott5_file = os.path.join(protein_subfolder, "prott5.npy")
        output_file = os.path.join(protein_subfolder, "features.npy")
        esm_features = np.load(esm_file)
        prott5_features = np.load(prott5_file)
        combined_features = np.concatenate((esm_features, prott5_features), axis=1)
        np.save(output_file, combined_features)
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

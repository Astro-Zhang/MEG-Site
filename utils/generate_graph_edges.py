"""Construct the multimodal protein graph described in the MEG-Site paper.

The edge set is the union of radius edges and 3D k-nearest-neighbor edges:
E = E_spatial ∪ E_knn. Edge features contain inverse distance and 16
radial-basis values (17 invariant dimensions in total).
"""

import argparse
import os
import sys
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from tqdm import tqdm


AMINO_ACID_MAP = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLU": 5,
    "GLN": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}


def compute_centers_and_types_from_pdb(pdb_path: str) -> tuple:
    """Compute residue centroids and types from a PDB file."""
    residue_coords = defaultdict(list)
    residue_types = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    residue_name = line[17:20].strip()
                    residue_id = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    if residue_name in AMINO_ACID_MAP:
                        residue_coords[residue_id].append([x, y, z])
                        residue_types[residue_id] = residue_name
                except (ValueError, IndexError):
                    continue

    sorted_ids = sorted(residue_coords.keys())
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
    centers = [np.mean(np.array(residue_coords[rid]), axis=0) for rid in sorted_ids]
    types = [AMINO_ACID_MAP[residue_types[rid]] for rid in sorted_ids]

    return np.array(centers), np.array(types), id_map, len(sorted_ids)


def compute_adjacency_matrix(
    centers: np.ndarray, radius: float = 2.0, num_neighbors: int = 10
) -> np.ndarray:
    """Return the union of spatial-cutoff and 3D KNN edges.

    Edges are directed in both directions, as expected by PyG message passing.
    The diagonal is always removed.
    """
    distances = cdist(centers, centers)
    spatial = distances <= radius
    np.fill_diagonal(spatial, False)

    knn = np.zeros_like(spatial, dtype=bool)
    if len(centers) > 1 and num_neighbors > 0:
        k = min(num_neighbors, len(centers) - 1)
        neighbor_ids = np.argsort(distances, axis=1)[:, 1 : k + 1]
        rows = np.repeat(np.arange(len(centers)), k)
        knn[rows, neighbor_ids.reshape(-1)] = True
        # Protein residue interactions are represented as bidirectional edges.
        knn |= knn.T

    adjacency = spatial | knn
    np.fill_diagonal(adjacency, False)
    return adjacency.astype(np.float32)


def adjacency_to_edges_and_features(
    adj_matrix: np.ndarray, centers: np.ndarray, types: np.ndarray
) -> tuple:
    """Convert adjacency matrix to edge indices and features."""
    row, col = np.nonzero(adj_matrix)
    edge_index = np.vstack((row, col))

    if edge_index.shape[1] == 0:
        return edge_index, np.empty((0, 17), dtype=np.float32)

    centers_rbf = np.linspace(0, 10, 16)
    sigma = 0.5
    edge_attr = []
    for i, j in zip(row, col):
        dist = np.linalg.norm(centers[j] - centers[i])
        inv_dist = 1.0 / (dist + 1e-8)
        rbf = np.exp(-((dist - centers_rbf) ** 2) / (2 * sigma**2))
        edge_attr.append(np.concatenate(([inv_dist], rbf)))

    return edge_index, np.asarray(edge_attr, dtype=np.float32)


def process_protein(protein_folder: str, radius: float = 2.0, num_neighbors: int = 10):
    """Generate MEG-Site graph arrays for every protein subdirectory."""
    processed_count = 0
    for folder_name in tqdm(os.listdir(protein_folder)):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        if not os.path.isdir(protein_subfolder):
            continue
        pdb_file = os.path.join(protein_subfolder, "protein.pdb")
        centers, types, _, residue_count = compute_centers_and_types_from_pdb(pdb_file)
        adj_matrix = compute_adjacency_matrix(
            centers, radius=radius, num_neighbors=num_neighbors
        )
        edge_index, edge_attr = adjacency_to_edges_and_features(
            adj_matrix, centers, types
        )
        radius_label = str(float(radius))
        np.save(
            os.path.join(protein_subfolder, f"edge_index_{radius_label}.npy"),
            edge_index,
        )
        np.save(
            os.path.join(protein_subfolder, f"edge_attr_{radius_label}.npy"),
            edge_attr,
        )
        processed_count += 1

    return processed_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein-root", default="datasets/protad")
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Spatial cutoff r in angstroms (paper default: 2.0).",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=10,
        help="Number k of nearest 3D neighbors per residue.",
    )
    args = parser.parse_args()

    protein_directory = args.protein_root
    if not os.path.isdir(protein_directory):
        print(f"Error: {protein_directory} is not a valid directory.")
        sys.exit(1)

    processed_count = process_protein(
        protein_directory,
        radius=args.radius,
        num_neighbors=args.num_neighbors,
    )
    print(f"Processed {processed_count} proteins and generated graph edges.")


if __name__ == "__main__":
    main()

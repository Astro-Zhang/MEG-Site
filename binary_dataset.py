"""ProTAD/MMSite binary data interface for the MEG-Site paper."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from utils.util_functions import get_col_name_list


class BinaryActiveSiteDataset(Dataset):
    """Load residue graphs and the 17 protein-level functional attributes."""
    def __init__(self, config, tsv_file, align=True):
        self.align = align
        self.config = config
        self.protad_root = config.dataset.protad_root

        # Load TSV
        self.data = pd.read_csv(tsv_file, sep="\t")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data["Entry"][idx]

        # Load Text Features (Pre-calculated)
        texts = self._load_text_features(entry)
        # Load Structure Features (ESM + ProtT5) and Geometry
        structure_data = self._load_structure_and_features(entry)

        # Labels
        L = structure_data["seq_features"].shape[0]
        # Active site handling
        labels = np.zeros(L)
        # Check if Active site is string or nan
        active_site_val = self.data["Active site"][idx]
        if isinstance(active_site_val, str):
            active_sites = active_site_val.split(",")
            for site in active_sites:
                s_idx = int(site) - 1  # 1-based index in TSV
                if 0 <= s_idx < L:
                    labels[s_idx] = 1

        # Separate PyG Data and other attributes to avoid collation issues
        graph_data = Data(
            x=structure_data["seq_features"],
            pos=structure_data["coords"],
            edge_index=structure_data["edge_index"],
            edge_attr=structure_data["edge_attr"],
            y=torch.from_numpy(labels).float(),
        )

        return {"graph": graph_data, "texts": texts, "entry": entry}

    def _load_structure_and_features(self, entry):
        prot_dir = os.path.join(self.protad_root, entry)
        try:
            features = np.load(
                os.path.join(prot_dir, self.config.dataset.features_file)
            )
            coords = np.load(
                os.path.join(prot_dir, self.config.dataset.coords_file)
            )  # (L, 3)
            edge_index = np.load(
                os.path.join(prot_dir, self.config.dataset.edge_index_file)
            )  # (2, E)
            edge_attr = np.load(
                os.path.join(prot_dir, self.config.dataset.edge_attr_file)
            )  # (E, 17), or legacy (E, 20)
            return {
                "seq_features": torch.from_numpy(features).float(),
                "coords": torch.from_numpy(coords).float(),
                "edge_index": torch.from_numpy(edge_index).long(),
                "edge_attr": torch.from_numpy(edge_attr).float(),
            }
        except Exception as e:
            raise RuntimeError(f"Error loading data for entry {entry}: {e}")

    def _load_text_features(self, entry):
        prot_dir = os.path.join(self.protad_root, entry)
        if self.align:
            try:
                path = os.path.join(
                    prot_dir, self.config.dataset.align_text_feature_file
                )
                texts = []
                with np.load(path) as data:
                    for col in get_col_name_list():
                        texts.append(torch.from_numpy(data[col]).float())
                return texts
            except Exception as e:
                raise RuntimeError(
                    f"Error loading text features for {entry} from {prot_dir}: {e}"
                )
        else:
            try:
                path = os.path.join(
                    prot_dir, self.config.dataset.text_feature_file
                )
                data = np.load(path)
                return torch.from_numpy(data).float()
            except Exception as e:
                raise RuntimeError(
                    f"Error loading text features for {entry} from {prot_dir}: {e}"
                )


def binary_collate_fn(batch):
    # batch is a list of dicts returned by __getitem__
    # Extract components
    graph_list = [item["graph"] for item in batch]
    texts_list = [item["texts"] for item in batch]
    entry_list = [item["entry"] for item in batch]

    # Batch graphs
    batch_data = Batch.from_data_list(graph_list)

    # Attach entries
    batch_data.entry = entry_list

    # Handle 'texts'
    # Check if the first item's texts is a list (multi-attribute) or Tensor (single text)
    if isinstance(texts_list[0], list):
        # Multi-attribute case: texts_list is [[Tensor, Tensor...], [Tensor, Tensor...]...]
        num_attrs = len(texts_list[0])
        assert num_attrs == 17, "Expected 17 text attributes"
        collated_texts = []
        # We want to collate per attribute across the batch
        for i in range(num_attrs):
            # i-th attribute across all samples
            # Stack (Batch, MaxLen, Hidden)
            attr_batch = torch.stack([s[i] for s in texts_list])
            collated_texts.append(attr_batch)

        # batch_data.texts will be a list of Tensors
        batch_data.texts = collated_texts

    else:
        # Single text case: texts_list is [Tensor, Tensor...]
        batch_data.texts = torch.stack(texts_list)

    return batch_data

"""M3Site graph and label loader used by the multiclass experiment."""

import json
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class MulticlassActiveSiteDataset(Dataset):
    """Dataset for M3Site preprocessed esm3_abs PyG Data files.

    Each item is a torch_geometric.data.Data object with residue-level labels:
        y: [num_nodes], class id in {0, 1, 2, 3, 4, 5, 6}
    """

    def __init__(self, data_root, split="0.1", split_name="train"):
        self.data_root = data_root
        self.split = str(split)
        self.split_name = split_name
        self.esm3_dir = os.path.join(data_root, "esm3_abs")
        self.label_path = os.path.join(data_root, "label.json")
        self.split_path = os.path.join(
            data_root, "splits", split_name, f"{split_name}_{self.split}.tsv"
        )

        self.data = pd.read_csv(self.split_path, sep="\t")
        with open(self.label_path) as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data.iloc[idx]["Entry"]
        path = os.path.join(self.esm3_dir, f"{entry}.pt")
        graph = torch.load(path, map_location="cpu")
        for key, value in graph:
            if torch.is_tensor(value):
                graph[key] = value.detach().clone()

        if entry not in self.labels:
            raise KeyError(f"Missing labels for {entry} in {self.label_path}")

        y = torch.tensor(self.labels[entry], dtype=torch.long)
        num_nodes = graph.x.size(0)
        if y.numel() != num_nodes:
            raise ValueError(
                f"Label length mismatch for {entry}: labels={y.numel()}, nodes={num_nodes}"
            )

        graph.y = y
        graph.entry = entry
        return graph


def multiclass_collate_fn(batch):
    return Batch.from_data_list(batch)

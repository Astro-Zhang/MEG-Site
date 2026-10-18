# MEG-Site Code Release

This repository contains the code for two related residue-level active-site
prediction settings:

1. **Binary prediction** identifies whether each residue is an active site.
2. **Multiclass prediction** follows the M3Site setting and assigns one of seven
   residue classes.

The two workflows share environment and utility code but use separate dataset,
model, training, and evaluation modules. Large precomputed features and model
checkpoints are intentionally excluded from this submission package.

Terminology follows the accompanying manuscript
`MEGSite_AAAI_0731.pdf`: GAEN is the Geometry-Aware Equivariant Network, DSAF
is Dynamic Site-Aware Fusion, and the residue graph is the multimodal protein
graph built from PLM features, coordinates, and spatial/KNN edges.

## Repository layout

```text
.
├── binary_dataset.py             # binary dataset and collation
├── binary_model.py               # proposed binary model
├── train_binary.py               # binary training entry point
├── evaluate_binary.py            # binary checkpoint evaluation
├── multiclass_dataset.py         # M3Site seven-class dataset
├── multiclass_model.py           # M3Site seven-class model
├── train_multiclass.py           # multiclass training entry point
├── evaluate_multiclass.py        # multiclass evaluation and export
├── conf/
│   ├── binary_train.yaml
│   ├── binary_eval.yaml
│   ├── multiclass_train.yaml
│   └── multiclass_eval.yaml
├── datasets/
│   ├── train.tsv, valid.tsv, test.tsv, mmsite.tsv
│   ├── protad/                   # binary precomputed features (not included)
│   └── M3Site_data/
│       ├── label.json
│       ├── splits/               # 0.1/0.3/0.5/0.7/0.9 splits
│       └── esm3_abs/             # multiclass graphs (not included)
├── utils/                        # shared metrics and feature preparation
├── pretrained_weights/           # optional checkpoints (not included)
└── environment.yaml
```

## Environment

Create the Conda environment:

```bash
conda env create -f environment.yaml
conda activate megsite
```

The reference environment uses Python 3.10, PyTorch 2.5.1, CUDA 12.1, Hydra
1.3.2, and PyTorch Geometric 2.6. If CUDA extension wheels are required, install
the PyTorch Geometric wheels matching the local PyTorch and CUDA versions.

## Binary active-site prediction

### Data

The included TSV files use tab-separated columns:

- `Entry`: protein identifier;
- `Sequence`: amino-acid sequence;
- `Active site`: comma-separated, one-based active-site residue indices.

For each entry, place the precomputed arrays in
`datasets/protad/<Entry>/`:

```text
features.npy                 # residue features, (L, 2304)
coords.npy                   # C-alpha coordinates, (L, 3)
edge_index_2.0.npy           # graph connectivity, (2, E)
edge_attr_2.0.npy            # invariant edge features, (E, 17)
text_features_align.npz      # aligned text attributes used for training
text_features_prior.npy      # protein-level text representation
```

Names and feature dimensions can be changed in
`conf/binary_train.yaml` and `conf/binary_eval.yaml`.

### Training

```bash
python train_binary.py
```

Hydra overrides may be supplied from the command line:

```bash
python train_binary.py seed=43 train.batch_size=16 train.epochs=50
```

The best checkpoint is selected by validation F1 and saved under
`outputs/binary/train/<date>/<time>/`.

### Evaluation

Place a checkpoint in `pretrained_weights/` or override its path:

```bash
python evaluate_binary.py \
  test.best_model_path=/path/to/best_model.pt
```

The binary workflow reports Accuracy, Precision, Recall, F1, Fmax, AUROC,
AUPRC, MCC, overlap score, and false-positive ratio. Fmax uses MMSite's
protein-centric threshold sweep, and AUPRC is its pair-centric micro average
precision over all protein-residue pairs.

## M3Site multiclass prediction

### Data

Split definitions and residue labels are included in
`datasets/M3Site_data/`. Place each preprocessed PyTorch Geometric graph at:

```text
datasets/M3Site_data/esm3_abs/<Entry>.pt
```

Each graph must provide:

- `x`: amino-acid identifiers;
- `esm_rep`: residue-level ESM representations;
- `prop`: residue property features;
- `pos`: 3D residue coordinates;
- `edge_index` and `edge_attr`: sparse graph structure;
- `func`: protein-level functional representation.

`label.json` provides the seven-class residue labels. Available split ratios are
`0.1`, `0.3`, `0.5`, `0.7`, and `0.9`.

### Training

```bash
python train_multiclass.py
```

The defaults are defined in `conf/multiclass_train.yaml`. Override individual
settings with Hydra syntax:

```bash
python train_multiclass.py \
  dataset.split=0.3 \
  model.fusion_mode=macross \
  train.batch_size=4 \
  train.epochs=100
```

The model section explicitly records the M3Site graph dimensions: 1536-D ESM
representations, 62-D residue properties, 768-D function embeddings, and
one-dimensional edge attributes. Legacy binary graphs with 20-D edge features
remain readable: their three Cartesian direction components are discarded at
runtime because treating vector components as scalar MLP inputs breaks rotation
equivariance.

### Evaluation

```bash
python evaluate_multiclass.py \
  dataset.split=0.1 \
  evaluate.checkpoint=outputs/multiclass/0.1/best_multiclass_0.1.pt
```

This entry point reports per-protein macro Accuracy, Precision, Recall, F1,
AUROC, AUPRC, and MCC, together with flattened reference metrics. Results can be
exported to JSON and CSV. Evaluation defaults are defined in
`conf/multiclass_eval.yaml`; its model settings must match the training
checkpoint.

## Feature preparation

Utilities are provided for coordinate extraction, graph construction, ESM and
ProtT5 feature generation, text feature generation, and feature fusion:

```bash
python utils/generate_coordinates.py
python utils/generate_graph_edges.py
python utils/generate_esm_features.py
python utils/generate_prott5_features.py
python utils/generate_text_feature.py
python utils/fuse_features.py
```

These scripts require raw structures and pretrained language models that are not
distributed in this code-only archive. Review their input paths before running.


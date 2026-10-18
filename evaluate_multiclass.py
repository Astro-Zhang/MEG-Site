import csv
import json
import os
import warnings
from types import SimpleNamespace

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from multiclass_dataset import MulticlassActiveSiteDataset, multiclass_collate_fn
from multiclass_model import MulticlassActiveSiteModel
from utils.util_functions import set_seed

warnings.filterwarnings("ignore")


def cal_auroc(y_true, y_proba, average="macro", multi_class="ovo"):
    unique_classes = np.unique(y_true)
    y_proba_trimmed = y_proba[:, unique_classes]
    y_proba_normalized = y_proba_trimmed / y_proba_trimmed.sum(axis=1, keepdims=True)
    if len(unique_classes) == 1:
        return 0.0
    if len(unique_classes) == 2:
        return roc_auc_score(y_true, y_proba_normalized[:, 1], average=average)
    return roc_auc_score(
        y_true, y_proba_normalized, multi_class=multi_class, average=average
    )


def cal_auprc(y_true, y_proba, average="macro"):
    unique_classes = np.unique(y_true)
    y_proba_trimmed = y_proba[:, unique_classes]
    y_true_onehot = np.zeros_like(y_proba_trimmed)
    for i, cls in enumerate(unique_classes):
        y_true_onehot[:, i] = (y_true == cls).astype(int)
    return average_precision_score(y_true_onehot, y_proba_trimmed, average=average)


def measure_graph_official(labels, preds, ptr, batch):
    accuracy, precision, recall, f1, auroc, auprc, mcc = 0, 0, 0, 0, 0, 0, 0
    graph_count = len(ptr) - 1
    for graph_idx in range(graph_count):
        mask = batch == graph_idx
        label = labels[mask].detach().cpu().numpy()
        pred = preds[mask].argmax(dim=-1).detach().cpu().numpy()
        prob = preds[mask].detach().cpu().numpy()
        accuracy += accuracy_score(label, pred)
        precision += precision_score(label, pred, average="macro")
        recall += recall_score(label, pred, average="macro")
        f1 += f1_score(label, pred, average="macro")
        auroc += cal_auroc(label, prob, average="macro", multi_class="ovo")
        auprc += cal_auprc(label, prob, average="macro")
        mcc += matthews_corrcoef(label, pred)

    return {
        "accuracy": accuracy / graph_count,
        "precision": precision / graph_count,
        "recall": recall / graph_count,
        "f1": f1 / graph_count,
        "auroc": auroc / graph_count,
        "auprc": auprc / graph_count,
        "mcc": mcc / graph_count,
    }


def compute_flat_reference_metrics(y_true, y_pred, y_prob, num_classes):
    all_labels = list(range(num_classes))
    return {
        "flat_accuracy": accuracy_score(y_true, y_pred),
        "flat_macro_precision": precision_score(
            y_true, y_pred, labels=all_labels, average="macro", zero_division=0
        ),
        "flat_macro_recall": recall_score(
            y_true, y_pred, labels=all_labels, average="macro", zero_division=0
        ),
        "flat_macro_f1": f1_score(
            y_true, y_pred, labels=all_labels, average="macro", zero_division=0
        ),
        "flat_mcc": matthews_corrcoef(y_true, y_pred),
    }


@torch.no_grad()
def evaluate(args):
    set_seed(args.seed)
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
    )

    dataset = MulticlassActiveSiteDataset(args.data_root, args.split, args.eval_split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
        collate_fn=multiclass_collate_fn,
    )

    model = MulticlassActiveSiteModel(
        num_classes=args.num_classes,
        hidden_features=args.hidden_features,
        num_layers=args.num_layers,
        dropout=args.dropout,
        aa_vocab_size=args.aa_vocab_size,
        aa_emb_dim=args.aa_embedding_dim,
        esm_dim=args.esm_dim,
        prop_dim=args.property_dim,
        func_dim=args.function_dim,
        edge_dim=args.edge_dim,
        use_func=not args.no_func,
        fusion_mode=args.fusion_mode,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    losses = []
    official_metrics = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "mcc": 0.0,
    }
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    for batch in tqdm(loader, desc=f"Evaluate {args.eval_split}_{args.split}"):
        batch = batch.to(device)
        logits = model(batch)["token_logits"]
        target = batch.y.long()
        loss = criterion(logits, target)
        prob = torch.softmax(logits, dim=1)

        losses.append(loss.item())
        batch_metrics = measure_graph_official(target, prob, batch.ptr, batch.batch)
        for key, value in batch_metrics.items():
            official_metrics[key] += value

        y_true_list.append(target.cpu().numpy())
        y_pred_list.append(prob.argmax(dim=1).cpu().numpy())
        y_prob_list.append(prob.cpu().numpy())

    for key in official_metrics:
        official_metrics[key] /= len(loader)

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_prob = np.concatenate(y_prob_list, axis=0)

    metrics = dict(official_metrics)
    metrics["loss"] = float(np.mean(losses))
    metrics.update(compute_flat_reference_metrics(y_true, y_pred, y_prob, args.num_classes))
    metrics["data_root"] = args.data_root
    metrics["split"] = args.split
    metrics["eval_split"] = args.eval_split
    metrics["checkpoint"] = args.checkpoint
    metrics["num_residues"] = int(y_true.shape[0])
    metrics["num_active_residues"] = int((y_true > 0).sum())

    print("\nM3Site official-style metrics (per-graph macro, then averaged):")
    print("Precision\tRecall\tF1\tAUROC\tAUPRC\tMCC")
    print(
        f"{metrics['precision']:.6f}\t{metrics['recall']:.6f}\t"
        f"{metrics['f1']:.6f}\t{metrics['auroc']:.6f}\t"
        f"{metrics['auprc']:.6f}\t{metrics['mcc']:.6f}"
    )

    print("\nAll metrics:")
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)

    if args.output_csv:
        csv_exists = os.path.exists(args.output_csv)
        fieldnames = [
            "eval_split",
            "split",
            "precision",
            "recall",
            "f1",
            "auroc",
            "auprc",
            "mcc",
            "loss",
            "accuracy",
            "checkpoint",
        ]
        with open(args.output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not csv_exists:
                writer.writeheader()
            writer.writerow({key: metrics[key] for key in fieldnames})


def build_args(cfg):
    return SimpleNamespace(
        seed=cfg.seed,
        data_root=cfg.dataset.root,
        split=str(cfg.dataset.split),
        eval_split=cfg.dataset.eval_split,
        gpu_id=cfg.runtime.gpu_id,
        num_classes=cfg.model.num_classes,
        hidden_features=cfg.model.hidden_features,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        aa_vocab_size=cfg.model.aa_vocab_size,
        aa_embedding_dim=cfg.model.aa_embedding_dim,
        esm_dim=cfg.model.esm_dim,
        property_dim=cfg.model.property_dim,
        function_dim=cfg.model.function_dim,
        edge_dim=cfg.model.edge_dim,
        no_func=not cfg.model.use_function,
        fusion_mode=cfg.model.fusion_mode,
        batch_size=cfg.evaluate.batch_size,
        num_workers=cfg.evaluate.num_workers,
        drop_last=cfg.evaluate.drop_last,
        checkpoint=cfg.evaluate.checkpoint,
        output_json=cfg.evaluate.output_json,
        output_csv=cfg.evaluate.output_csv,
    )


@hydra.main(config_path="conf", config_name="multiclass_eval", version_base="1.3")
def main(cfg: DictConfig):
    evaluate(build_args(cfg))


if __name__ == "__main__":
    main()

import random
import logging
import importlib
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
import torch.nn.functional as F


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_model(config):
    module = importlib.import_module("binary_model")
    logging.info(f"init model: {config.model.name}")
    model_class = getattr(module, config.model.name)
    model = model_class(config=config)
    logging.info(f"total params: {sum(p.numel() for p in model.parameters())}")
    logging.info(
        f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    return model


def get_max_length(col):
    if col == "Protein names":
        return 48
    if col == "Organism":
        return 24
    if col == "Taxonomic lineage":
        return 128
    if col == "Function":
        return 128
    if col == "Caution":
        return 72
    if col == "Miscellaneous":
        return 64
    if col == "Subunit structure":
        return 72
    if col == "Induction":
        return 64
    if col == "Tissue specificity":
        return 64
    if col == "Developmental stage":
        return 64
    if col == "Allergenic properties":
        return 48
    if col == "Biotechnological use":
        return 128
    if col == "Pharmaceutical use":
        return 64
    if col == "Involvement in disease":
        return 256
    if col == "Subcellular location":
        return 64
    if col == "Post-translational modification":
        return 96
    if col == "Sequence similarities":
        return 32


def get_str(data, col, idx):
    if col == "Sequence":
        return False
    if col == "Protein names":
        return f"The name of protein is {data[col][idx]}."
    if col == "Organism":
        organism = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"The organism is {organism}."
    if col == "Taxonomic lineage":
        taxonomic = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"The taxonomic lineage of this protein includes {taxonomic}."
    if col == "Function":
        function = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"The function of this protein includes {function}."
    if col == "Caution":
        caution = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"Caution includes {caution}."
    if col == "Miscellaneous":
        miscellaneous = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"Some miscellaneous things of this protein includes {miscellaneous}."
    if col == "Subunit structure":
        subunit = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"Subunit structure of this protein is {subunit}."
    if col == "Induction":
        induction = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"The induction of this protein includes {induction}."
    if col == "Tissue specificity":
        tissue = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"The tissue specificity of this protein is {tissue}."
    if col == "Developmental stage":
        developmental = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"The developmental stage of this protein is {developmental}."
    if col == "Allergenic properties":
        allergenic = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"The allergenic properties includes {allergenic}."
    if col == "Biotechnological use":
        biotechnologicaluse = (
            data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        )
        return f"The biotechnological use includes {biotechnologicaluse}."
    if col == "Pharmaceutical use":
        pharmaceutical_use = (
            data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        )
        return f"The pharmaceutical use includes {pharmaceutical_use}."
    if col == "Involvement in disease":
        diseases = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"The diseases it could lead involves include {diseases}."
    if col == "Subcellular location":
        subcellular = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"This protein is usually located in subcellular {subcellular}."
    if col == "Post-translational modification":
        ptm = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"The post-translational modification of this protein includes {ptm}."
    if col == "Sequence similarities":
        smilarities = data[col][idx] if isinstance(data[col][idx], str) else "unknown"
        return f"The sequence similarities of this protein includes {smilarities}."


def get_col_name_list():
    col_name_list = [
        "Protein names",
        "Organism",
        "Taxonomic lineage",
        "Function",
        "Caution",
        "Miscellaneous",
        "Subunit structure",
        "Induction",
        "Tissue specificity",
        "Developmental stage",
        "Allergenic properties",
        "Biotechnological use",
        "Pharmaceutical use",
        "Involvement in disease",
        "Subcellular location",
        "Post-translational modification",
        "Sequence similarities",
    ]
    return col_name_list


def find_continuous_ones(arr):
    continuous_ones = []
    start = None
    for i, l in enumerate(arr):  # noqa: E741
        if l == 1 and start is None:
            start = i
        elif l == 0 and start is not None:
            continuous_ones.append((start, i))
            start = None
    if start is not None:
        continuous_ones.append((start, len(arr)))
    return continuous_ones


def cal_os(label, pred):
    conti_ones_label = find_continuous_ones(label)
    conti_ones_pred = find_continuous_ones(pred)
    fenzi, fenmu = 0, 0
    for i in range(len(conti_ones_pred)):
        for j in range(len(conti_ones_label)):
            l = min(conti_ones_pred[i][1], conti_ones_label[j][1])  # noqa: E741
            r = max(conti_ones_pred[i][0], conti_ones_label[j][0])
            fenzi += max(0, l - r)
    if len(conti_ones_label) == 0:
        return 1
    for i in range(len(conti_ones_label)):
        fenmu += conti_ones_label[i][1] - conti_ones_label[i][0]
    return fenzi / fenmu


def intersect(a1, b1, a2, b2):
    if a2 >= b1 or a1 >= b2:
        return False
    return True


def cal_fpr(label, pred):
    conti_ones_label = find_continuous_ones(label)
    conti_ones_pred = find_continuous_ones(pred)
    fenzi, fenmu = 0, 0
    if len(conti_ones_pred) == 0:
        return 1
    for i in range(len(conti_ones_pred)):
        to_add = conti_ones_pred[i][1] - conti_ones_pred[i][0]
        flag = 0
        for j in range(len(conti_ones_label)):
            if intersect(
                conti_ones_pred[i][0],
                conti_ones_pred[i][1],
                conti_ones_label[j][0],
                conti_ones_label[j][1],
            ):
                flag = 1
                break
        if flag == 0:
            fenzi += to_add
        fenmu += to_add
    return fenzi / fenmu


def cos_sim(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def kl_loss(logits, softlabel, tau=0.5, way="mean", use_loss="kl"):
    sim_targets = F.softmax(softlabel / tau, dim=1)
    logit_inputs = F.log_softmax(logits / tau, dim=1)

    if use_loss == "kl":
        loss = F.kl_div(logit_inputs, sim_targets, reduction="batchmean")
    elif use_loss == "contrastive":
        if way == "mean":
            loss = -torch.mean(logit_inputs * sim_targets, dim=1).mean()
        elif way == "sum":
            loss = -torch.sum(logit_inputs * sim_targets, dim=1).mean()
    return loss


def calculate_metrics(all_labels, all_preds, all_probs):
    """
    Calculate metrics for binary classification.
    all_labels: list of numpy arrays, each (N_i,)
    all_preds: list of numpy arrays, each (N_i,)
    all_probs: list of numpy arrays, each (N_i,)
    """
    accuracy, precision, recall, f1, auc, prc, mcc, oss, fpr = 0, 0, 0, 0, 0, 0, 0, 0, 0
    num_samples = len(all_labels)

    valid_samples = 0

    for i in range(num_samples):
        label = all_labels[i]
        pred = all_preds[i]
        prob = all_probs[i]

        if len(np.unique(label)) < 2:
            raise ValueError(
                f"Sample {i} has only one class in labels, cannot compute AUC/PRC."
            )

        accuracy += accuracy_score(label, pred)
        precision += precision_score(label, pred, zero_division=0)
        recall += recall_score(label, pred, zero_division=0)
        f1 += f1_score(label, pred, zero_division=0)
        mcc += matthews_corrcoef(label, pred)
        oss += cal_os(label, pred)
        fpr += cal_fpr(label, pred)
        auc += roc_auc_score(label, prob)
        prc += average_precision_score(label, prob)
        valid_samples += 1

    if valid_samples == 0:
        return {
            k: 0.0
            for k in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "prc",
                "mcc",
                "oss",
                "fpr",
            ]
        }

    return {
        "accuracy": accuracy / valid_samples,
        "precision": precision / valid_samples,
        "recall": recall / valid_samples,
        "f1": f1 / valid_samples,
        "auc": auc / valid_samples,
        "prc": prc / valid_samples,
        "mcc": mcc / valid_samples,
        "oss": oss / valid_samples,
        "fpr": fpr / valid_samples,
    }


def calculate_metrics_safe(all_labels, all_preds):
    """
    Calculate metrics for binary classification.
    all_labels: list of numpy arrays, each (N_i,)
    all_preds: list of numpy arrays, each (N_i,)
    """
    accuracy, precision, recall, f1, mcc, oss, fpr = 0, 0, 0, 0, 0, 0, 0
    num_samples = len(all_labels)

    valid_samples = 0

    for i in range(num_samples):
        label = all_labels[i]
        pred = all_preds[i]

        accuracy += accuracy_score(label, pred)
        precision += precision_score(label, pred, zero_division=0)
        recall += recall_score(label, pred, zero_division=0)
        f1 += f1_score(label, pred, zero_division=0)
        mcc += matthews_corrcoef(label, pred)
        oss += cal_os(label, pred)
        fpr += cal_fpr(label, pred)
        valid_samples += 1

    if valid_samples == 0:
        return {
            k: 0.0
            for k in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "mcc",
                "oss",
                "fpr",
            ]
        }

    return {
        "accuracy": accuracy / valid_samples,
        "precision": precision / valid_samples,
        "recall": recall / valid_samples,
        "f1": f1 / valid_samples,
        "mcc": mcc / valid_samples,
        "oss": oss / valid_samples,
        "fpr": fpr / valid_samples,
    }

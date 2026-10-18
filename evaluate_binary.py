import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import logging
import numpy as np
from sklearn.metrics import average_precision_score
from binary_dataset import BinaryActiveSiteDataset, binary_collate_fn
from utils.util_functions import set_seed, calculate_metrics, get_model


def calculate_mmsite_fmax(all_labels, all_probs, num_thresholds=101):
    """Calculate protein-centric Fmax following the MMSite definition.

    At each threshold, precision is averaged over proteins with at least one
    predicted positive residue, while recall is averaged over all proteins.
    Fmax is the largest harmonic mean of those two averages.
    """
    if not all_labels:
        return 0.0

    labels = [np.asarray(label).reshape(-1).astype(bool) for label in all_labels]
    probs = [np.asarray(prob).reshape(-1) for prob in all_probs]
    if len(labels) != len(probs):
        raise ValueError("all_labels and all_probs must contain the same proteins")
    for protein_idx, (label, prob) in enumerate(zip(labels, probs)):
        if label.shape != prob.shape:
            raise ValueError(
                f"Protein {protein_idx} has mismatched label/probability lengths"
            )

    fmax = 0.0
    for threshold in np.linspace(0.0, 1.0, num_thresholds):
        precisions = []
        recalls = []
        for label, prob in zip(labels, probs):
            predicted = prob >= threshold
            true_positives = np.count_nonzero(predicted & label)
            predicted_positives = np.count_nonzero(predicted)
            actual_positives = np.count_nonzero(label)

            # MMSite averages precision over M(theta): proteins on which at
            # least one positive residue is predicted.
            if predicted_positives > 0:
                precisions.append(true_positives / predicted_positives)
            recalls.append(
                true_positives / actual_positives if actual_positives > 0 else 0.0
            )

        precision = float(np.mean(precisions)) if precisions else 0.0
        recall = float(np.mean(recalls))
        if precision + recall > 0:
            fmax = max(fmax, 2.0 * precision * recall / (precision + recall))
    return fmax


def calculate_mmsite_auprc(all_labels, all_probs):
    """Calculate MMSite pair-centric AUPRC (micro average precision).

    Every protein-residue pair contributes equally to a single precision-recall
    curve, matching the metric definition in the MMSite paper.
    """
    if not all_labels:
        return 0.0
    labels = np.concatenate(
        [np.asarray(label).reshape(-1) for label in all_labels]
    ).astype(int)
    probs = np.concatenate([np.asarray(prob).reshape(-1) for prob in all_probs])
    if labels.shape != probs.shape:
        raise ValueError("Flattened labels and probabilities have different lengths")
    return float(average_precision_score(labels, probs))


def add_mmsite_ranking_metrics(metrics, all_labels, all_probs):
    """Replace the legacy per-protein PRC with MMSite Fmax and AUPRC."""
    metrics.pop("prc", None)
    metrics["fmax"] = calculate_mmsite_fmax(all_labels, all_probs)
    metrics["auprc"] = calculate_mmsite_auprc(all_labels, all_probs)
    return metrics


@torch.no_grad()
def evaluate_for_mmsite(model, loader, device, desc="Evaluating"):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        if isinstance(batch.texts, list):
            for i in range(len(batch.texts)):
                batch.texts[i] = batch.texts[i].to(device)
        else:
            batch.texts = batch.texts.to(device)

        # Target is (N_total,) from PyG Data collation
        target = batch.y.float()
        # Token Classification Logic
        output = model(batch)["token_logits"].squeeze(-1)  # (N_total,)

        # Convert to dense (B, 512) and get mask
        # mask indicates valid nodes for each graph in batch
        output_dense, mask = to_dense_batch(output, batch.batch, max_num_nodes=512)
        target_dense, _ = to_dense_batch(target, batch.batch, max_num_nodes=512)

        # Collect Metrics
        output_prob = torch.sigmoid(output_dense)
        output_np = output_prob.detach().cpu().numpy()
        target_np = target_dense.detach().cpu().numpy()
        mask_np = mask.cpu().numpy()

        for i in range(output_np.shape[0]):
            valid_mask = mask_np[i]
            prob_i = output_np[i][valid_mask]
            label_i = target_np[i][valid_mask]
            pred_i = (prob_i > 0.5).astype(float)

            all_probs.append(prob_i)
            all_labels.append(label_i)
            all_preds.append(pred_i)

    del batch
    torch.cuda.empty_cache()

    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    return add_mmsite_ranking_metrics(metrics, all_labels, all_probs)


@torch.no_grad()
def evaluate(model, loader, device, desc="Evaluating"):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        if isinstance(batch.texts, list):
            for i in range(len(batch.texts)):
                batch.texts[i] = batch.texts[i].to(device)
        else:
            batch.texts = batch.texts.to(device)

        # Target is (N_total,) from PyG Data collation
        target = batch.y.float()

        # Forward
        output = model(batch)["token_logits"].squeeze(-1)  # (N_total,)
        loss = criterion(output, target)
        total_loss += loss.item()

        output_prob = torch.sigmoid(output)
        output_np = output_prob.cpu().numpy().flatten()
        target_np = target.cpu().numpy().flatten()
        batch_idx_np = batch.batch.cpu().numpy()
        for i in range(batch.num_graphs):
            mask = batch_idx_np == i
            prob_i = output_np[mask]
            label_i = target_np[mask]
            pred_i = (prob_i > 0.5).astype(float)

            all_probs.append(prob_i)
            all_labels.append(label_i)
            all_preds.append(pred_i)

    del batch
    torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    add_mmsite_ranking_metrics(metrics, all_labels, all_probs)
    metrics["loss"] = avg_loss
    return metrics


@hydra.main(config_path="conf", config_name="binary_eval", version_base="1.3")
def main(cfg: DictConfig):
    # Set seed
    set_seed(cfg.seed if "seed" in cfg else 42)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load Datasets
    logging.info("Loading datasets...")
    mmsite_tsv = cfg.dataset.mmsite_tsv
    mmsite_dataset = BinaryActiveSiteDataset(cfg, mmsite_tsv, align=False)

    mmsite_loader = DataLoader(
        mmsite_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        collate_fn=binary_collate_fn,
        num_workers=cfg.test.num_workers,
    )

    # 2. Testing Phase
    logging.info("=== Starting Testing Phase ===")
    model = get_model(cfg).to(device)

    best_model_path = cfg.test.best_model_path
    logging.info(f"Loading best model from: {best_model_path}")

    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )
    mmsite_metrics = evaluate_for_mmsite(model, mmsite_loader, device, desc="MMSite")
    logging.info("=== MMSite Results ===")
    for k, v in mmsite_metrics.items():
        logging.info(f"MMSite {k}: {v:.4f}")


if __name__ == "__main__":
    main()

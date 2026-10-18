import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import logging
import json
from binary_dataset import BinaryActiveSiteDataset, binary_collate_fn
from utils.util_functions import set_seed, calculate_metrics, get_model
from utils.util_classes import WarmupCosineAnnealingLR


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
        output = model(batch)["token_logits"]

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
    return metrics


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
        output = model(batch)["token_logits"]
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
    metrics["loss"] = avg_loss
    return metrics


def train(
    train_loader, valid_loader, test_loader, mmsite_loader, cfg, device, output_dir
):
    logging.info("=== Starting Training Phase ===")
    model = get_model(cfg).to(device)
    epochs = cfg.train.epochs

    # Optimizer for all parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.learning_rate,
    )
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        total_steps=epochs * len(train_loader),
        warmup_ratio=cfg.train.warmup_ratio,
    )
    criterion_bce = nn.BCEWithLogitsLoss()

    best_f1 = float("-inf")
    best_model_path = os.path.join(output_dir, "best_model.pt")

    # Track history
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_con_loss = 0.0
        train_token_loss = 0.0

        w_con = cfg.train.lamda

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch = batch.to(device)
            if isinstance(batch.texts, list):
                for i in range(len(batch.texts)):
                    batch.texts[i] = batch.texts[i].to(device)
            else:
                batch.texts = batch.texts.to(device)

            target = batch.y.float()

            optimizer.zero_grad()
            output_dict = model(batch)

            # Token classification loss is always used.
            token_logits = output_dict["token_logits"]
            loss_token = criterion_bce(token_logits, target)

            # Add contrastive loss only when enabled and returned by the model.
            contrastive_loss = output_dict.get("contrastive_loss")
            if w_con != 0 and contrastive_loss is not None:
                loss_con = contrastive_loss * w_con
                loss = loss_token + loss_con
            else:
                loss_con = loss_token.new_zeros(())
                loss = loss_token

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_con_loss += loss_con.item()
            train_token_loss += loss_token.item()

        del batch
        torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(train_loader)
        avg_con_loss = train_con_loss / len(train_loader)
        avg_token_loss = train_token_loss / len(train_loader)

        # Evaluate on Validation Set (using 'fuse' mode effectively to check F1)
        val_metrics = evaluate(model, valid_loader, device, desc=f"Valid {epoch + 1}")

        history["train"].append(avg_train_loss)
        history["val"].append(val_metrics["loss"])

        log_msg = (
            f"Epoch {epoch + 1}: Total Train Loss={avg_train_loss:.8f} "
            f"(Con={avg_con_loss:.8f}, Token={avg_token_loss:.8f}), "
            f"Val Loss={val_metrics['loss']:.8f}"
        )
        for k, v in val_metrics.items():
            if k != "loss":
                log_msg += f", {k}={v:.4f}"
        logging.info(log_msg)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved! (F1: {best_f1:.4f})")

    # Save history
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Testing
    logging.info("=== Starting Testing with Best Model ===")
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )
    valid_metrics = evaluate(model, valid_loader, device, desc="Valid")
    logging.info("=== Valid Results ===")
    for k, v in valid_metrics.items():
        logging.info(f"Valid {k}: {v:.4f}")
    test_metrics = evaluate(model, test_loader, device, desc="Test")
    logging.info("=== Test Results ===")
    for k, v in test_metrics.items():
        logging.info(f"Test {k}: {v:.4f}")
    mmsite_metrics = evaluate_for_mmsite(model, mmsite_loader, device, desc="MMSite")
    logging.info("=== MMSite Results ===")
    for k, v in mmsite_metrics.items():
        logging.info(f"MMSite {k}: {v:.4f}")

    return best_model_path


@hydra.main(config_path="conf", config_name="binary_train", version_base="1.3")
def main(cfg: DictConfig):
    # Set seed
    set_seed(cfg.seed if "seed" in cfg else 42)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load Datasets
    logging.info("Loading datasets...")
    train_tsv = cfg.dataset.train_tsv
    valid_tsv = cfg.dataset.valid_tsv
    test_tsv = cfg.dataset.test_tsv
    mmsite_tsv = cfg.dataset.mmsite_tsv
    train_dataset = BinaryActiveSiteDataset(cfg, train_tsv, align=True)
    valid_dataset = BinaryActiveSiteDataset(cfg, valid_tsv, align=False)
    test_dataset = BinaryActiveSiteDataset(cfg, test_tsv, align=False)
    mmsite_dataset = BinaryActiveSiteDataset(cfg, mmsite_tsv, align=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=binary_collate_fn,
        num_workers=cfg.train.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=binary_collate_fn,
        num_workers=cfg.train.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        collate_fn=binary_collate_fn,
        num_workers=cfg.train.num_workers,
    )
    mmsite_loader = DataLoader(
        mmsite_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        collate_fn=binary_collate_fn,
        num_workers=cfg.train.num_workers,
    )
    output_dir = HydraConfig.get().runtime.output_dir

    # 2. Training Phase
    train(
        train_loader, valid_loader, test_loader, mmsite_loader, cfg, device, output_dir
    )


if __name__ == "__main__":
    main()

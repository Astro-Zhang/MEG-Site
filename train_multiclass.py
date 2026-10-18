import json
import os
from collections import Counter
from types import SimpleNamespace

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from multiclass_dataset import MulticlassActiveSiteDataset, multiclass_collate_fn
from multiclass_model import MulticlassActiveSiteModel
from evaluate_multiclass import measure_graph_official
from utils.util_functions import set_seed
from utils.util_classes import WarmupCosineAnnealingLR


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        if self.gamma <= 0:
            return ce.mean()
        prob = torch.softmax(logits, dim=1)
        pt = prob.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def build_loaders(args):
    train_set = MulticlassActiveSiteDataset(args.data_root, args.split, "train")
    valid_set = MulticlassActiveSiteDataset(args.data_root, args.split, "valid")
    test_set = MulticlassActiveSiteDataset(args.data_root, args.split, "test")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
        collate_fn=multiclass_collate_fn,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=multiclass_collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=multiclass_collate_fn,
    )
    return train_set, train_loader, valid_loader, test_loader


def class_weights_from_labels(dataset, num_classes, mode="official"):
    if mode == "none":
        return None
    if mode == "official":
        if num_classes != 7:
            raise ValueError("official class weights are defined for 7 M3Site classes")
        return torch.tensor([0.01, 1.0, 0.7, 1.0, 0.5, 2.5, 2.0], dtype=torch.float)

    counter = Counter()
    for entry in dataset.data["Entry"]:
        counter.update(int(x) for x in dataset.labels[entry])
    counts = torch.tensor([counter.get(i, 0) for i in range(num_classes)], dtype=torch.float)
    inv = counts.sum() / counts.clamp_min(1.0)
    if mode == "balanced_sqrt":
        weights = torch.sqrt(inv)
    elif mode == "balanced":
        weights = inv
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")
    weights = weights / weights.mean()
    return weights


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc):
    model.eval()
    losses = []
    metric_sums = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "mcc": 0.0,
    }

    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        logits = model(batch)["token_logits"]
        target = batch.y.long()
        loss = criterion(logits, target)
        losses.append(loss.item())

        probs = torch.softmax(logits, dim=1)
        batch_metrics = measure_graph_official(target, probs, batch.ptr, batch.batch)
        for key, value in batch_metrics.items():
            metric_sums[key] += value

    metrics = {key: value / len(loader) for key, value in metric_sums.items()}
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
    )

    train_set, train_loader, valid_loader, test_loader = build_loaders(args)
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

    if args.use_class_weights and args.class_weight_mode == "none":
        args.class_weight_mode = "balanced_sqrt"
    weight = class_weights_from_labels(
        train_set, args.num_classes, mode=args.class_weight_mode
    )
    if weight is not None:
        weight = weight.to(device)
        print("class weights:", weight.detach().cpu().tolist())

    criterion = FocalCrossEntropyLoss(
        weight=weight, gamma=args.focal_gamma, label_smoothing=args.label_smoothing
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        total_steps=args.epochs * len(train_loader),
        warmup_ratio=args.warmup_ratio,
    )

    best_f1 = -1.0
    best_path = os.path.join(args.output_dir, f"best_multiclass_{args.split}.pt")
    history = {"train_loss": [], "valid": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)["token_logits"]
            loss = criterion(logits, batch.y.long())
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        valid_metrics = evaluate(model, valid_loader, criterion, device, f"Valid {epoch}")
        history["train_loss"].append(train_loss)
        history["valid"].append(valid_metrics)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, "
            f"valid_loss={valid_metrics['loss']:.6f}, "
            f"valid_f1={valid_metrics['f1']:.4f}, "
            f"valid_mcc={valid_metrics['mcc']:.4f}"
        )

        if valid_metrics["f1"] > best_f1:
            best_f1 = valid_metrics["f1"]
            torch.save(model.state_dict(), best_path)
            print(f"saved best checkpoint: {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device, "Test")
    history["test"] = test_metrics
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")


def build_args(cfg):
    return SimpleNamespace(
        seed=cfg.seed,
        data_root=cfg.dataset.root,
        split=str(cfg.dataset.split),
        output_dir=cfg.runtime.output_dir,
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
        class_weight_mode=cfg.loss.class_weight_mode,
        use_class_weights=False,
        focal_gamma=cfg.loss.focal_gamma,
        label_smoothing=cfg.loss.label_smoothing,
        batch_size=cfg.train.batch_size,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        epochs=cfg.train.epochs,
        num_workers=cfg.train.num_workers,
        warmup_ratio=cfg.train.warmup_ratio,
        grad_clip=cfg.train.gradient_clip,
        drop_last=cfg.train.drop_last,
    )


@hydra.main(config_path="conf", config_name="multiclass_train", version_base="1.3")
def main(cfg: DictConfig):
    train(build_args(cfg))


if __name__ == "__main__":
    main()

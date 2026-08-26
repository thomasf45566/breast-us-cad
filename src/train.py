"""Train a binary benign/malignant classifier on BUS-BRA.

Research prototype — not for diagnostic use.

Usage: python src/train.py --config configs/baseline.yaml [--epochs N]
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import wandb
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn

from data import build_master_df, make_dataloaders


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg: dict) -> nn.Module:
    return timm.create_model(
        cfg["model"]["name"], pretrained=cfg["model"]["pretrained"], num_classes=1
    )


def resolve_pos_weight(cfg: dict, train_labels: pd.Series) -> float:
    """`auto` = n_benign / n_malignant over the train folds; else a literal number."""
    setting = cfg["train"]["pos_weight"]
    if setting == "auto":
        return float((train_labels == 0).sum() / (train_labels == 1).sum())
    return float(setting)


def build_loss(cfg: dict, pos_weight: float, device: str) -> nn.Module:
    name = cfg["train"]["loss"]
    if name != "bce_with_logits":
        raise ValueError(f"Unknown train.loss: {name!r}")
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )


def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    name = cfg["train"]["optimizer"]
    if name != "adamw":
        raise ValueError(f"Unknown train.optimizer: {name!r}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer, epochs: int):
    name = cfg["train"]["scheduler"]
    if name != "cosine":
        raise ValueError(f"Unknown train.scheduler: {name!r}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


@torch.no_grad()
def predict(model: nn.Module, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) over a loader."""
    model.eval()
    probs, labels = [], []
    for images, targets in loader:
        logits = model(images.to(device)).squeeze(1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--epochs", type=int, help="override config epochs (smoke tests)")
    parser.add_argument("--run-name", help="override wandb run name")
    parser.add_argument("--ckpt-out", help="override checkpoint output path")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    epochs = args.epochs or cfg["train"]["epochs"]
    ckpt_path = Path(args.ckpt_out or cfg["paths"]["ckpt"])
    device = cfg["train"]["device"]
    set_seed(cfg["train"]["seed"])

    df = build_master_df(cfg["data"]["root"])
    train_loader, val_loader = make_dataloaders(
        df,
        train_folds=cfg["data"]["train_folds"],
        val_folds=cfg["data"]["val_folds"],
        batch_size=cfg["data"]["batch_size"],
        img_size=cfg["data"]["img_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    # pos_weight from train-fold class frequencies only (val stays untouched)
    train_labels = df[df["fold"].isin(cfg["data"]["train_folds"])]["label"]
    pos_weight = resolve_pos_weight(cfg, train_labels)

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg, pos_weight, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, epochs)

    run = wandb.init(
        project=cfg["wandb"]["project"],
        name=args.run_name or cfg["wandb"]["run_name"],
        config={**cfg, "epochs": epochs, "pos_weight": float(pos_weight)},
    )
    print(
        f"train: {len(train_loader.dataset)} images | val: {len(val_loader.dataset)} "
        f"images | pos_weight: {pos_weight:.3f} | device: {device} | epochs: {epochs}"
    )

    best_auc = 0.0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            loss = criterion(model(images).squeeze(1), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)
            n_seen += len(targets)
        scheduler.step()
        train_loss = running_loss / n_seen

        probs, labels = predict(model, val_loader, device)
        val_auc = roc_auc_score(labels, probs)
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": val_auc,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_auc": val_auc,
                },
                ckpt_path,
            )
            marker = f" -> saved {ckpt_path}"
        print(
            f"epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} | "
            f"val_auc {val_auc:.4f}{marker}"
        )

    wandb.summary["best_val_auc"] = best_auc
    run.finish()
    print(f"best val AUC: {best_auc:.4f} | checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

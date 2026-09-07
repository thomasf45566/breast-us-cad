"""Train a binary benign/malignant classifier on BUS-BRA.

Research prototype — not for diagnostic use.

Usage: python src/train.py --config configs/baseline.yaml [--epochs N]
"""

import argparse
import random
import time
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
    model = timm.create_model(
        cfg["model"]["name"], pretrained=cfg["model"]["pretrained"], num_classes=1
    )
    init_sd = cfg["model"].get("init_state_dict")
    if init_sd:
        state = torch.load(init_sd, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or set(missing) != {"head.weight", "head.bias"}:
            raise ValueError(
                f"init_state_dict {init_sd} is not a full backbone: "
                f"missing={missing} unexpected={unexpected}"
            )
        print(f"backbone initialized from {init_sd} ({len(state)} tensors; head random)")
    return model


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
    warmup = cfg["train"].get("warmup_epochs", 0)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup, 1)
    )
    if warmup <= 0:
        return cosine
    linear = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [linear, cosine], milestones=[warmup]
    )


def checkpoint_payload(
    cfg: dict,
    model_state: dict,
    *,
    epoch: int,
    val_auc: float,
    train_folds: list[int],
    val_folds: list[int],
) -> dict:
    """Checkpoint dict whose stored config records the folds ACTUALLY used.

    cross_validate.py passes per-fold `train_folds`/`val_folds` that differ
    from the YAML's `data.train_folds`/`data.val_folds`; before 2026-09-07 the
    raw YAML was stored, so every existing checkpoint (models/cv_vit_fold{1-5}.pt,
    cv_convnext_*, cv_effb0_*, cv_vit_cdrop_*, v2_biomedclip_fold{1-5}.pt)
    carries `train_folds=[1,2,3,4], val_folds=[5]` regardless of fold — the
    folds really used are in the wandb run configs (audit 2026-09-06 §1).
    `cfg` is not mutated.
    """
    resolved = {**cfg, "data": {**cfg["data"], "train_folds": list(train_folds), "val_folds": list(val_folds)}}
    return {"model_state": model_state, "config": resolved, "epoch": epoch, "val_auc": val_auc}


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


def train_one_fold(
    cfg: dict,
    *,
    train_folds: list[int],
    val_folds: list[int],
    epochs: int,
    run_name: str,
    ckpt_path: str | Path,
    group: str | None = None,
) -> dict:
    """Train on `train_folds`, select best checkpoint by AUC on `val_folds`.

    One wandb run per call (optionally grouped). Returns
    {"best_auc", "best_epoch", "ckpt_path"}.
    """
    ckpt_path = Path(ckpt_path)
    device = cfg["train"]["device"]
    set_seed(cfg["train"]["seed"])

    df = build_master_df(cfg["data"]["root"])
    train_loader, val_loader = make_dataloaders(
        df,
        train_folds=train_folds,
        val_folds=val_folds,
        batch_size=cfg["data"]["batch_size"],
        img_size=cfg["data"]["img_size"],
        num_workers=cfg["data"]["num_workers"],
        coarse_dropout=cfg["data"].get("coarse_dropout"),
    )

    # pos_weight from train-fold class frequencies only (val stays untouched)
    train_labels = df[df["fold"].isin(train_folds)]["label"]
    pos_weight = resolve_pos_weight(cfg, train_labels)

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg, pos_weight, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, epochs)

    run = wandb.init(
        project=cfg["wandb"]["project"],
        name=run_name,
        group=group,
        config={
            **cfg,
            "epochs": epochs,
            "pos_weight": float(pos_weight),
            "train_folds": train_folds,
            "val_folds": val_folds,
        },
    )
    print(
        f"train: {len(train_loader.dataset)} images | val: {len(val_loader.dataset)} "
        f"images | pos_weight: {pos_weight:.3f} | device: {device} | epochs: {epochs}"
    )

    best_auc, best_epoch = 0.0, 0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        epoch_start = time.monotonic()
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
            best_auc, best_epoch = val_auc, epoch
            torch.save(
                checkpoint_payload(
                    cfg,
                    model.state_dict(),
                    epoch=epoch,
                    val_auc=val_auc,
                    train_folds=train_folds,
                    val_folds=val_folds,
                ),
                ckpt_path,
            )
            marker = f" -> saved {ckpt_path}"
        print(
            f"epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} | "
            f"val_auc {val_auc:.4f} | {time.monotonic() - epoch_start:.1f}s{marker}"
        )

    run.summary["best_val_auc"] = best_auc
    run.finish()
    print(f"best val AUC: {best_auc:.4f} | checkpoint: {ckpt_path}")
    return {"best_auc": best_auc, "best_epoch": best_epoch, "ckpt_path": str(ckpt_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--epochs", type=int, help="override config epochs (smoke tests)")
    parser.add_argument("--run-name", help="override wandb run name")
    parser.add_argument("--ckpt-out", help="override checkpoint output path")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    train_one_fold(
        cfg,
        train_folds=cfg["data"]["train_folds"],
        val_folds=cfg["data"]["val_folds"],
        epochs=args.epochs or cfg["train"]["epochs"],
        run_name=args.run_name or cfg["wandb"]["run_name"],
        ckpt_path=args.ckpt_out or cfg["paths"]["ckpt"],
    )


if __name__ == "__main__":
    main()

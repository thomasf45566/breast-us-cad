"""Train a U-Net lesion segmentation model on BUS-BRA masks (bundled demo).

Research prototype — not for diagnostic use.

Deliberately minimal scope: one model, one run, no CV, no external eval.
Best checkpoint is selected by mean per-image Dice on the validation fold.

Usage: python src/train_seg.py --config configs/seg.yaml [--epochs N]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import wandb
import yaml
from torch import nn

from data import build_master_df, make_seg_dataloaders
from train import build_optimizer, build_scheduler, set_seed


def build_seg_model(cfg: dict) -> nn.Module:
    arch = cfg["model"]["arch"]
    if arch != "unet":
        raise ValueError(f"Unknown model.arch: {arch!r}")
    return smp.Unet(
        encoder_name=cfg["model"]["encoder"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=3,
        classes=1,
    )


class DiceBceLoss(nn.Module):
    """DiceLoss + BCEWithLogitsLoss, equal weight."""

    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice(logits, targets) + self.bce(logits, targets)


def build_seg_loss(cfg: dict) -> nn.Module:
    name = cfg["train"]["loss"]
    if name != "dice_bce":
        raise ValueError(f"Unknown train.loss: {name!r}")
    return DiceBceLoss()


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> tuple[float, float]:
    """Per-image Dice and IoU on binary masks."""
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    total = float(pred.sum() + gt.sum())
    return (2 * inter + eps) / (total + eps), (inter + eps) / (union + eps)


@torch.no_grad()
def predict_masks(
    model: nn.Module, loader, device: str, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_masks, gt_masks) as uint8 arrays of shape (N, H, W)."""
    model.eval()
    preds, gts = [], []
    for images, masks in loader:
        probs = torch.sigmoid(model(images.to(device)).squeeze(1))
        preds.append((probs >= threshold).cpu().numpy().astype(np.uint8))
        gts.append(masks.squeeze(1).numpy().astype(np.uint8))
    return np.concatenate(preds), np.concatenate(gts)


def mean_dice(preds: np.ndarray, gts: np.ndarray) -> float:
    return float(np.mean([dice_iou(p, g)[0] for p, g in zip(preds, gts)]))


def train_seg(
    cfg: dict,
    *,
    epochs: int,
    run_name: str,
    ckpt_path: str | Path,
    log_every: int = 0,
) -> dict:
    """Train on the config's train folds, select best checkpoint by val Dice."""
    ckpt_path = Path(ckpt_path)
    device = cfg["train"]["device"]
    set_seed(cfg["train"]["seed"])

    df = build_master_df(cfg["data"]["root"])
    train_folds = cfg["data"]["train_folds"]
    val_folds = cfg["data"]["val_folds"]
    train_loader, val_loader = make_seg_dataloaders(
        df,
        train_folds=train_folds,
        val_folds=val_folds,
        batch_size=cfg["data"]["batch_size"],
        img_size=cfg["data"]["img_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = build_seg_model(cfg).to(device)
    criterion = build_seg_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, epochs)

    run = wandb.init(
        project=cfg["wandb"]["project"],
        name=run_name,
        config={**cfg, "epochs": epochs, "train_folds": train_folds, "val_folds": val_folds},
    )
    print(
        f"train: {len(train_loader.dataset)} images | val: {len(val_loader.dataset)} "
        f"images | device: {device} | epochs: {epochs}"
    )

    best_dice, best_epoch = 0.0, 0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        running_loss, n_seen = 0.0, 0
        window_loss, window_n = 0.0, 0
        for i, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(masks)
            n_seen += len(masks)
            window_loss += loss.item() * len(masks)
            window_n += len(masks)
            if log_every and i % log_every == 0:
                print(f"  batch {i:3d}/{len(train_loader)} | loss {window_loss / window_n:.4f}")
                window_loss, window_n = 0.0, 0
        scheduler.step()
        train_loss = running_loss / n_seen

        preds, gts = predict_masks(model, val_loader, device)
        val_dice = mean_dice(preds, gts)
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_dice": val_dice,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        marker = ""
        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_dice": val_dice,
                },
                ckpt_path,
            )
            marker = f" -> saved {ckpt_path}"
        print(
            f"epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} | "
            f"val_dice {val_dice:.4f} | {time.monotonic() - epoch_start:.1f}s{marker}"
        )

    run.summary["best_val_dice"] = best_dice
    run.finish()
    print(f"best val Dice: {best_dice:.4f} (epoch {best_epoch}) | checkpoint: {ckpt_path}")
    return {"best_dice": best_dice, "best_epoch": best_epoch, "ckpt_path": str(ckpt_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/seg.yaml")
    parser.add_argument("--epochs", type=int, help="override config epochs (smoke tests)")
    parser.add_argument("--run-name", help="override wandb run name")
    parser.add_argument("--ckpt-out", help="override checkpoint output path")
    parser.add_argument(
        "--log-every", type=int, default=0, help="print train loss every N batches (smoke tests)"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    train_seg(
        cfg,
        epochs=args.epochs or cfg["train"]["epochs"],
        run_name=args.run_name or cfg["wandb"]["run_name"],
        ckpt_path=args.ckpt_out or cfg["paths"]["ckpt"],
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()

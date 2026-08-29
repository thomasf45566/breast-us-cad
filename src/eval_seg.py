"""Evaluate a segmentation checkpoint: mean Dice/IoU on the validation fold.

Research prototype — not for diagnostic use.

Saves a 12x3 example grid (image | ground-truth mask | predicted mask) for
the 4 best / 4 median / 4 worst per-image Dice cases, plus a per-image
Dice/IoU CSV.

Usage: python src/eval_seg.py --ckpt models/seg_unet_effb0.pt
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import BusSegDataset, build_master_df, get_transforms
from train_seg import build_seg_model, dice_iou, predict_masks


def load_display_pair(row: pd.Series, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Original image and GT mask resized to model resolution for display."""
    img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        raise FileNotFoundError(row["image_path"])
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return img, gt


def save_examples(
    eval_df: pd.DataFrame,
    dices: np.ndarray,
    preds: np.ndarray,
    img_size: int,
    path: Path,
) -> None:
    """12x3 grid: 4 best / 4 median / 4 worst Dice cases, one row per case."""
    order = np.argsort(dices)[::-1]  # best -> worst
    mid = len(order) // 2
    groups = [
        ("good", order[:4]),
        ("median", order[mid - 2 : mid + 2]),
        ("worst", order[-4:]),
    ]
    fig, axes = plt.subplots(12, 3, figsize=(7.5, 30))
    r = 0
    for group_name, idxs in groups:
        for idx in idxs:
            row = eval_df.iloc[idx]
            img, gt = load_display_pair(row, img_size)
            axes[r, 0].imshow(img, cmap="gray")
            axes[r, 0].set_ylabel(
                f"{group_name}\n{Path(row['image_path']).stem}\nDice {dices[idx]:.3f}",
                fontsize=8,
            )
            axes[r, 1].imshow(gt, cmap="gray", vmin=0, vmax=255)
            axes[r, 2].imshow(preds[idx], cmap="gray", vmin=0, vmax=1)
            for c in range(3):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
            r += 1
    for c, title in enumerate(["Image", "Ground truth", "Predicted"]):
        axes[0, c].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/seg_unet_effb0.pt")
    parser.add_argument("--tag", help="suffix for the metrics CSV (default: ckpt stem)")
    parser.add_argument("--out-fig", default="reports/seg_examples.png")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = cfg["train"]["device"]
    img_size = cfg["data"]["img_size"]
    tag = args.tag or Path(args.ckpt).stem

    model = build_seg_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    df = build_master_df(cfg["data"]["root"])
    eval_df = df[df["fold"].isin(cfg["data"]["val_folds"])].reset_index(drop=True)
    loader = DataLoader(
        BusSegDataset(eval_df, get_transforms("val", img_size)),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    preds, gts = predict_masks(model, loader, device)
    per_image = np.array([dice_iou(p, g) for p, g in zip(preds, gts)])
    dices, ious = per_image[:, 0], per_image[:, 1]

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.out_fig)
    save_examples(eval_df, dices, preds, img_size, fig_path)

    csv_path = reports_dir / f"seg_metrics_{tag}.csv"
    pd.DataFrame(
        {
            "image_path": eval_df["image_path"].values,
            "patient_id": eval_df["patient_id"].values,
            "dice": dices,
            "iou": ious,
        }
    ).to_csv(csv_path, index=False)

    print(f"checkpoint: {args.ckpt} (epoch {ckpt['epoch']}) | val folds: "
          f"{cfg['data']['val_folds']} | n = {len(dices)}")
    print(f"Mean Dice:   {dices.mean():.4f} (median {np.median(dices):.4f})")
    print(f"Mean IoU:    {ious.mean():.4f} (median {np.median(ious):.4f})")
    print(f"saved: {fig_path}, {csv_path}")


if __name__ == "__main__":
    main()

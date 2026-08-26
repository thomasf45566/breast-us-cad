"""Evaluate a trained checkpoint: AUC, sensitivity, specificity at Youden threshold.

Research prototype — not for diagnostic use.

Usage: python src/evaluate.py --ckpt models/best.pt --split val

`val` evaluates the official BUS-BRA validation folds from the checkpoint's
config. External test sets (BrEaST, Curated BUSI) are evaluated only after
the model is frozen and get their own loaders when that time comes.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from data import BusDataset, build_master_df, get_transforms
from train import build_model, predict


def youden_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, probs)
    return float(thresholds[np.argmax(tpr - fpr)])


def save_roc(labels, probs, auc, path: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — BUS-BRA validation")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    classes = ["benign", "malignant"]
    ax.set_xticks([0, 1], classes)
    ax.set_yticks([0, 1], classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (Youden threshold)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/best.pt")
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--tag", help="suffix for report filenames (default: ckpt stem)")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = cfg["train"]["device"]
    tag = args.tag or Path(args.ckpt).stem

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    df = build_master_df(cfg["data"]["root"])
    eval_df = df[df["fold"].isin(cfg["data"]["val_folds"])]
    loader = DataLoader(
        BusDataset(eval_df, get_transforms("val", cfg["data"]["img_size"])),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    probs, labels = predict(model, loader, device)
    run_name = cfg["wandb"]["run_name"]
    auc = roc_auc_score(labels, probs)
    thr = youden_threshold(labels, probs)
    preds = (probs >= thr).astype(int)
    cm = confusion_matrix(labels, preds)  # rows: true benign, malignant
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    roc_path = reports_dir / f"roc_{tag}.png"
    cm_path = reports_dir / f"cm_{tag}.png"
    save_roc(labels, probs, auc, roc_path)
    save_confusion(cm, cm_path)

    preds_path = reports_dir / f"preds_{run_name}.csv"
    pd.DataFrame(
        {
            "image_path": eval_df["image_path"].values,
            "patient_id": eval_df["patient_id"].values,
            "y_true": labels.astype(int),
            "y_prob": probs,
        }
    ).to_csv(preds_path, index=False)

    print(f"checkpoint: {args.ckpt} (epoch {ckpt['epoch']}) | split: {args.split} "
          f"| n = {len(labels)}")
    print(f"AUC:         {auc:.4f}")
    print(f"Youden thr:  {thr:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"saved: {roc_path}, {cm_path}, {preds_path}")


if __name__ == "__main__":
    main()

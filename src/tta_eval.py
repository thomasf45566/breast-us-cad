"""TTA (hflip only) evaluation of the frozen cv_vit CV checkpoints.

Research prototype — not for diagnostic use.

Each fold's checkpoint scores its own held-out fold once; original and
flipped probs come from the same pass, so plain and TTA settings share
identical forward passes on the originals. Reports per-fold AUC and
pooled OOF AUC for both settings; saves pooled OOF preds for the
calibration step.

Usage: python src/tta_eval.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import BusDataset, build_master_df, get_transforms
from evaluate import compute_metrics, predict_hflip_pair, save_confusion, save_roc
from train import build_model

ALL_FOLDS = [1, 2, 3, 4, 5]
CKPT_PATTERN = "models/cv_vit_fold{k}.pt"


def load_model(ckpt_path: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device)


def main() -> None:
    cfg = yaml.safe_load(Path("configs/vit.yaml").read_text())
    device = cfg["train"]["device"]
    df = build_master_df(cfg["data"]["root"])

    rows, oof = [], []
    for k in ALL_FOLDS:
        ckpt_path = CKPT_PATTERN.format(k=k)
        model = load_model(ckpt_path, device)
        fold_df = df[df["fold"] == k]
        loader = DataLoader(
            BusDataset(fold_df, get_transforms("val", cfg["data"]["img_size"])),
            batch_size=cfg["data"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
        )
        p_orig, p_flip, labels = predict_hflip_pair(model, loader, device)
        del model
        p_tta = (p_orig + p_flip) / 2

        m_plain = compute_metrics(labels, p_orig)
        m_tta = compute_metrics(labels, p_tta)
        rows.append(
            {
                "fold": k,
                "n": len(labels),
                "auc_plain": m_plain["auc"],
                "auc_tta": m_tta["auc"],
                "sens_plain": m_plain["sensitivity"],
                "sens_tta": m_tta["sensitivity"],
                "spec_plain": m_plain["specificity"],
                "spec_tta": m_tta["specificity"],
                "ckpt": ckpt_path,
            }
        )
        oof.append(
            pd.DataFrame(
                {
                    "image_path": fold_df["image_path"].values,
                    "patient_id": fold_df["patient_id"].values,
                    "fold": k,
                    "y_true": labels.astype(int),
                    "y_prob_plain": p_orig,
                    "y_prob_tta": p_tta,
                }
            )
        )
        print(
            f"fold {k}: n={len(labels)} | AUC plain {m_plain['auc']:.4f} "
            f"| AUC tta {m_tta['auc']:.4f}"
        )

    summary = pd.DataFrame(rows)
    oof_df = pd.concat(oof, ignore_index=True)
    y = oof_df["y_true"].values
    pooled = {
        "plain": roc_auc_score(y, oof_df["y_prob_plain"]),
        "tta": roc_auc_score(y, oof_df["y_prob_tta"]),
    }

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "tta_vit_summary.csv"
    summary.to_csv(summary_path, index=False)
    oof_path = reports_dir / "oof_vit_preds.csv"
    oof_df.to_csv(oof_path, index=False)

    for setting in ("plain", "tta"):
        probs = oof_df[f"y_prob_{setting}"].values
        m = compute_metrics(y, probs)
        save_roc(y, probs, m["auc"], reports_dir / f"roc_oof_vit_{setting}.png")
        save_confusion(m["cm"], reports_dir / f"cm_oof_vit_{setting}.png")
        print(
            f"pooled OOF {setting:5s}: AUC {m['auc']:.4f} | "
            f"sens {m['sensitivity']:.4f} | spec {m['specificity']:.4f} "
            f"(Youden {m['threshold']:.4f}, n={len(y)})"
        )

    print("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        f"mean fold AUC: plain {summary['auc_plain'].mean():.4f} ± "
        f"{summary['auc_plain'].std(ddof=1):.4f} | tta "
        f"{summary['auc_tta'].mean():.4f} ± {summary['auc_tta'].std(ddof=1):.4f}"
    )
    print(f"pooled OOF AUC: plain {pooled['plain']:.4f} | tta {pooled['tta']:.4f}")
    print(f"saved: {summary_path}, {oof_path}")


if __name__ == "__main__":
    main()

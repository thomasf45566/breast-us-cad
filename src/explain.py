"""Grad-CAM galleries for the frozen-v1 model (manual Step 10).

Research prototype — not for diagnostic use.

CAMs are generated with the fold-5 checkpoint only, at blocks[-2].norm1
(the frozen saliency method) — single-model visualizations, while the
frozen pipeline's DECISIONS come from the 5-ckpt hflip-TTA ensemble.
CAM target is the predicted class ("why did the model call it that").

1. Internal gallery (reports/gradcam_gallery_internal.png): 4x4 grid of
   TP/TN/FP/FN (4 each). Outcomes come from the pooled OOF calibrated
   probs at the frozen threshold, restricted to fold-5 images so the
   fold-5 CAM checkpoint is held-out for everything shown. Within each
   cell, cases with calibrated prob inside [0.3, 0.9] are preferred over
   saturated ones (CAM quality degrades at saturation); negatives sit
   below the threshold, so the rule picks their least-saturated cases.
2. External FP study (reports/gradcam_external_fp.png): the 8
   highest-calibrated-prob benign false positives from SYSUCC and GDPH
   (4 each) — what the model fixates on in atypical benign lesions
   under domain shift.

CAMs are computed on the original (unflipped) orientation only.

Usage: python src/explain.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from compare_backbones import load_cpu_model
from data import get_transforms
from external_val import CALIBRATION_JSON, OPERATING_POINT_JSON, calibrate_probs
from gradcam_check import IMG_SIZE, load_rgb01, vit_reshape_transform

CAM_CKPT = "models/cv_vit_fold5.pt"
OOF_PREDS_CSV = Path("reports/oof_vit_preds.csv")
EXTERNAL_PREDS_CSV = {name: Path(f"reports/external_{name}_preds.csv") for name in ("sysucc", "gdph")}
PREFERRED_BAND = (0.3, 0.9)  # calibrated-prob range where CAMs stay informative
N_PER_CELL = 4

CAM_CAVEAT = (
    "Single-model CAMs (fold-5 ckpt, Grad-CAM @ blocks[-2].norm1); decisions come from "
    "the 5-model hflip-TTA ensemble. Research use only — not for diagnostic use."
)


def band_priority(p: pd.Series) -> pd.Series:
    """Sort key: 0 inside the preferred band, distance to it outside."""
    lo, hi = PREFERRED_BAND
    return (lo - p).clip(lower=0) + (p - hi).clip(lower=0)


def make_cam(model: torch.nn.Module) -> GradCAM:
    return GradCAM(
        model=model,
        target_layers=[model.blocks[-2].norm1],
        reshape_transform=vit_reshape_transform,
    )


def cam_overlay(cam: GradCAM, image_path: str, target_class: int):
    """Grad-CAM overlay (RGB uint8) for one image, targeting `target_class`."""
    rgb = load_rgb01(image_path)
    x = get_transforms("val", IMG_SIZE)(image=(rgb * 255).astype("uint8"))["image"]
    gcam = cam(
        input_tensor=x.unsqueeze(0),
        targets=[BinaryClassifierOutputTarget(target_class)],
    )[0]
    return show_cam_on_image(rgb, gcam, use_rgb=True)


def render_gallery(cam, rows, n_cols, title, out_path):
    """`rows` is a list of (image_path, panel_title, cam_target) per panel."""
    n_rows = len(rows) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 3.7 * n_rows), gridspec_kw={"hspace": 0.3}
    )
    for ax, (image_path, panel_title, target) in zip(axes.flat, rows):
        ax.imshow(cam_overlay(cam, image_path, target))
        ax.set_title(panel_title, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{title}\n{CAM_CAVEAT}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def pick_internal(threshold: float, temperature: float) -> list[tuple[str, str, int]]:
    """4 cases per outcome cell from fold-5 rows of the pooled OOF preds."""
    oof = pd.read_csv(OOF_PREDS_CSV)
    oof["p_cal"] = calibrate_probs(oof["y_prob_tta"].values, temperature)
    oof["y_pred"] = (oof["p_cal"] >= threshold).astype(int)
    fold5 = oof[oof["fold"] == 5].copy()
    fold5["priority"] = band_priority(fold5["p_cal"])

    cells = {
        "TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0),
    }  # name -> (y_true, y_pred)
    rows = []
    for cell, (y_true, y_pred) in cells.items():
        sub = fold5[(fold5.y_true == y_true) & (fold5.y_pred == y_pred)]
        picks = sub.sort_values("priority").head(N_PER_CELL)
        print(f"internal {cell}: {len(sub)} candidates in fold 5, picked "
              + ", ".join(f"{Path(r.image_path).stem} (p={r.p_cal:.3f})"
                          for r in picks.itertuples()))
        for r in picks.itertuples():
            rows.append((
                r.image_path,
                f"{cell} — {Path(r.image_path).stem}\ncalibrated p = {r.p_cal:.3f}",
                int(r.y_pred),  # CAM target = predicted class
            ))
    return rows


def pick_external_fps() -> list[tuple[str, str, int]]:
    """Top-4 highest-calibrated-prob benign FPs per external cohort."""
    rows = []
    for name, csv_path in EXTERNAL_PREDS_CSV.items():
        preds = pd.read_csv(csv_path)
        fps = preds[(preds.y_true == 0) & (preds.y_pred == 1)]
        picks = fps.sort_values("y_prob_calibrated", ascending=False).head(N_PER_CELL)
        print(f"external {name}: {len(fps)} benign FPs, picked "
              + ", ".join(f"{Path(r.image_path).stem} (p={r.y_prob_calibrated:.3f})"
                          for r in picks.itertuples()))
        for r in picks.itertuples():
            rows.append((
                r.image_path,
                f"{name.upper()} FP — {Path(r.image_path).stem}\n"
                f"calibrated p = {r.y_prob_calibrated:.3f}",
                1,  # all flagged malignant
            ))
    return rows


def main() -> None:
    temperature = json.loads(CALIBRATION_JSON.read_text())["temperature"]
    threshold = json.loads(OPERATING_POINT_JSON.read_text())["threshold"]
    cam = make_cam(load_cpu_model(CAM_CKPT))
    Path("reports").mkdir(exist_ok=True)

    render_gallery(
        cam,
        pick_internal(threshold, temperature),
        n_cols=N_PER_CELL,
        title=("Internal Grad-CAM gallery — BUS-BRA fold-5 OOF outcomes at the frozen "
               f"threshold ({threshold:.4f}); rows: TP / TN / FP / FN"),
        out_path="reports/gradcam_gallery_internal.png",
    )
    render_gallery(
        cam,
        pick_external_fps(),
        n_cols=N_PER_CELL,
        title=("External benign false positives — highest calibrated probs, "
               "SYSUCC (top) and GDPH (bottom), frozen threshold"),
        out_path="reports/gradcam_external_fp.png",
    )


if __name__ == "__main__":
    main()

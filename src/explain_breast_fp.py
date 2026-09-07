"""External false-positive Grad-CAM gallery on BrEaST (CC BY 4.0) — replaces
reports/gradcam_external_fp.png, which embedded GDPH/SYSUCC images (dataset
without an explicit licence) and is excluded from distribution (P4c,
2026-09-07; RESULTS.md Errata).

Research prototype — not for diagnostic use.

Same design as explain.py's external-FP study: the highest-calibrated-prob
benign false positives at the frozen threshold, read from the SAVED
reports/external_breast_preds.csv (no new inference for the selection);
fold-5 checkpoint, Grad-CAM @ blocks[-2].norm1, target = predicted class
(malignant). Single-model visualization; decisions come from the ensemble.

Usage: python src/explain_breast_fp.py
"""

from pathlib import Path

import pandas as pd

from compare_backbones import load_cpu_model
from explain import CAM_CKPT, N_PER_CELL, make_cam, render_gallery

BREAST_PREDS_CSV = Path("reports/external_breast_preds.csv")
OUT_PNG = Path("reports/gradcam_external_fp_breast.png")
N_MAX = 8


def pick_breast_fps(n_max: int = N_MAX) -> list[tuple[str, str, int]]:
    preds = pd.read_csv(BREAST_PREDS_CSV)
    fps = preds[(preds.y_true == 0) & (preds.y_pred == 1)]
    picks = fps.sort_values("y_prob_calibrated", ascending=False).head(n_max)
    print(f"BrEaST: {len(fps)} benign FPs at the frozen threshold, showing top {len(picks)}: "
          + ", ".join(f"{Path(r.image_path).stem} (p={r.y_prob_calibrated:.3f})" for r in picks.itertuples()))
    return [
        (r.image_path, f"BrEaST FP — {Path(r.image_path).stem}\ncalibrated p = {r.y_prob_calibrated:.3f}", 1)
        for r in picks.itertuples()
    ]


def main() -> None:
    cam = make_cam(load_cpu_model(CAM_CKPT))
    render_gallery(
        cam,
        pick_breast_fps(),
        n_cols=N_PER_CELL,
        title=("External benign false positives — highest calibrated probs, BrEaST "
               "(CC BY 4.0, Pawłowska et al. 2024), frozen threshold"),
        out_path=str(OUT_PNG),
    )


if __name__ == "__main__":
    main()

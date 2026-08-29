"""Descriptive analysis of the saved external predictions — probability
shift under domain shift. Research prototype — not for diagnostic use.

Reads the frozen single-shot prediction CSVs only: NO model, NO inference,
NO threshold changes. Internal reference = pooled OOF TTA probs
(reports/oof_vit_preds.csv) passed through the frozen calibration.

Outputs: reports/prob_shift_external.png (Figure A, 1x5 panel row) and a
per-cohort table (median calibrated prob benign/malignant, fraction of
benign above the frozen threshold) printed as markdown.

Usage: python src/plot_prob_shift.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from external_val import CALIBRATION_JSON, OPERATING_POINT_JSON, calibrate_probs

OOF_CSV = Path("reports/oof_vit_preds.csv")
FIG_PATH = Path("reports/prob_shift_external.png")
COHORTS = [("breast", "BrEaST"), ("busi", "BUSI"), ("gdph", "GDPH"), ("sysucc", "SYSUCC")]
BENIGN_COLOR, MALIGNANT_COLOR = "#2b7bba", "#d1495b"


def load_panels(temperature: float) -> list[tuple[str, pd.DataFrame]]:
    """(display name, df with y_true + y_prob_calibrated), internal first."""
    oof = pd.read_csv(OOF_CSV)
    internal = pd.DataFrame(
        {
            "y_true": oof["y_true"],
            "y_prob_calibrated": calibrate_probs(oof["y_prob_tta"].values, temperature),
        }
    )
    panels = [("Internal (OOF)", internal)]
    for name, title in COHORTS:
        panels.append((title, pd.read_csv(f"reports/external_{name}_preds.csv")))
    return panels


def sens_spec(df: pd.DataFrame, thr: float) -> tuple[float, float]:
    pred = df["y_prob_calibrated"] >= thr
    pos, neg = df["y_true"] == 1, df["y_true"] == 0
    return pred[pos].mean(), (~pred[neg]).mean()


def plot_figure(panels, thr: float) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 31)
    for ax, (title, df) in zip(axes, panels):
        p = df["y_prob_calibrated"]
        ax.hist(p[df["y_true"] == 0], bins=bins, density=True, alpha=0.55,
                color=BENIGN_COLOR, label="benign")
        ax.hist(p[df["y_true"] == 1], bins=bins, density=True, alpha=0.55,
                color=MALIGNANT_COLOR, label="malignant")
        ax.axvline(thr, color="black", linestyle="--", linewidth=1.2)
        sens, spec = sens_spec(df, thr)
        ax.set_title(f"{title} (n={len(df)})")
        ax.text(0.97, 0.95, f"sens {sens:.3f}\nspec {spec:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_xlabel("calibrated probability")
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        f"Calibrated probability by class — internal vs external cohorts "
        f"(frozen threshold {thr:.4f}, dashed)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"saved: {FIG_PATH}")


def print_table(panels, thr: float) -> None:
    print("\n| Cohort | median p (benign) | median p (malignant) | frac benign ≥ thr (=1−spec) |")
    print("|---|---|---|---|")
    for title, df in panels:
        p = df["y_prob_calibrated"]
        benign, malignant = p[df["y_true"] == 0], p[df["y_true"] == 1]
        print(f"| {title} | {benign.median():.4f} | {malignant.median():.4f} "
              f"| {(benign >= thr).mean():.4f} |")


def main() -> None:
    calib = json.loads(CALIBRATION_JSON.read_text())
    thr = json.loads(OPERATING_POINT_JSON.read_text())["threshold"]
    panels = load_panels(calib["temperature"])
    plot_figure(panels, thr)
    print_table(panels, thr)


if __name__ == "__main__":
    main()

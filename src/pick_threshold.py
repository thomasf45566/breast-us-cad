"""Operating point on calibrated pooled OOF probabilities: sens >= 0.90, max spec.

Research prototype — not for diagnostic use.

Applies the frozen temperature (models/calibration.json) to the pooled OOF
hflip-TTA probabilities, picks the threshold with maximum specificity among
those achieving sensitivity >= 0.90, and reports sens/spec/PPV/NPV with 95%
patient-level bootstrap CIs (resample patients, 2000 iterations). Saves
models/operating_point.json and an ROC slide asset with the point marked.

Usage: python src/pick_threshold.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from calibrate import probs_to_logits

OOF_CSV = Path("reports/oof_vit_preds.csv")
CALIBRATION_JSON = Path("models/calibration.json")
OUT_JSON = Path("models/operating_point.json")
ROC_PNG = Path("reports/roc_oof_operating_point.png")
SENS_FLOOR = 0.90
N_BOOT = 2000
BOOT_SEED = 42


def pick_operating_point(labels: np.ndarray, probs: np.ndarray) -> float:
    """Highest threshold with sensitivity >= SENS_FLOOR (=> max specificity)."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    ok = tpr >= SENS_FLOOR
    if not ok.any():
        raise RuntimeError(f"No threshold reaches sensitivity >= {SENS_FLOOR}")
    # roc_curve thresholds are decreasing; among sens-feasible points the one
    # with minimum FPR is the first True entry.
    i = int(np.argmax(ok))
    return float(thresholds[i])


def confusion_counts(labels: np.ndarray, probs: np.ndarray, thr: float) -> tuple[int, int, int, int]:
    preds = probs >= thr
    pos, neg = labels == 1, labels == 0
    tp = int((preds & pos).sum())
    fn = int((~preds & pos).sum())
    tn = int((~preds & neg).sum())
    fp = int((preds & neg).sum())
    return tp, fp, fn, tn


def point_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    return {
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp),
        "npv": tn / (tn + fn),
    }


def patient_bootstrap_ci(
    df: pd.DataFrame, thr: float, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> dict:
    """95% percentile CIs for sens/spec at a fixed threshold, resampling patients."""
    preds = df["prob_cal"].values >= thr
    y = df["y_true"].values
    counts = (
        pd.DataFrame(
            {
                "patient_id": df["patient_id"].values,
                "tp": preds & (y == 1),
                "fn": ~preds & (y == 1),
                "tn": ~preds & (y == 0),
                "fp": preds & (y == 0),
            }
        )
        .groupby("patient_id")
        .sum()
        .to_numpy(dtype=np.int64)  # columns: tp, fn, tn, fp
    )
    n_patients = counts.shape[0]
    rng = np.random.default_rng(seed)
    sens, spec = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_patients, n_patients)
        tp, fn, tn, fp = counts[idx].sum(axis=0)
        sens[b] = tp / (tp + fn) if tp + fn else np.nan
        spec[b] = tn / (tn + fp) if tn + fp else np.nan
    return {
        "n_patients": int(n_patients),
        "n_boot": n_boot,
        "seed": seed,
        "sensitivity_ci95": [float(np.nanpercentile(sens, 2.5)), float(np.nanpercentile(sens, 97.5))],
        "specificity_ci95": [float(np.nanpercentile(spec, 2.5)), float(np.nanpercentile(spec, 97.5))],
    }


def plot_roc_with_point(labels, probs, thr, metrics, auc, path: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    op_x, op_y = 1 - metrics["specificity"], metrics["sensitivity"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color="#4c72b0", lw=2, label=f"pooled OOF ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.plot(op_x, op_y, "o", color="#c44e52", ms=9, zorder=5,
            label=f"operating point (thr = {thr:.3f})")
    ax.annotate(
        f"sens {metrics['sensitivity']:.3f}\nspec {metrics['specificity']:.3f}",
        xy=(op_x, op_y), xytext=(op_x + 0.08, op_y - 0.14), fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "0.3"},
    )
    ax.axhline(SENS_FLOOR, color="#c44e52", lw=0.8, ls=":", alpha=0.6)
    ax.text(0.985, SENS_FLOOR - 0.03, f"sens ≥ {SENS_FLOOR:.2f}", fontsize=8,
            color="#c44e52", ha="right")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("BUS-BRA pooled OOF — vit_b16 ensemble + hflip TTA, calibrated\n"
                 "Research use only — not for diagnosis", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    calib = json.loads(CALIBRATION_JSON.read_text())
    temperature = calib["temperature"]
    oof = pd.read_csv(OOF_CSV)
    oof["prob_cal"] = 1 / (1 + np.exp(-probs_to_logits(oof["y_prob_tta"].values) / temperature))
    y = oof["y_true"].values
    p = oof["prob_cal"].values

    auc = roc_auc_score(y, p)
    thr = pick_operating_point(y, p)
    tp, fp, fn, tn = confusion_counts(y, p, thr)
    metrics = point_metrics(tp, fp, fn, tn)
    ci = patient_bootstrap_ci(oof, thr)

    OUT_JSON.write_text(
        json.dumps(
            {
                "rule": f"sensitivity >= {SENS_FLOOR} with maximum specificity",
                "threshold": thr,
                "probability_space": "calibrated (temperature-scaled) hflip-TTA ensemble prob",
                "temperature": temperature,
                "calibration_file": str(CALIBRATION_JSON),
                "fit_on": f"pooled OOF, {OOF_CSV}, column y_prob_tta",
                "n_images": int(len(y)),
                "auc": float(auc),
                "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                **{k: float(v) for k, v in metrics.items()},
                "bootstrap": ci,
            },
            indent=2,
        )
        + "\n"
    )
    plot_roc_with_point(y, p, thr, metrics, auc, ROC_PNG)

    s_lo, s_hi = ci["sensitivity_ci95"]
    sp_lo, sp_hi = ci["specificity_ci95"]
    print(f"n = {len(y)} images, {ci['n_patients']} patients | T = {temperature:.4f} "
          f"| AUC (calibrated) = {auc:.4f}")
    print(f"threshold:   {thr:.4f}  (rule: sens >= {SENS_FLOOR}, max spec)")
    print(f"sensitivity: {metrics['sensitivity']:.4f}  (95% CI {s_lo:.4f}-{s_hi:.4f})")
    print(f"specificity: {metrics['specificity']:.4f}  (95% CI {sp_lo:.4f}-{sp_hi:.4f})")
    print(f"PPV:         {metrics['ppv']:.4f}")
    print(f"NPV:         {metrics['npv']:.4f}")
    print(f"confusion:   tp {tp} | fp {fp} | fn {fn} | tn {tn}")
    print(f"saved: {OUT_JSON}, {ROC_PNG}")


if __name__ == "__main__":
    main()

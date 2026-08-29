"""Temperature scaling of the pooled OOF hflip-TTA probabilities.

Research prototype — not for diagnostic use.

Fits a single scalar temperature T on the pooled out-of-fold TTA
probabilities (reports/oof_vit_preds.csv, column y_prob_tta), saves it to
models/calibration.json, and writes reliability diagrams + ECE (15
equal-width bins) to reports/. AUC is unchanged by construction
(monotone transform) and asserted.

Usage: python src/calibrate.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

OOF_CSV = Path("reports/oof_vit_preds.csv")
OUT_JSON = Path("models/calibration.json")
REPORTS_DIR = Path("reports")
N_BINS = 15
EPS = 1e-6


def probs_to_logits(p: np.ndarray) -> np.ndarray:
    """Convention: logit of the TTA-averaged probability, log(p/(1-p)) with p
    clipped to [1e-6, 1-1e-6] — NOT the average of the two flips' logits.
    Inference averages sigmoid probs, so calibration sits downstream of that."""
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Single scalar T minimizing NLL of sigmoid(logit / T), via LBFGS."""
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    t = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(z / t, y)
        loss.backward()
        return loss

    opt.step(closure)
    temperature = float(t.item())
    if temperature <= 0:
        raise RuntimeError(f"LBFGS produced non-positive temperature: {temperature}")
    return temperature


def nll(labels: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs, EPS, 1 - EPS)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def bin_stats(labels: np.ndarray, probs: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Per-bin count, mean confidence, and observed positive rate; equal-width bins."""
    bin_id = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_id == b
        rows.append(
            {
                "bin": b,
                "count": int(mask.sum()),
                "confidence": probs[mask].mean() if mask.any() else np.nan,
                "accuracy": labels[mask].mean() if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ece(stats: pd.DataFrame, n_total: int) -> float:
    filled = stats[stats["count"] > 0]
    return float(
        (filled["count"] / n_total * (filled["accuracy"] - filled["confidence"]).abs()).sum()
    )


def plot_reliability(ax, stats: pd.DataFrame, title: str, ece_value: float) -> None:
    width = 1.0 / N_BINS
    centers = (stats["bin"] + 0.5) * width
    ax.bar(
        centers,
        stats["accuracy"].fillna(0),
        width=width * 0.9,
        color="#4c72b0",
        label="observed malignant fraction",
    )
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(centers, stats["confidence"], ".", color="#c44e52", label="bin mean prob")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (malignant)")
    ax.set_ylabel("Observed malignant fraction")
    ax.set_title(f"{title}\nECE = {ece_value:.4f} ({N_BINS} equal-width bins)")
    ax.legend(loc="upper left", fontsize=8)


def main() -> None:
    oof = pd.read_csv(OOF_CSV)
    y = oof["y_true"].values.astype(float)
    p_raw = oof["y_prob_tta"].values
    logits = probs_to_logits(p_raw)

    temperature = fit_temperature(logits, y)
    p_cal = 1 / (1 + np.exp(-logits / temperature))

    auc_raw = roc_auc_score(y, p_raw)
    auc_cal = roc_auc_score(y, p_cal)
    assert abs(auc_cal - auc_raw) < 1e-10, (
        f"AUC changed under temperature scaling: {auc_raw} -> {auc_cal}"
    )

    stats_raw = bin_stats(y, p_raw)
    stats_cal = bin_stats(y, p_cal)
    ece_raw = ece(stats_raw, len(y))
    ece_cal = ece(stats_cal, len(y))
    nll_raw = nll(y, p_raw)
    nll_cal = nll(y, p_cal)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "method": "temperature_scaling",
                "temperature": temperature,
                "logit_convention": "logit of TTA-averaged prob, p clipped to [1e-6, 1-1e-6]",
                "fit_on": "pooled OOF, reports/oof_vit_preds.csv, column y_prob_tta",
                "n": int(len(y)),
                "ece_15bin_before": ece_raw,
                "ece_15bin_after": ece_cal,
                "nll_before": nll_raw,
                "nll_after": nll_cal,
                "auc": float(auc_raw),
            },
            indent=2,
        )
        + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_reliability(axes[0], stats_raw, "Before calibration (OOF, hflip TTA)", ece_raw)
    plot_reliability(axes[1], stats_cal, f"After temperature scaling (T = {temperature:.3f})", ece_cal)
    fig.suptitle("Reliability — pooled OOF, cv_vit + hflip TTA. Research use only.")
    fig.tight_layout()
    fig_path = REPORTS_DIR / "reliability_oof_vit_tta.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"n = {len(y)} pooled OOF preds (hflip TTA)")
    print(f"temperature T: {temperature:.4f}")
    print(f"AUC:           {auc_raw:.4f} (unchanged after scaling — asserted)")
    print(f"NLL:           {nll_raw:.4f} -> {nll_cal:.4f}")
    print(f"ECE (15 bins): {ece_raw:.4f} -> {ece_cal:.4f}")
    print(f"saved: {OUT_JSON}, {fig_path}")


if __name__ == "__main__":
    main()

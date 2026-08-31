"""v2 POST-HOC descriptive analysis: cross-site calibration transfer.

Research prototype — not for diagnostic use.

Calibrate on ONE cohort's full saved predictions and evaluate on ANOTHER —
all ordered pairs among the four external cohorts, plus the internal
BUS-BRA pooled OOF as the reference calibration source (which reproduces
the frozen pipeline exactly, asserted). Saved prediction CSVs only; no
inference, no training, no model changes; the frozen pipeline is
unchanged. POST-HOC: this analysis is not part of pre-registered
Amendment 2 (which covers within-cohort learning curves) and is labeled
descriptive wherever reported.

Methods (full source data, no draws):
  M1  threshold re-selected on the source's calibrated probs with the
      frozen rule (highest thr with sens >= 0.90), transferred to the
      target's calibrated probs. (Selecting the threshold under a
      source-refit temperature gives the SAME decisions — monotone map,
      same critical sample — so it is not a separate variant here.)
  M2  temperature refit on the source's logits (calibrate.py LBFGS
      fitter), transferred to the target's logits, frozen threshold
      0.2683 (the Amendment 2 M2a form).

Diagonal cells (source == target) are in-sample and marked as such.

Usage: python src/v2_cross_site.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrate import fit_temperature, probs_to_logits
from pick_threshold import confusion_counts, pick_operating_point

CALIBRATION_JSON = Path("models/calibration.json")
OPERATING_POINT_JSON = Path("models/operating_point.json")
OOF_CSV = Path("reports/oof_vit_preds.csv")
OUT_CSV = Path("reports/v2_cross_site.csv")
OUT_PNG = Path("reports/v2_cross_site_matrix.png")

COHORTS = ["breast", "busi", "gdph", "sysucc"]
SOURCES = ["internal"] + COHORTS
LABEL = {"internal": "Internal (OOF)", "breast": "BrEaST", "busi": "BUSI",
         "gdph": "GDPH", "sysucc": "SYSUCC"}


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(labels, logits) — logits are log(p/(1-p)) of the raw TTA prob,
    the calibrate.probs_to_logits convention."""
    if name == "internal":
        df = pd.read_csv(OOF_CSV)
        return df["y_true"].to_numpy(), probs_to_logits(df["y_prob_tta"].to_numpy())
    df = pd.read_csv(f"reports/external_{name}_preds.csv")
    return df["y_true"].to_numpy(), probs_to_logits(df["y_prob_raw"].to_numpy())


def sens_spec(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[float, float]:
    tp, fp, fn, tn = confusion_counts(y, p, thr)
    return tp / (tp + fn), tn / (tn + fp)


def main() -> None:
    t_frozen = json.loads(CALIBRATION_JSON.read_text())["temperature"]
    thr_frozen = json.loads(OPERATING_POINT_JSON.read_text())["threshold"]
    data = {name: load(name) for name in SOURCES}

    # Per-source calibrations on the FULL source data
    src_thr, src_temp = {}, {}
    for s in SOURCES:
        y, z = data[s]
        src_thr[s] = pick_operating_point(y, 1 / (1 + np.exp(-z / t_frozen)))
        src_temp[s] = fit_temperature(z, y.astype(float))

    # The internal source must reproduce the frozen pipeline exactly
    assert abs(src_thr["internal"] - thr_frozen) < 1e-9, \
        f"internal M1 threshold {src_thr['internal']} != frozen {thr_frozen}"
    assert abs(src_temp["internal"] - t_frozen) < 1e-3, \
        f"internal M2 temperature {src_temp['internal']} != frozen {t_frozen}"
    print(f"internal source reproduces frozen pipeline: thr {src_thr['internal']:.4f} "
          f"== {thr_frozen:.4f}, T {src_temp['internal']:.4f} ~= {t_frozen:.4f}  OK")

    rows = []
    for s in SOURCES:
        for t in COHORTS:
            y_t, z_t = data[t]
            p_cal = 1 / (1 + np.exp(-z_t / t_frozen))
            sens1, spec1 = sens_spec(y_t, p_cal, src_thr[s])
            p_m2 = 1 / (1 + np.exp(-z_t / src_temp[s]))
            sens2, spec2 = sens_spec(y_t, p_m2, thr_frozen)
            rows.append({
                "source": s, "target": t, "in_sample": s == t,
                "src_threshold_M1": src_thr[s], "src_temperature_M2": src_temp[s],
                "sens_M1": sens1, "spec_M1": spec1,
                "sens_M2": sens2, "spec_M2": spec2,
            })
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    # 5x4 matrix figure, one panel per method: cell color = specificity,
    # annotated spec (large) + sens (small); diagonal marked in-sample.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    for ax, m, title in [
        (axes[0], "M1", "M1: transferred threshold (source-selected, sens≥0.90 rule)"),
        (axes[1], "M2", "M2: transferred temperature + frozen thr 0.2683"),
    ]:
        mat = np.array([[res[(res["source"] == s) & (res["target"] == t)][f"spec_{m}"].iloc[0]
                         for t in COHORTS] for s in SOURCES])
        im = ax.imshow(mat, cmap="Blues", vmin=0.2, vmax=0.9, aspect="auto")
        for i, s in enumerate(SOURCES):
            for j, t in enumerate(COHORTS):
                r = res[(res["source"] == s) & (res["target"] == t)].iloc[0]
                dark = mat[i, j] > 0.62
                note = "\n(in-sample)" if r["in_sample"] else ""
                ax.text(j, i, f"{r[f'spec_{m}']:.2f}\nsens {r[f'sens_{m}']:.2f}{note}",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if dark else "#1a1a2e")
        ax.set_xticks(range(len(COHORTS)), [LABEL[t] for t in COHORTS], fontsize=9)
        ax.set_yticks(range(len(SOURCES)), [LABEL[s] for s in SOURCES], fontsize=9)
        ax.set_xlabel("evaluation target")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85, label="specificity")
    axes[0].set_ylabel("calibration source (full data)")
    fig.suptitle("v2 POST-HOC: cross-site calibration transfer — saved predictions only, "
                 "frozen model unchanged\nResearch use only — not for diagnosis", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nsource calibrations: " + ", ".join(
        f"{LABEL[s]} thr {src_thr[s]:.4f} / T {src_temp[s]:.4f}" for s in SOURCES))
    for m in ["M1", "M2"]:
        print(f"\n{m} specificity (sens) — rows: source, cols: target")
        header = "".join(f"{LABEL[t]:>22s}" for t in COHORTS)
        print(f"{'':14s}{header}")
        for s in SOURCES:
            cells = ""
            for t in COHORTS:
                r = res[(res["source"] == s) & (res["target"] == t)].iloc[0]
                tag = "*" if r["in_sample"] else " "
                cells += f"{r[f'spec_{m}']:>12.4f} ({r[f'sens_{m}']:.2f}){tag}"
            print(f"{LABEL[s]:14s}{cells}")
    print("* = in-sample (source == target)")
    print(f"\nsaved: {OUT_CSV}, {OUT_PNG}")


if __name__ == "__main__":
    main()

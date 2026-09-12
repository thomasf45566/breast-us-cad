"""POST-HOC: leave-fold-out THRESHOLD SENSITIVITY on the fixed OOF artifact.

Research prototype — not for diagnostic use.

Responds to audit 2026-09-06 §7: the frozen threshold 0.2683 was fitted on
the same pooled calibrated OOF predictions on which sens 0.9028 / spec 0.7713
are reported (in-sample). Here, from reports/oof_vit_preds.csv with the
FROZEN temperature (models/calibration.json), for each fold k the
sens ≥ 0.90 threshold is picked (frozen pick_threshold routine) on the
other four folds' calibrated OOF probabilities and evaluated on fold k.
Reports the five thresholds and mean ± SD sens/spec (plus pooled
leave-fold-out decisions) next to the in-sample values, which are
asserted to reproduce models/operating_point.json. No model, calibration,
or threshold is changed.

WHAT THIS IS NOT (re-audit 2026-09-11, docs/AUDIT2_astra_2026-09-11.md
§2.2/§7): this is NOT a nested out-of-sample model evaluation. Only the
threshold-fitting rows are held out. The OOF scores on folds j ≠ k come
from checkpoints that were TRAINED on fold k, and the checkpoint scoring
fold k was itself selected at its best-val-AUC epoch on fold k; the frozen
T was fitted on all OOF labels (a positive common transform, so it does
not move the rank boundary). There is no outer retraining loop. The
numbers quantify how sensitive the operating point is to which folds the
threshold is fitted on — a threshold-selection sensitivity analysis — not
an unbiased estimate of out-of-sample sens/spec. The file name is kept
for artifact continuity (reports/posthoc_nested_threshold.csv is cited in
RESULTS.md POST-HOC 2); "nested" in that name is historical.

Usage: python src/posthoc_nested_threshold.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate import probs_to_logits
from pick_threshold import confusion_counts, pick_operating_point, point_metrics

OOF_CSV = Path("reports/oof_vit_preds.csv")
CALIBRATION_JSON = Path("models/calibration.json")
OPERATING_POINT_JSON = Path("models/operating_point.json")
OUT_CSV = Path("reports/posthoc_nested_threshold.csv")


def calibrated_probs(df: pd.DataFrame, temperature: float) -> np.ndarray:
    """Frozen convention: sigmoid(logit(p_tta) / T)."""
    z = probs_to_logits(df["y_prob_tta"].to_numpy(dtype=np.float64)) / temperature
    return 1.0 / (1.0 + np.exp(-z))


def assert_in_sample_reproduces(labels: np.ndarray, probs: np.ndarray) -> dict:
    op = json.loads(OPERATING_POINT_JSON.read_text())
    thr = pick_operating_point(labels, probs)
    assert abs(thr - op["threshold"]) < 1e-9, f"in-sample threshold {thr} != {op['threshold']}"
    m = point_metrics(*confusion_counts(labels, probs, thr))
    assert abs(m["sensitivity"] - op["sensitivity"]) < 1e-12
    assert abs(m["specificity"] - op["specificity"]) < 1e-12
    print(f"assert: in-sample threshold {thr:.4f}, sens {m['sensitivity']:.4f}, spec {m['specificity']:.4f} "
          f"reproduce {OPERATING_POINT_JSON} — OK")
    return {"threshold": thr, **m}


def main() -> None:
    df = pd.read_csv(OOF_CSV)
    temperature = json.loads(CALIBRATION_JSON.read_text())["temperature"]
    labels = df["y_true"].to_numpy()
    probs = calibrated_probs(df, temperature)
    folds = df["fold"].to_numpy()
    in_sample = assert_in_sample_reproduces(labels, probs)

    rows, pooled_pred = [], np.zeros(len(df), dtype=bool)
    for k in sorted(np.unique(folds)):
        fit, ev = folds != k, folds == k
        thr = pick_operating_point(labels[fit], probs[fit])
        tp, fp, fn, tn = confusion_counts(labels[ev], probs[ev], thr)
        m = point_metrics(tp, fp, fn, tn)
        pooled_pred[ev] = probs[ev] >= thr
        rows.append({"fold": int(k), "n": int(ev.sum()), "threshold_from_other_folds": thr,
                     "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    table = pd.DataFrame(rows)
    pos, neg = labels == 1, labels == 0
    pooled_sens = float((pooled_pred & pos).sum() / pos.sum())
    pooled_spec = float((~pooled_pred & neg).sum() / neg.sum())
    table.to_csv(OUT_CSV, index=False)

    sd = lambda s: s.std(ddof=1)
    print("\nPOST-HOC — leave-fold-out threshold sensitivity (threshold from the other 4 folds, evaluated on fold k;")
    print("           NOT a nested model evaluation: checkpoints were trained on the outer fold — see module docstring)")
    print(f"frozen T = {temperature:.4f}; rule = frozen pick_threshold routine (sens ≥ 0.90 among roc_curve "
          f"drop_intermediate=True points)")
    print("| fold k | n | threshold (folds ≠ k) | sens on k | spec on k |")
    print("|---|---|---|---|---|")
    for r in table.itertuples():
        print(f"| {r.fold} | {r.n} | {r.threshold_from_other_folds:.4f} | {r.sensitivity:.4f} | {r.specificity:.4f} |")
    print(f"| **mean ± SD** | — | {table['threshold_from_other_folds'].mean():.4f} ± "
          f"{sd(table['threshold_from_other_folds']):.4f} | {table['sensitivity'].mean():.4f} ± "
          f"{sd(table['sensitivity']):.4f} | {table['specificity'].mean():.4f} ± {sd(table['specificity']):.4f} |")
    print(f"| pooled leave-fold-out decisions (n=1875) | — | — | {pooled_sens:.4f} | {pooled_spec:.4f} |")
    print(f"| in-sample (frozen 0.2683, reference) | 1875 | {in_sample['threshold']:.4f} | "
          f"{in_sample['sensitivity']:.4f} | {in_sample['specificity']:.4f} |")
    print(f"written: {OUT_CSV}")


if __name__ == "__main__":
    main()

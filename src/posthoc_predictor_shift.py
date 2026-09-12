"""POST-HOC: separate the predictor-change from the cohort-change component
of the internal→external calibration/operating-point shift.

Research prototype — not for diagnostic use.

Responds to the 2026-09-11 re-audits (docs/AUDIT2_astra_2026-09-11.md §2.1,
§7; docs/AUDIT2_claude_2026-09-11.md §7): the internal reference numbers
(pooled OOF AUC 0.9254, benign median p_cal 0.0625, spec 0.7713 at the
frozen threshold) describe ONE HELD-OUT CHECKPOINT PER IMAGE + hflip TTA,
whereas every external number describes the deployed 5-checkpoint
ensemble + hflip TTA. The temperature T and the threshold were fitted on
the single-held-out-checkpoint distribution and applied to the ensemble
externally, so the internal→external comparison changes the predictor as
well as the cohort. This script puts, per external cohort and side by
side, (a) the fold-5 single checkpoint + TTA and (b) the 5-checkpoint
ensemble + TTA — both from the SAVED member probabilities
(reports/v2_members_{cohort}.csv, Amendment 3 (n)) — next to the internal
single-held-out values from reports/oof_vit_preds.csv. Frozen T and
threshold throughout; no inference, no model, no artifact is changed.

Reading the table: the (a)-external minus (a)-internal(fold 5) difference
is the cohort-change component for a fixed predictor; the (b) minus (a)
difference on the same external cohort is the predictor-change component.

Outputs: reports/posthoc_predictor_shift.csv (+ per-member table
reports/posthoc_predictor_shift_members.csv) and a markdown table on stdout.

Usage: python src/posthoc_predictor_shift.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from calibrate import probs_to_logits
from pick_threshold import confusion_counts, point_metrics

OOF_CSV = Path("reports/oof_vit_preds.csv")
CALIBRATION_JSON = Path("models/calibration.json")
OPERATING_POINT_JSON = Path("models/operating_point.json")
LOCO_SUMMARY_CSV = Path("reports/v2_loco_summary.csv")
OUT_CSV = Path("reports/posthoc_predictor_shift.csv")
OUT_MEMBERS_CSV = Path("reports/posthoc_predictor_shift_members.csv")

COHORTS = ["breast", "busi", "gdph", "sysucc"]
LABEL = {"breast": "BrEaST", "busi": "BUSI", "gdph": "GDPH", "sysucc": "SYSUCC"}
CAM_FOLD = 5  # the single checkpoint compared externally (Amendment 4 (t) comparator)


def calibrate(p_raw: np.ndarray, temperature: float) -> np.ndarray:
    """Frozen convention: sigmoid(logit(p) / T), float64."""
    z = probs_to_logits(np.asarray(p_raw, dtype=np.float64)) / temperature
    return 1.0 / (1.0 + np.exp(-z))


def describe(y: np.ndarray, p_cal: np.ndarray, thr: float) -> dict:
    tp, fp, fn, tn = confusion_counts(y, p_cal, thr)
    m = point_metrics(tp, fp, fn, tn)
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p_cal)),
        "benign_median_p_cal": float(np.median(p_cal[y == 0])),
        "malignant_median_p_cal": float(np.median(p_cal[y == 1])),
        "sens": m["sensitivity"],
        "spec": m["specificity"],
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def internal_rows(temperature: float, thr: float, op: dict) -> list[dict]:
    oof = pd.read_csv(OOF_CSV)
    y = oof["y_true"].to_numpy()
    p = calibrate(oof["y_prob_tta"].to_numpy(), temperature)
    pooled = describe(y, p, thr)
    # assert the pooled row reproduces the frozen operating-point record
    assert (pooled["tp"], pooled["fp"], pooled["fn"], pooled["tn"]) == tuple(
        op["confusion"][k] for k in ("tp", "fp", "fn", "tn")), "pooled OOF confusion != operating_point.json"
    assert abs(pooled["auc"] - op["auc"]) < 1e-12
    print(f"assert: pooled OOF (one held-out checkpoint per image + TTA) reproduces "
          f"models/operating_point.json — tp/fp/fn/tn {op['confusion']}, AUC {op['auc']:.4f} OK")
    f5 = oof["fold"].to_numpy() == CAM_FOLD
    fold5 = describe(y[f5], p[f5], thr)
    return [
        {"cohort": "internal (BUS-BRA)", "predictor": "one held-out ckpt per image + TTA (pooled OOF)", **pooled},
        {"cohort": "internal (BUS-BRA fold 5)", "predictor": "fold-5 single ckpt + TTA (fold-5 OOF)", **fold5},
    ]


def external_rows(cohort: str, temperature: float, thr: float, loco_summary: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    members = pd.read_csv(f"reports/v2_members_{cohort}.csv")
    ext = pd.read_csv(f"reports/external_{cohort}_preds.csv")
    df = members.merge(ext[["image_path", "y_true", "y_prob_raw", "y_prob_calibrated"]],
                       on="image_path", suffixes=("", "_ext"), validate="1:1")
    assert len(df) == len(members) == len(ext)
    assert (df["y_true"] == df["y_true_ext"]).all()
    y = df["y_true"].to_numpy()

    # (b) ensemble: frozen reduction order — per-ckpt (orig+flip)/2, then mean over 5
    pairs = np.stack([(df[f"m{k}_orig"].to_numpy(np.float32) + df[f"m{k}_flip"].to_numpy(np.float32)) / np.float32(2)
                      for k in range(1, 6)], axis=1)
    p_raw_ens = pairs.mean(axis=1, dtype=np.float32)
    assert np.array_equal(p_raw_ens, df["y_prob_raw_recomputed"].to_numpy(np.float32)), \
        f"{cohort}: member reduction does not reproduce y_prob_raw_recomputed"
    max_gap = float(np.abs(p_raw_ens.astype(np.float64) - df["y_prob_raw"].to_numpy(np.float64)).max())
    assert max_gap < 1e-6, f"{cohort}: ensemble from members != saved y_prob_raw (max gap {max_gap:.2e})"
    p_ens = calibrate(p_raw_ens, temperature)
    ens = describe(y, p_ens, thr)
    ens_saved = describe(y, df["y_prob_calibrated"].to_numpy(), thr)
    assert (ens["tp"], ens["fp"], ens["fn"], ens["tn"]) == (ens_saved["tp"], ens_saved["fp"], ens_saved["fn"], ens_saved["tn"]), \
        f"{cohort}: ensemble confusion from members != external-v1 saved decisions"

    # (a) fold-5 single + TTA (Amendment 4 (t) comparator, asserted vs Q1b summary)
    p_single = calibrate(pairs[:, CAM_FOLD - 1], temperature)
    single = describe(y, p_single, thr)
    s = loco_summary[loco_summary["hold_out"] == cohort].iloc[0]
    for got, ref, name in [(single["auc"], s["v1single_auc"], "auc"), (single["spec"], s["v1single_spec"], "spec"),
                           (single["sens"], s["v1single_sens"], "sens"),
                           (single["benign_median_p_cal"], s["v1single_benign_median"], "benign median")]:
        assert abs(got - ref) < 1e-6, f"{cohort}: v1-single {name} {got!r} != reports/v2_loco_summary.csv {ref!r}"
    print(f"assert: {LABEL[cohort]:6s} ensemble-from-members == external-v1 decisions (max raw gap {max_gap:.1e}); "
          f"fold-5 single reproduces v2_loco_summary.csv  OK")

    rows = [
        {"cohort": LABEL[cohort], "predictor": "fold-5 single ckpt + TTA", **single},
        {"cohort": LABEL[cohort], "predictor": "5-ckpt ensemble + TTA (deployed; external-v1)", **ens},
    ]
    member_rows = []
    for k in range(1, 6):
        d = describe(y, calibrate(pairs[:, k - 1], temperature), thr)
        member_rows.append({"cohort": LABEL[cohort], "member": f"cv_vit_fold{k} + TTA", **d})
    member_rows.append({"cohort": LABEL[cohort], "member": "5-ckpt ensemble + TTA", **ens})
    return rows, member_rows


def main() -> None:
    temperature = json.loads(CALIBRATION_JSON.read_text())["temperature"]
    op = json.loads(OPERATING_POINT_JSON.read_text())
    thr = op["threshold"]
    loco_summary = pd.read_csv(LOCO_SUMMARY_CSV)
    print(f"frozen T = {temperature:.4f}, frozen threshold = {thr:.4f} (both applied unchanged)\n")

    rows = internal_rows(temperature, thr, op)
    member_rows = []
    for c in COHORTS:
        r, m = external_rows(c, temperature, thr, loco_summary)
        rows += r
        member_rows += m
    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)
    pd.DataFrame(member_rows).to_csv(OUT_MEMBERS_CSV, index=False)

    print("\nPOST-HOC — predictor-change vs cohort-change (frozen T, frozen threshold 0.2683)")
    print("| cohort | predictor | n | AUC | benign median p_cal | sens | spec |")
    print("|---|---|---|---|---|---|---|")
    for r in table.itertuples():
        print(f"| {r.cohort} | {r.predictor} | {r.n} | {r.auc:.4f} | {r.benign_median_p_cal:.4f} | {r.sens:.4f} | {r.spec:.4f} |")

    print("\nDecomposition per external cohort (benign median p_cal / spec):")
    f5 = table.iloc[1]
    for c in COHORTS:
        a = table[(table["cohort"] == LABEL[c]) & table["predictor"].str.startswith("fold-5")].iloc[0]
        b = table[(table["cohort"] == LABEL[c]) & table["predictor"].str.startswith("5-ckpt")].iloc[0]
        print(f"  {LABEL[c]:6s} cohort change (fold-5 single, internal fold 5 → external): "
              f"benign median {f5.benign_median_p_cal:.4f} → {a.benign_median_p_cal:.4f} ({a.benign_median_p_cal - f5.benign_median_p_cal:+.4f}), "
              f"spec {f5.spec:.4f} → {a.spec:.4f} ({a.spec - f5.spec:+.4f}); "
              f"predictor change (single → ensemble, same cohort): benign median {a.benign_median_p_cal:.4f} → {b.benign_median_p_cal:.4f} "
              f"({b.benign_median_p_cal - a.benign_median_p_cal:+.4f}), spec {a.spec:.4f} → {b.spec:.4f} ({b.spec - a.spec:+.4f})")

    print("\nPer-member specificity at 0.2683 (all five single checkpoints + TTA vs the ensemble):")
    mt = pd.DataFrame(member_rows)
    for c in COHORTS:
        sub = mt[mt["cohort"] == LABEL[c]]
        print(f"  {LABEL[c]:6s} spec " + " / ".join(f"{v:.3f}" for v in sub["spec"]) +
              "   benign median " + " / ".join(f"{v:.3f}" for v in sub["benign_median_p_cal"]) + "  (m1..m5, ensemble)")
    print(f"\nwritten: {OUT_CSV}, {OUT_MEMBERS_CSV}")


if __name__ == "__main__":
    main()

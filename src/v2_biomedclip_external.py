"""Q2c: single-shot external evaluation of the frozen v2 BiomedCLIP pipeline.

Research prototype — not for diagnostic use.

Mirrors external_val.py with only the frozen artifact set replaced:
5 x models/v2_biomedclip_fold{k}.pt + hflip TTA, T and threshold from
models/v2_biomedclip_{calibration,operating_point}.json. Cohort builders,
preprocessing, and bootstrap are imported from external_val unchanged.
external-v1 files are read-only inputs for the comparison table; the v1
side is recomputed from saved reports/external_*_preds.csv and asserted
against the recorded external-v1 AUCs before use.

--self-check exercises the inference.ensemble_tta_probs_from_loader path
on BUS-BRA fold-5 with only the fold-5 ckpt and must reproduce
reports/v2_biomedclip_tta_summary.csv's fold-5 auc_tta digit-for-digit
(the freeze script computed it via evaluate.predict_hflip_pair, so this
cross-checks two implementations, as v1's self-check did).

The external run happens ONCE (Amendment 4 (u)); --confirm required.

Usage: python src/v2_biomedclip_external.py --self-check
       python src/v2_biomedclip_external.py --run --confirm
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from data import build_master_df  # noqa: E402
from evaluate import save_confusion, save_roc  # noqa: E402
from external_val import DATASETS, ensemble_tta_probs, patient_bootstrap  # noqa: E402
from inference import calibrate_probs  # noqa: E402
from pick_threshold import confusion_counts, point_metrics  # noqa: E402

CONFIG = "configs/v2_biomedclip.yaml"
V2_CKPTS = [f"models/v2_biomedclip_fold{k}.pt" for k in (1, 2, 3, 4, 5)]
V2_CALIBRATION = Path("models/v2_biomedclip_calibration.json")
V2_OPERATING_POINT = Path("models/v2_biomedclip_operating_point.json")
V2_TTA_SUMMARY = Path("reports/v2_biomedclip_tta_summary.csv")
V2_OOF = Path("reports/v2_biomedclip_oof_preds.csv")

V1_CALIBRATION = Path("models/calibration.json")
V1_OPERATING_POINT = Path("models/operating_point.json")
V1_OOF = Path("reports/oof_vit_preds.csv")
# RESULTS.md "External validation (single-shot, frozen-v1)" — assertion targets
V1_RECORDED_AUC = {"breast": 0.8542, "busi": 0.9339, "gdph": 0.9154, "sysucc": 0.8380}

COMPARISON_CSV = Path("reports/v2_biomedclip_external_comparison.csv")
SHRINK_FACTOR = 0.70  # criterion (u): |shift_v2| <= 0.70 * |shift_v1|
AUC_MARGIN = 0.01


def benign_median_oof(oof_csv: Path, temperature: float) -> float:
    oof = pd.read_csv(oof_csv)
    p_cal = calibrate_probs(oof["y_prob_tta"].values, temperature)
    return float(np.median(p_cal[oof["y_true"].values == 0]))


def cohort_row(name: str, df: pd.DataFrame, probs: np.ndarray, thr: float,
               benign_median_own_oof: float) -> dict:
    y = df["label"].values
    auc = roc_auc_score(y, probs)
    tp, fp, fn, tn = confusion_counts(y, probs, thr)
    m = point_metrics(tp, fp, fn, tn)
    ci = patient_bootstrap(df, probs, thr)
    benign_median = float(np.median(probs[y == 0]))
    return {
        "cohort": name,
        "n": len(y),
        "auc": auc,
        "auc_lo": ci["auc_ci95"][0],
        "auc_hi": ci["auc_ci95"][1],
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "benign_median_pcal": benign_median,
        "shift": benign_median - benign_median_own_oof,
    }


def self_check(cfg: dict) -> None:
    expected = float(
        pd.read_csv(V2_TTA_SUMMARY).set_index("fold").loc[5, "auc_tta"]
    )
    df = build_master_df(cfg["data"]["root"])
    fold5 = df[df["fold"] == 5]
    raw, labels = ensemble_tta_probs(fold5, [V2_CKPTS[4]], cfg)
    auc = roc_auc_score(labels, raw)
    print(f"self-check: fold-5 val, v2 fold-5 ckpt only, n = {len(labels)}")
    print(f"  expected ({V2_TTA_SUMMARY}):          {expected:.16f}")
    print(f"  reproduced by external code path:     {auc:.16f}")
    if abs(auc - expected) > 1e-12:
        print("  SELF-CHECK FAILED — pipeline bug, do not run external validation.")
        sys.exit(1)
    calib = json.loads(V2_CALIBRATION.read_text())
    op = json.loads(V2_OPERATING_POINT.read_text())
    probs = calibrate_probs(raw, calib["temperature"])
    assert abs(roc_auc_score(labels, probs) - auc) < 1e-10
    n_pos = int((probs >= op["threshold"]).sum())
    print(f"  calibration (T={calib['temperature']:.4f}) + threshold "
          f"({op['threshold']:.4f}) applied: {n_pos}/{len(labels)} flagged malignant")
    print("  SELF-CHECK PASSED — AUC reproduced digit-for-digit.")


def run(cfg: dict) -> None:
    t_v2 = json.loads(V2_CALIBRATION.read_text())["temperature"]
    thr_v2 = json.loads(V2_OPERATING_POINT.read_text())["threshold"]
    t_v1 = json.loads(V1_CALIBRATION.read_text())["temperature"]
    thr_v1 = json.loads(V1_OPERATING_POINT.read_text())["threshold"]
    oof_med_v2 = benign_median_oof(V2_OOF, t_v2)
    oof_med_v1 = benign_median_oof(V1_OOF, t_v1)
    print(f"own-OOF benign median p_cal: v1 {oof_med_v1:.4f} | v2 {oof_med_v2:.4f}")

    reports_dir = Path("reports")
    rows = []
    for name, (builder, patient_level) in DATASETS.items():
        df = builder()

        # v1 side from SAVED preds only (no v1 inference), asserted vs record
        v1_preds = pd.read_csv(reports_dir / f"external_{name}_preds.csv")
        assert len(v1_preds) == len(df)
        v1_df = pd.DataFrame(
            {"patient_id": v1_preds["patient_id"], "label": v1_preds["y_true"]}
        )
        p1 = v1_preds["y_prob_calibrated"].values
        r1 = cohort_row(name, v1_df, p1, thr_v1, oof_med_v1)
        assert abs(r1["auc"] - V1_RECORDED_AUC[name]) < 5e-5, (
            f"{name}: recomputed v1 AUC {r1['auc']:.4f} != recorded "
            f"{V1_RECORDED_AUC[name]} — wrong input files, aborting"
        )

        # v2 side: THE single-shot inference
        raw, labels = ensemble_tta_probs(df, V2_CKPTS, cfg)
        assert (labels == df["label"].values).all()
        p2 = calibrate_probs(raw, t_v2)
        r2 = cohort_row(name, df, p2, thr_v2, oof_med_v2)

        preds_path = reports_dir / f"v2_biomedclip_external_{name}_preds.csv"
        pd.DataFrame(
            {
                "image_path": df["image_path"].values,
                "patient_id": df["patient_id"].values,
                "y_true": labels.astype(int),
                "y_prob_raw": raw,
                "y_prob_calibrated": p2,
                "y_pred": (p2 >= thr_v2).astype(int),
            }
        ).to_csv(preds_path, index=False)
        save_roc(labels, p2, r2["auc"], reports_dir / f"v2_biomedclip_roc_external_{name}.png")
        tp, fp, fn, tn = confusion_counts(labels, p2, thr_v2)
        save_confusion(np.array([[tn, fp], [fn, tp]]),
                       reports_dir / f"v2_biomedclip_cm_external_{name}.png")

        ratio = abs(r2["shift"]) / abs(r1["shift"])
        rows.append({
            **{f"v1_{k}": v for k, v in r1.items() if k not in ("cohort", "n")},
            **{f"v2_{k}": v for k, v in r2.items() if k not in ("cohort", "n")},
            "cohort": name,
            "n": r2["n"],
            "shift_ratio": ratio,
            "delta_auc": r2["auc"] - r1["auc"],
            "bootstrap": "patient-level" if patient_level else "image-level",
        })
        print(f"\n=== {name} (n={r2['n']}, single-shot v2) ===")
        print(f"  AUC:  v1 {r1['auc']:.4f} ({r1['auc_lo']:.4f}-{r1['auc_hi']:.4f}) | "
              f"v2 {r2['auc']:.4f} ({r2['auc_lo']:.4f}-{r2['auc_hi']:.4f}) | "
              f"dAUC {r2['auc'] - r1['auc']:+.4f}")
        print(f"  sens: v1 {r1['sensitivity']:.4f} | v2 {r2['sensitivity']:.4f}   "
              f"spec: v1 {r1['specificity']:.4f} | v2 {r2['specificity']:.4f}")
        print(f"  benign median p_cal: v1 {r1['benign_median_pcal']:.4f} "
              f"(shift {r1['shift']:+.4f}) | v2 {r2['benign_median_pcal']:.4f} "
              f"(shift {r2['shift']:+.4f}) | ratio {ratio:.3f}")

    comp = pd.DataFrame(rows)
    comp.to_csv(COMPARISON_CSV, index=False)

    # ---- Amendment 4 (u) criterion, mechanical
    auc_wins = int((comp["delta_auc"] >= AUC_MARGIN).sum())
    shrink_wins = int((comp["shift_ratio"] <= SHRINK_FACTOR).sum())
    branch_auc = auc_wins >= 3
    branch_shift = shrink_wins >= 3
    met = branch_auc or branch_shift
    print("\n=== Amendment 4 (u) criterion (mechanical) ===")
    print(f"  branch A — v2 AUC >= v1 + {AUC_MARGIN} on >= 3/4: "
          f"{auc_wins}/4 -> {'MET' if branch_auc else 'NOT MET'}")
    print(f"  branch B — |shift_v2| <= {SHRINK_FACTOR}*|shift_v1| on >= 3/4: "
          f"{shrink_wins}/4 -> {'MET' if branch_shift else 'NOT MET'}")
    print(f"  Q2 VERDICT: domain pretraining reduces shift — "
          f"{'CLAIMED' if met else 'NOT CLAIMED'}")
    print(f"  rule (v): Q1 LOCO backbone = "
          f"{'BiomedCLIP init' if met else 'v1 ImageNet vit_base_patch16_224'}")
    print(f"saved: {COMPARISON_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--confirm", action="store_true",
                        help="required: the external run is SINGLE-SHOT")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(CONFIG).read_text())

    if args.self_check:
        self_check(cfg)
        return
    if not args.run:
        parser.error("provide --self-check or --run")
    if not args.confirm:
        parser.error("Q2c is SINGLE-SHOT (Amendment 4 (u)); re-run with --confirm "
                     "only when the one real run is intended")
    run(cfg)


if __name__ == "__main__":
    main()

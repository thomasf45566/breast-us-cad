"""Q2b freeze steps for the v2 BiomedCLIP CV ensemble (Amendment 4 (u)).

Research prototype — not for diagnostic use.

Replicates the v1 post-CV pipeline with only artifact names changed:
  1. pooled OOF hflip-TTA over models/v2_biomedclip_fold{1-5}.pt
     (tta_eval.py structure: evaluate.predict_hflip_pair, one pass/fold),
  2. one temperature fit on pooled OOF TTA probs (calibrate.py convention:
     logit of the TTA-averaged prob, LBFGS NLL),
  3. threshold = highest with sens >= 0.90 on calibrated OOF
     (pick_threshold.py rule + patient-level bootstrap CIs),
then FREEZES models/v2_biomedclip_{calibration,operating_point}.json.
v1 artifacts are never written.

Usage: python src/v2_biomedclip_freeze.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from calibrate import bin_stats, ece, fit_temperature, nll, plot_reliability, probs_to_logits  # noqa: E402
from data import BusDataset, build_master_df, get_transforms  # noqa: E402
from evaluate import compute_metrics, predict_hflip_pair, save_confusion, save_roc  # noqa: E402
from pick_threshold import (  # noqa: E402
    SENS_FLOOR,
    confusion_counts,
    patient_bootstrap_ci,
    pick_operating_point,
    point_metrics,
)
from tta_eval import load_model  # noqa: E402

ALL_FOLDS = [1, 2, 3, 4, 5]
CKPT_PATTERN = "models/v2_biomedclip_fold{k}.pt"
CONFIG = "configs/v2_biomedclip.yaml"
V1_POOLED_OOF_TTA_AUC = 0.9254  # RESULTS.md TTA section, reference only

TTA_SUMMARY_CSV = Path("reports/v2_biomedclip_tta_summary.csv")
OOF_CSV = Path("reports/v2_biomedclip_oof_preds.csv")
CALIBRATION_JSON = Path("models/v2_biomedclip_calibration.json")
OPERATING_POINT_JSON = Path("models/v2_biomedclip_operating_point.json")
RELIABILITY_PNG = Path("reports/v2_biomedclip_reliability.png")
ROC_OP_PNG = Path("reports/v2_biomedclip_roc_oof_operating_point.png")

V1_ARTIFACTS = [
    Path("models/calibration.json"),
    Path("models/operating_point.json"),
    Path("reports/oof_vit_preds.csv"),
]


def oof_tta(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        m_plain, m_tta = compute_metrics(labels, p_orig), compute_metrics(labels, p_tta)
        rows.append(
            {
                "fold": k,
                "n": len(labels),
                "auc_plain": m_plain["auc"],
                "auc_tta": m_tta["auc"],
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
        print(f"fold {k}: n={len(labels)} | AUC plain {m_plain['auc']:.4f} "
              f"| AUC tta {m_tta['auc']:.4f}")
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def main() -> None:
    v1_before = {p: p.stat().st_mtime for p in V1_ARTIFACTS}
    cfg = yaml.safe_load(Path(CONFIG).read_text())
    reports_dir = Path(cfg["paths"]["reports_dir"])

    # ---- 1. pooled OOF hflip TTA
    summary, oof_df = oof_tta(cfg)
    y = oof_df["y_true"].values.astype(float)
    p_raw = oof_df["y_prob_tta"].values
    pooled_plain = roc_auc_score(y, oof_df["y_prob_plain"])
    pooled_tta = roc_auc_score(y, p_raw)
    summary.to_csv(TTA_SUMMARY_CSV, index=False)
    oof_df.to_csv(OOF_CSV, index=False)
    for setting in ("plain", "tta"):
        probs = oof_df[f"y_prob_{setting}"].values
        m = compute_metrics(y, probs)
        save_roc(y, probs, m["auc"], reports_dir / f"v2_biomedclip_roc_oof_{setting}.png")
        save_confusion(m["cm"], reports_dir / f"v2_biomedclip_cm_oof_{setting}.png")
    print(f"\npooled OOF AUC: plain {pooled_plain:.4f} | tta {pooled_tta:.4f} "
          f"(v1 reference tta: {V1_POOLED_OOF_TTA_AUC})")

    # ---- 2. temperature on pooled OOF TTA (calibrate.py convention)
    logits = probs_to_logits(p_raw)
    temperature = fit_temperature(logits, y)
    p_cal = 1 / (1 + np.exp(-logits / temperature))
    assert abs(roc_auc_score(y, p_cal) - pooled_tta) < 1e-10
    stats_raw, stats_cal = bin_stats(y, p_raw), bin_stats(y, p_cal)
    ece_raw, ece_cal = ece(stats_raw, len(y)), ece(stats_cal, len(y))
    nll_raw, nll_cal = nll(y, p_raw), nll(y, p_cal)
    CALIBRATION_JSON.write_text(
        json.dumps(
            {
                "method": "temperature_scaling",
                "temperature": temperature,
                "logit_convention": "logit of TTA-averaged prob, p clipped to [1e-6, 1-1e-6]",
                "fit_on": f"pooled OOF, {OOF_CSV}, column y_prob_tta",
                "n": int(len(y)),
                "ece_15bin_before": ece_raw,
                "ece_15bin_after": ece_cal,
                "nll_before": nll_raw,
                "nll_after": nll_cal,
                "auc": float(pooled_tta),
            },
            indent=2,
        )
        + "\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_reliability(axes[0], stats_raw, "Before calibration (OOF, hflip TTA)", ece_raw)
    plot_reliability(axes[1], stats_cal, f"After temperature scaling (T = {temperature:.3f})", ece_cal)
    fig.suptitle("Reliability — pooled OOF, v2 BiomedCLIP ensemble + hflip TTA. Research use only.")
    fig.tight_layout()
    fig.savefig(RELIABILITY_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"temperature T: {temperature:.4f} | NLL {nll_raw:.4f} -> {nll_cal:.4f} | "
          f"ECE {ece_raw:.4f} -> {ece_cal:.4f}")

    # ---- 3. operating point on calibrated OOF (pick_threshold.py rule)
    thr = pick_operating_point(y, p_cal)
    tp, fp, fn, tn = confusion_counts(y, p_cal, thr)
    metrics = point_metrics(tp, fp, fn, tn)
    ci = patient_bootstrap_ci(oof_df.assign(prob_cal=p_cal), thr)
    OPERATING_POINT_JSON.write_text(
        json.dumps(
            {
                "rule": f"sensitivity >= {SENS_FLOOR} with maximum specificity",
                "threshold": thr,
                "probability_space": "calibrated (temperature-scaled) hflip-TTA ensemble prob",
                "temperature": temperature,
                "calibration_file": str(CALIBRATION_JSON),
                "fit_on": f"pooled OOF, {OOF_CSV}, column y_prob_tta",
                "n_images": int(len(y)),
                "auc": float(pooled_tta),
                "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                **{k: float(v) for k, v in metrics.items()},
                "bootstrap": ci,
            },
            indent=2,
        )
        + "\n"
    )
    fpr, tpr, _ = roc_curve(y, p_cal)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color="#4c72b0", lw=2, label=f"pooled OOF ROC (AUC = {pooled_tta:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.plot(1 - metrics["specificity"], metrics["sensitivity"], "o", color="#c44e52",
            ms=9, zorder=5, label=f"operating point (thr = {thr:.3f})")
    ax.axhline(SENS_FLOOR, color="#c44e52", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("BUS-BRA pooled OOF — v2 BiomedCLIP ensemble + hflip TTA, calibrated\n"
                 "Research use only — not for diagnosis", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROC_OP_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)

    s_lo, s_hi = ci["sensitivity_ci95"]
    sp_lo, sp_hi = ci["specificity_ci95"]
    print(f"threshold:   {thr:.4f}  (rule: sens >= {SENS_FLOOR}, max spec)")
    print(f"sensitivity: {metrics['sensitivity']:.4f}  (95% CI {s_lo:.4f}-{s_hi:.4f})")
    print(f"specificity: {metrics['specificity']:.4f}  (95% CI {sp_lo:.4f}-{sp_hi:.4f})")
    print(f"PPV {metrics['ppv']:.4f} | NPV {metrics['npv']:.4f} | "
          f"confusion tp {tp} fp {fp} fn {fn} tn {tn}")

    for p, mtime in v1_before.items():
        assert p.stat().st_mtime == mtime, f"v1 artifact modified: {p}"
    print("\nv1 artifacts untouched (asserted).")
    print(f"FROZEN: {CALIBRATION_JSON}, {OPERATING_POINT_JSON}")
    print(f"saved: {TTA_SUMMARY_CSV}, {OOF_CSV}, {RELIABILITY_PNG}, {ROC_OP_PNG}")


if __name__ == "__main__":
    main()

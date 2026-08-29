"""External validation of the frozen-v1 model — single-shot, per data/external_protocol.md.

Research prototype — not for diagnostic use.

Frozen inference path (all parameters read from the frozen artifacts):
mean sigmoid prob over 5 checkpoints x {original, hflip} (10 passes/image),
temperature from models/calibration.json applied to the logit of the averaged
prob, decision threshold from models/operating_point.json. Preprocessing is
exactly the BUS-BRA val path (grayscale -> 3ch, resize 224, ImageNet norm).

The external run happens ONCE; --confirm is required so it cannot start by
accident. --self-check exercises the same code path on BUS-BRA fold-5 with
only the fold-5 checkpoint and must reproduce the recorded TTA AUC
digit-for-digit (reports/tta_vit_summary.csv) before any external run.

Usage: python src/external_val.py --self-check
       python src/external_val.py --dataset breast --confirm
       python src/external_val.py --dataset busi --confirm
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from calibrate import probs_to_logits
from data import BusDataset, build_master_df, get_transforms
from evaluate import predict_hflip_pair, save_confusion, save_roc
from pick_threshold import confusion_counts, point_metrics
from torch.utils.data import DataLoader
from tta_eval import load_model

CKPT_PATHS = [f"models/cv_vit_fold{k}.pt" for k in (1, 2, 3, 4, 5)]
CALIBRATION_JSON = Path("models/calibration.json")
OPERATING_POINT_JSON = Path("models/operating_point.json")
BREAST_DIR = Path("data/raw/breast_poland")
BREAST_XLSX = BREAST_DIR / "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
BUSI_CLEAN_CSV = Path("data/splits/busi_clean.csv")
TTA_SUMMARY_CSV = Path("reports/tta_vit_summary.csv")
N_BOOT = 2000
BOOT_SEED = 42


def ensemble_tta_probs(df: pd.DataFrame, ckpt_paths: list[str], cfg: dict):
    """Frozen inference core: per-ckpt hflip-TTA probs, averaged over ckpts.

    Identical code path for self-check (one ckpt) and external run (five).
    Returns (raw ensemble probs, labels).
    """
    device = cfg["train"]["device"]
    loader = DataLoader(
        BusDataset(df, get_transforms("val", cfg["data"]["img_size"])),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    per_model, labels = [], None
    for path in ckpt_paths:
        model = load_model(path, device)
        p_orig, p_flip, labels = predict_hflip_pair(model, loader, device)
        per_model.append((p_orig + p_flip) / 2)
        del model
    return np.mean(per_model, axis=0), labels


def calibrate_probs(raw: np.ndarray, temperature: float) -> np.ndarray:
    return 1 / (1 + np.exp(-probs_to_logits(raw) / temperature))


def build_breast_df() -> pd.DataFrame:
    """BrEaST per protocol: Classification column, normals excluded, main images only."""
    meta = pd.read_excel(BREAST_XLSX)
    meta = meta[meta["Classification"] != "normal"]
    df = pd.DataFrame(
        {
            "image_path": meta["Image_filename"].map(lambda f: str(BREAST_DIR / f)),
            "patient_id": meta["CaseID"],
            "label": meta["Classification"].map({"benign": 0, "malignant": 1}),
        }
    )
    assert df["label"].notna().all()
    return df


def build_busi_df() -> pd.DataFrame:
    """BUSI per protocol: frozen dedup keep-list; NO patient IDs exist, so
    patient_id = filename and the bootstrap is image-level."""
    keep = pd.read_csv(BUSI_CLEAN_CSV)
    return pd.DataFrame(
        {
            "image_path": keep["image_path"],
            "patient_id": keep["filename"],
            "label": keep["label"],
        }
    )


def patient_bootstrap(df, probs, thr, n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """95% percentile CIs for AUC/sens/spec, resampling patient_ids."""
    by_patient = df.reset_index(drop=True).groupby("patient_id").indices
    patients = list(by_patient)
    y = df["label"].values
    rng = np.random.default_rng(seed)
    auc_b, sens_b, spec_b = [], [], []
    for _ in range(n_boot):
        idx = np.concatenate(
            [by_patient[p] for p in rng.choice(patients, len(patients))]
        )
        yb, pb = y[idx], probs[idx]
        if yb.min() == yb.max():
            continue  # degenerate resample, no AUC
        auc_b.append(roc_auc_score(yb, pb))
        tp, fp, fn, tn = confusion_counts(yb, pb, thr)
        sens_b.append(tp / (tp + fn) if tp + fn else np.nan)
        spec_b.append(tn / (tn + fp) if tn + fp else np.nan)
    pct = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    return {
        "n_patients": len(patients),
        "n_boot_valid": len(auc_b),
        "auc_ci95": pct(auc_b),
        "sensitivity_ci95": pct(sens_b),
        "specificity_ci95": pct(spec_b),
    }


def run_external(name: str, df: pd.DataFrame, cfg: dict, patient_level: bool) -> None:
    calib = json.loads(CALIBRATION_JSON.read_text())
    op = json.loads(OPERATING_POINT_JSON.read_text())
    thr = op["threshold"]

    raw, labels = ensemble_tta_probs(df, CKPT_PATHS, cfg)
    assert (labels == df["label"].values).all()
    probs = calibrate_probs(raw, calib["temperature"])

    auc = roc_auc_score(labels, probs)
    tp, fp, fn, tn = confusion_counts(labels, probs, thr)
    m = point_metrics(tp, fp, fn, tn)
    ci = patient_bootstrap(df, probs, thr)
    prevalence = labels.mean()
    boot_kind = "patient-level" if patient_level else "IMAGE-level (no patient IDs published)"

    reports_dir = Path("reports")
    preds_path = reports_dir / f"external_{name}_preds.csv"
    pd.DataFrame(
        {
            "image_path": df["image_path"].values,
            "patient_id": df["patient_id"].values,
            "y_true": labels.astype(int),
            "y_prob_raw": raw,
            "y_prob_calibrated": probs,
            "y_pred": (probs >= thr).astype(int),
        }
    ).to_csv(preds_path, index=False)
    save_roc(labels, probs, auc, reports_dir / f"roc_external_{name}.png")
    save_confusion(np.array([[tn, fp], [fn, tp]]), reports_dir / f"cm_external_{name}.png")

    print(f"\n=== EXTERNAL: {name} (frozen-v1, single-shot) ===")
    print(f"n = {len(labels)} images, {ci['n_patients']} patients | "
          f"prevalence {prevalence:.3f} | bootstrap: {boot_kind}")
    print(f"AUC:         {auc:.4f}  (95% CI {ci['auc_ci95'][0]:.4f}-{ci['auc_ci95'][1]:.4f})")
    print(f"threshold:   {thr:.4f} (frozen, calibrated prob, T={calib['temperature']:.4f})")
    print(f"sensitivity: {m['sensitivity']:.4f}  "
          f"(95% CI {ci['sensitivity_ci95'][0]:.4f}-{ci['sensitivity_ci95'][1]:.4f})")
    print(f"specificity: {m['specificity']:.4f}  "
          f"(95% CI {ci['specificity_ci95'][0]:.4f}-{ci['specificity_ci95'][1]:.4f})")
    print(f"PPV: {m['ppv']:.4f} | NPV: {m['npv']:.4f} (at prevalence {prevalence:.3f})")
    print(f"confusion:   tp {tp} | fp {fp} | fn {fn} | tn {tn}")
    print(f"saved: {preds_path}, reports/roc_external_{name}.png, "
          f"reports/cm_external_{name}.png")


def self_check(cfg: dict) -> None:
    """Same code path, fold-5 ckpt only, BUS-BRA fold-5 val set: the raw TTA
    AUC must reproduce the recorded fold-5 auc_tta digit-for-digit."""
    expected = float(
        pd.read_csv(TTA_SUMMARY_CSV).set_index("fold").loc[5, "auc_tta"]
    )
    df = build_master_df(cfg["data"]["root"])
    fold5 = df[df["fold"] == 5]
    raw, labels = ensemble_tta_probs(fold5, ["models/cv_vit_fold5.pt"], cfg)
    auc = roc_auc_score(labels, raw)
    print(f"self-check: fold-5 val, fold-5 ckpt only, n = {len(labels)}")
    print(f"  expected (reports/tta_vit_summary.csv): {expected:.16f}")
    print(f"  reproduced by external_val code path:   {auc:.16f}")
    if abs(auc - expected) > 1e-12:
        print("  SELF-CHECK FAILED — pipeline bug, do not run external validation.")
        sys.exit(1)
    # calibration + threshold stages execute too, so the whole path is exercised
    calib = json.loads(CALIBRATION_JSON.read_text())
    op = json.loads(OPERATING_POINT_JSON.read_text())
    probs = calibrate_probs(raw, calib["temperature"])
    assert abs(roc_auc_score(labels, probs) - auc) < 1e-10  # monotone
    n_pos = int((probs >= op["threshold"]).sum())
    print(f"  calibration (T={calib['temperature']:.4f}) + threshold "
          f"({op['threshold']:.4f}) applied: {n_pos}/{len(labels)} flagged malignant")
    print("  SELF-CHECK PASSED — AUC reproduced digit-for-digit.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["breast", "busi", "all"])
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="required for external datasets: the external run is SINGLE-SHOT",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("configs/vit.yaml").read_text())

    if args.self_check:
        self_check(cfg)
        return
    if not args.dataset:
        parser.error("provide --self-check or --dataset")
    if not args.confirm:
        parser.error(
            "external validation is SINGLE-SHOT (data/external_protocol.md); "
            "re-run with --confirm only when the one real run is intended"
        )
    if args.dataset in ("breast", "all"):
        run_external("breast", build_breast_df(), cfg, patient_level=True)
    if args.dataset in ("busi", "all"):
        run_external("busi", build_busi_df(), cfg, patient_level=False)


if __name__ == "__main__":
    main()

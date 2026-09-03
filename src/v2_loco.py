"""Q1b: one multi-source LOCO run (Amendment 4 (t)) — train, finalize, evaluate.

Research prototype — not for diagnostic use.

Backbone fixed by rule (v) after the Q2 verdict (NOT MET): v1 ImageNet
vit_base_patch16_224. Data from v2_data.build_multisource_df(hold_out);
hyperparameters identical to v1 (configs/v2_loco.yaml) plus early stopping
on pooled-validation AUC. All selection (early stop, temperature,
threshold) happens on the training-side validation pool only; the held-out
cohort is touched exactly once, in --evaluate.

Stages (run in order; later stages refuse to run out of order):
  --train      train ONE model, save models/v2_loco_{hold_out}.pt
  --finalize   hflip-TTA the pooled val split, fit this run's temperature
               (calibrate.py convention) and sens >= 0.90 threshold, save
               models/v2_loco_{hold_out}_{calibration,operating_point}.json
  --evaluate   SINGLE-SHOT held-out eval (--confirm required): AUC + CI,
               sens/spec at the run threshold, benign median p_cal; the
               v1-single comparator (fold-5 ckpt + TTA) is recomputed from
               the SAVED reports/v2_members_{hold_out}.csv only, per (t);
               appends the row to reports/v2_loco_summary.csv.

Usage: python src/v2_loco.py --hold-out breast --train [--epochs N ...]
       python src/v2_loco.py --hold-out breast --finalize
       python src/v2_loco.py --hold-out breast --evaluate --confirm
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from calibrate import fit_temperature, probs_to_logits  # noqa: E402
from data import BusDataset, get_transforms  # noqa: E402
from evaluate import predict_hflip_pair, save_confusion, save_roc  # noqa: E402
from external_val import DATASETS, patient_bootstrap  # noqa: E402
from inference import calibrate_probs  # noqa: E402
from pick_threshold import SENS_FLOOR, confusion_counts, pick_operating_point, point_metrics  # noqa: E402
from train import build_loss, build_model, build_optimizer, build_scheduler, predict, set_seed  # noqa: E402
from v2_data import EXTERNAL_COHORTS, build_multisource_df, make_loco_dataloaders  # noqa: E402

CONFIG = "configs/v2_loco.yaml"
V1_CALIBRATION = Path("models/calibration.json")
V1_OPERATING_POINT = Path("models/operating_point.json")
SUMMARY_CSV = Path("reports/v2_loco_summary.csv")


def paths_for(hold_out: str) -> dict[str, Path]:
    return {
        "ckpt": Path(f"models/v2_loco_{hold_out}.pt"),
        "calibration": Path(f"models/v2_loco_{hold_out}_calibration.json"),
        "operating_point": Path(f"models/v2_loco_{hold_out}_operating_point.json"),
        "preds": Path(f"reports/v2_loco_{hold_out}_preds.csv"),
        "roc": Path(f"reports/v2_loco_roc_{hold_out}.png"),
        "cm": Path(f"reports/v2_loco_cm_{hold_out}.png"),
    }


def val_loader_for(df: pd.DataFrame, cfg: dict) -> DataLoader:
    """Deterministic loader over the pooled training-side val split."""
    val_df = df[df["split"] == "val"].rename(columns={"y_true": "label"})
    return DataLoader(
        BusDataset(val_df, get_transforms("val", cfg["data"]["img_size"])),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )


def train_stage(hold_out: str, cfg: dict, epochs: int, ckpt_path: Path, run_name: str) -> None:
    device = cfg["train"]["device"]
    patience = cfg["train"]["early_stop_patience"]
    set_seed(cfg["train"]["seed"])

    df = build_multisource_df(hold_out)
    train_loader, val_loader = make_loco_dataloaders(
        df,
        batch_size=cfg["data"]["batch_size"],
        img_size=cfg["data"]["img_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    train_labels = df[df["split"] == "train"]["y_true"]
    pos_weight = float((train_labels == 0).sum() / (train_labels == 1).sum())

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg, pos_weight, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, epochs)

    run = wandb.init(
        project=cfg["wandb"]["project"],
        name=run_name,
        group=cfg["wandb"]["group"],
        config={**cfg, "hold_out": hold_out, "epochs": epochs,
                "pos_weight": pos_weight,
                "n_train": int((df["split"] == "train").sum()),
                "n_val": int((df["split"] == "val").sum())},
    )
    print(f"LOCO hold_out={hold_out}: train {len(train_loader.dataset)} | "
          f"val {len(val_loader.dataset)} | pos_weight {pos_weight:.3f} | "
          f"device {device} | epochs {epochs} (patience {patience})")

    best_auc, best_epoch = 0.0, 0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()
        model.train()
        running, n_seen = 0.0, 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            loss = criterion(model(images).squeeze(1), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(targets)
            n_seen += len(targets)
        scheduler.step()
        train_loss = running / n_seen

        probs, labels = predict(model, val_loader, device)
        val_auc = roc_auc_score(labels, probs)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc,
                   "lr": scheduler.get_last_lr()[0]})
        marker = ""
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save({"model_state": model.state_dict(), "config": cfg,
                        "hold_out": hold_out, "epoch": epoch, "val_auc": val_auc},
                       ckpt_path)
            marker = f" -> saved {ckpt_path}"
        print(f"epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} | "
              f"val_auc {val_auc:.4f} | {time.monotonic() - t0:.1f}s{marker}")
        if epoch - best_epoch >= patience:
            print(f"early stop: no val-AUC improvement in {patience} epochs")
            break

    run.summary["best_val_auc"] = best_auc
    run.summary["best_epoch"] = best_epoch
    run.finish()
    print(f"best val AUC: {best_auc:.4f} (epoch {best_epoch}) | checkpoint: {ckpt_path}")


def tta_probs_single(ckpt_path: Path, loader: DataLoader, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    p_orig, p_flip, labels = predict_hflip_pair(model, loader, device)
    del model
    return (p_orig + p_flip) / 2, labels, ckpt


def finalize_stage(hold_out: str, cfg: dict, p: dict[str, Path]) -> None:
    assert p["ckpt"].exists(), f"train first: {p['ckpt']} missing"
    df = build_multisource_df(hold_out)
    loader = val_loader_for(df, cfg)
    p_tta, y, ckpt = tta_probs_single(p["ckpt"], loader, cfg["train"]["device"])
    y = y.astype(float)

    val_auc_tta = roc_auc_score(y, p_tta)
    temperature = fit_temperature(probs_to_logits(p_tta), y)
    p_cal = calibrate_probs(p_tta, temperature)
    assert abs(roc_auc_score(y, p_cal) - val_auc_tta) < 1e-10
    thr = pick_operating_point(y, p_cal)
    tp, fp, fn, tn = confusion_counts(y, p_cal, thr)
    m = point_metrics(tp, fp, fn, tn)

    p["calibration"].write_text(json.dumps({
        "method": "temperature_scaling",
        "temperature": temperature,
        "logit_convention": "logit of TTA-averaged prob, p clipped to [1e-6, 1-1e-6]",
        "fit_on": f"LOCO hold_out={hold_out} pooled training-side val (hflip TTA)",
        "n": int(len(y)),
        "auc_val_tta": float(val_auc_tta),
        "best_epoch": int(ckpt["epoch"]),
    }, indent=2) + "\n")
    p["operating_point"].write_text(json.dumps({
        "rule": f"sensitivity >= {SENS_FLOOR} with maximum specificity",
        "threshold": thr,
        "probability_space": "calibrated (temperature-scaled) hflip-TTA single-model prob",
        "temperature": temperature,
        "calibration_file": str(p["calibration"]),
        "fit_on": f"LOCO hold_out={hold_out} pooled training-side val",
        "n_images": int(len(y)),
        "auc": float(val_auc_tta),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        **{k: float(v) for k, v in m.items()},
    }, indent=2) + "\n")
    print(f"val (pooled, TTA): AUC {val_auc_tta:.4f} | T {temperature:.4f} | "
          f"thr {thr:.4f} | sens {m['sensitivity']:.4f} | spec {m['specificity']:.4f}")
    print(f"FROZEN for this run: {p['calibration']}, {p['operating_point']}")
    print("LIMITATION (Amendment 4 (t)): external val slices are image-level; "
          "correlated images may sit on both sides, and the pooled mixture's "
          "prevalence matches no single site.")


def evaluate_stage(hold_out: str, cfg: dict, p: dict[str, Path]) -> None:
    for k in ("ckpt", "calibration", "operating_point"):
        assert p[k].exists(), f"missing {p[k]} — run earlier stages first"
    assert not p["preds"].exists(), (
        f"{p['preds']} already exists — the held-out eval is SINGLE-SHOT"
    )
    temperature = json.loads(p["calibration"].read_text())["temperature"]
    thr = json.loads(p["operating_point"].read_text())["threshold"]

    builder, patient_level = DATASETS[hold_out]
    df = builder()
    loader = DataLoader(
        BusDataset(df.rename(columns={"y_true": "label"}), get_transforms("val", cfg["data"]["img_size"])),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    p_tta, labels, _ = tta_probs_single(p["ckpt"], loader, cfg["train"]["device"])
    assert (labels == df["label"].values).all()
    p_cal = calibrate_probs(p_tta, temperature)

    auc = roc_auc_score(labels, p_cal)
    tp, fp, fn, tn = confusion_counts(labels, p_cal, thr)
    m = point_metrics(tp, fp, fn, tn)
    ci = patient_bootstrap(df, p_cal, thr)
    benign_median = float(np.median(p_cal[labels == 0]))

    pd.DataFrame({
        "image_path": df["image_path"].values,
        "patient_id": df["patient_id"].values,
        "y_true": labels.astype(int),
        "y_prob_raw": p_tta,
        "y_prob_calibrated": p_cal,
        "y_pred": (p_cal >= thr).astype(int),
    }).to_csv(p["preds"], index=False)
    save_roc(labels, p_cal, auc, p["roc"])
    save_confusion(np.array([[tn, fp], [fn, tp]]), p["cm"])

    # ---- v1-single comparator from SAVED members only (no v1 inference)
    members = pd.read_csv(f"reports/v2_members_{hold_out}.csv")
    mdf = df.merge(members, on="image_path", suffixes=("", "_m"), validate="1:1")
    assert len(mdf) == len(df) and (mdf["y_true"] == mdf["label"]).all()
    t_v1 = json.loads(V1_CALIBRATION.read_text())["temperature"]
    thr_v1 = json.loads(V1_OPERATING_POINT.read_text())["threshold"]
    p1_raw = (mdf["m5_orig"].values + mdf["m5_flip"].values) / 2
    p1_cal = calibrate_probs(p1_raw, t_v1)
    y1 = mdf["label"].values
    auc1 = roc_auc_score(y1, p1_cal)
    tp1, fp1, fn1, tn1 = confusion_counts(y1, p1_cal, thr_v1)
    m1 = point_metrics(tp1, fp1, fn1, tn1)
    ci1 = patient_bootstrap(mdf, p1_cal, thr_v1)
    benign_median1 = float(np.median(p1_cal[y1 == 0]))

    row = {
        "hold_out": hold_out,
        "n": len(labels),
        "bootstrap": "patient-level" if patient_level else "image-level",
        "loco_auc": auc, "loco_auc_lo": ci["auc_ci95"][0], "loco_auc_hi": ci["auc_ci95"][1],
        "loco_sens": m["sensitivity"], "loco_spec": m["specificity"],
        "loco_benign_median": benign_median,
        "loco_threshold": thr, "loco_temperature": temperature,
        "v1single_auc": auc1, "v1single_auc_lo": ci1["auc_ci95"][0],
        "v1single_auc_hi": ci1["auc_ci95"][1],
        "v1single_sens": m1["sensitivity"], "v1single_spec": m1["specificity"],
        "v1single_benign_median": benign_median1,
        "delta_auc": auc - auc1,
        "delta_spec": m["specificity"] - m1["specificity"],
    }
    header = not SUMMARY_CSV.exists()
    pd.DataFrame([row]).to_csv(SUMMARY_CSV, mode="a", header=header, index=False)

    print(f"\n=== LOCO hold_out={hold_out} — SINGLE-SHOT held-out eval ===")
    print(f"n = {len(labels)} | bootstrap {row['bootstrap']}")
    print(f"LOCO:      AUC {auc:.4f} ({ci['auc_ci95'][0]:.4f}-{ci['auc_ci95'][1]:.4f}) | "
          f"sens {m['sensitivity']:.4f} | spec {m['specificity']:.4f} | "
          f"benign median p_cal {benign_median:.4f} (thr {thr:.4f}, T {temperature:.4f})")
    print(f"v1-single: AUC {auc1:.4f} ({ci1['auc_ci95'][0]:.4f}-{ci1['auc_ci95'][1]:.4f}) | "
          f"sens {m1['sensitivity']:.4f} | spec {m1['specificity']:.4f} | "
          f"benign median p_cal {benign_median1:.4f} (thr {thr_v1:.4f}, T {t_v1:.4f})")
    print(f"delta:     AUC {row['delta_auc']:+.4f} | spec {row['delta_spec']:+.4f}")
    print(f"saved: {p['preds']}, {p['roc']}, {p['cm']}, {SUMMARY_CSV}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold-out", required=True, choices=list(EXTERNAL_COHORTS))
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--confirm", action="store_true",
                    help="required for --evaluate: the held-out eval is SINGLE-SHOT")
    ap.add_argument("--epochs", type=int, help="override config epochs (smoke tests)")
    ap.add_argument("--run-name", help="override wandb run name")
    ap.add_argument("--ckpt-out", help="override checkpoint output path (smoke tests)")
    args = ap.parse_args()
    if sum([args.train, args.finalize, args.evaluate]) != 1:
        ap.error("choose exactly one of --train / --finalize / --evaluate")

    cfg = yaml.safe_load(Path(CONFIG).read_text())
    p = paths_for(args.hold_out)

    if args.train:
        train_stage(
            args.hold_out, cfg,
            epochs=args.epochs or cfg["train"]["epochs"],
            ckpt_path=Path(args.ckpt_out) if args.ckpt_out else p["ckpt"],
            run_name=args.run_name or f"v2_loco_{args.hold_out}",
        )
    elif args.finalize:
        finalize_stage(args.hold_out, cfg, p)
    else:
        if not args.confirm:
            ap.error("the held-out eval is SINGLE-SHOT (Amendment 4 (t)); "
                     "re-run with --confirm only when the one real run is intended")
        evaluate_stage(args.hold_out, cfg, p)


if __name__ == "__main__":
    main()

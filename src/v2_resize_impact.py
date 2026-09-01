"""Deployment impact of the resize-kernel-dispatch finding — read-only diagnostic.

Research prototype — not for diagnostic use.

Quantifies, for the external keep-list images where cv2.resize INTER_LINEAR
and inference.resize_bilinear_frozen disagree (<= 1 uint8 gray level,
size-dependent KleidiCV dispatch — see src/v2_data.py), how much the frozen
classifier's output moves between the two preprocessing paths:

- CANONICAL path (albumentations/cv2) — the path that produced external-v1.
- SPACE path (inference.preprocess_gray, KleidiCV port).

Both paths are recomputed HERE with the frozen 5-ckpt x hflip pipeline and
identical batching, so their difference isolates the resize effect exactly.
Deltas are reported against the SAVED external-v1 y_prob_calibrated
(primary, deployment-relevant) and against the in-script canonical rerun
(pure resize effect). The in-script canonical rerun differs from saved
values only by MPS batch-composition noise (subset batches != the original
full-cohort batches), reported as the noise floor.

Harness sanity, asserted per cohort: for sampled UNAFFECTED images the two
paths produce IDENTICAL tensors and must agree bitwise; the noise floor vs
saved values must be < 1e-6. No frozen artifact, model, or Space code is
modified; the only output is reports/v2_resize_dispatch_impact.csv + stdout.

Usage: python src/v2_resize_impact.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from inference import (
    CKPT_FILES,
    IMG_SIZE,
    calibrate_probs,
    load_classifier,
    predict_hflip_pair,
    preprocess_gray,
    resize_bilinear_frozen,
)

COHORTS = ("breast", "busi", "gdph", "sysucc")
EXPECTED_AFFECTED = {"breast": 99, "busi": 0, "gdph": 309, "sysucc": 1013}
N_SANITY_PER_COHORT = 5
NOISE_FLOOR_TOL = 1e-6  # MPS batch-composition noise vs the saved full-cohort run
OUT_CSV = Path("reports/v2_resize_dispatch_impact.csv")


class SpacePathDataset(Dataset):
    """Images through the Space's preprocessing (inference.preprocess_gray)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        gray = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(row["image_path"])
        return preprocess_gray(gray).squeeze(0), torch.tensor(row["y_true"], dtype=torch.long)


def resize_paths_differ(image_path: str) -> bool:
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    a = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return not (a == resize_bilinear_frozen(rgb)).all()


def ensemble_raw_probs(dataset: Dataset, y_true: np.ndarray, cfg: dict) -> np.ndarray:
    """Frozen 5-ckpt x hflip ensemble mean over the given dataset's tensors."""
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    device = cfg["train"]["device"]
    per_model = []
    for filename in CKPT_FILES:
        model = load_classifier(filename, device)
        p_orig, p_flip, labels = predict_hflip_pair(model, loader, device)
        assert (labels == y_true).all()
        per_model.append((p_orig + p_flip) / 2)
        del model
    return np.mean(per_model, axis=0)


def both_path_raw_probs(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(canonical albumentations path, Space port path), identical batching."""
    from data import BusDataset, get_transforms

    canon_ds = BusDataset(
        df.rename(columns={"y_true": "label"}), get_transforms("val", IMG_SIZE)
    )
    y = df["y_true"].values
    return (
        ensemble_raw_probs(canon_ds, y, cfg),
        ensemble_raw_probs(SpacePathDataset(df), y, cfg),
    )


def main() -> None:
    cfg = yaml.safe_load(Path("configs/vit.yaml").read_text())
    temperature = json.loads(Path("models/calibration.json").read_text())["temperature"]
    threshold = json.loads(Path("models/operating_point.json").read_text())["threshold"]

    frames = []
    for cohort in COHORTS:
        preds = pd.read_csv(f"reports/external_{cohort}_preds.csv")
        affected = preds["image_path"].map(resize_paths_differ)
        assert affected.sum() == EXPECTED_AFFECTED[cohort], (
            f"{cohort}: {affected.sum()} affected != expected {EXPECTED_AFFECTED[cohort]}"
        )
        print(f"[{cohort}] affected {affected.sum()}/{len(preds)}")

        # harness sanity on unaffected images: both paths build IDENTICAL
        # tensors there, so their ensemble probs must agree bitwise; the
        # residual vs saved values is pure MPS batch-composition noise
        clean = preds[~affected]
        if len(clean):
            sanity = clean.sample(
                min(N_SANITY_PER_COHORT, len(clean)), random_state=42
            ).reset_index(drop=True)
            raw_c, raw_s = both_path_raw_probs(sanity, cfg)
            assert (raw_c == raw_s).all(), f"{cohort}: paths differ on identical tensors"
            floor = float(np.abs(raw_c - sanity["y_prob_raw"].values).max())
            assert floor < NOISE_FLOOR_TOL, f"{cohort}: noise floor {floor:.3g} too high"
            print(f"[{cohort}] sanity: {len(sanity)} unaffected images — paths agree "
                  f"bitwise; batch-noise floor vs saved = {floor:.3g}")

        if not affected.any():
            continue
        aff = preds[affected].reset_index(drop=True)
        raw_canon, raw_space = both_path_raw_probs(aff, cfg)
        cal_canon = calibrate_probs(raw_canon, temperature)
        cal_space = calibrate_probs(raw_space, temperature)
        floor_aff = float(np.abs(cal_canon - aff["y_prob_calibrated"].values).max())
        print(f"[{cohort}] canonical rerun vs saved (noise floor, calibrated): "
              f"max = {floor_aff:.3g}")
        frames.append(pd.DataFrame(
            {
                "cohort": cohort,
                "image_path": aff["image_path"],
                "y_true": aff["y_true"],
                "p_cal_canonical": aff["y_prob_calibrated"],
                "p_cal_space": cal_space,
                "abs_delta": np.abs(cal_space - aff["y_prob_calibrated"].values),
                "abs_delta_pure_resize": np.abs(cal_space - cal_canon),
                "pred_canonical": (aff["y_prob_calibrated"].values >= threshold).astype(int),
                "pred_space": (cal_space >= threshold).astype(int),
            }
        ))

    out = pd.concat(frames, ignore_index=True)
    out["decision_flip"] = (out["pred_canonical"] != out["pred_space"]).astype(int)
    out.to_csv(OUT_CSV, index=False)

    print(f"\n=== resize-dispatch impact (frozen pipeline, T={temperature:.4f}, "
          f"thr={threshold:.4f}) ===")
    for cohort, g in out.groupby("cohort"):
        print(f"{cohort}: n={len(g)}  max |dp_cal|={g['abs_delta'].max():.6f}  "
              f"median={g['abs_delta'].median():.6f}  "
              f"(pure-resize max {g['abs_delta_pure_resize'].max():.6f})  "
              f"flips={g['decision_flip'].sum()}")
    print(f"ALL: n={len(out)}  max |dp_cal|={out['abs_delta'].max():.6f}  "
          f"median={out['abs_delta'].median():.6f}  "
          f"(pure-resize max {out['abs_delta_pure_resize'].max():.6f}, "
          f"median {out['abs_delta_pure_resize'].median():.6f})  "
          f"flips={out['decision_flip'].sum()}")
    if out["decision_flip"].any():
        flips = out[out["decision_flip"] == 1]
        print("flipped images:")
        for _, r in flips.iterrows():
            print(f"  {r['cohort']}  {r['image_path']}  "
                  f"p_canonical={r['p_cal_canonical']:.4f} -> p_space={r['p_cal_space']:.4f}")
    print(f"saved: {OUT_CSV}")


if __name__ == "__main__":
    main()

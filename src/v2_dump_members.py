"""Amendment 3 (n): persist the 10 per-member ensemble probabilities — ONLY.

Research prototype — not for diagnostic use.

Authorized constrained re-run of the frozen 5-ckpt x hflip pipeline on the
four external cohorts (data/external_protocol.md Amendment 3 (n)). Sole
purpose: write the 10 per-member sigmoid probabilities per image to
reports/v2_members_{cohort}.csv. Hard guarantee, asserted per image: the
recomputed ensemble mean reproduces the saved y_prob_raw column of the
external-v1 preds CSV bitwise at the stored float32 precision, or to
within 1e-9 absolute in float64. On ANY mismatch: abort, write nothing
further. NO metrics are computed here — no AUC, no sens/spec, no errors.

Usage: python src/v2_dump_members.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader

from data import BusDataset, get_transforms
from external_val import DATASETS
from inference import CKPT_FILES, load_classifier, predict_hflip_pair

REPORTS = Path("reports")
MEMBER_COLS = [f"m{k}_{view}" for k in (1, 2, 3, 4, 5) for view in ("orig", "flip")]


def dump_cohort(name: str, df: pd.DataFrame, cfg: dict) -> None:
    saved = pd.read_csv(
        REPORTS / f"external_{name}_preds.csv", dtype={"y_prob_raw": str}
    )
    # exactly the external-v1 rows, same order — hard requirement
    assert list(saved["image_path"]) == list(df["image_path"]), f"{name}: row mismatch"
    assert (saved["y_true"].values == df["label"].values).all(), f"{name}: label mismatch"

    loader = DataLoader(
        BusDataset(df, get_transforms("val", cfg["data"]["img_size"])),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    device = cfg["train"]["device"]
    members, per_model = {}, []
    for k, filename in enumerate(CKPT_FILES, start=1):
        model = load_classifier(filename, device)
        p_orig, p_flip, labels = predict_hflip_pair(model, loader, device)
        assert (labels == df["label"].values).all()
        members[f"m{k}_orig"] = p_orig
        members[f"m{k}_flip"] = p_flip
        per_model.append((p_orig + p_flip) / 2)  # frozen per-ckpt reduction
        del model
    # exactly inference.ensemble_tta_probs_from_loader's reduction (float32)
    mean = np.mean(per_model, axis=0)

    # reproduction check: bitwise at the CSV's stored float32 precision,
    # with the pre-registered 1e-9 float64 fallback
    saved32 = saved["y_prob_raw"].astype(np.float32).values
    saved64 = saved["y_prob_raw"].astype(np.float64).values
    bitwise = mean == saved32
    diff64 = np.abs(mean.astype(np.float64) - saved64)
    ok = bitwise | (diff64 <= 1e-9)
    print(
        f"[{name}] reproduction vs external_{name}_preds.csv y_prob_raw: "
        f"n={len(mean)}, bitwise@float32 {int(bitwise.sum())}/{len(mean)}, "
        f"max|diff|_float64 = {diff64.max():.3e} "
        f"-> {'PASS' if ok.all() else 'FAIL'}"
    )
    if not ok.all():
        bad = np.flatnonzero(~ok)
        print(f"[{name}] MISMATCH on {len(bad)} images, first rows {bad[:5].tolist()}")
        print(f"[{name}] ABORTING per Amendment 3 (n) — nothing further is written.")
        sys.exit(1)

    out_path = REPORTS / f"v2_members_{name}.csv"
    pd.DataFrame(
        {
            "image_path": df["image_path"].values,
            "patient_id": df["patient_id"].values,
            "y_true": df["label"].values.astype(int),
            **members,
            "y_prob_raw_recomputed": mean,
        }
    ).to_csv(out_path, index=False)
    print(f"[{name}] saved {out_path} ({len(mean)} rows x 10 member columns)")


def main() -> None:
    cfg = yaml.safe_load(Path("configs/vit.yaml").read_text())
    for name, (builder, _) in DATASETS.items():
        dump_cohort(name, builder(), cfg)
    print("all cohorts reproduced — member persistence complete, no metrics computed")


if __name__ == "__main__":
    main()

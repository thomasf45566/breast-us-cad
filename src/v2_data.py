"""v2 line 4 (Amendment 4): multi-source LOCO data layer — splits only, no training.

Research prototype — not for diagnostic use.

build_multisource_df(hold_out) assembles one leave-one-cohort-out training
pool: BUS-BRA official folds 1-4 (train) / fold 5 (val), patient-level, plus
the three non-held-out external cohorts (frozen dedup keep-lists, reused from
external_val.DATASETS) split 85/15. External splits are IMAGE-level (no
patient IDs exist; BrEaST case-level == image-level per protocol (a)) —
a stated limitation of Amendment 4 (t). Each cohort's 15% val slice is drawn
once from seed 42, independently of hold_out, so it is identical across the
four LOCO runs. The held-out cohort is asserted entirely absent.

Preprocessing reuses the frozen pipeline unchanged (data.BusDataset +
data.get_transforms: grayscale imread collapses PNG/RGB/RGBA/BMP alike ->
3-channel replicate -> resize 224 -> ImageNet normalize) — the exact path
that produced external-v1 (reproduced bitwise by v2_dump_members.py).

verify_frozen_preprocessing() checks parity with the deployment path
(inference.preprocess_gray). FINDING (2026-08-31, full sweep of all 2454
external keep-list images): cv2.resize INTER_LINEAR dispatches the KleidiCV
kernel (which inference.resize_bilinear_frozen ports) only for SOME image
sizes; outside that envelope cv2 falls back to OpenCV's own bilinear kernel
and the two resizes differ by at most 1 uint8 gray level (breast 99/252,
busi 0/379, gdph 309/810, sysucc 1013/1013 images affected; BUS-BRA's 713
sizes all match, which is why the port's earlier verification passed). The
verifier therefore asserts the load/format/normalize stages are bitwise
IDENTICAL in both paths and that any tensor difference is exactly the
documented kernel dispatch, bounded at 1 gray level pre-normalization.
Reproduce the sweep with: python src/v2_data.py --sweep
"""

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data import BusDataset, build_master_df, get_transforms
from external_val import DATASETS

EXTERNAL_COHORTS = ("breast", "busi", "gdph", "sysucc")
BUSBRA_VAL_FOLD = 5
VAL_FRACTION = 0.15
SPLIT_SEED = 42
# Only BrEaST publishes patient IDs (one case = one image, protocol (a));
# the keep-list cohorts get patient_id = NaN and image-level handling.
_HAS_PATIENT_IDS = {"breast"}

V2_COLUMNS = ["image_path", "patient_id", "y_true", "cohort", "split"]


def _external_cohort_df(name: str) -> pd.DataFrame:
    """One external cohort (keep-list rows) with its frozen 85/15 split.

    Rows are sorted by image_path before permuting so the split depends only
    on (SPLIT_SEED, cohort), never on builder row order or hold_out.
    """
    builder, _ = DATASETS[name]
    df = builder().sort_values("image_path", kind="stable").reset_index(drop=True)
    n_val = int(round(len(df) * VAL_FRACTION))
    rng = np.random.default_rng([SPLIT_SEED, EXTERNAL_COHORTS.index(name)])
    split = np.full(len(df), "train", dtype=object)
    split[rng.permutation(len(df))[:n_val]] = "val"
    return pd.DataFrame(
        {
            "image_path": df["image_path"],
            "patient_id": df["patient_id"] if name in _HAS_PATIENT_IDS else np.nan,
            "y_true": df["label"].astype(int),
            "cohort": name,
            "split": split,
        }
    )


def build_multisource_df(hold_out: str) -> pd.DataFrame:
    """Training pool for the LOCO run holding out `hold_out` (Amendment 4 (t)).

    Columns [image_path, patient_id, y_true, cohort, split]; cohort is
    'busbra' or an external cohort name; split in {'train', 'val'}.
    """
    if hold_out not in EXTERNAL_COHORTS:
        raise ValueError(f"hold_out must be one of {EXTERNAL_COHORTS}, got {hold_out!r}")

    bus = build_master_df()
    busbra = pd.DataFrame(
        {
            "image_path": bus["image_path"],
            "patient_id": bus["patient_id"],
            "y_true": bus["label"].astype(int),
            "cohort": "busbra",
            "split": np.where(bus["fold"] == BUSBRA_VAL_FOLD, "val", "train"),
        }
    )
    parts = [busbra] + [_external_cohort_df(c) for c in EXTERNAL_COHORTS if c != hold_out]
    df = pd.concat(parts, ignore_index=True)[V2_COLUMNS]

    # Amendment 4 (s): the held-out cohort is never seen in ANY form.
    held_paths = set(DATASETS[hold_out][0]()["image_path"])
    assert hold_out not in set(df["cohort"]), f"held-out cohort {hold_out} present"
    assert set(df["image_path"]).isdisjoint(held_paths), (
        f"held-out {hold_out} images leaked into the pool"
    )
    assert not df["image_path"].duplicated().any()
    assert set(df["split"]) == {"train", "val"}
    assert df["y_true"].isin([0, 1]).all()
    return df


def make_loco_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Train/val loaders over a build_multisource_df frame.

    Train uses the v1 training augmentations, val the frozen val transforms
    (both via data.get_transforms, unchanged).
    """
    train_df = df[df["split"] == "train"].rename(columns={"y_true": "label"})
    val_df = df[df["split"] == "val"].rename(columns={"y_true": "label"})
    train_loader = DataLoader(
        BusDataset(train_df, get_transforms("train", img_size)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        BusDataset(val_df, get_transforms("val", img_size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def _normalize_frozen(rgb_resized: np.ndarray) -> "torch.Tensor":
    """inference.val_transform_rgb's normalize stage, verbatim, minus its resize."""
    import torch

    from data import IMAGENET_MEAN, IMAGENET_STD

    mean = np.array(IMAGENET_MEAN, dtype=np.float32) * 255.0
    denom = np.reciprocal(np.array(IMAGENET_STD, dtype=np.float32) * 255.0)
    img = (rgb_resized.astype(np.float32) - mean) * denom
    return torch.from_numpy(img.transpose(2, 0, 1)).contiguous()


def verify_frozen_preprocessing(
    df: pd.DataFrame, n: int = 20, seed: int = SPLIT_SEED
) -> tuple[int, int]:
    """Assert format-handling parity with the frozen inference path, per image.

    For n random EXTERNAL images (mixed grayscale/RGB/RGBA sources), asserts:
    (1) this layer's tensor == normalize(cv2.resize(gray->3ch)) bitwise — the
        load, channel-replicate, and normalize stages match the path that
        produced external-v1 exactly;
    (2) inference.preprocess_gray == normalize(resize_bilinear_frozen(...))
        bitwise — the deployment path is exactly its documented kernel;
    (3) the two resize kernels differ by <= 1 uint8 gray level, so any
        tensor difference is confined to cv2's size-dependent kernel
        dispatch (see module docstring), never to format handling.
    Returns (n, count of images where the full tensors are bitwise equal).
    """
    import cv2
    import torch

    from inference import IMG_SIZE, preprocess_gray, resize_bilinear_frozen

    ext = df[df["cohort"] != "busbra"].reset_index(drop=True)
    rows = ext.iloc[np.random.default_rng(seed).choice(len(ext), size=n, replace=False)]
    ds = BusDataset(rows.rename(columns={"y_true": "label"}), get_transforms("val"))
    n_bitwise = 0
    for i in range(len(ds)):
        path = ds.df.iloc[i]["image_path"]
        x_here, _ = ds[i]
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        r_cv2 = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        r_frozen = resize_bilinear_frozen(rgb)
        x_frozen = preprocess_gray(gray).squeeze(0)
        assert torch.equal(x_here, _normalize_frozen(r_cv2)), f"format handling: {path}"
        assert torch.equal(x_frozen, _normalize_frozen(r_frozen)), f"inference path: {path}"
        kernel_delta = int(np.abs(r_cv2.astype(int) - r_frozen.astype(int)).max())
        assert kernel_delta <= 1, f"resize kernels differ by {kernel_delta} > 1: {path}"
        n_bitwise += torch.equal(x_here, x_frozen)
    return n, n_bitwise


def kernel_dispatch_sweep() -> None:
    """Full-cohort cv2-vs-frozen-port resize comparison (module-docstring FINDING)."""
    import cv2

    from inference import IMG_SIZE, resize_bilinear_frozen

    for name in EXTERNAL_COHORTS:
        builder, _ = DATASETS[name]
        df = builder()
        n_mismatch, cmax = 0, 0
        for p in df["image_path"]:
            gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            a = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            b = resize_bilinear_frozen(rgb)
            d = int(np.abs(a.astype(int) - b.astype(int)).max())
            n_mismatch += d > 0
            cmax = max(cmax, d)
        print(f"{name}: {n_mismatch}/{len(df)} images differ, max |delta| = {cmax}")


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["cohort", "split"])["y_true"]
    out = g.agg(n="size", n_malignant="sum", prevalence="mean").reset_index()
    out["prevalence"] = out["prevalence"].round(4)
    return out


if __name__ == "__main__":
    import sys

    if "--sweep" in sys.argv:
        kernel_dispatch_sweep()
        sys.exit(0)
    for hold_out in EXTERNAL_COHORTS:
        df = build_multisource_df(hold_out)
        n_train, n_val = (df["split"] == "train").sum(), (df["split"] == "val").sum()
        print(f"\n=== LOCO hold_out={hold_out}: {len(df)} images "
              f"(train {n_train} / val {n_val}) ===")
        print(_summarize(df).to_string(index=False))
        n, n_bitwise = verify_frozen_preprocessing(df)
        print(f"preprocessing parity vs inference.py: {n}/{n} format/normalize "
              f"stages identical; {n_bitwise}/{n} tensors bitwise equal (rest "
              f"differ only by cv2's size-dependent resize kernel, <= 1 gray level)")

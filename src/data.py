"""BUS-BRA data pipeline: master DataFrame, Dataset, and dataloader factory.

Research prototype — not for diagnostic use.

Splits are patient-level via the official BUS-BRA 5-fold assignments
(5-fold-cv.csv, `kFold` column); never split by image.
"""

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LABEL_MAP = {"benign": 0, "malignant": 1}


def build_master_df(root: str | Path = "data/raw/busbra") -> pd.DataFrame:
    """Merge BUS-BRA metadata with the official 5-fold assignments.

    Returns one row per image with columns
    [image_path, mask_path, patient_id, label, birads, fold].
    """
    root = Path(root)
    meta = pd.read_csv(root / "bus_data.csv")
    folds = pd.read_csv(root / "5-fold-cv.csv", usecols=["ID", "kFold"])
    df = meta.merge(folds, on="ID", validate="one_to_one")

    out = pd.DataFrame(
        {
            "image_path": df["ID"].map(lambda i: str(root / "Images" / f"{i}.png")),
            "mask_path": df["ID"].map(
                lambda i: str(root / "Masks" / f"{i.replace('bus_', 'mask_', 1)}.png")
            ),
            "patient_id": df["Case"],
            "label": df["Pathology"].map(LABEL_MAP),
            "birads": df["BIRADS"],
            "fold": df["kFold"],
        }
    )

    if out["label"].isna().any():
        bad = df.loc[out["label"].isna(), "Pathology"].unique()
        raise ValueError(f"Unmapped pathology values: {bad}")
    missing = [p for p in out["image_path"].head(5) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Image files not found, e.g. {missing[0]}")
    return out


class MeanFillCoarseDropout(A.ImageOnlyTransform):
    """CoarseDropout filling holes with the per-image mean intensity."""

    def __init__(
        self,
        min_holes: int = 2,
        max_holes: int = 4,
        hole_size: int = 32,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.min_holes = min_holes
        self.max_holes = max_holes
        self.hole_size = hole_size

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.copy()
        h, w = img.shape[:2]
        fill = img.mean()
        for _ in range(random.randint(self.min_holes, self.max_holes)):
            y = random.randint(0, max(h - self.hole_size, 0))
            x = random.randint(0, max(w - self.hole_size, 0))
            img[y : y + self.hole_size, x : x + self.hole_size] = fill
        return img


def get_transforms(
    split: str, img_size: int = 224, coarse_dropout: dict | None = None
) -> A.Compose:
    """Albumentations pipeline. `split` is 'train' or 'val'/'test'.

    `coarse_dropout` (train only): kwargs for MeanFillCoarseDropout, applied
    after Resize (hole size is at input scale) and before Normalize.
    """
    aug = []
    if split == "train":
        aug = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5
            ),
            A.RandomBrightnessContrast(p=0.5),
        ]
        if coarse_dropout:
            aug.append(MeanFillCoarseDropout(**coarse_dropout))
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            *aug,
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class BusDataset(Dataset):
    """Grayscale ultrasound images replicated to 3 channels for ImageNet backbones."""

    def __init__(self, df: pd.DataFrame, transforms: A.Compose):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(row["image_path"])
        img = np.repeat(img[:, :, None], 3, axis=2)
        img = self.transforms(image=img)["image"]
        return img, torch.tensor(row["label"], dtype=torch.long)


def make_dataloaders(
    df: pd.DataFrame,
    train_folds: list[int],
    val_folds: list[int],
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0,
    coarse_dropout: dict | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders from official fold indices (patient-level splits)."""
    overlap = set(train_folds) & set(val_folds)
    if overlap:
        raise ValueError(f"Folds appear in both train and val: {sorted(overlap)}")

    train_df = df[df["fold"].isin(train_folds)]
    val_df = df[df["fold"].isin(val_folds)]

    train_loader = DataLoader(
        BusDataset(train_df, get_transforms("train", img_size, coarse_dropout)),
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


if __name__ == "__main__":
    df = build_master_df()
    print(f"master df: {len(df)} images, {df['patient_id'].nunique()} patients")
    print(df["label"].value_counts().rename({0: "benign", 1: "malignant"}).to_string())

    train_loader, val_loader = make_dataloaders(df, train_folds=[1, 2, 3, 4], val_folds=[5])
    print(f"train: {len(train_loader.dataset)} images | val: {len(val_loader.dataset)} images")

    images, labels = next(iter(train_loader))
    print(f"batch images: {tuple(images.shape)} dtype={images.dtype}")
    print(f"batch labels: {tuple(labels.shape)} labels[:8]={labels[:8].tolist()}")

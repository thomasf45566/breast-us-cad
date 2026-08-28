"""Grad-CAM quality check: convnext vs vit fold-5 CV checkpoints.

Research prototype — not for diagnostic use.

Picks the same 8 clear-cut fold-5 val images (4 malignant TP, 4 benign TN,
confident under BOTH models) and saves original-vs-overlay grids to
reports/gradcam_check_{convnext,vit}.png. Read-only evaluation on CPU.

Usage: python src/gradcam_check.py
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torch.utils.data import DataLoader

from compare_backbones import load_cpu_model
from data import BusDataset, build_master_df, get_transforms
from train import predict

IMG_SIZE = 224


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """(B, 1+196, C) tokens -> (B, C, 14, 14) spatial map (CLS dropped)."""
    t = tensor[:, 1:, :]
    h = w = int(t.shape[1] ** 0.5)
    return t.reshape(t.size(0), h, w, t.size(2)).permute(0, 3, 1, 2)


def load_rgb01(image_path: str) -> np.ndarray:
    """Raw grayscale image as float RGB in [0, 1], resized to model input."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return np.repeat(img[:, :, None], 3, axis=2).astype(np.float32) / 255.0


def score(model: torch.nn.Module, df) -> np.ndarray:
    loader = DataLoader(
        BusDataset(df, get_transforms("val", IMG_SIZE)),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    probs, _ = predict(model, loader, "cpu")
    return probs


def make_grid(model, target_layers, reshape_transform, rows, out_path, model_label):
    cam = GradCAM(
        model=model, target_layers=target_layers, reshape_transform=reshape_transform
    )
    tf = get_transforms("val", IMG_SIZE)

    fig, axes = plt.subplots(2, len(rows), figsize=(3 * len(rows), 6.5))
    for col, row in enumerate(rows):
        rgb = load_rgb01(row.image_path)
        raw = (rgb * 255).astype(np.uint8)
        x = tf(image=raw)["image"].unsqueeze(0)
        gcam = cam(
            input_tensor=x, targets=[BinaryClassifierOutputTarget(int(row.label))]
        )[0]
        overlay = show_cam_on_image(rgb, gcam, use_rgb=True)

        name = Path(row.image_path).stem
        cls = "malignant TP" if row.label == 1 else "benign TN"
        axes[0, col].imshow(rgb, cmap="gray")
        axes[0, col].set_title(
            f"{name}\n{cls}\ncn {row.p_cn:.3f} | vit {row.p_vit:.3f}", fontsize=8
        )
        axes[1, col].imshow(overlay)
        for r in (0, 1):
            axes[r, col].axis("off")

    fig.suptitle(
        f"Grad-CAM check — {model_label} (fold-5 CV ckpt). Research use only.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/busbra")
    parser.add_argument("--n-per-class", type=int, default=4)
    args = parser.parse_args()

    df = build_master_df(args.data_root)
    val_df = df[df["fold"] == 5].reset_index(drop=True)

    convnext = load_cpu_model("models/cv_convnext_fold5.pt")
    vit = load_cpu_model("models/cv_vit_fold5.pt")

    print(f"scoring {len(val_df)} fold-5 images on CPU...")
    val_df["p_cn"] = score(convnext, val_df)
    val_df["p_vit"] = score(vit, val_df)

    mal = val_df[val_df.label == 1].copy()
    ben = val_df[val_df.label == 0].copy()
    mal["conf"] = mal[["p_cn", "p_vit"]].min(axis=1)  # confidently malignant for both
    ben["conf"] = 1 - ben[["p_cn", "p_vit"]].max(axis=1)  # confidently benign for both
    picks = list(mal.nlargest(args.n_per_class, "conf").itertuples()) + list(
        ben.nlargest(args.n_per_class, "conf").itertuples()
    )

    print("\nselected images (same 8 for both models):")
    for r in picks:
        print(
            f"  {Path(r.image_path).stem} | label={'malignant' if r.label else 'benign'} "
            f"| p_convnext={r.p_cn:.4f} | p_vit={r.p_vit:.4f}"
        )

    Path("reports").mkdir(exist_ok=True)
    make_grid(
        convnext,
        [convnext.stages[-1]],
        None,
        picks,
        "reports/gradcam_check_convnext.png",
        "convnext_small",
    )
    make_grid(
        vit,
        [vit.blocks[-1].norm1],
        vit_reshape_transform,
        picks,
        "reports/gradcam_check_vit.png",
        "vit_base_patch16_224",
    )


if __name__ == "__main__":
    main()

"""Pre-registered adoption evaluation: ViT vs ViT+CoarseDropout.

Research prototype — not for diagnostic use.

Gate 1: pooled OOF AUC(cdrop) >= pooled OOF AUC(plain) - 0.01
        (each fold's val predictions from its own CV checkpoint, concatenated).
Gate 2 (visual, user judges): Grad-CAM at blocks[-2].norm1 on the malignant
        TP set from the interpretability audit — plain vs cdrop, same images.

Usage: python src/adoption_eval.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from compare_backbones import load_cpu_model
from data import BusDataset, build_master_df, get_transforms
from gradcam_check import IMG_SIZE, load_rgb01, vit_reshape_transform
from train import predict

ALL_FOLDS = [1, 2, 3, 4, 5]
# malignant TPs from the interpretability audit (check 1 saturated + check 2 moderate)
TP_IDS = [
    "bus_0450-l", "bus_0758-s", "bus_0370-r", "bus_0980-r",
    "bus_0230-l", "bus_0227-r", "bus_0328-l",
]


def pooled_oof(df, ckpt_pattern: str) -> tuple[np.ndarray, np.ndarray]:
    probs_all, labels_all = [], []
    for k in ALL_FOLDS:
        model = load_cpu_model(ckpt_pattern.format(k=k))
        fold_df = df[df["fold"] == k].reset_index(drop=True)
        loader = DataLoader(
            BusDataset(fold_df, get_transforms("val", IMG_SIZE)),
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )
        probs, labels = predict(model, loader, "cpu")
        probs_all.append(probs)
        labels_all.append(labels)
        del model
    return np.concatenate(probs_all), np.concatenate(labels_all)


def vit_cam_maps(model, rows):
    cam = GradCAM(
        model=model,
        target_layers=[model.blocks[-2].norm1],
        reshape_transform=vit_reshape_transform,
    )
    tf = get_transforms("val", IMG_SIZE)
    maps, probs = [], []
    for row in rows:
        raw = (load_rgb01(row.image_path) * 255).astype(np.uint8)
        x = tf(image=raw)["image"].unsqueeze(0)
        with torch.no_grad():
            probs.append(torch.sigmoid(model(x)).item())
        maps.append(cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)])[0])
    return maps, probs


def main() -> None:
    df = build_master_df()

    print("pooled OOF (plain ViT)...")
    p_plain, y = pooled_oof(df, "models/cv_vit_fold{k}.pt")
    print("pooled OOF (ViT + CoarseDropout)...")
    p_cdrop, y2 = pooled_oof(df, "models/cv_vit_cdrop_fold{k}.pt")
    assert (y == y2).all()

    auc_plain = roc_auc_score(y, p_plain)
    auc_cdrop = roc_auc_score(y, p_cdrop)
    gate1 = auc_cdrop >= auc_plain - 0.01
    print(f"\npooled OOF AUC plain : {auc_plain:.4f}  ({len(y)} images)")
    print(f"pooled OOF AUC cdrop : {auc_cdrop:.4f}")
    print(f"gate 1 (cdrop >= plain - 0.01): {'PASS' if gate1 else 'FAIL'}")

    # ---- saliency comparison on fold-5 checkpoints -------------------------
    rows = [
        df[df["image_path"].str.contains(f"/{i}.png")].iloc[0] for i in TP_IDS
    ]
    rows = [type("R", (), {"image_path": r["image_path"], "label": r["label"]}) for r in rows]
    plain = load_cpu_model("models/cv_vit_fold5.pt")
    cdrop = load_cpu_model("models/cv_vit_cdrop_fold5.pt")
    maps_p, probs_p = vit_cam_maps(plain, rows)
    del plain
    maps_c, probs_c = vit_cam_maps(cdrop, rows)

    fig, axes = plt.subplots(3, len(rows), figsize=(2.6 * len(rows), 9))
    for col, row in enumerate(rows):
        rgb = load_rgb01(row.image_path)
        axes[0, col].imshow(rgb, cmap="gray")
        axes[0, col].set_title(
            f"{Path(row.image_path).stem}\nplain {probs_p[col]:.2f} | "
            f"cdrop {probs_c[col]:.2f}",
            fontsize=8,
        )
        axes[1, col].imshow(show_cam_on_image(rgb, maps_p[col], use_rgb=True))
        axes[2, col].imshow(show_cam_on_image(rgb, maps_c[col], use_rgb=True))
    for r, lab in enumerate(["original", "plain ViT", "ViT + cdrop"]):
        axes[r, 0].set_ylabel(lab, fontsize=10)
        for c in range(len(rows)):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_visible(False)
    fig.suptitle(
        "Adoption gate 2 — Grad-CAM blk[-2].norm1, malignant TPs, fold-5 ckpts. "
        "Research use only.",
        fontsize=11,
    )
    fig.tight_layout()
    out = "reports/gradcam_adoption_vit_cdrop.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

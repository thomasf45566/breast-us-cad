"""Three-part backbone diagnostic before the winner decision.

Research prototype — not for diagnostic use.

1. Saliency re-run on moderate-confidence malignant TPs (0.6 <= p <= 0.95 for
   both models) + the 4 most-confident benign TNs. ConvNeXt Grad-CAM vs ViT
   Grad-CAM (blocks[-2].norm1) and ViT attention rollout.
2. (Artifact audit is done by eye on the raw PNGs — no code here.)
3. Border-occlusion probe: mask outer 15% of each fold-5 malignant image with
   the image mean, report mean probability drop per model.

Read-only on models. Usage: python src/backbone_diagnostic.py
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torch.utils.data import DataLoader

from compare_backbones import load_cpu_model
from data import BusDataset, build_master_df, get_transforms
from gradcam_check import IMG_SIZE, load_rgb01, vit_reshape_transform
from train import predict

BORDER_FRAC = 0.15


def score(model: torch.nn.Module, dataset) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    probs, _ = predict(model, loader, "cpu")
    return probs


class BorderMaskedBusDataset(BusDataset):
    """BusDataset variant that fills the outer border with the image mean."""

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(row["image_path"])
        h, w = img.shape
        bh, bw = int(round(h * BORDER_FRAC)), int(round(w * BORDER_FRAC))
        masked = np.full_like(img, int(round(img.mean())))
        masked[bh : h - bh, bw : w - bw] = img[bh : h - bh, bw : w - bw]
        img = np.repeat(masked[:, :, None], 3, axis=2)
        img = self.transforms(image=img)["image"]
        return img, torch.tensor(row["label"], dtype=torch.long)


def attention_rollout(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Standard attention rollout for a timm ViT; returns a [0,1] map (224x224)."""
    attns = []
    hooks = []
    fused_prev = []
    for blk in model.blocks:
        fused_prev.append(blk.attn.fused_attn)
        blk.attn.fused_attn = False  # materialize attention matrices
        hooks.append(
            blk.attn.attn_drop.register_forward_hook(
                lambda m, i, o: attns.append(o.detach())
            )
        )
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()
        for blk, f in zip(model.blocks, fused_prev):
            blk.attn.fused_attn = f

    result = torch.eye(attns[0].size(-1))
    for a in attns:
        a = a.mean(dim=1)[0]  # average heads -> (N, N)
        a = a + torch.eye(a.size(-1))
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result
    cls_to_patches = result[0, 1:]
    n = int(cls_to_patches.numel() ** 0.5)
    m = cls_to_patches.reshape(n, n).numpy()
    m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m


def cam_maps(model, target_layers, reshape_transform, rows):
    cam = GradCAM(
        model=model, target_layers=target_layers, reshape_transform=reshape_transform
    )
    tf = get_transforms("val", IMG_SIZE)
    out = []
    for row in rows:
        raw = (load_rgb01(row.image_path) * 255).astype(np.uint8)
        x = tf(image=raw)["image"].unsqueeze(0)
        out.append(
            cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(int(row.label))])[0]
        )
    return out


def save_grid(rows, map_rows, row_labels, out_path, title):
    """map_rows: list of (label, [map per image]) drawn under the originals."""
    ncols = len(rows)
    nrows = 1 + len(map_rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 3.1 * nrows))
    for col, row in enumerate(rows):
        rgb = load_rgb01(row.image_path)
        name = Path(row.image_path).stem
        cls = "malig TP" if row.label == 1 else "benign TN"
        axes[0, col].imshow(rgb, cmap="gray")
        axes[0, col].set_title(
            f"{name}\n{cls} | cn {row.p_cn:.2f} vit {row.p_vit:.2f}", fontsize=8
        )
        for r, (_, maps) in enumerate(map_rows, start=1):
            axes[r, col].imshow(show_cam_on_image(rgb, maps[col], use_rgb=True))
    for r in range(nrows):
        label = "original" if r == 0 else map_rows[r - 1][0]
        axes[r, 0].set_ylabel(label, fontsize=9)
        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_visible(False)
    fig.suptitle(f"{title}. Research use only.", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    df = build_master_df()
    val_df = df[df["fold"] == 5].reset_index(drop=True)
    convnext = load_cpu_model("models/cv_convnext_fold5.pt")
    vit = load_cpu_model("models/cv_vit_fold5.pt")

    print(f"scoring {len(val_df)} fold-5 images (original)...")
    val_df["p_cn"] = score(convnext, BusDataset(val_df, get_transforms("val", IMG_SIZE)))
    val_df["p_vit"] = score(vit, BusDataset(val_df, get_transforms("val", IMG_SIZE)))

    # ---- Part 1: moderate-confidence TPs + confident TNs -------------------
    mal = val_df[val_df.label == 1].copy()
    ben = val_df[val_df.label == 0].copy()
    band = mal[(mal.p_cn.between(0.6, 0.95)) & (mal.p_vit.between(0.6, 0.95))].copy()
    band["mean_p"] = band[["p_cn", "p_vit"]].mean(axis=1)
    print(f"\nmalignant with both probs in [0.6, 0.95]: {len(band)} (taking 6)")
    tp6 = band.nlargest(6, "mean_p")
    ben["conf"] = 1 - ben[["p_cn", "p_vit"]].max(axis=1)
    tn4 = ben.nlargest(4, "conf")
    picks = list(tp6.itertuples()) + list(tn4.itertuples())
    for r in picks:
        print(
            f"  {Path(r.image_path).stem} | {'malignant' if r.label else 'benign'} "
            f"| p_cn={r.p_cn:.4f} | p_vit={r.p_vit:.4f}"
        )

    cn_maps = cam_maps(convnext, [convnext.stages[-1]], None, picks)
    save_grid(
        picks,
        [("Grad-CAM stages[-1]", cn_maps)],
        None,
        "reports/gradcam_check2_convnext.png",
        "Diagnostic 2 — convnext_small (fold-5 CV ckpt)",
    )

    vit_gc = cam_maps(vit, [vit.blocks[-2].norm1], vit_reshape_transform, picks)
    tf = get_transforms("val", IMG_SIZE)
    vit_ro = []
    for row in picks:
        raw = (load_rgb01(row.image_path) * 255).astype(np.uint8)
        vit_ro.append(attention_rollout(vit, tf(image=raw)["image"].unsqueeze(0)))
    save_grid(
        picks,
        [("Grad-CAM blk[-2].norm1", vit_gc), ("attention rollout", vit_ro)],
        None,
        "reports/gradcam_check2_vit.png",
        "Diagnostic 2 — vit_base_patch16_224 (fold-5 CV ckpt)",
    )

    # ---- Part 3: border occlusion probe (malignant fold-5) -----------------
    mal_df = mal.reset_index(drop=True)
    print(f"\nocclusion probe: re-scoring {len(mal_df)} malignant images "
          f"with outer {BORDER_FRAC:.0%} border masked...")
    masked_cn = score(convnext, BorderMaskedBusDataset(mal_df, get_transforms("val", IMG_SIZE)))
    masked_vit = score(vit, BorderMaskedBusDataset(mal_df, get_transforms("val", IMG_SIZE)))

    out = pd.DataFrame(
        {
            "image": mal_df["image_path"].map(lambda p: Path(p).stem),
            "p_cn": mal_df["p_cn"],
            "p_cn_masked": masked_cn,
            "p_vit": mal_df["p_vit"],
            "p_vit_masked": masked_vit,
        }
    )
    out["drop_cn"] = out.p_cn - out.p_cn_masked
    out["drop_vit"] = out.p_vit - out.p_vit_masked
    out.to_csv("reports/occlusion_border15.csv", index=False)
    print("saved: reports/occlusion_border15.csv")
    for tag, col in [("convnext", "drop_cn"), ("vit", "drop_vit")]:
        d = out[col]
        print(
            f"{tag:9s} mean drop {d.mean():+.4f} | median {d.median():+.4f} "
            f"| drop>0.2: {(d > 0.2).sum()}/{len(d)}"
        )


if __name__ == "__main__":
    main()

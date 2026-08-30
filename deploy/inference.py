"""Frozen-v1 inference path — the ONLY module the demo app depends on.

Research prototype — not for clinical use. Not a medical device.

Everything needed to run the frozen pipeline lives here: classifier /
segmentation model builders, the 5-ckpt hflip-TTA ensemble, temperature +
threshold loading, Grad-CAM generation, and the exact val preprocessing.
Dependency boundary: torch / timm / segmentation_models_pytorch /
pytorch_grad_cam / cv2 / numpy / huggingface_hub only — no wandb, no
albumentations, no training-side modules. The val transform below is a
plain cv2+numpy re-implementation verified bitwise-identical to the
albumentations pipeline used in training/eval (same cv2.INTER_LINEAR
resize, same float32 (x - 255*mean) * 1/(255*std) normalize).

Weights resolve from models/ when present (repo workflow unchanged),
otherwise from the Hugging Face Hub weights repo with the default local
cache (Spaces workflow).
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224

HF_WEIGHTS_REPO = os.environ.get("BREAST_US_CAD_WEIGHTS_REPO", "happytommy/breast-us-cad-weights")
MODELS_DIR = Path(os.environ.get("BREAST_US_CAD_MODELS_DIR", "models"))

CKPT_FILES = [f"cv_vit_fold{k}.pt" for k in (1, 2, 3, 4, 5)]
CAM_CKPT_FILE = "cv_vit_fold5.pt"  # frozen saliency: fold-5 ckpt @ blocks[-2].norm1
SEG_CKPT_FILE = "seg_unet_effb0.pt"
CALIBRATION_FILE = "calibration.json"
OPERATING_POINT_FILE = "operating_point.json"


# ---------------------------------------------------------------- weights

def resolve_weight(filename: str) -> str:
    """models/<filename> when present, else download from the HF weights repo."""
    local = MODELS_DIR / filename
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_WEIGHTS_REPO, filename=filename)


def load_classifier(filename: str, device: str = "cpu") -> torch.nn.Module:
    """Rebuild a cv_vit checkpoint from its stored config and load its weights."""
    ckpt = torch.load(resolve_weight(filename), map_location="cpu", weights_only=False)
    model = timm.create_model(
        ckpt["config"]["model"]["name"], pretrained=False, num_classes=1
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device)


def load_seg_model(filename: str = SEG_CKPT_FILE, device: str = "cpu") -> torch.nn.Module:
    """U-Net from the segmentation checkpoint's stored config."""
    import segmentation_models_pytorch as smp

    ckpt = torch.load(resolve_weight(filename), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if cfg["model"]["arch"] != "unet":
        raise ValueError(f"Unknown model.arch: {cfg['model']['arch']!r}")
    model = smp.Unet(
        encoder_name=cfg["model"]["encoder"],
        encoder_weights=None,  # state dict below overrides every parameter
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device)


def load_calibration() -> float:
    return json.loads(Path(resolve_weight(CALIBRATION_FILE)).read_text())["temperature"]


def load_operating_point() -> float:
    return json.loads(Path(resolve_weight(OPERATING_POINT_FILE)).read_text())["threshold"]


# ---------------------------------------------------------- preprocessing

def _bilinear_maps(src_len: int, dst_len: int):
    """Per-axis source indices + 8-bit fractions, KleidiCV fixed-point scheme."""
    d = np.arange(dst_len, dtype=np.int64)
    nom = ((d << 16) + 32768) * src_len + (dst_len // 2)
    sfix = nom // dst_len - 32768 + 128  # center-aligned, pre-rounded for 8-bit frac
    s = sfix >> 16
    frac8 = (sfix & 0xFFFF) >> 8
    s0 = np.clip(s, 0, src_len - 1)
    s1 = np.clip(s + 1, 0, src_len - 1)
    return s0, s1, np.where(s < 0, 0, frac8).astype(np.int64)


def _lerp8(a: np.ndarray, b: np.ndarray, w) -> np.ndarray:
    return (a * 256 + (b - a) * w + 128) >> 8


def resize_bilinear_frozen(rgb: np.ndarray, out: int = IMG_SIZE) -> np.ndarray:
    """Deterministic integer bilinear resize — the frozen pipeline's resize.

    Pure-integer port of the Arm KleidiCV bilinear kernel (the HAL that
    serves cv2.resize INTER_LINEAR in the OpenCV build the frozen numbers
    were produced with): 16-bit fixed-point center-aligned coordinates,
    fraction truncated to 8 bits after a +2^7 pre-round, vertical lerp
    with round-half-up to u8, then horizontal. Verified bitwise-identical
    to that cv2.resize on every BUS-BRA image (1879 images, 713 distinct
    sizes) and all bundled examples; unlike cv2.resize, it produces the
    same bits on every platform (cv2's INTER_LINEAR differs between ARM
    and x86 builds, which is why this port exists).
    """
    h, w = rgb.shape[:2]
    sx0, sx1, fx = _bilinear_maps(w, out)
    sy0, sy1, fy = _bilinear_maps(h, out)
    img = rgb.astype(np.int64)
    left = _lerp8(img[sy0][:, sx0, :], img[sy1][:, sx0, :], fy[:, None, None])
    right = _lerp8(img[sy0][:, sx1, :], img[sy1][:, sx1, :], fy[:, None, None])
    return np.clip(_lerp8(left, right, fx[None, :, None]), 0, 255).astype(np.uint8)


def val_transform_rgb(rgb: np.ndarray, img_size: int = IMG_SIZE) -> torch.Tensor:
    """HWC uint8 RGB -> normalized (3, S, S) float32 tensor.

    Bitwise-identical to the training-side albumentations val pipeline
    (Resize INTER_LINEAR -> Normalize(ImageNet) -> ToTensorV2) as realized
    on the machine that produced the frozen numbers.
    """
    img = resize_bilinear_frozen(rgb, img_size)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32) * 255.0
    denom = np.reciprocal(np.array(IMAGENET_STD, dtype=np.float32) * 255.0)
    img = (img.astype(np.float32) - mean) * denom
    return torch.from_numpy(img.transpose(2, 0, 1)).contiguous()


def preprocess_gray(gray: np.ndarray) -> torch.Tensor:
    """Grayscale ultrasound -> 3-channel -> frozen val transform, batched (1,3,S,S)."""
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    return val_transform_rgb(rgb).unsqueeze(0)


# ------------------------------------------------- ensemble + hflip TTA

@torch.no_grad()
def predict_hflip_pair(model, loader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One pass over the loader; returns (probs_orig, probs_hflip, labels).

    TTA = mean of the two prob arrays. Horizontal flip only — vertical would
    break ultrasound depth orientation.
    """
    model.eval()
    p_orig, p_flip, labels = [], [], []
    for images, targets in loader:
        images = images.to(device)
        p_orig.append(torch.sigmoid(model(images).squeeze(1)).cpu().numpy())
        flipped = torch.flip(images, dims=[3])
        p_flip.append(torch.sigmoid(model(flipped).squeeze(1)).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(p_orig), np.concatenate(p_flip), np.concatenate(labels)


def ensemble_tta_probs_from_loader(loader, ckpt_files: list[str], device: str):
    """Frozen inference core: per-ckpt hflip-TTA probs, averaged over ckpts.

    Identical code path for self-check (one ckpt) and external run (five).
    Returns (raw ensemble probs, labels).
    """
    per_model, labels = [], None
    for filename in ckpt_files:
        model = load_classifier(filename, device)
        p_orig, p_flip, labels = predict_hflip_pair(model, loader, device)
        per_model.append((p_orig + p_flip) / 2)
        del model
    return np.mean(per_model, axis=0), labels


@torch.no_grad()
def ensemble_prob(x: torch.Tensor, models: list[torch.nn.Module]) -> float:
    """Single-image frozen decision path: mean sigmoid over ckpts x {orig, hflip}."""
    xs = torch.cat([x, torch.flip(x, dims=[3])])
    per_model = [torch.sigmoid(m(xs).squeeze(1)).mean().item() for m in models]
    return float(np.mean(per_model))


# ------------------------------------------------------------ calibration

def probs_to_logits(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convention: logit of the TTA-averaged probability, log(p/(1-p)) with p
    clipped to [1e-6, 1-1e-6] — NOT the average of the two flips' logits.
    Inference averages sigmoid probs, so calibration sits downstream of that."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibrate_probs(raw: np.ndarray, temperature: float) -> np.ndarray:
    return 1 / (1 + np.exp(-probs_to_logits(raw) / temperature))


# ----------------------------------------------------------- segmentation

@torch.no_grad()
def seg_prob_map(seg_model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """(1,3,S,S) input -> (S,S) sigmoid probability map."""
    return torch.sigmoid(seg_model(x)).squeeze().cpu().numpy()


# --------------------------------------------------------------- Grad-CAM

def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """(B, 1+196, C) tokens -> (B, C, 14, 14) spatial map (CLS dropped)."""
    t = tensor[:, 1:, :]
    h = w = int(t.shape[1] ** 0.5)
    return t.reshape(t.size(0), h, w, t.size(2)).permute(0, 3, 1, 2)


def load_rgb01(image_path: str) -> np.ndarray:
    """Raw grayscale image as float RGB in [0, 1], resized to model input."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    rgb = np.repeat(img[:, :, None], 3, axis=2)
    return resize_bilinear_frozen(rgb).astype(np.float32) / 255.0


def make_cam(model: torch.nn.Module) -> GradCAM:
    """Frozen saliency method: Grad-CAM @ blocks[-2].norm1 of a ViT classifier."""
    return GradCAM(
        model=model,
        target_layers=[model.blocks[-2].norm1],
        reshape_transform=vit_reshape_transform,
    )


def cam_overlay(cam: GradCAM, image_path: str, target_class: int) -> np.ndarray:
    """Grad-CAM overlay (RGB uint8) for one image, targeting `target_class`."""
    rgb = load_rgb01(image_path)
    x = val_transform_rgb((rgb * 255).astype("uint8"))
    gcam = cam(
        input_tensor=x.unsqueeze(0),
        targets=[BinaryClassifierOutputTarget(target_class)],
    )[0]
    return show_cam_on_image(rgb, gcam, use_rgb=True)

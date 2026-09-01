"""Q2a fallback: export the BiomedCLIP ViT-B/16 image encoder to a timm state dict.

Per Amendment 4 (u) + the 2026-09-01 substitution note: USFM did not load
cleanly, so Q2's backbone is BiomedCLIP
(hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224). Its
vision tower is exactly a timm vit_base_patch16_224 (open_clip TimmModel
trunk), so the export is a rename, not a remap. This script applies the
same weight-coverage check and forward-pass checks as src/v2_load_usfm.py
and saves models/pretrained/biomedclip_vitb16_timm.pt for train.py's
model.init_state_dict.

Usage: python src/v2_export_biomedclip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import timm
import torch

sys.path.insert(0, str(Path(__file__).parent))
from inference import load_rgb01, val_transform_rgb  # noqa: E402
from v2_load_usfm import (  # noqa: E402
    MODEL_NAME,
    SAMPLE_IMAGE_GLOB,
    component_report,
    forward_check,
)

HF_REF = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
OUT_PATH = Path("models/pretrained/biomedclip_vitb16_timm.pt")
TRUNK_PREFIX = "visual.trunk."


def main() -> int:
    # Load the checkpoint file directly rather than via
    # open_clip.create_model_and_transforms: building the model would pull in
    # the PubMedBERT text tower (a transformers dependency) that the vision
    # trunk export never touches.
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(
        HF_REF.removeprefix("hf-hub:"), "open_clip_pytorch_model.bin"
    )
    full_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    trunk = {
        k[len(TRUNK_PREFIX):]: v
        for k, v in full_sd.items()
        if k.startswith(TRUNK_PREFIX)
    }
    dropped = [k for k in full_sd if not k.startswith(TRUNK_PREFIX)]
    print(f"source: {HF_REF}")
    print(f"vision-trunk tensors: {len(trunk)} | dropped (text tower + "
          f"projection heads): {len(dropped)}")

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
    model_sd = model.state_dict()
    backbone_keys = [k for k in model_sd if not k.startswith("head.")]

    mapped = {k: v for k, v in trunk.items()
              if k in model_sd and model_sd[k].shape == v.shape}
    unused = sorted(set(trunk) - set(mapped))
    unfilled = sorted(set(backbone_keys) - set(mapped))

    print(f"\ntimm {MODEL_NAME} backbone tensors: {len(backbone_keys)}")
    print(f"loaded from BiomedCLIP trunk: {len(mapped)} "
          f"({100 * len(mapped) / len(backbone_keys):.1f}%)")
    print(f"UNFILLED model tensors ({len(unfilled)}): {unfilled}")
    print(f"UNUSED trunk tensors ({len(unused)}): {unused}")

    comps = component_report(set(mapped), model_sd)
    print("\ncomponent coverage:")
    for name, ok in comps.items():
        print(f"  {name}: {'LOADED' if ok else 'MISSING'}")
    clean = all(comps.values()) and not unfilled and not unused
    print(f"\nweight-coverage check: {'CLEAN' if clean else 'NOT CLEAN'}")
    if not clean:
        return 1

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    assert missing == ["head.weight", "head.bias"] and not unexpected, (missing, unexpected)

    sample = sorted(Path().glob(SAMPLE_IMAGE_GLOB))[0]
    x = val_transform_rgb(load_rgb01(str(sample))).unsqueeze(0)
    print(f"\nsample image: {sample}")
    feats_cpu = forward_check(model, x, "cpu")
    print(f"CPU forward_features: shape {feats_cpu.shape}, no NaNs, "
          f"mean {feats_cpu.mean():.4f}, std {feats_cpu.std():.4f}")
    if torch.backends.mps.is_available():
        feats_mps = forward_check(model, x, "mps")
        print(f"MPS forward_features: no NaNs, max |MPS-CPU| "
              f"{np.abs(feats_mps - feats_cpu).max():.2e}")

    imagenet = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1)
    feats_in = forward_check(imagenet, x, "cpu")
    cls_b, cls_in = feats_cpu[0, 0], feats_in[0, 0]
    cos = float(np.dot(cls_b, cls_in) / (np.linalg.norm(cls_b) * np.linalg.norm(cls_in)))
    print(f"vs ImageNet init: CLS-feature cosine {cos:.4f} "
          f"(weights {'DIFFER — BiomedCLIP actually loaded' if cos < 0.99 else 'IDENTICAL?!'})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mapped, OUT_PATH)
    print(f"\nsaved {len(mapped)} tensors -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

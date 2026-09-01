"""Q2a feasibility spike: load USFM pretrained weights into timm vit_base_patch16_224.

Amendment 4 (u): loading must be verified by a stated weight-coverage check
before any training. This script
  1. reads the USFM checkpoint (models/pretrained/USFM_latest.pth),
  2. maps its keys onto timm's vit_base_patch16_224 skeleton,
  3. prints the coverage report: matched / unmatched checkpoint keys,
     unfilled model tensors, and the four named components
     (patch embed, pos embed, all transformer blocks, final norm),
  4. runs one forward pass on CPU and one on MPS with a sample image:
     NaN check + feature distance vs an ImageNet-initialized ViT.

Exit code 0 = clean load per the criteria above; 1 = not clean.

Usage: python src/v2_load_usfm.py [--ckpt models/pretrained/USFM_latest.pth]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import timm
import torch

sys.path.insert(0, str(Path(__file__).parent))
from inference import load_rgb01, val_transform_rgb  # noqa: E402

MODEL_NAME = "vit_base_patch16_224"
DEFAULT_CKPT = "models/pretrained/USFM_latest.pth"
SAMPLE_IMAGE_GLOB = "data/raw/busbra/Images/*.png"

STRIP_PREFIXES = ("module.", "backbone.", "encoder.", "model.")


def unwrap_checkpoint(path: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "module"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
    state = {}
    for k, v in ckpt.items():
        if not torch.is_tensor(v):
            continue
        for p in STRIP_PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
        state[k] = v
    return state


def map_usfm_to_timm(
    usfm: dict[str, torch.Tensor], model_sd: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Return (mapped state, consumed checkpoint keys, notes)."""
    mapped: dict[str, torch.Tensor] = {}
    consumed: list[str] = []
    notes: list[str] = []

    # 1) identical names with identical shapes
    for k, v in usfm.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            mapped[k] = v
            consumed.append(k)

    # 2) BEiT-style separate q/v biases -> fused qkv.bias (k-bias fixed at 0)
    n_blocks = len({k.split(".")[1] for k in model_sd if k.startswith("blocks.")})
    for i in range(n_blocks):
        qb, vb = f"blocks.{i}.attn.q_bias", f"blocks.{i}.attn.v_bias"
        dst = f"blocks.{i}.attn.qkv.bias"
        if qb in usfm and vb in usfm and dst in model_sd and dst not in mapped:
            fused = torch.cat([usfm[qb], torch.zeros_like(usfm[qb]), usfm[vb]])
            if fused.shape == model_sd[dst].shape:
                mapped[dst] = fused
                consumed += [qb, vb]
                notes.append(f"{dst}: fused from q_bias + zeros(k) + v_bias")

    # 3) BEiT LayerScale gamma_1/gamma_2 -> timm ls1/ls2 (exist only if the
    #    skeleton was built with init_values; vanilla vit_base has neither)
    for i in range(n_blocks):
        for g, ls in ((f"blocks.{i}.gamma_1", f"blocks.{i}.ls1.gamma"),
                      (f"blocks.{i}.gamma_2", f"blocks.{i}.ls2.gamma")):
            if g in usfm and ls in model_sd and model_sd[ls].shape == usfm[g].shape:
                mapped[ls] = usfm[g]
                consumed.append(g)
                notes.append(f"{ls}: from {g}")

    # 4) BEiT mean-pooling final norm (fc_norm) -> timm final norm
    if "fc_norm.weight" in usfm and "norm.weight" in model_sd and "norm.weight" not in mapped:
        if usfm["fc_norm.weight"].shape == model_sd["norm.weight"].shape:
            mapped["norm.weight"] = usfm["fc_norm.weight"]
            mapped["norm.bias"] = usfm["fc_norm.bias"]
            consumed += ["fc_norm.weight", "fc_norm.bias"]
            notes.append("norm.{weight,bias}: from fc_norm (BEiT mean pooling)")

    return mapped, consumed, notes


def component_report(mapped: set[str], model_sd: dict[str, torch.Tensor]) -> dict[str, bool]:
    def full(prefix: str) -> bool:
        keys = [k for k in model_sd if k.startswith(prefix)]
        return bool(keys) and all(k in mapped for k in keys)

    n_blocks = len({k.split(".")[1] for k in model_sd if k.startswith("blocks.")})
    return {
        "patch_embed": full("patch_embed."),
        "pos_embed": "pos_embed" in mapped,
        "cls_token": "cls_token" in mapped,
        "all_blocks": all(full(f"blocks.{i}.") for i in range(n_blocks)),
        "final_norm": full("norm."),
    }


def forward_check(model: torch.nn.Module, x: torch.Tensor, device: str) -> np.ndarray:
    model = model.to(device).eval()
    with torch.no_grad():
        feats = model.forward_features(x.to(device))
    out = feats.float().cpu().numpy()
    assert not np.isnan(out).any(), f"NaNs in forward_features on {device}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    args = ap.parse_args()

    usfm = unwrap_checkpoint(args.ckpt)
    print(f"checkpoint: {args.ckpt}")
    print(f"checkpoint tensors after unwrap/prefix-strip: {len(usfm)}")

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
    model_sd = model.state_dict()
    backbone_keys = [k for k in model_sd if not k.startswith("head.")]

    mapped, consumed, notes = map_usfm_to_timm(usfm, model_sd)
    unused_ckpt = sorted(set(usfm) - set(consumed))
    unfilled = sorted(set(backbone_keys) - set(mapped))

    print(f"\ntimm {MODEL_NAME} backbone tensors: {len(backbone_keys)}")
    print(f"loaded from USFM: {len(mapped)} "
          f"({100 * len(mapped) / len(backbone_keys):.1f}% of backbone tensors)")
    for n in notes[:5]:
        print(f"  note: {n}")
    if len(notes) > 5:
        print(f"  ... {len(notes) - 5} more remapping notes")

    print(f"\nUNFILLED model tensors ({len(unfilled)}):")
    for k in unfilled:
        print(f"  {k}  {tuple(model_sd[k].shape)}")
    print(f"\nUNUSED checkpoint tensors ({len(unused_ckpt)}):")
    for k in unused_ckpt:
        print(f"  {k}  {tuple(usfm[k].shape)}")

    comps = component_report(set(mapped), model_sd)
    print("\ncomponent coverage:")
    for name, ok in comps.items():
        print(f"  {name}: {'LOADED' if ok else 'MISSING'}")

    clean = all(comps.values())
    print(f"\nweight-coverage check: {'CLEAN' if clean else 'NOT CLEAN'}")

    # forward passes regardless of verdict — evidence for the report
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    assert not unexpected, f"unexpected keys slipped through mapping: {unexpected}"
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
    else:
        print("MPS unavailable — skipped")

    imagenet = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1)
    feats_in = forward_check(imagenet, x, "cpu")
    cls_usfm, cls_in = feats_cpu[0, 0], feats_in[0, 0]
    cos = float(np.dot(cls_usfm, cls_in) / (np.linalg.norm(cls_usfm) * np.linalg.norm(cls_in)))
    print(f"\nvs ImageNet init: CLS-feature cosine {cos:.4f}, "
          f"L2 dist {np.linalg.norm(cls_usfm - cls_in):.2f} "
          f"(weights {'DIFFER — USFM actually loaded' if cos < 0.99 else 'IDENTICAL?!'})")

    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())

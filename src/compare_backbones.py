"""Final backbone comparison: CV AUC stats + params + CPU latency.

Research prototype — not for diagnostic use.

Usage: python src/compare_backbones.py
Appends the table to RESULTS.md (re-running appends again).
"""

import argparse
import statistics
import time
from pathlib import Path

import pandas as pd
import timm
import torch

DEFAULT_ENTRIES = [
    ("effnet_b0", "models/cv_effb0_fold1.pt", "reports/cv_summary.csv"),
    ("convnext_small", "models/cv_convnext_fold1.pt", "reports/cv_convnext_summary.csv"),
    ("vit_base_patch16_224", "models/cv_vit_fold1.pt", "reports/cv_vit_summary.csv"),
]


def auc_stats(summary_csv: str) -> dict:
    """Mean ± sd and min-max range over the fold rows of a CV summary CSV."""
    df = pd.read_csv(summary_csv)
    aucs = df[~df["fold"].isin(["mean", "sd"])]["auc"].astype(float)
    return {
        "mean": aucs.mean(),
        "sd": aucs.std(ddof=1),
        "lo": aucs.min(),
        "hi": aucs.max(),
    }


def load_cpu_model(ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    name = ckpt["config"]["model"]["name"]
    model = timm.create_model(name, pretrained=False, num_classes=1)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def median_cpu_latency_ms(model: torch.nn.Module, runs: int, img_size: int = 224) -> float:
    torch.manual_seed(0)
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-md", default="RESULTS.md")
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    rows = []
    for name, ckpt_path, summary_csv in DEFAULT_ENTRIES:
        s = auc_stats(summary_csv)
        model = load_cpu_model(ckpt_path)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        latency = median_cpu_latency_ms(model, args.runs)
        rows.append(
            f"| {name} | {s['mean']:.4f} ± {s['sd']:.4f} "
            f"| {s['lo']:.4f}–{s['hi']:.4f} | {params_m:.1f} | {latency:.0f} |"
        )
        del model

    cn, vit = auc_stats(DEFAULT_ENTRIES[1][2]), auc_stats(DEFAULT_ENTRIES[2][2])
    gap = abs(cn["mean"] - vit["mean"])
    note = (
        f"ConvNeXt–ViT mean AUC gap: {gap:.4f} — "
        f"{'exceeds' if gap > min(cn['sd'], vit['sd']) else 'well below'} both models' "
        f"fold SD (convnext {cn['sd']:.4f}, vit {vit['sd']:.4f})."
    )

    table = "\n".join(
        [
            "| Model | CV AUC mean ± SD | Per-fold AUC range | Params (M) "
            f"| CPU inference (ms, median of {args.runs}) |",
            "|---|---|---|---|---|",
            *rows,
        ]
    )
    block = f"\n### Final comparison (single-image CPU latency, 224×224)\n\n{table}\n\n{note}\n"
    print(block)
    with open(args.results_md, "a") as f:
        f.write(block)
    print(f"appended to {args.results_md}")


if __name__ == "__main__":
    main()

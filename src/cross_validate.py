"""5-fold cross-validation over the official BUS-BRA folds.

Research prototype — not for diagnostic use.

Usage: python src/cross_validate.py --config configs/baseline.yaml
       python src/cross_validate.py --folds 1 --epochs 1   # smoke test
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data import BusDataset, build_master_df, get_transforms
from evaluate import compute_metrics
from train import build_model, predict, train_one_fold

ALL_FOLDS = [1, 2, 3, 4, 5]


def eval_fold(cfg: dict, ckpt_path: str, val_folds: list[int]) -> dict:
    """Load the fold's best checkpoint and score it on the held-out fold."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = cfg["train"]["device"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    df = build_master_df(cfg["data"]["root"])
    val_df = df[df["fold"].isin(val_folds)]
    loader = DataLoader(
        BusDataset(val_df, get_transforms("val", cfg["data"]["img_size"])),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    probs, labels = predict(model, loader, device)
    return compute_metrics(labels, probs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--folds", type=int, nargs="+", default=ALL_FOLDS)
    parser.add_argument("--epochs", type=int, help="override config epochs (smoke tests)")
    parser.add_argument("--prefix", default="cv_effb0")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    epochs = args.epochs or cfg["train"]["epochs"]

    rows = []
    for k in args.folds:
        train_folds = [f for f in ALL_FOLDS if f != k]
        run_name = f"{args.prefix}_fold{k}"
        ckpt_path = f"models/{args.prefix}_fold{k}.pt"
        print(f"\n=== fold {k}: train {train_folds}, val [{k}] ===")
        result = train_one_fold(
            cfg,
            train_folds=train_folds,
            val_folds=[k],
            epochs=epochs,
            run_name=run_name,
            ckpt_path=ckpt_path,
            group=args.prefix,
        )
        m = eval_fold(cfg, result["ckpt_path"], [k])
        rows.append(
            {
                "fold": k,
                "auc": m["auc"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "threshold": m["threshold"],
                "best_epoch": result["best_epoch"],
                "ckpt": result["ckpt_path"],
            }
        )

    summary = pd.DataFrame(rows)
    metrics = ["auc", "sensitivity", "specificity"]
    stats = pd.DataFrame(
        {
            "fold": ["mean", "sd"],
            **{c: [summary[c].mean(), summary[c].std(ddof=1)] for c in metrics},
        }
    )
    out = pd.concat([summary, stats], ignore_index=True)

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"{args.prefix}_summary.csv"
    out.to_csv(out_path, index=False)

    print("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for c in metrics:
        print(f"{c}: {summary[c].mean():.4f} ± {summary[c].std(ddof=1):.4f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

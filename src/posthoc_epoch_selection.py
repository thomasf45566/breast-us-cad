"""POST-HOC: epoch-selection sensitivity of the v1 cv_vit CV AUCs.

Research prototype — not for diagnostic use.

Responds to audit 2026-09-06 §1: train.py saves the checkpoint at the epoch
with the best AUC on the held-out fold and cross_validate.py reports that
same fold's AUC, so every per-fold CV AUC is a max-over-epochs on the fold
it is reported on. This script reads the five cv_vit wandb run histories
(local wandb/run-*/run-*.wandb datastores — no retraining, no inference)
and tabulates, per fold: best-epoch val AUC (the reported value), final
epoch (30) val AUC, and val AUC at ONE fixed epoch chosen as the median of
the five best epochs. Optimism = best − fixed. The best-epoch column is
asserted to reproduce reports/cv_vit_summary.csv.

Usage: python src/posthoc_epoch_selection.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

WANDB_DIR = Path("wandb")
SUMMARY_CSV = Path("reports/cv_vit_summary.csv")
OUT_CSV = Path("reports/posthoc_epoch_selection.csv")
CV_VIT_ARGS = ["--config", "configs/vit.yaml", "--prefix", "cv_vit"]


def find_cv_vit_runs() -> dict[int, Path]:
    """Map val fold -> run directory for the five cross_validate cv_vit runs."""
    runs: dict[int, Path] = {}
    for run_dir in sorted(WANDB_DIR.glob("run-*")):
        meta_path = run_dir / "files" / "wandb-metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("args") != CV_VIT_ARGS:
            continue
        import yaml  # local import: only needed for the matching runs

        cfg = yaml.safe_load((run_dir / "files" / "config.yaml").read_text())
        (fold,) = cfg["val_folds"]["value"]
        assert fold not in runs, f"two cv_vit runs claim val fold {fold}"
        runs[fold] = run_dir
    assert sorted(runs) == [1, 2, 3, 4, 5], f"expected 5 cv_vit runs, found {sorted(runs)}"
    return runs


def read_history(run_dir: Path) -> pd.DataFrame:
    """Per-epoch logged metrics from the run's local .wandb datastore."""
    (store_path,) = run_dir.glob("run-*.wandb")
    store = datastore.DataStore()
    store.open_for_scan(str(store_path))
    rows = []
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in record.history.item:
            key = item.key or "/".join(item.nested_key)
            row[key] = json.loads(item.value_json)
        rows.append(row)
    hist = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    assert list(hist["epoch"]) == list(range(1, len(hist) + 1)), "non-contiguous epochs"
    return hist[["epoch", "val_auc"]]


def tabulate(histories: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    best_epochs = {k: int(h.loc[h["val_auc"].idxmax(), "epoch"]) for k, h in histories.items()}
    fixed_epoch = int(np.median(list(best_epochs.values())))
    rows = []
    for k, h in sorted(histories.items()):
        auc_at = dict(zip(h["epoch"], h["val_auc"]))
        best = h["val_auc"].max()
        rows.append(
            {
                "fold": k,
                "n_epochs": len(h),
                "best_epoch": best_epochs[k],
                "auc_best_epoch": best,
                "auc_final_epoch": auc_at[len(h)],
                "fixed_epoch": fixed_epoch,
                "auc_fixed_epoch": auc_at[fixed_epoch],
                "optimism_best_minus_fixed": best - auc_at[fixed_epoch],
                "optimism_best_minus_final": best - auc_at[len(h)],
            }
        )
    return pd.DataFrame(rows), fixed_epoch


def assert_reproduces_summary(table: pd.DataFrame) -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    summary = summary[summary["fold"].astype(str).str.isdigit()].astype({"fold": int})
    merged = table.merge(summary[["fold", "auc", "best_epoch"]], on="fold", suffixes=("", "_csv"), validate="1:1")
    max_delta = (merged["auc_best_epoch"] - merged["auc"]).abs().max()
    assert max_delta < 1e-12, f"best-epoch AUC does not reproduce {SUMMARY_CSV} (max |Δ| {max_delta})"
    assert (merged["best_epoch"] == merged["best_epoch_csv"].astype(int)).all(), "best_epoch mismatch"
    print(f"assert: best-epoch AUC and best_epoch reproduce {SUMMARY_CSV} (max |Δ| {max_delta:.1e}) — OK")


def mean_sd(values: pd.Series) -> str:
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"


def main() -> None:
    runs = find_cv_vit_runs()
    histories = {k: read_history(d) for k, d in runs.items()}
    table, fixed_epoch = tabulate(histories)
    assert_reproduces_summary(table)

    OUT_CSV.parent.mkdir(exist_ok=True)
    table.to_csv(OUT_CSV, index=False)

    print("\nPOST-HOC — epoch-selection sensitivity (cv_vit, val AUC on the held-out fold)")
    print(f"fixed epoch = median of the five best epochs {sorted(table['best_epoch'])} = {fixed_epoch}")
    print("| fold | best epoch | AUC @ best epoch (reported) | AUC @ epoch 30 | AUC @ fixed epoch | optimism (best − fixed) |")
    print("|---|---|---|---|---|---|")
    for r in table.itertuples():
        print(
            f"| {r.fold} | {r.best_epoch} | {r.auc_best_epoch:.4f} | {r.auc_final_epoch:.4f} | "
            f"{r.auc_fixed_epoch:.4f} | {r.optimism_best_minus_fixed:+.4f} |"
        )
    print(
        f"| **mean ± SD** | — | {mean_sd(table['auc_best_epoch'])} | {mean_sd(table['auc_final_epoch'])} | "
        f"{mean_sd(table['auc_fixed_epoch'])} | {mean_sd(table['optimism_best_minus_fixed'])} |"
    )
    print(f"optimism best − final: {mean_sd(table['optimism_best_minus_final'])}")
    print(f"written: {OUT_CSV}")


if __name__ == "__main__":
    main()

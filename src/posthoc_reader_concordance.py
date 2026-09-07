"""POST-HOC: model false positives vs radiologist BI-RADS calls (GDPH, SYSUCC).

Research prototype — not for diagnostic use.

Descriptive only, from SAVED files: reports/external_{gdph,sysucc}_preds.csv
joined 1:1 with the two reader columns of BIRADS&FOLD.xlsx (same join and
≥ 4a normalization as src/birads_comparison.py). On BENIGN images, cross-tab
the frozen model decision (thr 0.2683) against each reader's ≥ 4a call and
report the fraction of model false positives that the reader also called
≥ 4a, next to the same fraction among model true negatives. No per-image
concordance was pre-registered; this is not a test of anything.

Usage: python src/posthoc_reader_concordance.py
"""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # birads_comparison imports pyplot

import pandas as pd

from birads_comparison import COHORTS, POSITIVE, READERS, VALID, load_cohort

OUT_CSV = Path("reports/posthoc_reader_concordance.csv")


def concordance_rows(name: str, df: pd.DataFrame) -> list[dict]:
    benign = df[df["y_true"] == 0]
    rows = []
    for reader in READERS:
        sub = benign[benign[reader].isin(VALID)]
        reader_pos = sub[reader].isin(POSITIVE)
        model_fp = sub["y_pred"] == 1
        rows.append(
            {
                "cohort": name,
                "reader": reader,
                "n_benign": len(sub),
                "n_model_fp": int(model_fp.sum()),
                "n_model_tn": int((~model_fp).sum()),
                "reader_ge4a_among_model_fp": float(reader_pos[model_fp].mean()),
                "reader_ge4a_among_model_tn": float(reader_pos[~model_fp].mean()),
                "reader_ge4a_all_benign": float(reader_pos.mean()),
                "n_fp_and_reader_ge4a": int((model_fp & reader_pos).sum()),
                "n_fp_and_reader_lt4a": int((model_fp & ~reader_pos).sum()),
                "n_tn_and_reader_ge4a": int((~model_fp & reader_pos).sum()),
                "n_tn_and_reader_lt4a": int((~model_fp & ~reader_pos).sum()),
            }
        )
    return rows


def main() -> None:
    rows = []
    for name in COHORTS:
        rows += concordance_rows(name, load_cohort(name))
    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)
    print("POST-HOC — benign images: model decision × reader ≥ 4a call")
    print("| cohort | reader | n benign | model FP | FP also ≥4a by reader | TN called ≥4a by reader | reader ≥4a on all benign |")
    print("|---|---|---|---|---|---|---|")
    for r in table.itertuples():
        print(f"| {r.cohort.upper()} | {r.reader} | {r.n_benign} | {r.n_model_fp} | "
              f"{r.n_fp_and_reader_ge4a}/{r.n_model_fp} = {r.reader_ge4a_among_model_fp:.3f} | "
              f"{r.n_tn_and_reader_ge4a}/{r.n_model_tn} = {r.reader_ge4a_among_model_tn:.3f} | "
              f"{r.reader_ge4a_all_benign:.3f} |")
    print(f"written: {OUT_CSV}")


if __name__ == "__main__":
    main()

"""POST-HOC: overconfidence count on the v1 OOF predictions.

Research prototype — not for diagnostic use.

Responds to audit 2026-09-06 §2 item 6: the report once stated "121 malignant,
only 3 with probability in 0.6–0.95" without a committed source. This script
counts, from reports/oof_vit_preds.csv, malignant images whose RAW
(pre-temperature) hflip-TTA OOF probability lies in [0.6, 0.95], for fold 5
and for all folds (the plain no-TTA column is shown alongside). Descriptive
only; nothing frozen is touched.

Usage: python src/posthoc_overconfidence.py
"""

from pathlib import Path

import pandas as pd

OOF_CSV = Path("reports/oof_vit_preds.csv")
OUT_CSV = Path("reports/posthoc_overconfidence.csv")
LO, HI = 0.6, 0.95


def count_band(df: pd.DataFrame, col: str) -> tuple[int, int]:
    mal = df[df["y_true"] == 1]
    return int(mal[col].between(LO, HI, inclusive="both").sum()), len(mal)


def main() -> None:
    df = pd.read_csv(OOF_CSV)
    rows = []
    for scope, sub in (("fold 5", df[df["fold"] == 5]), ("all folds", df)):
        for col in ("y_prob_tta", "y_prob_plain"):
            n_band, n_mal = count_band(sub, col)
            rows.append({"scope": scope, "probability": col, "n_malignant": n_mal,
                         f"n_in_[{LO},{HI}]": n_band, "fraction": n_band / n_mal,
                         "n_above_0.95": int(sub[(sub["y_true"] == 1)][col].gt(HI).sum()),
                         "n_below_0.6": int(sub[(sub["y_true"] == 1)][col].lt(LO).sum())})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)
    print("POST-HOC — malignant OOF images by raw (pre-temperature) probability band")
    print(table.to_string(index=False))
    print(f"written: {OUT_CSV}")


if __name__ == "__main__":
    main()

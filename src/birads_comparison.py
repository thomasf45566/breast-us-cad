"""Pre-registered secondary analysis (protocol §h): frozen model vs the two
BI-RADS reader columns on GDPH and SYSUCC. Research prototype — not for
diagnostic use.

Runs from the SAVED single-shot predictions only — no model, no inference,
no threshold changes. Reader positive call: normalized BI-RADS ∈
{4a, 4b, 4c, 5}; the single row with stray reader value 'c' is excluded
from reader comparisons only (the image stays in the primary analysis).
Descriptive comparison only, per cohort, never pooled.

Usage: python src/birads_comparison.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve

XLSX = Path("data/raw/gdph_sysucc/BIRADS&FOLD.xlsx")
READERS = ["BIRADS-reader1", "BIRADS-reader2"]
VALID = {"2", "3", "4a", "4b", "4c", "5"}
POSITIVE = {"4a", "4b", "4c", "5"}  # >= 4a, pre-registered
COHORTS = ["gdph", "sysucc"]


def normalize(v) -> str:
    """4A -> 4a; numeric 2/3/5 as-is (xlsx may parse them as ints)."""
    return str(v).strip().lower()


def load_cohort(name: str) -> pd.DataFrame:
    """Keep-list predictions joined 1:1 with reader BI-RADS on image ID."""
    preds = pd.read_csv(f"reports/external_{name}_preds.csv")
    preds["ID"] = preds["image_path"].map(lambda p: Path(p).stem)
    meta = pd.read_excel(XLSX)
    for r in READERS:
        meta[r] = meta[r].map(normalize)
    df = preds.merge(meta[["ID", *READERS]], on="ID", how="left", validate="1:1")
    assert df[READERS].notna().all().all(), f"{name}: unmatched IDs in join"
    return df


def sens_spec(y_true: pd.Series, positive: pd.Series) -> tuple[float, float]:
    pos, neg = y_true == 1, y_true == 0
    return positive[pos].mean(), (~positive[neg]).mean()


def cohort_table(name: str, df: pd.DataFrame) -> dict:
    rows = [("model (thr 0.2683)", df["y_pred"] == 1, df)]
    for r in READERS:
        sub = df[df[r].isin(VALID)]  # stray 'c' row drops out here only
        rows.append((r, sub[r].isin(POSITIVE), sub))
    print(f"\n=== {name.upper()} — model vs BI-RADS readers (positive = >= 4a) ===")
    print("| Rater | n | Sensitivity | Specificity |")
    print("|---|---|---|---|")
    out = {}
    for label, positive, sub in rows:
        sens, spec = sens_spec(sub["y_true"], positive)
        out[label] = (sens, spec)
        print(f"| {label} | {len(sub)} | {sens:.4f} | {spec:.4f} |")
    both = df[df[READERS[0]].isin(VALID) & df[READERS[1]].isin(VALID)]
    r1 = both[READERS[0]].isin(POSITIVE)
    r2 = both[READERS[1]].isin(POSITIVE)
    agree = (r1 == r2).mean()
    kappa = cohen_kappa_score(r1, r2)
    print(f"reader1 vs reader2 (binary >= 4a, n={len(both)}): "
          f"agreement {agree:.4f}, Cohen's kappa {kappa:.4f}")
    out["agreement"], out["kappa"], out["n_both"] = agree, kappa, len(both)
    return out


def cohort_figure(name: str, df: pd.DataFrame, table: dict) -> None:
    fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob_calibrated"])
    auc = roc_auc_score(df["y_true"], df["y_prob_calibrated"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2b7bba", label=f"model ROC (AUC {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    marks = [("model (thr 0.2683)", "o", "black"),
             ("BIRADS-reader1", "s", "#d1495b"), ("BIRADS-reader2", "^", "#66a182")]
    for label, marker, color in marks:
        sens, spec = table[label]
        ax.plot(1 - spec, sens, marker, color=color, markersize=9, label=label)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.set_title(f"{name.upper()}: frozen model vs BI-RADS readers (>= 4a)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    path = Path(f"reports/birads_comparison_{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")


def main() -> None:
    for name in COHORTS:
        df = load_cohort(name)
        table = cohort_table(name, df)
        cohort_figure(name, df, table)


if __name__ == "__main__":
    main()

"""Amendment 3 (o)-(q): ensemble disagreement as an abstention signal.

Research prototype — not for diagnostic use.

Inputs: reports/v2_members_{cohort}.csv (from v2_dump_members.py, already
reproduction-checked against external-v1) + the saved external-v1 preds
CSVs. No inference, no model changes; external-v1 numbers are never edited.

Signals (o): U_std = std (ddof=0) of the 10 member probs; U_range =
max - min; baseline margin M = |y_prob_calibrated - thr| with uncertainty
direction -M (LOW margin = uncertain).

Analyses (p), per cohort, never pooled:
  (a) error-prediction AUROC of each signal vs error = (y_pred != y_true)
      at the frozen operating point, 95% percentile bootstrap (2000, seed
      42), case-level for BrEaST (== image-level), image-level otherwise;
  (b) abstention curves at q in {5,10,20,30}%: abstain the top-q% most
      uncertain (ties broken by stable sort on image_path), report
      retained-set sens/spec at the frozen threshold + error enrichment
      ratio in the abstained set;
  (c) the same curves for the margin baseline (same loop).

Verdict (q), applied mechanically per cohort:
  1. error-AUROC(U_std) >= 0.65
  2. at q=10%: retained spec >= frozen spec + 0.05 AND retained sens >=
     frozen sens (frozen = full-cohort external-v1 values)
  3. error-AUROC(U_std) > error-AUROC(margin)

Outputs: reports/v2_abstention_{metrics,curves}.csv,
reports/v2_disagreement_auroc.png, reports/v2_abstention_curves.png.

Usage: python src/v2_disagreement.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPORTS = Path("reports")
OPERATING_POINT_JSON = Path("models/operating_point.json")
COHORTS = ["breast", "busi", "gdph", "sysucc"]
COHORT_LABELS = {"breast": "BrEaST", "busi": "BUSI", "gdph": "GDPH", "sysucc": "SYSUCC"}
MEMBER_COLS = [f"m{k}_{view}" for k in (1, 2, 3, 4, 5) for view in ("orig", "flip")]
Q_GRID = [5, 10, 20, 30]
N_BOOT, BOOT_SEED = 2000, 42
AUROC_CRITERION, SPEC_GAIN, Q_STAR = 0.65, 0.05, 10
# Okabe-Ito, CVD-validated (dataviz validator): signal -> hue, fixed order
SIGNAL_COLORS = {"U_std": "#0072B2", "U_range": "#E69F00", "margin": "#009E73"}
SIGNALS = list(SIGNAL_COLORS)


def load_cohort(name: str, thr: float) -> pd.DataFrame:
    mem = pd.read_csv(REPORTS / f"v2_members_{name}.csv")
    preds = pd.read_csv(REPORTS / f"external_{name}_preds.csv")
    assert list(mem["image_path"]) == list(preds["image_path"]), f"{name}: row mismatch"
    assert (mem["y_true"].values == preds["y_true"].values).all()
    # frozen decisions exactly as saved
    assert (preds["y_pred"].values == (preds["y_prob_calibrated"].values >= thr)).all()

    P = mem[MEMBER_COLS].values
    df = pd.DataFrame(
        {
            "image_path": mem["image_path"],
            "patient_id": mem["patient_id"],
            "y_true": preds["y_true"],
            "y_pred": preds["y_pred"],
            "error": (preds["y_pred"] != preds["y_true"]).astype(int),
            "U_std": P.std(axis=1, ddof=0),
            "U_range": P.max(axis=1) - P.min(axis=1),
            "margin": -np.abs(preds["y_prob_calibrated"].values - thr),
        }
    )
    return df


def error_auroc_ci(df: pd.DataFrame, signal: str) -> dict:
    """Point AUROC + 95% percentile bootstrap CI, resampling patient_ids
    (BrEaST: case-level == image-level; others: patient_id = filename)."""
    point = roc_auc_score(df["error"], df[signal])
    by_patient = df.reset_index(drop=True).groupby("patient_id").indices
    patients = list(by_patient)
    err, sig = df["error"].values, df[signal].values
    rng = np.random.default_rng(BOOT_SEED)
    boot = []
    for _ in range(N_BOOT):
        idx = np.concatenate(
            [by_patient[p] for p in rng.choice(patients, len(patients))]
        )
        if err[idx].min() == err[idx].max():
            continue  # degenerate resample
        boot.append(roc_auc_score(err[idx], sig[idx]))
    return {
        "auroc": point,
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
        "n_boot_valid": len(boot),
    }


def sens_spec(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    pos, neg = y_true == 1, y_true == 0
    sens = (y_pred[pos] == 1).mean() if pos.any() else np.nan
    spec = (y_pred[neg] == 0).mean() if neg.any() else np.nan
    return float(sens), float(spec)


def abstention_rows(df: pd.DataFrame, name: str, signal: str) -> list[dict]:
    """Retained-set sens/spec + abstained-set error enrichment per q.
    q=0 row = the frozen full-cohort operating point (reference)."""
    n = len(df)
    full_err = df["error"].mean()
    # top-q% most uncertain abstain; ties broken by stable sort on image id
    ordered = df.sort_values(
        [signal, "image_path"], ascending=[False, True], kind="mergesort"
    )
    rows = []
    for q in [0, *Q_GRID]:
        n_abs = int(round(q / 100 * n))
        abst, ret = ordered.iloc[:n_abs], ordered.iloc[n_abs:]
        sens, spec = sens_spec(ret["y_true"].values, ret["y_pred"].values)
        rows.append(
            {
                "cohort": name,
                "signal": signal,
                "q_pct": q,
                "n_abstained": n_abs,
                "n_retained": len(ret),
                "retained_prevalence": float(ret["y_true"].mean()),
                "retained_sens": sens,
                "retained_spec": spec,
                "abstained_error_rate": float(abst["error"].mean()) if n_abs else np.nan,
                "error_enrichment": float(abst["error"].mean() / full_err)
                if n_abs and full_err > 0
                else np.nan,
            }
        )
    return rows


def verdict(metrics: pd.DataFrame, curves: pd.DataFrame, name: str) -> dict:
    m = metrics.set_index("signal")
    c = curves.set_index(["signal", "q_pct"])
    frozen_sens = c.loc[("U_std", 0), "retained_sens"]  # q=0 == frozen, signal-invariant
    frozen_spec = c.loc[("U_std", 0), "retained_spec"]
    c1 = m.loc["U_std", "auroc"] >= AUROC_CRITERION
    c2 = (
        c.loc[("U_std", Q_STAR), "retained_spec"] >= frozen_spec + SPEC_GAIN
        and c.loc[("U_std", Q_STAR), "retained_sens"] >= frozen_sens
    )
    c3 = m.loc["U_std", "auroc"] > m.loc["margin", "auroc"]
    return {
        "cohort": name,
        "c1_auroc_ge_0.65": bool(c1),
        "c2_q10_spec_sens": bool(c2),
        "c3_beats_margin": bool(c3),
        "useful": bool(c1 and c2 and c3),
        "frozen_sens": float(frozen_sens),
        "frozen_spec": float(frozen_spec),
    }


def plot_auroc(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(COHORTS))
    width = 0.26
    for i, sig in enumerate(SIGNALS):
        sub = metrics[metrics["signal"] == sig].set_index("cohort").loc[COHORTS]
        pos = x + (i - 1) * width
        ax.bar(
            pos,
            sub["auroc"],
            width - 0.02,  # 2px-equivalent gap between adjacent bars
            color=SIGNAL_COLORS[sig],
            label={"margin": "margin |p_cal − thr| (baseline)"}.get(sig, sig),
        )
        ax.errorbar(
            pos, sub["auroc"],
            yerr=[sub["auroc"] - sub["ci_lo"], sub["ci_hi"] - sub["auroc"]],
            fmt="none", ecolor="#333333", elinewidth=1, capsize=2,
        )
        for xp, v in zip(pos, sub["auroc"]):
            ax.annotate(f"{v:.2f}", (xp, 0.02), ha="center", va="bottom",
                        fontsize=8, color="white", fontweight="bold")
    ax.axhline(AUROC_CRITERION, color="#666666", ls="--", lw=1)
    ax.annotate("criterion (q)1: 0.65", (len(COHORTS) - 0.55, AUROC_CRITERION + 0.008),
                fontsize=8, color="#666666")
    ax.axhline(0.5, color="#bbbbbb", ls=":", lw=1)
    ax.annotate("chance", (-0.45, 0.505), fontsize=8, color="#999999")
    ax.set_xticks(x, [COHORT_LABELS[c] for c in COHORTS])
    ax.set_ylabel("error-prediction AUROC (95% bootstrap CI)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Amendment 3: can uncertainty signals predict frozen-pipeline errors?")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(curves: pd.DataFrame, verdicts: pd.DataFrame, path: Path) -> None:
    sens_c, spec_c = "#0072B2", "#D55E00"  # CVD-validated pair
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, COHORTS):
        v = verdicts.set_index("cohort").loc[name]
        for sig, ls in (("U_std", "-"), ("margin", "--")):
            sub = curves[(curves["cohort"] == name) & (curves["signal"] == sig)]
            ax.plot(sub["q_pct"], sub["retained_sens"], ls, color=sens_c,
                    marker="o", ms=4, lw=2)
            ax.plot(sub["q_pct"], sub["retained_spec"], ls, color=spec_c,
                    marker="o", ms=4, lw=2)
        ax.axhline(v["frozen_spec"] + SPEC_GAIN, color=spec_c, ls=":", lw=1, alpha=0.6)
        ax.annotate("spec target (q)2", (30, v["frozen_spec"] + SPEC_GAIN),
                    fontsize=7, color=spec_c, ha="right", va="bottom", alpha=0.8)
        ax.set_title(f"{COHORT_LABELS[name]} — "
                     f"{'USEFUL' if v['useful'] else 'not useful'} per (q)", fontsize=10)
        ax.set_xticks([0, *Q_GRID])
        ax.grid(color="#eeeeee", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("abstained fraction q (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("retained-set value")
    handles = [
        plt.Line2D([], [], color=sens_c, lw=2, label="sensitivity"),
        plt.Line2D([], [], color=spec_c, lw=2, label="specificity"),
        plt.Line2D([], [], color="#555555", lw=2, ls="-", label="abstain by U_std"),
        plt.Line2D([], [], color="#555555", lw=2, ls="--", label="abstain by margin"),
    ]
    fig.legend(handles=handles, ncol=4, frameon=False, fontsize=9,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Amendment 3: abstention curves at the frozen threshold "
                 "(q=0 = frozen operating point)", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    thr = json.loads(OPERATING_POINT_JSON.read_text())["threshold"]
    metrics_rows, curve_rows, verdicts = [], [], []
    for name in COHORTS:
        df = load_cohort(name, thr)
        for sig in SIGNALS:
            metrics_rows.append(
                {"cohort": name, "signal": sig, "n": len(df),
                 "n_errors": int(df["error"].sum()), **error_auroc_ci(df, sig)}
            )
            curve_rows.extend(abstention_rows(df, name, sig))
    metrics = pd.DataFrame(metrics_rows)
    curves = pd.DataFrame(curve_rows)
    for name in COHORTS:
        verdicts.append(
            verdict(metrics[metrics["cohort"] == name],
                    curves[curves["cohort"] == name], name)
        )
    verdicts = pd.DataFrame(verdicts)

    metrics.to_csv(REPORTS / "v2_abstention_metrics.csv", index=False)
    curves.to_csv(REPORTS / "v2_abstention_curves.csv", index=False)
    plot_auroc(metrics, REPORTS / "v2_disagreement_auroc.png")
    plot_curves(curves, verdicts, REPORTS / "v2_abstention_curves.png")

    print("=== Amendment 3 — error-prediction AUROC (95% bootstrap CI) ===")
    for _, r in metrics.iterrows():
        print(f"  {r['cohort']:7s} {r['signal']:8s} {r['auroc']:.4f} "
              f"({r['ci_lo']:.4f}-{r['ci_hi']:.4f})  "
              f"errors {r['n_errors']}/{r['n']}")
    print("\n=== abstention at q=10% (U_std vs margin), frozen threshold ===")
    at10 = curves[curves["q_pct"].isin([0, 10])]
    for name in COHORTS:
        for sig in ("U_std", "margin"):
            r0 = at10[(at10.cohort == name) & (at10.signal == sig) & (at10.q_pct == 0)].iloc[0]
            r = at10[(at10.cohort == name) & (at10.signal == sig) & (at10.q_pct == 10)].iloc[0]
            print(f"  {name:7s} {sig:7s} sens {r0['retained_sens']:.4f}->{r['retained_sens']:.4f} "
                  f"spec {r0['retained_spec']:.4f}->{r['retained_spec']:.4f} "
                  f"enrich x{r['error_enrichment']:.2f}")
    print("\n=== pre-registered verdict (q), applied mechanically ===")
    for _, v in verdicts.iterrows():
        print(f"  {v['cohort']:7s} c1(AUROC>=0.65)={v['c1_auroc_ge_0.65']} "
              f"c2(q10 spec+0.05, sens held)={v['c2_q10_spec_sens']} "
              f"c3(beats margin)={v['c3_beats_margin']} "
              f"=> {'USEFUL' if v['useful'] else 'NOT USEFUL'}")
    print("\nsaved: reports/v2_abstention_metrics.csv, "
          "reports/v2_abstention_curves.csv, reports/v2_disagreement_auroc.png, "
          "reports/v2_abstention_curves.png")
    print("note: no internal OOF reference exists (Amendment 3 (r)).")


if __name__ == "__main__":
    main()

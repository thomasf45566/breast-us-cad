"""Q1c: LOCO summary table, delta CIs, figure, mechanical (t) verdict.

Research prototype — not for diagnostic use.

Post-hoc REPORTING ONLY (Amendment 4 (t)): every number here is derived
from SAVED artifacts — reports/v2_loco_{cohort}_preds.csv (the single-shot
held-out LOCO evals), reports/v2_members_{cohort}.csv (v1 fold-5 single
model + TTA, m5 columns, per (t)), reports/external_{cohort}_preds.csv
(v1 full-ensemble external-v1 records, reference only) and the run JSONs.
No inference, no training, no model file is touched. Point values are
asserted to reproduce reports/v2_loco_summary.csv and the external-v1
AUCs before anything is written.

Adds what Q1b's per-run rows lack: paired bootstrap CIs on ΔAUC and Δspec
(LOCO − v1-single; same resamples applied to both models, protocol (d)
units: case-level BrEaST, image-level otherwise, 2000 iterations, seed
42), the v1-ensemble reference column, the 2×4 summary figure, and the
mechanical (t) verdict over all four runs.

Outputs: reports/v2_loco_q1c_table.csv, reports/v2_loco_summary.png,
verdict + markdown table on stdout.

Usage: python src/v2_loco_report.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from external_val import BOOT_SEED, N_BOOT, patient_bootstrap  # noqa: E402
from inference import calibrate_probs  # noqa: E402
from pick_threshold import confusion_counts, point_metrics  # noqa: E402

COHORTS = ["breast", "busi", "gdph", "sysucc"]
LABELS = {"breast": "BrEaST", "busi": "BUSI", "gdph": "GDPH", "sysucc": "SYSUCC"}
SUMMARY_CSV = Path("reports/v2_loco_summary.csv")
TABLE_CSV = Path("reports/v2_loco_q1c_table.csv")
FIG_PNG = Path("reports/v2_loco_summary.png")
V1_T = json.loads(Path("models/calibration.json").read_text())["temperature"]
V1_THR = json.loads(Path("models/operating_point.json").read_text())["threshold"]
# external-v1 AUCs as recorded in RESULTS.md — reproduction guard for the
# v1-ensemble reference column
EXTERNAL_V1_AUC = {"breast": 0.8542, "busi": 0.9339, "gdph": 0.9154, "sysucc": 0.8380}

C_V1 = "#4c72b0"
C_LOCO = "#dd8452"


def load_cohort(cohort: str) -> dict:
    """Merge saved LOCO preds, v1-single (from members), v1-ensemble preds."""
    loco = pd.read_csv(f"reports/v2_loco_{cohort}_preds.csv")
    members = pd.read_csv(f"reports/v2_members_{cohort}.csv")
    ens = pd.read_csv(f"reports/external_{cohort}_preds.csv")
    df = loco.merge(
        members[["image_path", "y_true", "m5_orig", "m5_flip"]],
        on="image_path", suffixes=("", "_m"), validate="1:1",
    ).merge(
        ens[["image_path", "y_true", "y_prob_calibrated"]].rename(
            columns={"y_true": "y_true_e", "y_prob_calibrated": "p_ens"}),
        on="image_path", validate="1:1",
    )
    assert len(df) == len(loco) == len(members) == len(ens)
    assert (df["y_true"] == df["y_true_m"]).all() and (df["y_true"] == df["y_true_e"]).all()
    df["p_loco"] = df["y_prob_calibrated"]
    df["p_v1s"] = calibrate_probs((df["m5_orig"].values + df["m5_flip"].values) / 2, V1_T)
    op = json.loads(Path(f"models/v2_loco_{cohort}_operating_point.json").read_text())
    return {"df": df, "thr_loco": op["threshold"], "temperature": op["temperature"]}


def model_metrics(df: pd.DataFrame, prob_col: str, thr: float) -> dict:
    y = df["y_true"].values
    p = df[prob_col].values
    m = point_metrics(*confusion_counts(y, p, thr))
    ci = patient_bootstrap(df.rename(columns={"y_true": "label"}), p, thr)
    return {
        "auc": roc_auc_score(y, p),
        "auc_lo": ci["auc_ci95"][0], "auc_hi": ci["auc_ci95"][1],
        "sens": m["sensitivity"], "spec": m["specificity"],
        "benign_median": float(np.median(p[y == 0])),
        "threshold": thr,
    }


def paired_delta_ci(df: pd.DataFrame, thr_loco: float) -> dict:
    """Bootstrap CIs on ΔAUC / Δspec (LOCO − v1-single), SAME resamples for
    both models; units per protocol (d) via patient_id groups (case-level
    for BrEaST; elsewhere patient_id is unique per image ⇒ image-level),
    matching patient_bootstrap exactly (N_BOOT, BOOT_SEED, percentile)."""
    d = df.reset_index(drop=True)
    by_patient = d.groupby("patient_id").indices
    patients = list(by_patient)
    y = d["y_true"].values
    p_loco, p_v1s = d["p_loco"].values, d["p_v1s"].values
    rng = np.random.default_rng(BOOT_SEED)
    d_auc, d_spec = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([by_patient[p] for p in rng.choice(patients, len(patients))])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        d_auc.append(roc_auc_score(yb, p_loco[idx]) - roc_auc_score(yb, p_v1s[idx]))
        _, fp_l, _, tn_l = confusion_counts(yb, p_loco[idx], thr_loco)
        _, fp_v, _, tn_v = confusion_counts(yb, p_v1s[idx], V1_THR)
        spec_l = tn_l / (tn_l + fp_l) if tn_l + fp_l else np.nan
        spec_v = tn_v / (tn_v + fp_v) if tn_v + fp_v else np.nan
        d_spec.append(spec_l - spec_v)
    pct = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    return {"n_boot_valid": len(d_auc),
            "delta_auc_ci95": pct(d_auc), "delta_spec_ci95": pct(d_spec)}


def check_reproduction(cohort: str, rows: dict, summary: pd.DataFrame) -> None:
    """Point values must reproduce the committed Q1b summary + external-v1."""
    s = summary[summary["hold_out"] == cohort].iloc[0]
    pairs = [
        (rows["loco"]["auc"], s["loco_auc"]), (rows["loco"]["sens"], s["loco_sens"]),
        (rows["loco"]["spec"], s["loco_spec"]),
        (rows["loco"]["benign_median"], s["loco_benign_median"]),
        (rows["v1_single"]["auc"], s["v1single_auc"]),
        (rows["v1_single"]["sens"], s["v1single_sens"]),
        (rows["v1_single"]["spec"], s["v1single_spec"]),
        (rows["v1_single"]["benign_median"], s["v1single_benign_median"]),
        (rows["loco"]["auc_lo"], s["loco_auc_lo"]), (rows["loco"]["auc_hi"], s["loco_auc_hi"]),
        (rows["v1_single"]["auc_lo"], s["v1single_auc_lo"]),
        (rows["v1_single"]["auc_hi"], s["v1single_auc_hi"]),
    ]
    for got, ref in pairs:
        # 1e-6: preds CSVs store float32 probs (shortest repr), so medians
        # re-read from disk can differ from the committed float64 at ~1e-9
        assert abs(got - ref) < 1e-6, f"{cohort}: {got!r} != committed {ref!r}"
    assert abs(rows["v1_ensemble"]["auc"] - EXTERNAL_V1_AUC[cohort]) < 5e-5, (
        f"{cohort}: v1-ensemble AUC {rows['v1_ensemble']['auc']:.4f} != "
        f"external-v1 record {EXTERNAL_V1_AUC[cohort]}"
    )


def make_figure(data: dict) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    for j, cohort in enumerate(COHORTS):
        d = data[cohort]
        df, thr_loco = d["df"], d["thr_loco"]
        y = df["y_true"].values

        ax = axes[0, j]
        for col, thr, color, name in [
            ("p_v1s", V1_THR, C_V1, "v1-single"),
            ("p_loco", thr_loco, C_LOCO, "LOCO"),
        ]:
            p = df[col].values
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            ax.plot(fpr, tpr, color=color, lw=1.6, label=f"{name} AUC {auc:.3f}")
            tp, fp, fn, tn = confusion_counts(y, p, thr)
            ax.plot(fp / (fp + tn), tp / (tp + fn), "o", color=color, ms=7,
                    mec="black", mew=0.6, zorder=5,
                    label=f"  @thr {thr:.3f}: sens {tp / (tp + fn):.2f}, spec {tn / (tn + fp):.2f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
        ax.set_title(f"{LABELS[cohort]} (n={len(df)}, held out)")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate" if j == 0 else "")
        ax.legend(loc="lower right", fontsize=7.5)

        ax = axes[1, j]
        bins = np.linspace(0, 1, 31)
        for col, thr, color, name in [
            ("p_v1s", V1_THR, C_V1, "v1-single"),
            ("p_loco", thr_loco, C_LOCO, "LOCO"),
        ]:
            pb = df.loc[df["y_true"] == 0, col].values
            ax.hist(pb, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
                    color=color, edgecolor=color,
                    label=f"{name} (median {np.median(pb):.3f})")
            ax.axvline(thr, color=color, ls="--", lw=1.4,
                       label=f"  thr {thr:.3f}")
        ax.set_xlabel("Calibrated probability (benign images)")
        ax.set_ylabel("Density" if j == 0 else "")
        ax.legend(loc="upper right", fontsize=7.5)
    fig.suptitle(
        "v2 Q1 LOCO vs v1-single (fold-5 + TTA) on held-out cohorts — single-shot evals, "
        "saved predictions only (Amendment 4 (t))",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    assert list(summary["hold_out"]) == COHORTS
    data, out_rows = {}, []
    for cohort in COHORTS:
        d = load_cohort(cohort)
        data[cohort] = d
        rows = {
            "loco": model_metrics(d["df"], "p_loco", d["thr_loco"]),
            "v1_single": model_metrics(d["df"], "p_v1s", V1_THR),
            "v1_ensemble": model_metrics(d["df"], "p_ens", V1_THR),
        }
        check_reproduction(cohort, rows, summary)
        deltas = paired_delta_ci(d["df"], d["thr_loco"])
        d["rows"], d["deltas"] = rows, deltas
        units = "case-level" if cohort == "breast" else "image-level"
        for model, r in rows.items():
            rec = {"cohort": cohort, "n": len(d["df"]), "bootstrap": units,
                   "model": model, **r}
            if model == "loco":
                rec.update({
                    "delta_auc": r["auc"] - rows["v1_single"]["auc"],
                    "delta_auc_lo": deltas["delta_auc_ci95"][0],
                    "delta_auc_hi": deltas["delta_auc_ci95"][1],
                    "delta_spec": r["spec"] - rows["v1_single"]["spec"],
                    "delta_spec_lo": deltas["delta_spec_ci95"][0],
                    "delta_spec_hi": deltas["delta_spec_ci95"][1],
                    "n_boot_valid": deltas["n_boot_valid"],
                })
            out_rows.append(rec)
    pd.DataFrame(out_rows).to_csv(TABLE_CSV, index=False)
    make_figure(data)

    # ---- markdown table
    print("\n| Held-out cohort | Model | AUC (95% CI) | Sens | Spec | Benign median p_cal "
          "| ΔAUC vs v1-single (95% CI) | Δspec (95% CI) |")
    print("|---|---|---|---|---|---|---|---|")
    for cohort in COHORTS:
        d = data[cohort]
        for model, name in [("v1_single", "v1-single"), ("v1_ensemble", "v1-ensemble (ref)"),
                            ("loco", "v2-LOCO")]:
            r = d["rows"][model]
            da = ds = ""
            if model == "loco":
                v1 = d["rows"]["v1_single"]
                ci_a, ci_s = d["deltas"]["delta_auc_ci95"], d["deltas"]["delta_spec_ci95"]
                da = f"**{r['auc'] - v1['auc']:+.4f}** ({ci_a[0]:+.4f} to {ci_a[1]:+.4f})"
                ds = f"**{r['spec'] - v1['spec']:+.4f}** ({ci_s[0]:+.4f} to {ci_s[1]:+.4f})"
            label = f"{LABELS[cohort]} (n={len(d['df'])})" if model == "v1_single" else ""
            print(f"| {label} | {name} | {r['auc']:.4f} ({r['auc_lo']:.4f}–{r['auc_hi']:.4f}) "
                  f"| {r['sens']:.4f} | {r['spec']:.4f} | {r['benign_median']:.4f} | {da} | {ds} |")

    # ---- mechanical (t) verdict
    print("\n=== Pre-registered criterion (t), applied mechanically ===")
    a_hits, b_hits = [], []
    for cohort in COHORTS:
        d = data[cohort]
        loco, v1s = d["rows"]["loco"], d["rows"]["v1_single"]
        da, dsp = loco["auc"] - v1s["auc"], loco["spec"] - v1s["spec"]
        a = da >= 0.01
        b = (dsp >= 0.10) and (loco["sens"] >= 0.85)
        a_hits.append(a)
        b_hits.append(b)
        print(f"{LABELS[cohort]:7s} | branch A (ΔAUC {da:+.4f} >= +0.01): {'PASS' if a else 'fail'} "
              f"| branch B (Δspec {dsp:+.4f} >= +0.10 AND sens {loco['sens']:.4f} >= 0.85): "
              f"{'PASS' if b else 'fail'}")
    na, nb = sum(a_hits), sum(b_hits)
    met = na >= 3 or nb >= 3
    print(f"branch A: {na}/4 (need >= 3) | branch B: {nb}/4 (need >= 3)")
    print(f"VERDICT: criterion {'MET — claim' if met else 'NOT MET — no claim'}: "
          f"'multi-source training reduces the domain-shift specificity collapse'"
          f"{' via branch A (AUC)' if met and na >= 3 else ''}")
    print(f"\nsaved: {TABLE_CSV}, {FIG_PNG}")


if __name__ == "__main__":
    main()

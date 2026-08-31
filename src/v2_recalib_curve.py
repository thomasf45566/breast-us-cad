"""v2 secondary study: site-specific recalibration learning curves.

Research prototype — not for diagnostic use.

Implements data/external_protocol.md Amendment 2 (M1, primary method):
per external cohort, draw k labeled images at natural prevalence, re-select
the decision threshold on the draw with the frozen rule (highest threshold
with sens >= 0.90, pick_threshold.pick_operating_point), and evaluate
sens/spec on the n-k held-out images. Inputs are the saved external-v1
prediction CSVs ONLY — no inference, no training, no model changes.
External-v1 numbers are never altered; outputs go to reports/v2_* and the
"v2: Site-specific recalibration" section of RESULTS.md.

Usage: python src/v2_recalib_curve.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pick_threshold import confusion_counts, pick_operating_point

OPERATING_POINT_JSON = Path("models/operating_point.json")
DRAWS_CSV = Path("reports/v2_recalib_M1_draws.csv")
SUMMARY_CSV = Path("reports/v2_recalib_M1_summary.csv")
CURVES_PNG = Path("reports/v2_recalib_M1_curves.png")

# Fixed cohort order — also the index used in per-draw seed derivation.
COHORTS = ["breast", "busi", "gdph", "sysucc"]
COHORT_LABEL = {"breast": "BrEaST", "busi": "BUSI (clean)", "gdph": "GDPH", "sysucc": "SYSUCC"}
PREDS_CSV = {c: Path(f"reports/external_{c}_preds.csv") for c in COHORTS}

K_GRID = [10, 20, 30, 50, 100, 200]  # Amendment 2 (j): drop k >= n/2 per cohort
R_DRAWS = 500
MASTER_SEED = 42
SENS_FLOOR = 0.90  # frozen rule, same constant as pick_threshold.py

# Amendment 2 decision metric (m)
KSTAR_RECOVERY = 0.80
KSTAR_SENS = 0.85

# External-v1 reference values as recorded in RESULTS.md ("External
# validation (single-shot, frozen-v1)") — sanity check (a) must reproduce
# these digit-for-digit from the saved CSVs before any v2 computation runs.
EXTERNAL_V1_REFERENCE = {
    "breast": {"confusion": (90, 91, 8, 63), "sens": "0.9184", "spec": "0.4091"},
    "busi": {"confusion": (156, 80, 7, 136), "sens": "0.9571", "spec": "0.6296"},
    "gdph": {"confusion": (364, 238, 11, 197), "sens": "0.9707", "spec": "0.4529"},
    "sysucc": {"confusion": (674, 152, 50, 137), "sens": "0.9309", "spec": "0.4740"},
}


def load_cohort(cohort: str) -> pd.DataFrame:
    df = pd.read_csv(PREDS_CSV[cohort])
    if cohort == "breast":
        # Amendment 2 (j): BrEaST sampling is patient-level; the protocol
        # states one case = one image = one patient, so row-level sampling
        # IS patient-level — assert that this still holds.
        assert df["patient_id"].nunique() == len(df), "BrEaST is no longer 1 image/patient"
    return df


def sens_spec(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[float, float]:
    tp, fp, fn, tn = confusion_counts(y, p, thr)
    return tp / (tp + fn), tn / (tn + fp)


def select_threshold_m1(y_k: np.ndarray, p_k: np.ndarray, thr_frozen: float) -> tuple[float, bool]:
    """Frozen rule on the k local calibrated probs; degenerate draws
    (single-class k-set) fall back to the frozen threshold per Amendment 2 (j)."""
    if y_k.min() == y_k.max():
        return thr_frozen, True
    return pick_operating_point(y_k, p_k), False


def run_curve(cohort: str, method: str, ks: list[int], R: int, seed: int) -> pd.DataFrame:
    """One cohort x method learning curve; tidy rows, one per (k, draw).

    Draw seeds derive deterministically from (seed, cohort index, k, draw)
    so every cell is reproducible in isolation.
    """
    if method != "M1":
        raise NotImplementedError(f"method {method} not implemented yet")
    op = json.loads(OPERATING_POINT_JSON.read_text())
    thr_frozen = op["threshold"]

    df = load_cohort(cohort)
    y = df["y_true"].to_numpy()
    p = df["y_prob_calibrated"].to_numpy()
    n = len(df)
    thr_oracle = pick_operating_point(y, p)  # in-sample oracle, all n images
    ci = COHORTS.index(cohort)

    rows = []
    for k in [k for k in ks if k < n / 2]:
        for draw in range(R):
            rng = np.random.default_rng([seed, ci, k, draw])
            cal_idx = rng.choice(n, size=k, replace=False)
            eval_mask = np.ones(n, dtype=bool)
            eval_mask[cal_idx] = False

            thr, degenerate = select_threshold_m1(y[cal_idx], p[cal_idx], thr_frozen)
            sens, spec = sens_spec(y[eval_mask], p[eval_mask], thr)
            _, spec_frozen = sens_spec(y[eval_mask], p[eval_mask], thr_frozen)
            _, spec_oracle = sens_spec(y[eval_mask], p[eval_mask], thr_oracle)
            denom = spec_oracle - spec_frozen
            recovery = (spec - spec_frozen) / denom if denom > 0 else np.nan

            rows.append({
                "cohort": cohort, "method": method, "k": k, "draw": draw,
                "threshold": thr, "sens": sens, "spec": spec,
                "spec_frozen_holdout": spec_frozen, "spec_oracle_holdout": spec_oracle,
                "recovery": recovery, "degenerate": degenerate,
            })
    return pd.DataFrame(rows)


def summarize(draws: pd.DataFrame) -> pd.DataFrame:
    def q(s, x):
        return s.quantile(x)

    g = draws.groupby(["cohort", "k"])
    out = pd.DataFrame({
        "sens_median": g["sens"].median(),
        "sens_p2.5": g["sens"].apply(q, 0.025),
        "sens_p97.5": g["sens"].apply(q, 0.975),
        "spec_median": g["spec"].median(),
        "spec_p2.5": g["spec"].apply(q, 0.025),
        "spec_p97.5": g["spec"].apply(q, 0.975),
        "recovery_median": g["recovery"].median(),
        "recovery_p2.5": g["recovery"].apply(q, 0.025),
        "recovery_p97.5": g["recovery"].apply(q, 0.975),
        "frac_degenerate": g["degenerate"].mean(),
    }).reset_index()
    return out


def k_star(summary: pd.DataFrame, cohort: str) -> int | None:
    """Amendment 2 (m): smallest k with median recovery >= 0.80 AND median
    sens >= 0.85. None = not reached (a valid outcome)."""
    rows = summary[summary["cohort"] == cohort].sort_values("k")
    ok = rows[(rows["recovery_median"] >= KSTAR_RECOVERY) & (rows["sens_median"] >= KSTAR_SENS)]
    return int(ok["k"].iloc[0]) if len(ok) else None


def sanity_checks(thr_frozen: float) -> dict[str, dict]:
    """Amendment 2 pre-run checks (a)-(c); prints each, raises on failure.
    Returns per-cohort frozen/oracle full-cohort operating stats for reuse."""
    stats = {}
    print("=== Sanity checks (pre-registered, before full run) ===")
    for c in COHORTS:
        df = load_cohort(c)
        y = df["y_true"].to_numpy()
        p = df["y_prob_calibrated"].to_numpy()

        # (a) k=0: frozen threshold reproduces external-v1 digit-for-digit
        assert np.array_equal((p >= thr_frozen).astype(int), df["y_pred"].to_numpy()), \
            f"{c}: recomputed decisions != saved y_pred"
        conf = confusion_counts(y, p, thr_frozen)
        ref = EXTERNAL_V1_REFERENCE[c]
        sens, spec = sens_spec(y, p, thr_frozen)
        assert conf == ref["confusion"], f"{c}: confusion {conf} != external-v1 {ref['confusion']}"
        assert (f"{sens:.4f}", f"{spec:.4f}") == (ref["sens"], ref["spec"]), \
            f"{c}: sens/spec {sens:.4f}/{spec:.4f} != external-v1 {ref['sens']}/{ref['spec']}"
        print(f"(a) {COHORT_LABEL[c]:12s} k=0 frozen: tp/fp/fn/tn {conf} "
              f"sens {sens:.4f} spec {spec:.4f} == external-v1  OK")

        # (b) oracle on all n satisfies its own constraint by construction
        thr_o = pick_operating_point(y, p)
        sens_o, spec_o = sens_spec(y, p, thr_o)
        assert sens_o >= SENS_FLOOR, f"{c}: oracle sens {sens_o:.4f} < {SENS_FLOOR}"
        print(f"(b) {COHORT_LABEL[c]:12s} oracle thr {thr_o:.4f}: "
              f"sens {sens_o:.4f} (>= {SENS_FLOOR}) spec {spec_o:.4f}  OK")
        stats[c] = {"spec_frozen": spec, "thr_oracle": thr_o, "spec_oracle": spec_o}

    # (c) one draw: calibration and evaluation index sets are disjoint
    c, k, draw = "breast", 30, 0
    n = len(load_cohort(c))
    rng = np.random.default_rng([MASTER_SEED, COHORTS.index(c), k, draw])
    cal_idx = rng.choice(n, size=k, replace=False)
    eval_mask = np.ones(n, dtype=bool)
    eval_mask[cal_idx] = False
    eval_idx = np.flatnonzero(eval_mask)
    assert len(set(cal_idx) & set(eval_idx)) == 0
    assert len(cal_idx) + len(eval_idx) == n
    print(f"(c) {COHORT_LABEL[c]} k={k} draw={draw}: |cal|={len(cal_idx)} "
          f"|eval|={len(eval_idx)} disjoint, union covers n={n}  OK")
    return stats


def plot_curves(summary: pd.DataFrame, stats: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharey=True)
    for ax, c in zip(axes.ravel(), COHORTS):
        rows = summary[summary["cohort"] == c].sort_values("k")
        ks = rows["k"].to_numpy()
        ax.fill_between(ks, rows["spec_p2.5"], rows["spec_p97.5"],
                        color="#4c72b0", alpha=0.18, lw=0, label="spec 95% band (500 draws)")
        ax.plot(ks, rows["spec_median"], "o-", color="#4c72b0", lw=2, ms=5,
                label="specificity (median)")
        ax.plot(ks, rows["sens_median"], "s--", color="#c44e52", lw=1.8, ms=5,
                label="sensitivity (median)")
        ax.axhline(stats[c]["spec_frozen"], color="0.35", ls="--", lw=1,
                   label="frozen spec (external-v1)")
        ax.axhline(stats[c]["spec_oracle"], color="0.35", ls=":", lw=1.2,
                   label="oracle spec (all n)")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_ylim(0.3, 1.02)
        ax.grid(True, which="major", axis="y", color="0.9", lw=0.8)
        ax.set_title(f"{COHORT_LABEL[c]} (n={len(load_cohort(c))})", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("k local labeled images (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("proportion on held-out n−k")
    axes[0, 0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("v2 M1: local threshold re-selection learning curves "
                 "(frozen model unchanged)\nResearch use only — not for diagnosis",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    op = json.loads(OPERATING_POINT_JSON.read_text())
    thr_frozen = op["threshold"]
    print(f"frozen threshold {thr_frozen:.4f} (models/operating_point.json) | "
          f"method M1 | R={R_DRAWS} draws | master seed {MASTER_SEED}")

    stats = sanity_checks(thr_frozen)

    print("\n=== M1 full run ===")
    draws = pd.concat(
        [run_curve(c, "M1", K_GRID, R_DRAWS, MASTER_SEED) for c in COHORTS],
        ignore_index=True,
    )
    draws.to_csv(DRAWS_CSV, index=False)
    summary = summarize(draws)
    summary.to_csv(SUMMARY_CSV, index=False)
    plot_curves(summary, stats, CURVES_PNG)

    for c in COHORTS:
        rows = summary[summary["cohort"] == c].sort_values("k")
        print(f"\n{COHORT_LABEL[c]} — frozen spec {stats[c]['spec_frozen']:.4f}, "
              f"oracle spec {stats[c]['spec_oracle']:.4f} (thr {stats[c]['thr_oracle']:.4f})")
        for _, r in rows.iterrows():
            print(f"  k={int(r['k']):3d}: sens {r['sens_median']:.4f} "
                  f"[{r['sens_p2.5']:.4f}-{r['sens_p97.5']:.4f}]  "
                  f"spec {r['spec_median']:.4f} [{r['spec_p2.5']:.4f}-{r['spec_p97.5']:.4f}]  "
                  f"recovery {r['recovery_median']:.3f} "
                  f"[{r['recovery_p2.5']:.3f}-{r['recovery_p97.5']:.3f}]  "
                  f"degen {r['frac_degenerate']:.1%}")
        ks = k_star(summary, c)
        print(f"  k* (median recovery >= {KSTAR_RECOVERY} AND median sens >= {KSTAR_SENS}): "
              f"{ks if ks is not None else 'not reached'}")

    print(f"\nsaved: {DRAWS_CSV} ({len(draws)} rows), {SUMMARY_CSV}, {CURVES_PNG}")


if __name__ == "__main__":
    main()

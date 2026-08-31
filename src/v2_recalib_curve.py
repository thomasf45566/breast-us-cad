"""v2 secondary study: site-specific recalibration learning curves.

Research prototype — not for diagnostic use.

Implements data/external_protocol.md Amendment 2: per external cohort,
draw k labeled images at natural prevalence and evaluate on the n-k
held-out images. Methods:
  M1  (primary)   threshold re-selected on the k calibrated probs with the
                  frozen rule (highest thr with sens >= 0.90).
  M2a (secondary) temperature refit on the k local logits (calibrate.py
                  LBFGS fitter), frozen internal threshold 0.2683.
  M2b (secondary) same temperature refit, local sens >= 0.90 rule.
  M3  (secondary) Platt scaling (a, b) on the k local logits, local rule.
Logits are log(p/(1-p)) of the raw (pre-temperature) TTA probability,
the calibrate.probs_to_logits convention. Inputs are the saved external-v1
prediction CSVs ONLY — no inference, no training, no model changes.
External-v1 numbers are never altered; outputs go to reports/v2_* and the
"v2: Site-specific recalibration" section of RESULTS.md.

k_reliable (smallest k with 2.5th-pct recovery >= 0.5) is a POST-HOC
reliability metric, not part of pre-registered Amendment 2 — labeled as
such everywhere it is reported.

Usage: python src/v2_recalib_curve.py
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from calibrate import fit_temperature, probs_to_logits
from pick_threshold import confusion_counts, pick_operating_point

OPERATING_POINT_JSON = Path("models/operating_point.json")
DRAWS_CSV = Path("reports/v2_recalib_M1_draws.csv")
SUMMARY_CSV = Path("reports/v2_recalib_M1_summary.csv")
CURVES_PNG = Path("reports/v2_recalib_M1_curves.png")
METHODS_DRAWS_CSV = Path("reports/v2_recalib_methods_draws.csv")
METHODS_SUMMARY_CSV = Path("reports/v2_recalib_methods_summary.csv")
METHODS_PNG = Path("reports/v2_recalib_methods.png")

# Fixed cohort order — also the index used in per-draw seed derivation.
COHORTS = ["breast", "busi", "gdph", "sysucc"]
COHORT_LABEL = {"breast": "BrEaST", "busi": "BUSI (clean)", "gdph": "GDPH", "sysucc": "SYSUCC"}
PREDS_CSV = {c: Path(f"reports/external_{c}_preds.csv") for c in COHORTS}

METHODS = ["M1", "M2a", "M2b", "M3"]
K_GRID = [10, 20, 30, 50, 100, 200]  # Amendment 2 (j): drop k >= n/2 per cohort
R_DRAWS = 500
MASTER_SEED = 42
SENS_FLOOR = 0.90  # frozen rule, same constant as pick_threshold.py

# Amendment 2 decision metric (m); k_reliable threshold is POST-HOC
KSTAR_RECOVERY = 0.80
KSTAR_SENS = 0.85
KRELIABLE_RECOVERY_P2 = 0.50

# External-v1 reference values as recorded in RESULTS.md ("External
# validation (single-shot, frozen-v1)") — sanity check (a) must reproduce
# these digit-for-digit from the saved CSVs before any v2 computation runs.
EXTERNAL_V1_REFERENCE = {
    "breast": {"confusion": (90, 91, 8, 63), "sens": "0.9184", "spec": "0.4091"},
    "busi": {"confusion": (156, 80, 7, 136), "sens": "0.9571", "spec": "0.6296"},
    "gdph": {"confusion": (364, 238, 11, 197), "sens": "0.9707", "spec": "0.4529"},
    "sysucc": {"confusion": (674, 152, 50, 137), "sens": "0.9309", "spec": "0.4740"},
}

# One temperature fit per (cohort, k, draw), shared by M2a and M2b.
# None = the LBFGS fit produced a non-positive temperature (ill-posed local
# fit, e.g. a locally anti-correlated or separable k-set) — the draw falls
# back to the frozen pipeline, mirroring the degenerate-draw rule.
_M2_CACHE: dict[tuple[str, int, int], float | None] = {}


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


def fit_platt(z_k: np.ndarray, y_k: np.ndarray) -> LogisticRegression:
    """Unregularized logistic fit p = sigmoid(a*z + b) on the k local logits.
    Separable k-sets are allowed: coefficients grow large but the transform
    stays monotone, so downstream threshold selection remains well defined."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return LogisticRegression(penalty=None, max_iter=1000).fit(z_k.reshape(-1, 1), y_k)


def draw_metrics(
    method: str, cohort: str, k: int, draw: int,
    y: np.ndarray, p_cal: np.ndarray, z: np.ndarray,
    cal_idx: np.ndarray, eval_mask: np.ndarray, thr_frozen: float,
) -> tuple[float, float, float, bool]:
    """(threshold, sens, spec, fit_failed) on the held-out set for one draw.

    Degenerate single-class k-sets fall back to the frozen pipeline
    (frozen calibration + frozen threshold) for every method, per the
    Amendment 2 (j) fallback; local fits are ill-posed with one class.
    An M2 temperature fit that lands non-positive (ill-posed local NLL)
    takes the same fallback and is flagged fit_failed.
    """
    y_k, z_k = y[cal_idx], z[cal_idx]
    degenerate = y_k.min() == y_k.max()
    if degenerate:
        sens, spec = sens_spec(y[eval_mask], p_cal[eval_mask], thr_frozen)
        return thr_frozen, sens, spec, False

    if method == "M1":
        thr = pick_operating_point(y_k, p_cal[cal_idx])
        p_eval = p_cal[eval_mask]
    elif method in ("M2a", "M2b"):
        key = (cohort, k, draw)
        if key not in _M2_CACHE:
            try:
                _M2_CACHE[key] = fit_temperature(z_k, y_k.astype(float))
            except RuntimeError:
                _M2_CACHE[key] = None
        t_local = _M2_CACHE[key]
        if t_local is None:
            sens, spec = sens_spec(y[eval_mask], p_cal[eval_mask], thr_frozen)
            return thr_frozen, sens, spec, True
        p_local = 1 / (1 + np.exp(-z / t_local))
        thr = thr_frozen if method == "M2a" else pick_operating_point(y_k, p_local[cal_idx])
        p_eval = p_local[eval_mask]
    elif method == "M3":
        platt = fit_platt(z_k, y_k)
        p_local = platt.predict_proba(z.reshape(-1, 1))[:, 1]
        thr = pick_operating_point(y_k, p_local[cal_idx])
        p_eval = p_local[eval_mask]
    else:
        raise NotImplementedError(f"method {method}")
    sens, spec = sens_spec(y[eval_mask], p_eval, thr)
    return thr, sens, spec, False


def run_curve(cohort: str, method: str, ks: list[int], R: int, seed: int) -> pd.DataFrame:
    """One cohort x method learning curve; tidy rows, one per (k, draw).

    Draw seeds derive deterministically from (seed, cohort index, k, draw)
    — independent of method, so all methods see identical draws.
    """
    op = json.loads(OPERATING_POINT_JSON.read_text())
    thr_frozen = op["threshold"]

    df = load_cohort(cohort)
    y = df["y_true"].to_numpy()
    p_cal = df["y_prob_calibrated"].to_numpy()
    z = probs_to_logits(df["y_prob_raw"].to_numpy())  # calibrate.py convention
    n = len(df)
    thr_oracle = pick_operating_point(y, p_cal)  # in-sample oracle, all n images
    ci = COHORTS.index(cohort)

    rows = []
    for k in [k for k in ks if k < n / 2]:
        for draw in range(R):
            rng = np.random.default_rng([seed, ci, k, draw])
            cal_idx = rng.choice(n, size=k, replace=False)
            eval_mask = np.ones(n, dtype=bool)
            eval_mask[cal_idx] = False
            degenerate = y[cal_idx].min() == y[cal_idx].max()

            thr, sens, spec, fit_failed = draw_metrics(
                method, cohort, k, draw, y, p_cal, z, cal_idx, eval_mask, thr_frozen
            )
            _, spec_frozen = sens_spec(y[eval_mask], p_cal[eval_mask], thr_frozen)
            _, spec_oracle = sens_spec(y[eval_mask], p_cal[eval_mask], thr_oracle)
            denom = spec_oracle - spec_frozen
            recovery = (spec - spec_frozen) / denom if denom > 0 else np.nan

            rows.append({
                "cohort": cohort, "method": method, "k": k, "draw": draw,
                "threshold": thr, "sens": sens, "spec": spec,
                "spec_frozen_holdout": spec_frozen, "spec_oracle_holdout": spec_oracle,
                "recovery": recovery, "degenerate": degenerate,
                "fit_failed": fit_failed,
            })
    return pd.DataFrame(rows)


def summarize(draws: pd.DataFrame) -> pd.DataFrame:
    def q(s, x):
        return s.quantile(x)

    g = draws.groupby(["cohort", "method", "k"])
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
        "frac_fit_failed": g["fit_failed"].mean(),
    }).reset_index()
    return out


def k_star(summary: pd.DataFrame, cohort: str, method: str) -> int | None:
    """Amendment 2 (m): smallest k with median recovery >= 0.80 AND median
    sens >= 0.85. None = not reached (a valid outcome)."""
    rows = summary[(summary["cohort"] == cohort) & (summary["method"] == method)].sort_values("k")
    ok = rows[(rows["recovery_median"] >= KSTAR_RECOVERY) & (rows["sens_median"] >= KSTAR_SENS)]
    return int(ok["k"].iloc[0]) if len(ok) else None


def k_reliable(summary: pd.DataFrame, cohort: str, method: str) -> int | None:
    """POST-HOC (not in Amendment 2): smallest k with 2.5th-percentile
    recovery >= 0.5 — a draw-level reliability bar, not a median."""
    rows = summary[(summary["cohort"] == cohort) & (summary["method"] == method)].sort_values("k")
    ok = rows[rows["recovery_p2.5"] >= KRELIABLE_RECOVERY_P2]
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
    """M1-only figure: spec median + 95% band, sens median, reference lines."""
    m1 = summary[summary["method"] == "M1"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharey=True)
    for ax, c in zip(axes.ravel(), COHORTS):
        rows = m1[m1["cohort"] == c].sort_values("k")
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


METHOD_STYLE = {  # color, marker, linestyle — identity never by color alone
    "M1": ("#4c72b0", "o", "-"),
    "M2a": ("#dd8452", "s", "-"),
    "M2b": ("#55a868", "^", "--"),
    "M3": ("#8172b3", "D", ":"),
}


def plot_methods(summary: pd.DataFrame, stats: dict[str, dict], path: Path) -> None:
    """Methods comparison: rows = spec median / sens median / spec band width
    (the band-width row is the POST-HOC stability question), cols = cohorts."""
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 9), sharex="col")
    for j, c in enumerate(COHORTS):
        for m in METHODS:
            rows = summary[(summary["cohort"] == c) & (summary["method"] == m)].sort_values("k")
            ks = rows["k"].to_numpy()
            color, marker, ls = METHOD_STYLE[m]
            width = rows["spec_p97.5"] - rows["spec_p2.5"]
            axes[0, j].plot(ks, rows["spec_median"], marker=marker, ls=ls, color=color,
                            lw=1.6, ms=4.5, label=m)
            axes[1, j].plot(ks, rows["sens_median"], marker=marker, ls=ls, color=color,
                            lw=1.6, ms=4.5, label=m)
            axes[2, j].plot(ks, width, marker=marker, ls=ls, color=color,
                            lw=1.6, ms=4.5, label=m)
        axes[0, j].axhline(stats[c]["spec_frozen"], color="0.35", ls="--", lw=0.9)
        axes[0, j].axhline(stats[c]["spec_oracle"], color="0.35", ls=":", lw=1.1)
        axes[1, j].axhline(SENS_FLOOR, color="0.35", ls=":", lw=0.9)
        axes[0, j].set_title(f"{COHORT_LABEL[c]} (n={len(load_cohort(c))})", fontsize=10)
        for i in range(3):
            ax = axes[i, j]
            ax.set_xscale("log")
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
            ax.grid(True, which="major", axis="y", color="0.9", lw=0.8)
        axes[0, j].set_ylim(0.3, 1.0)
        axes[1, j].set_ylim(0.7, 1.0)
        axes[2, j].set_ylim(0, 1.0)
        axes[2, j].set_xlabel("k local labeled images (log)")
    axes[0, 0].set_ylabel("specificity (median)")
    axes[1, 0].set_ylabel("sensitivity (median)")
    axes[2, 0].set_ylabel("spec 95% band width\n(97.5−2.5 pct) — POST-HOC")
    axes[0, 0].legend(loc="lower right", fontsize=7.5, ncols=2)
    fig.suptitle("v2 methods comparison: M1 local thr · M2a refit T + frozen thr · "
                 "M2b refit T + local thr · M3 Platt + local thr\n"
                 "(row 3 band-width panel is POST-HOC) — "
                 "Research use only — not for diagnosis", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def at_k(summary: pd.DataFrame, cohort: str, method: str, k: int, col: str) -> float:
    rows = summary[(summary["cohort"] == cohort) & (summary["method"] == method)
                   & (summary["k"] == k)]
    return float(rows[col].iloc[0])


def main() -> None:
    op = json.loads(OPERATING_POINT_JSON.read_text())
    thr_frozen = op["threshold"]
    print(f"frozen threshold {thr_frozen:.4f} (models/operating_point.json) | "
          f"methods {METHODS} | R={R_DRAWS} draws | master seed {MASTER_SEED}")

    stats = sanity_checks(thr_frozen)

    print("\n=== Full run (all methods) ===")
    frames = []
    for m in METHODS:
        for c in COHORTS:
            frames.append(run_curve(c, m, K_GRID, R_DRAWS, MASTER_SEED))
            print(f"  {m} {COHORT_LABEL[c]}: {len(frames[-1])} draws done")
    draws = pd.concat(frames, ignore_index=True)

    # M1 rows must reproduce the committed first-run CSV (determinism check);
    # the committed M1 files are left untouched.
    if DRAWS_CSV.exists():
        prev = pd.read_csv(DRAWS_CSV).sort_values(["cohort", "k", "draw"]).reset_index(drop=True)
        cur = draws[draws["method"] == "M1"].sort_values(
            ["cohort", "k", "draw"]).reset_index(drop=True)
        assert np.allclose(prev[["threshold", "sens", "spec"]],
                           cur[["threshold", "sens", "spec"]]), \
            "M1 re-run does not reproduce committed v2_recalib_M1_draws.csv"
        print("M1 re-run reproduces committed v2_recalib_M1_draws.csv  OK")

    draws.to_csv(METHODS_DRAWS_CSV, index=False)
    summary = summarize(draws)
    summary.to_csv(METHODS_SUMMARY_CSV, index=False)
    plot_curves(summary, stats, CURVES_PNG)
    plot_methods(summary, stats, METHODS_PNG)

    print("\n=== Per cohort x method: k* (pre-registered) | k_reliable (POST-HOC) | "
          "medians at k=30 and k=100 ===")
    for c in COHORTS:
        print(f"\n{COHORT_LABEL[c]} — frozen spec {stats[c]['spec_frozen']:.4f}, "
              f"oracle spec {stats[c]['spec_oracle']:.4f}")
        for m in METHODS:
            ks_ = k_star(summary, c, m)
            kr = k_reliable(summary, c, m)
            print(f"  {m:3s}: k* {ks_ if ks_ is not None else 'not reached':>11} | "
                  f"k_reliable {kr if kr is not None else 'not reached':>11} | "
                  f"k=30 spec {at_k(summary, c, m, 30, 'spec_median'):.4f} "
                  f"sens {at_k(summary, c, m, 30, 'sens_median'):.4f} | "
                  f"k=100 spec {at_k(summary, c, m, 100, 'spec_median'):.4f} "
                  f"sens {at_k(summary, c, m, 100, 'sens_median'):.4f}")

    ff = summary[summary["frac_fit_failed"] > 0]
    if len(ff):
        print("\nM2 temperature-fit fallbacks (non-positive T -> frozen pipeline):")
        for _, r in ff[ff["method"] == "M2a"].iterrows():
            print(f"  {COHORT_LABEL[r['cohort']]} k={int(r['k'])}: {r['frac_fit_failed']:.1%}")
    print(f"\nsaved: {METHODS_DRAWS_CSV} ({len(draws)} rows), {METHODS_SUMMARY_CSV}, "
          f"{METHODS_PNG}; committed M1 files untouched (determinism-checked)")


if __name__ == "__main__":
    main()

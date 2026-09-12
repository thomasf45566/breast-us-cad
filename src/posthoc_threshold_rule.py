"""POST-HOC: the threshold rule as implemented vs the rule as described.

Research prototype — not for diagnostic use.

Responds to the 2026-09-11 re-audit (docs/AUDIT2_astra_2026-09-11.md §2.5
item 1 and item 2). pick_threshold.pick_operating_point selects "the
highest threshold with sensitivity >= 0.90" AMONG THE OPERATING POINTS
RETURNED BY sklearn.metrics.roc_curve WITH ITS DEFAULT
drop_intermediate=True, which drops collinear ROC points. The exhaustive
rule — the highest observed score at which sensitivity >= 0.90 holds —
can therefore differ from the frozen value. This script quantifies that,
from committed artifacts only (nothing frozen is changed):

1. Internal pooled OOF and the four external full-cohort oracles:
   frozen-routine threshold vs exhaustive threshold, with confusions.
2. Amendment 2 M1 counterfactual: the 11,000 committed M1 draws re-run
   with the exhaustive rule (same seeds, same k-sets); number of draws
   whose selected threshold changes, the largest held-out sens/spec
   change, and k* under both rules. Committed reports/v2_recalib_M1_*.csv
   are read, never written.
3. Amendment 2 M3 Platt fits: per (cohort, k), the number of draws whose
   fitted logistic slope a is NEGATIVE (a rank-reversing map), which the
   "structural equivalence" argument (RESULTS.md M2/M3 reading) must
   exclude — it holds for positive-slope monotone transforms only.

Outputs: reports/posthoc_threshold_rule.csv,
reports/posthoc_threshold_rule_M1.csv, reports/posthoc_platt_slopes.csv.

Usage: python src/posthoc_threshold_rule.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import v2_recalib_curve as rc
from calibrate import probs_to_logits
from pick_threshold import SENS_FLOOR, confusion_counts, pick_operating_point

OOF_CSV = Path("reports/oof_vit_preds.csv")
CALIBRATION_JSON = Path("models/calibration.json")
OPERATING_POINT_JSON = Path("models/operating_point.json")
M1_DRAWS_CSV = Path("reports/v2_recalib_M1_draws.csv")
OUT_RULE_CSV = Path("reports/posthoc_threshold_rule.csv")
OUT_M1_CSV = Path("reports/posthoc_threshold_rule_M1.csv")
OUT_PLATT_CSV = Path("reports/posthoc_platt_slopes.csv")


def pick_exhaustive(labels: np.ndarray, probs: np.ndarray) -> float:
    """Highest observed score s such that sens(p >= s) >= SENS_FLOOR.

    sens(p >= s) >= floor holds iff at least ceil(floor * n_pos) positives
    score >= s, so the highest such s is the ceil(floor * n_pos)-th largest
    positive score. Cross-checked against roc_curve(drop_intermediate=False).
    """
    pos = np.sort(probs[labels == 1])[::-1]
    need = int(np.ceil(SENS_FLOOR * len(pos) - 1e-12))
    thr = float(pos[need - 1])
    _, tpr, thresholds = roc_curve(labels, probs, drop_intermediate=False)
    ok = tpr >= SENS_FLOOR
    assert abs(float(thresholds[int(np.argmax(ok))]) - thr) < 1e-15
    return thr


def rule_rows(temperature: float, thr_frozen: float) -> pd.DataFrame:
    rows = []
    oof = pd.read_csv(OOF_CSV)
    y = oof["y_true"].to_numpy()
    p = 1 / (1 + np.exp(-probs_to_logits(oof["y_prob_tta"].to_numpy(np.float64)) / temperature))
    sets = [("internal pooled OOF", y, p)]
    for c in rc.COHORTS:
        df = rc.load_cohort(c)
        sets.append((f"{rc.COHORT_LABEL[c]} full-cohort oracle", df["y_true"].to_numpy(), df["y_prob_calibrated"].to_numpy()))
    for name, yy, pp in sets:
        t_routine = pick_operating_point(yy, pp)
        t_exh = pick_exhaustive(yy, pp)
        c_r, c_e = confusion_counts(yy, pp, t_routine), confusion_counts(yy, pp, t_exh)
        rows.append({"data": name, "n": len(yy), "threshold_routine": t_routine, "threshold_exhaustive": t_exh,
                     "tp_fp_fn_tn_routine": "/".join(map(str, c_r)), "tp_fp_fn_tn_exhaustive": "/".join(map(str, c_e)),
                     "sens_routine": c_r[0] / (c_r[0] + c_r[2]), "sens_exhaustive": c_e[0] / (c_e[0] + c_e[2]),
                     "spec_routine": c_r[3] / (c_r[3] + c_r[1]), "spec_exhaustive": c_e[3] / (c_e[3] + c_e[1]),
                     "images_differing": int(np.sum((pp >= t_routine) != (pp >= t_exh)))})
    table = pd.DataFrame(rows)
    assert abs(table.loc[0, "threshold_routine"] - thr_frozen) < 1e-15, "routine threshold != models/operating_point.json"
    return table


def m1_counterfactual() -> pd.DataFrame:
    committed = pd.read_csv(M1_DRAWS_CSV).sort_values(["cohort", "k", "draw"]).reset_index(drop=True)
    original_pick = rc.pick_operating_point
    rc.pick_operating_point = pick_exhaustive  # M1 only reaches the rule through this name
    try:
        frames = [rc.run_curve(c, "M1", rc.K_GRID, rc.R_DRAWS, rc.MASTER_SEED) for c in rc.COHORTS]
    finally:
        rc.pick_operating_point = original_pick
    cf = pd.concat(frames, ignore_index=True).sort_values(["cohort", "k", "draw"]).reset_index(drop=True)
    assert len(cf) == len(committed) == 11000
    assert (cf[["cohort", "k", "draw"]].values == committed[["cohort", "k", "draw"]].values).all()
    # the oracle-based denominator is recomputed under the exhaustive rule too;
    # k* below therefore uses each rule's own recovery definition
    committed_m1 = committed.assign(method="M1")
    if "fit_failed" not in committed_m1:
        committed_m1["fit_failed"] = False  # first-run CSV predates the M2 column
    summary_orig = rc.summarize(committed_m1)
    summary_cf = rc.summarize(cf)
    rows = []
    for c in rc.COHORTS:
        for k in sorted(committed[committed["cohort"] == c]["k"].unique()):
            a = committed[(committed["cohort"] == c) & (committed["k"] == k)]
            b = cf[(cf["cohort"] == c) & (cf["k"] == k)]
            changed = int((np.abs(a["threshold"].values - b["threshold"].values) > 1e-12).sum())
            rows.append({"cohort": rc.COHORT_LABEL[c], "k": int(k), "n_draws": len(a), "thresholds_changed": changed,
                         "max_abs_delta_spec": float(np.abs(a["spec"].values - b["spec"].values).max()),
                         "max_abs_delta_sens": float(np.abs(a["sens"].values - b["sens"].values).max()),
                         "spec_median_routine": float(a["spec"].median()), "spec_median_exhaustive": float(b["spec"].median()),
                         "sens_median_routine": float(a["sens"].median()), "sens_median_exhaustive": float(b["sens"].median()),
                         "kstar_routine": rc.k_star(summary_orig, c, "M1"), "kstar_exhaustive": rc.k_star(summary_cf, c, "M1")})
    return pd.DataFrame(rows)


def platt_slopes() -> pd.DataFrame:
    rows = []
    for c in rc.COHORTS:
        df = rc.load_cohort(c)
        y = df["y_true"].to_numpy()
        z = probs_to_logits(df["y_prob_raw"].to_numpy())
        n, ci = len(df), rc.COHORTS.index(c)
        for k in [k for k in rc.K_GRID if k < n / 2]:
            neg = 0
            for draw in range(rc.R_DRAWS):
                rng = np.random.default_rng([rc.MASTER_SEED, ci, k, draw])
                idx = rng.choice(n, size=k, replace=False)
                if y[idx].min() == y[idx].max():
                    continue  # degenerate draw: no fit (frozen fallback)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)  # sklearn penalty=None deprecation notice
                    slope = float(rc.fit_platt(z[idx], y[idx]).coef_[0, 0])
                if slope < 0:
                    neg += 1
            rows.append({"cohort": rc.COHORT_LABEL[c], "k": k, "n_draws": rc.R_DRAWS, "platt_negative_slope": neg})
    return pd.DataFrame(rows)


def main() -> None:
    temperature = json.loads(CALIBRATION_JSON.read_text())["temperature"]
    thr_frozen = json.loads(OPERATING_POINT_JSON.read_text())["threshold"]

    print("POST-HOC 6a — threshold rule: frozen routine (roc_curve, drop_intermediate=True) vs exhaustive")
    rule = rule_rows(temperature, thr_frozen)
    rule.to_csv(OUT_RULE_CSV, index=False)
    print("| data | thr routine | thr exhaustive | tp/fp/fn/tn routine → exhaustive | sens r→e | spec r→e | images differing |")
    print("|---|---|---|---|---|---|---|")
    for r in rule.itertuples():
        print(f"| {r.data} | {r.threshold_routine:.5f} | {r.threshold_exhaustive:.5f} | {r.tp_fp_fn_tn_routine} → {r.tp_fp_fn_tn_exhaustive} "
              f"| {r.sens_routine:.4f}→{r.sens_exhaustive:.4f} | {r.spec_routine:.4f}→{r.spec_exhaustive:.4f} | {r.images_differing} |")

    print("\nPOST-HOC 6b — Amendment 2 M1 re-run with the exhaustive rule (committed draws untouched)")
    m1 = m1_counterfactual()
    m1.to_csv(OUT_M1_CSV, index=False)
    print("| cohort | k | thresholds changed / draws | max |Δspec| | max |Δsens| | spec median r→e | sens median r→e |")
    print("|---|---|---|---|---|---|---|")
    for r in m1.itertuples():
        print(f"| {r.cohort} | {r.k} | {r.thresholds_changed}/{r.n_draws} | {r.max_abs_delta_spec:.3f} | {r.max_abs_delta_sens:.3f} "
              f"| {r.spec_median_routine:.3f}→{r.spec_median_exhaustive:.3f} | {r.sens_median_routine:.3f}→{r.sens_median_exhaustive:.3f} |")
    tot = m1.groupby("cohort")["thresholds_changed"].sum()
    print(f"changed thresholds: {int(m1['thresholds_changed'].sum())}/{int(m1['n_draws'].sum())} "
          f"({' / '.join(f'{c} {int(v)}' for c, v in tot.items())})")
    ks = m1.groupby("cohort")[["kstar_routine", "kstar_exhaustive"]].first()
    print("k* routine vs exhaustive: " + "; ".join(f"{c} {int(r.kstar_routine)} vs {int(r.kstar_exhaustive)}" for c, r in ks.iterrows()))

    print("\nPOST-HOC 6c — M3 Platt fits with NEGATIVE slope (rank-reversing), per cohort × k")
    pl = platt_slopes()
    pl.to_csv(OUT_PLATT_CSV, index=False)
    for c in pl["cohort"].unique():
        sub = pl[pl["cohort"] == c]
        print(f"  {c:6s} " + ", ".join(f"k={int(r.k)}: {int(r.platt_negative_slope)}/{int(r.n_draws)}" for r in sub.itertuples()))
    print(f"\nwritten: {OUT_RULE_CSV}, {OUT_M1_CSV}, {OUT_PLATT_CSV}")


if __name__ == "__main__":
    main()

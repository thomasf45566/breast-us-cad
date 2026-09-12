# Independent re-audit of breast-us-cad

Requested output: `AUDIT2_astra_2026-09-11.md`. Audit execution spanned 2026-09-11–12, Asia/Taipei. Repository: `/Users/thomaswang/projects/breast-us-cad-audit3`; audited HEAD: `4794ad65c182a02cf16e094152fd1ca19560e69e` (73 commits). All source references below are relative to that repository and that HEAD unless a historical commit is specified.

**Overall: FAIL — major — for the paper's claims as written.** The principal saved numerical results are reproducible. Their interpretation, predictor labeling, purported bias correction, and some protocol-compliance claims are not all correct. There is no evidence here of fabricated external scores or demonstrated patient leakage across the official BUS-BRA folds. There is substantial evidence that documentation corrections were incomplete, and the previous audit missed consequential issues.

I read `docs/AUDIT_2026-09-06.md` and `docs/AUDIT_RESPONSE_2026-09-06.md` first, as requested. I then checked the underlying code, data, histories, checkpoints, remote records and predictions independently. I did not read any other audit output under `/tmp`. No neural-network training or new external-image inference was performed. Internal fold-5 inference and the four explicitly authorized bundled demo examples were run. External metrics, recalibration experiments and comparisons were recomputed from saved predictions. Raw images were read for hashing, resize comparison and visual adjudication, without model inference.

The sole authored output is this file. No repository file, checkpoint, result or git ref was changed; final `git status --short` was empty. The no-write constraint required in-memory output adapters, mounted-path redirection, in-memory pytest temporary fixtures and suppressed runtime caches. An early matplotlib import attempted its ordinary temporary font-cache setup; subsequent runs used the no-disk harness reproduced below. I did **not** build a new environment, because that would write additional files. A fresh environment build therefore remains untested in this re-audit; a pre-existing environment was inspected and used. This limitation must not be confused with the previous auditor's environment-build result.

Evidence labels C1–C13 refer to the exact executed Python/heredoc commands reproduced in §11's command appendix. They preserve the actual local paths and audit adapters. Numerical checks used the clone's code and committed predictions, not another audit's calculations. Unless stated otherwise, bootstrap intervals use 2,000 resamples, seed 42; `±` for fold summaries is sample SD (`ddof=1`), not a CI. Verdict severity describes the residual problem, not the quality of the whole section.

## 1. Split integrity

**The official BUS-BRA partition is intact. Fold 5 is not a permanently untouched test set in five-fold CV.** It legitimately enters the training sets for checkpoints 1–4, while checkpoint 5 excludes it. That distinction matters when evaluating the five-model ensemble on internal data.

C1 independently joined raw `bus_data.csv` to the mounted official `5-fold-cv.csv`, then joined all saved OOF rows by image ID. Exact output:

```text
images 1875; patients 1064
patients occurring in multiple kFold values: 0
fold image counts: {1: 376, 2: 385, 3: 366, 4: 365, 5: 383}
OOF fold assignments matching official assignments: 1875/1875
OOF patient_id matching official Case: 1875/1875
patient pathology counts: benign 722; malignant 342
committed fold file == mounted official fold file: True
sha256: 8bed2d2efdf6ea370ab82a3cbda4c213f498274a68b434828ceeb636ce43a68d
```

`src/data.py:33` verifies the committed partition's SHA-256, with a verified raw-file fallback; `src/data.py:54` joins one-to-one and assigns `patient_id = Case`, not the image filename. The fallback and tamper-refusal tests pass. This is an exact comparison with the mounted release file, not a claim that a hash hardcoded by the author independently authenticates the publisher. The release is identified in the repository as [BUS-BRA, Zenodo 8231412](https://zenodo.org/doi/10.5281/zenodo.8231412).

Training trace: `src/cross_validate.py:59` constructs `[f for f in ALL_FOLDS if f != k]`; lines 65–66 pass those resolved lists into `train_one_fold`; `src/train.py` builds only BUS-BRA loaders for these runs. All five original wandb configs record the corresponding complement and held-out fold. Their recorded command arguments are `--config configs/vit.yaml --prefix cv_vit`. C3 read the five original checkpoints and wandb datastores read-only: epochs **29, 10, 18, 16, 19**, AUCs matching `reports/cv_vit_summary.csv` to at most 1.1e-16. The five wandb metadata records point to pre-rewrite commit `c9adedd`, mapped by identical tree to `0c23853`. Their starts are 2026-08-28 08:44:12, 10:08:07, 11:35:29, 15:10:03 and 16:42:40 +08.

The existing checkpoint YAMLs still misleadingly say `train_folds=[1,2,3,4], val_folds=[5]` in **all five** artifacts. P2 fixed `train.checkpoint_payload` for future runs (`src/train.py:94`, `:216`); it did not and should not silently rewrite frozen weights. This is a resolved future-code defect and a still-present artifact-provenance limitation. Wandb supplies the missing evidence locally; the self-description of the public weights alone remains insufficient.

Searches covered dataset construction, `train_folds`, `val_folds`, `build_master_df`, `DATASETS`, checkpoint loading, raw/network reads, configs, and historical training versions. No v1 loader includes BrEaST/BUSI/GDPH/SYSUCC. Their training use is isolated in `src/v2_data.py:74` and `src/v2_loco.py:80` under Amendment 4. The tests verify held-out-cohort exclusion, disjoint train/validation image paths, official BUS-BRA patient partitions and deterministic 85/15 external slices. External datasets without patient IDs cannot support an independent patient-level exclusion guarantee.

**Validation-label reuse is real even though split leakage is absent.** `train.py` selects the best validation epoch on the same fold later reported by `cross_validate.py`. Fold-5 model/backbone screening also uses the eventual reported fold. Saved OOF predictions inherit these decisions. P2 disclosure does not turn these into untouched test folds; see §2 and §7.

Actual suite execution (C2, all 15 tests, no omitted tests):

```text
BARE CLONE TEST SUITE (in-memory temp fixtures; no mounts)
7 passed, 8 errors in 2.20s
BARE_EXIT 1

MOUNTED SUITE (same source/tests; in-memory temp fixtures; read redirection only)
15 passed in 1.11s
MOUNTED_EXIT 0
```

The eight initial errors were missing `data/raw/...` files on the fresh clone, not failed split assertions. The mounted run redirected physical data reads to `$DATA_ROOT/raw`, keeping source/configs/keep-lists in this clone. It ran 3 checkpoint-config tests, 2 HF-path tests, 5 split tests and 5 v2 split tests. README:155 still says eight tests. The in-memory test fixture changes filesystem plumbing only; it cannot establish OS-level behavior of a newly created environment.

Had split leakage existed, I would have expected a patient with multiple fold values, an OOF/official mismatch, an actual run config including its own validation fold, an external path in the v1 training frame, or a failed exclusion test. None was found. Fold-5 reuse by the other four CV checkpoints is expected and is why an internal five-member ensemble OOF estimate does not exist.

**Verdict: CONCERN — major.** Official patient split integrity passes, and no v1 external-training contamination was found. Validation selection bias and misleading frozen-checkpoint fold metadata remain; an unqualified “patient-level splits only” claim is false for v2.

## 2. Provenance of every headline number

### 2.1 What the saved numbers actually describe

**New major finding N1: the “five-checkpoint ensemble pooled OOF” predictor does not exist.** README:33, report:22 and report:112 attach that label to 0.9254. `src/tta_eval.py:44` loops over folds, loads the corresponding single checkpoint, scores only that fold, averages original/hflip for that checkpoint, then concatenates fold predictions (`:73`, `:91`). It does not average five checkpoints for each OOF image. Protocol:364–371 explicitly says this, and RESULTS:169–171 correctly says the ensemble has no unbiased internal estimate. Report:100 also acknowledges it, contradicting its own abstract/table.

Thus the calibration temperature and threshold were fitted on a **single-held-out-checkpoint-per-image OOF distribution** and applied externally to a **five-checkpoint ensemble distribution**. The external evaluations genuinely measure the ensemble, but the internal/external operating-point comparison changes both the cohort and the predictor. The repository has not isolated how much of the probability shift comes from domain change versus that predictor change. It cannot label the internal row as ensemble performance or use it as a controlled same-predictor calibration comparison. The previous audit repeated the misleading OOF label and missed this.

### 2.2 Internal numbers: independently recomputed

| Claim / value | Recomputed result | Producer → input artifacts → RESULTS entry |
|---|---|---|
| BUS-BRA images/patients, folds, patient pathology | 1875 / 1064; 5; 722 benign / 342 malignant | `data.build_master_df`; raw metadata + committed official split; RESULTS:132, report:43; C1 |
| Image prevalence | 607/1875 = 0.3237333333 | OOF labels; `models/operating_point.json`; RESULTS:132 |
| ViT single-model CV AUC | **0.9306778395028115 ± 0.016090382884059153** | `cross_validate.py`; `reports/cv_vit_summary.csv`; RESULTS:14; C1/C3 |
| ConvNeXt / EfficientNet / CoarseDropout CV | 0.9304468697 ± 0.0172789329 / 0.8909861772 ± 0.0271385024 / 0.9329394048 ± 0.0198790712 | corresponding `cv_*_summary.csv`; RESULTS:14, :61; summary arithmetic reproduced, no retraining |
| Five ViT plain fold AUCs | .9525622822, .9411384615, .9192920054, .9131994261, .9271970223 | `tta_eval.py`; OOF CSV grouped by fold; RESULTS:84 |
| Five TTA fold AUCs | .9571124306, .9397230769, .9237804878, .9174011068, .9234433159 | same; `tta_vit_summary.csv`; RESULTS:84 |
| Pooled plain / TTA AUC | **0.9230942630405521 / 0.9254283620640372**; gain .0023340990 | `tta_eval.py`; `oof_vit_preds.csv`; RESULTS:84 |
| Temperature | **2.364384651184082**, exactly the JSON value | `calibrate.fit_temperature`, LBFGS refitted on OOF TTA logits; `models/calibration.json`; RESULTS:111 |
| ECE, 15 bins | .07156697875330724 → .04008772073058075 | `calibrate.py`; same OOF; RESULTS:111 |
| NLL | .44110173028383276 → .32365252038822134 | same |
| Frozen threshold, as implemented | **0.26832120350764344**, exactly JSON | `pick_threshold.pick_operating_point`; calibrated OOF; `models/operating_point.json`; RESULTS:132 |
| Internal confusion TP/FP/FN/TN | **548 / 290 / 59 / 978** | same |
| Sensitivity / specificity | .9028006589785832 / .7712933753943217 | same, fitted and evaluated on same OOF scores |
| Sensitivity CI | [.8741595649654773, .9278470605623914] | `patient_bootstrap_ci`, 1064 patients; RESULTS:132 |
| Specificity CI | [.7454055459205599, .7983060882247078] | same |
| PPV / NPV | .6539379474940334 / .9431051108968177 | same |
| Benign / malignant median calibrated OOF p | .06250689508979229 / .8533100421887019 | `plot_prob_shift.py`; OOF + T; RESULTS:212 |
| P2 fixed epoch | epoch 18; mean AUC .919539, SD .016396; best-minus-fixed .011138 ± .007262 | `posthoc_epoch_selection.py`; five wandb histories → `posthoc_epoch_selection.csv`; RESULTS:1102; C3 |
| P2 last epoch comparison | epoch 30; .923530 ± .018133; best-minus-final .007147 ± .005899 | same |
| P2 threshold holdout calculation | sens .9003 ± .0569, spec .7802 ± .1315; pooled decisions .9012 / .7784 | `posthoc_nested_threshold.py`; OOF + frozen T → `posthoc_nested_threshold.csv`; RESULTS:1132; C1 |
| P2 malignant probability-band count | fold 5 TTA: 32/121 in [.6,.95], 61 above .95, 28 below .6; plain band count 27 | `posthoc_overconfidence.py`; OOF → `posthoc_overconfidence.csv`; RESULTS:1163 |

“Fixed epoch” means selected post hoc as the median of the five best epochs, not specified before observing these validation curves. Calling the .0111 difference an unbiased estimate of optimism is unwarranted. The script is a useful sensitivity analysis. It does not retrain an independently selected model. Likewise, the P2 threshold analysis holds out direct threshold-fitting rows, but its input predictions are generated by overlapping training sets and validation-selected checkpoints. It is not a nested outer-model evaluation; §7 explains the dependency.

**New exact falsehood:** report:97 and RESULTS:1127–1131 say the fold ranking is unchanged. It changes from **1 > 2 > 5 > 3 > 4** at best epoch to **1 > 2 > 3 > 5 > 4** at epoch 18. The printed table itself refutes the sentence. The further RESULTS assertion that the .0111 estimate is “if anything, slightly understated” is not established by fold 3 having zero difference by construction.

### 2.3 All four external cohorts

C1 recomputed point metrics and all AUC/sensitivity/specificity CIs from `reports/external_{breast,busi,gdph,sysucc}_preds.csv` with `external_val.patient_bootstrap`. Producer: `src/external_val.py:133`; RESULTS:173–210. Each saved `y_pred` equals calibrated probability ≥ frozen threshold. Recalibrating saved raw probabilities agrees within float32 serialization precision. No fresh external inference was used.

| Cohort | n; TP/FP/FN/TN | AUC [95% CI] | Sensitivity [95% CI] | Specificity [95% CI] |
|---|---|---|---|---|
| BrEaST | 252; 90/91/8/63 | .854161145 [.801924051, .902435845] | .918367347 [.860458865, .967750515] | .409090909 [.333333333, .489655172] |
| BUSI | 379; 156/80/7/136 | .933878664 [.906165894, .958003804] | .957055215 [.925000000, .986754967] | .629629630 [.560331933, .692682927] |
| GDPH | 810; 364/238/11/197 | .915386973 [.894871447, .934271837] | .970666667 [.952873748, .986631016] | .452873563 [.406008253, .500000000] |
| SYSUCC | 1013; 674/152/50/137 | .838029785 [.809793575, .865535445] | .930939227 [.911888112, .948895028] | .474048443 [.416921369, .531838989] |

All README external table numbers and report abstract ranges round correctly. Total n = **2454**. There are **76 false negatives**, alongside **561 false positives**; these totals are a factual check of the “not missed cancers” wording, not a pooled performance analysis. Prevalences are 98/252, 163/379, 375/810 and 724/1013. SYSUCC PPV/NPV = .8159806295 / .7326203209.

| Cohort | Benign median | Malignant median | Fraction of benigns ≥ threshold |
|---|---|---|---|
| BrEaST | .30611414 | .75613073 | .590909091 |
| BUSI | .173856855 | .83086276 | .370370370 |
| GDPH | .28938693 | .76977134 | .547126437 |
| SYSUCC | .28692287 | .680447265 | .525951557 |

These reproduce `plot_prob_shift.py` → the printed shift table / saved predictions → RESULTS:212. They are descriptive distribution comparisons, not a demonstrated causal mechanism or an external calibration-error estimate.

### 2.4 Follow-up headline numbers

| Analysis | Recomputed result | Provenance and reproduction depth |
|---|---|---|
| M1 learning curves | **11,000/11,000 draws** match, max numeric difference 4.44e-16 | C7 calls `v2_recalib_curve.run_curve` on saved external CSVs; `v2_recalib_M1_draws.csv`, `_summary.csv`; RESULTS:419 |
| M1 k*, BrEaST/BUSI/GDPH/SYSUCC | **20 / 20 / 10 / 10** | exact ≥.80 recovery AND ≥.85 median sensitivity, no rounding before decision |
| k=10 median sensitivity | .829787 / .846411 / .866400 / .908648 | same |
| k=10 median specificity | .763514 / .900000 / .826553 / .554196 | same |
| Degenerate k=10 draw fraction | .008 / .006 / 0 / .024 | same; no redrawing |
| Methods summaries | all **44,000 saved method-draw rows** summarized; max difference 4.44e-16 | `_methods_draws.csv` → `_methods_summary.csv`; RESULTS:491. All M2/M3 fits were not rerun; summaries were recomputed, plus targeted independent Platt refits described below |
| M2a / M2b / M3 k* | M2a none on all; M2b 10/20/20/20; M3 20/20/10/10 | same |
| Post-hoc k_reliable | only GDPH at **200**, for M1/M2b/M3; otherwise not reached | `k_reliable`, recovery 2.5th percentile ≥.5; same artifacts; RESULTS:509, :556 |
| M2 fit-failed fraction at k=10 | .122 / .204 / .204 / .072 | saved method draws; protocol lacks this runtime fallback |
| Cross-site source threshold / T | .3755342508 / 2.2717347145; .3850788808 / 1.5803500414; .4484007610 / 1.3778190613; .3313646576 / 2.2253355980 | `v2_cross_site.py`; saved external CSVs → `v2_cross_site.csv`; RESULTS:565; all 20 source-target rows reproduced, max difference 2.22e-16 |
| Disagreement U_std AUROC | .7742127154 / .8562037474 / .8024468641 / .6256974033 | `v2_disagreement.py`; `v2_members_*.csv` + external CSVs → `v2_abstention_metrics.csv`; RESULTS:601; all 12 signal AUROCs and CIs reproduced |
| Margin AUROC | .6638938404 / .7536214769 / .7171395028 / .7470669385 | same |
| U_range AUROC | .7718360071 / .8510470792 / .7998339168 / .6256485698 | same |
| Abstention curves and verdict | all cells reproduced, max difference 4.44e-16; **0/4 useful** | `v2_abstention_curves.csv`; RESULTS:639, :663 |
| BiomedCLIP CV / pooled OOF | .9170228966 ± .0220302151 / .9108663905 | `v2_biomedclip_cv_summary.csv`, `v2_biomedclip_oof_preds.csv`; `v2_biomedclip_*` scripts; RESULTS:768, :792; C8 |
| BiomedCLIP T / threshold / internal sens/spec | 2.8644702434539795 / .2007056850 / .9011532125 / .7176656151 | refitted T and repicked threshold; threshold differs from stored by ≈4e-9 (float32 roundtrip); CIs reproduce; RESULTS:792 |
| BiomedCLIP external AUC | .8440895839 / .9101624631 / .8820659004 / .8308130532 | four `v2_biomedclip_external_*_preds.csv`; RESULTS:820 |
| BiomedCLIP ΔAUC vs v1 ensemble | −.0100715611 / −.0237162009 / −.0333210728 / −.0072167313 | same; point differences, no paired significance claim established |
| Benign shift ratios | **.8922516918 / .6890729145 / .9615164529 / .7001626175** | own OOF benign median .0808115642, external medians .29817024/.157539505/.29896015/.23793894; RESULTS:846 |
| BiomedCLIP criterion | A **0/4**, B **1/4**, **NOT CLAIMED** | SYSUCC .7001626175 > .70 fails; no beneficial rounding |
| LOCO criterion | A **4/4**, B **2/4**, **MET via A only** | `v2_loco_report.py`; `v2_loco_*_preds.csv`, saved members, JSONs → report tables; RESULTS:982, :1019; C8 |
| Segmentation | mean Dice .901602, mean IoU .832555; 280/383 Dice >.9 | `train_seg.py` / saved `seg_metrics_seg_unet_effb0.csv`; RESULTS:312. Saved-score aggregation, not mask inference rerun |
| Demo / resize | four example values and discrepancy counts reproduce; max resize probability difference **.007227** | §6; RESULTS:343, :1077 |

LOCO results, using the **registered v1 fold-5 single model** comparator:

| Held out | LOCO AUC | ΔAUC [paired 95% CI] | LOCO sens/spec | Δspec [paired 95% CI] |
|---|---|---|---|---|
| BrEaST | .865690432 | +.019016698 [−.015531937, +.055033187] | .857142857 / .753246753 | +.279220779 [.196058127, .361842105] |
| BUSI | .937144967 | +.030248807 [.007374244, .054995224] | .846625767 / .879629630 | +.273148148 [.208737864, .337839381] |
| GDPH | .945446743 | +.055558621 [.037067807, .074349824] | .960000000 / .712643678 | +.301149425 [.248350532, .357303371] |
| SYSUCC | .844414919 | +.012552811 [−.009894800, .034637792] | .799723757 / .716262976 | +.266435986 [.212221858, .320560795] |

All 12 model-by-cohort metric rows, their CIs and paired deltas reproduced to ≤1.11e-16. LOCO thresholds .4737956524/.4115960896/.5409913659/.4968149364; benign medians .2230459/.05122581/.2782694/.17059384. Against the deployed ensemble, ΔAUC is only **.0115292870/.0032663031/.0300597701/.0063851345**: two cohorts reach +.01. These qualifications are now correctly adjacent to the headline.

Reader metrics and P2 concordance were recomputed after the 2405-row XLSX/filename one-to-one check (`birads_comparison.py`, `posthoc_reader_concordance.py`; RESULTS:235, :1184). GDPH readers: .9760/.8943 and .9787/.5126, κ .5149; SYSUCC: .9130/.6505 and .9931/.1315, κ .2152. Among model FPs, reader-positive counts are GDPH 34/238 and 152/238; SYSUCC 81/152 and 143/152. The stray `c` belongs to **reader2**, not reader1 as protocol:166 states; its image is excluded by the keep-list, so it changes no evaluated metric.

### 2.5 New numerical or mathematical discrepancies

1. **Highest-threshold rule fails.** `pick_threshold.py:38` uses `roc_curve(labels, probs)` with its default `drop_intermediate=True`. That drops valid collinear operating points. As documented by [scikit-learn's primary API reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html), dropping them preserves AUC, not every possible threshold. Repeating with `drop_intermediate=False` gives the actual highest observed threshold meeting sensitivity ≥.90:

   | Data | Committed routine | Highest qualifying threshold | TP/FP/FN/TN, routine → highest |
   |---|---|---|---|
   | Internal OOF | .26832120350764344 | **.26878165315822555** | 548/290/59/978 → 547/290/60/978 |
   | BrEaST full-cohort oracle | .37553424 | **.3798318** | 90/58/8/96 → 89/58/9/96 |
   | BUSI full-cohort oracle | .3850789 | same | unchanged |
   | GDPH full-cohort oracle | .44840077 | same | unchanged |
   | SYSUCC full-cohort oracle | .33136466 | **.33504036** | 656/125/68/164 → 652/125/72/164 |

   Training-set specificity is unchanged in these examples: the defect is the explicit *highest-threshold* tie-breaking assertion, not failure to attain a maximum-specificity feasible point. The distinction matters on held-out data. An in-memory counterfactual rerun of all M1 draws with the exhaustive rule changed **2432/11000 selected thresholds** (259/2500, 372/2500, 604/3000, 1197/3000). Maximum absolute held-out specificity change by cohort was .393103/.212871/.272300/.520979; sensitivity .284091/.281046/.290411/.234463. These are maxima across draws, not typical effects. M1 k* remained **20/20/10/10**. No frozen file was changed. The historic named routine was used consistently; the strict textual description of that routine was false. Correcting it in the frozen line would require a new version and an explicit deviation, not retroactive replacement.

2. **Platt rank reversal is omitted from the “structural equivalence” explanation.** `v2_recalib_curve.py:99` fits unconstrained logistic slope `a`. Positive temperature scaling is order preserving; Platt scaling with negative slope reverses the order. Targeted refits of all k=10 draws found negative slopes on **12 BrEaST, 4 BUSI, 3 GDPH and 16 SYSUCC draws**. Examples: GDPH draw 75, slope −40.9764290141; SYSUCC draw 75, −26.6753208917. Thus report:22/:126/:160 is false when it attributes all small-k departures solely to fit-failure fallback and ties. The implementation follows the registered unconstrained `(a,b)` fit; the overclaim is mathematical interpretation, not a newly forbidden fit. Exact command C11.

3. **Temperature-difference arithmetic:** report:129 says BrEaST T 2.27 and SYSUCC T 2.23 differ from internal T by .03–.04. The actual differences are **.0926499367 and .1390490532**. RESULTS:565 and `v2_cross_site.csv` contain the temperatures that contradict the prose.

4. **Double-rounding:** the smallest P2 threshold is .23545219228291955, which rounds to **.235**, not .236 at three decimals (README:48, report:100, RESULTS:1157). SYSUCC oracle specificity = 164/289 = .5674740484, which rounds to **.567**, not .568 (report:126; previous audit also repeated .568). These are minor precision defects, with no criterion change.

5. **Resize bound:** README:99 asserts calibrated probability impact ≤7×10⁻³; saved maximum .007227 exceeds that bound. “Approximately .0072” is accurate; a rounded strict upper bound of .007 is not. Report's approximate maximum is defensible to one significant digit, but the exact bound is not.

6. **Unreproduced by the advertised artifact route:** CoarseDropout pooled OOF AUC .9139 is traced to its historical evaluation/report and code, but there is no committed CoarseDropout per-image OOF prediction CSV supporting an independent CSV recomputation. Historical latency, subjective CAM tallies and the old “GO” observation cannot all be recomputed from committed CSV/JSON files either. The original wandb histories needed for P2 are not part of the clone/archive. Therefore README:95 and report:83's universal artifact-recomputation claims are false even though the principal tables are highly reproducible. No new training or external inference was used to paper over missing provenance.

### 2.6 Consistency sweep: reproducible output, weak verification

C1 imported `scripts/consistency_sweep.py`, replaced only its output sink with an in-memory object, ran `main()`, and compared the generated text against `docs/consistency_check.md`.

```text
shared values: 156; unverified: 0
Shared values: 156 · verified: 147 · derived: 9 · unverified: 0
generated text == committed docs/consistency_check.md: True
sha256: e3705c3a9b6c57a98da47c54aab118b89d2c128bff45b0288c44fed4d7f21d12
```

That output does **not** validate claim-level consistency. `scripts/consistency_sweep.py:150` marks a token verified if any source has the same number; it ignores which cohort, metric, predictor or inequality the number belongs to. Only tokens shared by ≥2 documents enter the check (`:172`). Examples in the committed output: nested specificity SD **.1315** is “verified” by the unrelated SYSUCC reader2 specificity at RESULTS:266; nested sensitivity **.900** is matched to another result. The static `ADJUDICATION` string already says “0 UNVERIFIED”; `main` appends it without recomputing that prose (`:195`). Embedded About-text line numbers are relative to the extracted block, not always actual source lines. This check cannot catch the ensemble-label error, false temperature subtraction, swapped fold ranking, wrong inequality bound, or omitted single-document facts. The manual P6 consistency claim is therefore contradicted by the present re-audit, despite byte-identical regeneration.

**Verdict: FAIL — major.** Core numerical tables reproduce, but the internal predictor is misidentified, the stated threshold algorithm is not implemented literally, P2 is overinterpreted, and several numerical/mathematical sentences remain false. “Every number verified” is not supported by the token sweep.

## 3. Freeze-before-test timeline and claim-by-claim PROVENANCE verification

Commands included `git log --all --format='%h %aI %cI %s'`, `git for-each-ref refs/tags`, `git cat-file -p frozen-v1`, `git log --follow --name-status --format=... -- <path>`, `git log -p -- reports/external_*_preds.csv`, ancestry tests, `git diff external-v1 HEAD -- RESULTS.md`, and history/mtime checks in C3/C11. Times below are +08 unless marked UTC.

| Milestone | Commit/time independently read | Evidence / limit |
|---|---|---|
| baseline-v1; cv-v1 | 2d38e90, Aug 26 15:07:25; 5e78373, Aug 27 18:02:19 | correct lightweight tags |
| Calibration; frozen-v1 | 25ef75a, Aug 29 18:33:03; f167665, 18:38:34 | frozen tag annotated, tagger 18:38:35; ancestor of external-v1 |
| Initial external protocol | a16a64f, Aug 29 18:49:09 | after freeze, before saved external results |
| Amendment 1 | 436ab56, Aug 29 21:44:18 | includes dedup outcomes/code/artifacts; **10 files**, not a protocol-only commit |
| External-v1 | 2cbe1da, Aug 29 23:14:26 | four prediction CSVs each added exactly once; no later modification/deletion in reachable history |
| Amendment 2 → M1 | 610e87f, Aug 31 16:46:51 → fbc4bba, 17:56:44 | protocol-only amendment; original draw-file mtime 17:55:21 |
| Identity rewrite | 2c24500, Aug 31 17:58:46 | all 30 pre-rewrite mappings independently verified below |
| Amendment 3 → members → analysis | 78f3991, Aug 31 20:27:20 → 9fc3be1, 21:18:46 → 17b1e76, 21:20:19 | protocol-only amendment; first member CSV mtime 21:05:02 |
| Amendment 4 → v2 data | 3faa70b, Aug 31 22:53:53 → 2f92e65, 23:12:56 | protocol-only amendment; original `v2_data.py` mtime 23:11:34 |
| BiomedCLIP substitution → smoke | 4fc3fb2, Sep 1 18:38:49 → wandb 830ejnrt, 18:58:49 | note precedes training by 20 minutes |
| Q2 freeze → external | d1b45d7, Sep 2 14:16:38 → 568fdac, 14:34:40 | v2-biomedclip annotated tag at latter; NOT CLAIMED |
| Four LOCO evaluations | 99aeb3e/eeca3fc/59a58b1/7764aba, Sep 3 | four saved prediction files each one adding commit |
| LOCO summary | 14c3bf2, Sep 4 07:10:16 | v2-loco tag; criterion via A |
| Prior audit; P1; P2; P3 | 7a40e27 10:36:54; 5652f0a 10:59:35; 35bdb43/f74647f 11:11:39/11:13:21; 67915b0 12:12:38, Sep 7 | correct milestone table |
| Audited archive tag | fa193a8, Sep 7 13:05:16 | full hash and archive verified remotely below |
| Archive-hash documentation | f785258, Sep 7 13:05:17 | after tag; later ORCID/DOI/P6 changes also outside archived snapshot |

Original four external CSV mtimes are Aug 29 **23:00:57.303673 / 23:03:58.964636 / 23:08:16.734052 / 23:13:30.970024**. All precede their adding commit and follow freeze/protocol. `git log -p` shows new-file patches only. Calibration JSON, operating-point JSON, OOF CSV, relevant keep-lists, member files and LOCO predictions likewise have one adding commit in the inspected history. `RESULTS.md` and protocol have **zero removed lines** versus external-v1. The external-v1 numeric result section was not amended. Later licensing-driven removal of image figures is a disclosed exception to a literal “no file altered” interpretation, not a numerical rewrite.

These checks establish **reachable-history consistency**, not that nobody ever ran an uncommitted earlier evaluation, overwrote a local file before its first commit, moved a lightweight tag, or assigned dates after the fact. `--confirm` is not an immutability guarantee; §4 identifies the missing overwrite refusal. No public pre-freeze anchor exists. The current public archive cannot create one retroactively.

### PROVENANCE.md, every claim group

| Location / claim | Independent disposition and evidence |
|---|---|
| :9–12 sources, time zone, no notarization beyond §5/§7 | Source values checked against git/wandb/HF/Zenodo. The claim about what the author read on Sep 7 is historical process, not independently observable. +08/UTC conversions match except the approximate upload interval discussed below. |
| §1 :16–18 private then public; pushed after deposition | GitHub API now returns `private:false`, repository created Aug 26 02:45:16Z, pushed Sep 7 08:15:39Z. Zenodo published Sep 7 07:30:33Z, so the current recorded push follows it. Exact prior privacy and the visibility-transition instant are not recoverable from today's public API; historical privacy is disclosed, not independently notarized here. |
| §1 :19–23 every amendment precedes what it governs, append-only | Correct for model metrics and Amendments 2–4; **overbroad for Amendment 1's data audit**, which already contains its outcomes. :166 correctly qualifies that elsewhere. Append-only verified. |
| §1 :24–26 identity rewrite preserved dates/trees/messages | **Verified for all 30 pairs**, not merely inferred from matching dates. C3 read old objects from the original mounted working copy and current objects from the clone, comparing tree, author date, committer date and complete message. All equal. |
| §1 :27–30 “only public artifact” was weights, created Aug 30 04:25:49Z after external | Creation time and ordering verified. “Only” is literally false: the public Space was also created during the study (the document's own §5 lists it). Neither predates external-v1. |
| §1 :31–34 limitation permanent | Correct: internal consistency is not an independent freeze-before-test timestamp. |
| §2 :40–58 each tag/milestone row | Verified against refs, commits and run metadata; see timeline above. “Never modified” is supported only for inspected git history, not all historical filesystem states. |
| §2 :60–63 62 commits before P4, one identity, one author/committer-date exception | **Verified:** `git rev-list --count 1f196a9^` = 62; current HEAD = 73; only `6c180ee` differs by 6 seconds. |
| §3 :67–83 30 rewrites, 29 identities changed, 30th parent-only, preserved fields, recreated frozen tag | All 30 table mappings checked. The final pre-rewrite M1 commit already has the corrected identity. Current annotated frozen tag's date is consistent. The original old annotated tag object was not independently available as an external anchor; “nothing pushed” remains author testimony. |
| §3 :87–98 31 unreachable commits on author's disk, absent in clone | **Verified:** original object store retains 31 unreachable commits; clone has zero. A fresh clone/ZIP alone cannot rerun this comparison. |
| §3 :102–131 all mapping rows | **30/30 verified** for tree, author date, committer date and message, including all six wandb-cited prehashes. This is materially stronger evidence than the first audit had described. |
| §3 :132/:138–140 pre-amend draft 8e058bd | Verified: `git -C <original> diff --stat 8e058bd 6c180ee` shows only `reports/v2_biomedclip_cv_summary.csv`, **8 insertions**; author date same, committer 6 seconds later. |
| §3 :134–137 six wandb mappings | Verified in local wandb metadata; five cv_vit runs resolve via c9adedd→0c23853. These are local logs, not fetched server-side wandb timestamps. |
| §3 :142–149 no contents changed by identity rewrite; clocks consistent | Supported by identical trees and metadata checks. Does not establish that uncommitted alternate analyses never existed. |
| §4 :153–158 “each amendment was committed alone” | **False for Amendment 1**: 436ab56 changes 10 files including outputs; true for Amendments 2/3/4. README:179 correctly narrows this to 2/3/4. Numeric RESULTS append-only claim verified. |
| §4 :159–165 protocol→freeze→single run; “--confirm single-run guard” | **Incorrect order in prose:** actual freeze→protocol→external run. **Incorrect guard description:** `--confirm` authorizes execution but does not refuse an existing output. The paragraph's broader admission that evidence is self-assigned is correct. |
| §4 :166–170 Amendment 1 post-audit; TTA not preregistered | Verified by commit contents and TTA history; +.002334 pooled AUC was seen before TTA adoption. |
| §4 :171–173 CoarseDropout rule before first run | Rule at a14990f Aug 28 22:31:00; first wandb run 22:35:07, consistent. A visual gate remains subjective. |
| §5 :179–185 HF timestamps | Repo creation 04:25:49Z, fold2–4 uploads 04:28:36–04:31:27Z, fold5/seg/JSON 07:12:50–07:13:06Z, Space creation 09:24:53Z on Aug 30, Space push b7878b3 Sep 7 03:27:26Z, weights push 5ca5ba0 Sep 7 04:11:35Z all verified through HF API/commit history. **Omitted earlier first checkpoint:** fold1 uploaded 04:27:26Z. Calling fold2–4 the “first” uploads is inaccurate. |
| §5 :187–191 28/28 checksums, today's weights equal uploads, about 13 hours later | **28/28 local SHA-256 and HF checksums match.** Remote large files checked via LFS content SHA-256; small files via downloaded bytes. Historical HF file commits support original v1 weights unchanged. “About 13 hours” fits early uploads; fold5/seg arrive about **16 hours** after external-v1. None is pre-freeze evidence. |
| §6 :195–211 future OSF registration, no rewrite, freeze-time weights, Zenodo milestones | **Prospective promises**, not completed facts or a cure for v1. No new study was run in this audit. Current artifacts cannot prove future compliance. |
| §7 :215–222 archive tracked contents; metadata; no checkpoints inside | Verified by in-memory ZIP inspection. It is a git-archive snapshot, **not the git object database/history**. It contains the fold file, keep-lists, source, reports, checksum manifest and JSONs; binary weights are separate. “This commit” is stale at current HEAD. |
| §7 :223–227 tag, SHA-256, DOI, date, size, MD5 | **Verified exactly** against the live record and downloaded ZIP; details below. The history supports licensing changes and current tag target; the claim that an earlier archive was *never* uploaded/pushed cannot be established exhaustively from present refs, though the published file is the correct replacement. |
| §7 :228–229 ORCID added after archive; record authoritative | Verified: archive metadata predates 063a25a; current Zenodo creator ORCID **0009-0009-3066-3298** is present. |
| §7 :230–231 public after deposition | Current visibility and last push ordering support it; exact historical transition remains unobserved. |

Remote primary endpoints used: [Zenodo record](https://zenodo.org/records/22630912), [Zenodo record API](https://zenodo.org/api/records/22630912), [HF model API](https://huggingface.co/api/models/happytommy/breast-us-cad-weights), [HF model commits](https://huggingface.co/api/models/happytommy/breast-us-cad-weights/commits/main), [HF Space API](https://huggingface.co/api/spaces/happytommy/breast-us-cad), [GitHub repository API](https://api.github.com/repos/thomasf45566/breast-us-cad). Network requests were read-only, except the explicitly requested demo prediction calls. Initial sandbox DNS failures were retried with approved network access; an incomplete ZIP transfer was retried and only the complete download used.

### Zenodo archive identity

```text
DOI: 10.5281/zenodo.22630912
record created: 2026-09-07T07:30:33.179118+00:00
record modified: 2026-09-07T07:30:33.462566+00:00
publication date: 2026-09-07; published/done; version 1.0-audited
file: breast-us-cad-v1.0.zip
size: 26404667 bytes
MD5: 07a7c1fe4f4daf15338d51ea9620d067
SHA-256: bb45b685fa807fa14e83f01271f5256ab4cc933d5c43f53b176dd19b5b182c1e
ZIP comment: fa193a83aafc264c44a82a5e361900858585add4
remote bytes == git archive --format=zip --prefix=breast-us-cad-v1.0/ v1.0-audited: True
```

Both the remote ZIP and locally regenerated archive were held in memory, not written to another `/tmp` file. **The deposited object matches the advertised tag exactly.** However, its embedded `docs/PROVENANCE.md §7` still identifies the superseded **1f196a9** tag target and old **fcd270a5…** archive hash, with DOI pending. Current HEAD fixes those lines after the tag. A file cannot contain its own cryptographic hash, but it can avoid asserting the wrong archive identity: leave a neutral external-manifest reference, rather than a stale affirmative value. The published ZIP therefore contradicts its own provenance description even though today's web-facing SHA is correct. Also, current P6 script/check output and other post-tag changes are outside that deposition (14 changed paths since the tag); the DOI must not be presented as notarizing all current HEAD content.

**Verdict: CONCERN — major.** Reachable chronology, all 30 rewrite mappings and the Zenodo archive identity pass. Independent pre-run registration remains permanently absent; “each amendment alone” and “single-run guard” are false, and the archive carries stale self-identification. No evidence of committed external CSV replacement was found, but “never run/overwritten anywhere” is unprovable from git.

## 4. Protocol compliance

I read `data/external_protocol.md` end to end, including all four amendments and the substitution note. The following maps every rule group, including the exceptions rather than assuming a previous PASS.

| Rule | Enforcement/evidence | Result |
|---|---|---|
| Task definition: lesion benign/malignant; no normal class | `external_val.build_breast_df:66`, keep-list loaders; only binary labels in saved predictions | Followed |
| (a): BrEaST spreadsheet mapping, 4 normals excluded, main image only, 252 cases | Raw XLSX and output IDs/counts; `Classification != normal`, `Image_filename`; no mask paths in input frame | Followed; one case/image, **not an independently established one-patient-per-case identity** beyond release convention |
| (b): BUSI class prefix, normals/masks excluded, frozen dedup keep-list, image bootstrap | `external_val.py:81`; 386 raw lesion images→379, keep-list equality; filenames used as bootstrap IDs | Followed |
| (c): grayscale→3 channels→224 val transform, no masks/cropping/text removal; 5×2 probability average, logit/T then threshold | `external_val.py:50`, `BusDataset`, `get_transforms('val')`, `ensemble_probs_loader`; code trace and internal self-check | Followed for external validation; demo resize is a separately documented discrepancy (§6) |
| (d): separate cohorts, AUC/sens/spec percentile CIs n=2000 seed42, prevalence with PPV/NPV | `external_val.py:95`, `:105`, `:133`; four independent outputs/RESULTS entries; all CIs reproduced | Followed; image-unit limitation stated in RESULTS; no pooled external performance metric used |
| (d): single run, no retuning, outputs recorded regardless | One adding commit per external CSV; frozen T/threshold unchanged | History consistent; **no output-exists refusal in v1 code** |
| (e): GDPH/SYSUCC separate, prefix↔XLSX IDs, source folds ignored, dedup/conflict exclusion, prevalences | 2405 unique rows/files, exact ID set equality; own keep-list regeneration; `phash_sweep.py:76` | Followed; label source verified, no claim of independent pathology rereview |
| (f): BUSI_WHU excluded pending trustworthy labels | Absent from executable external/v2 registries, split lists and every tracked prediction/result CSV/JSON checked | Followed; comments discussing exclusion are not registry entries |
| (g): all raw-set/kept-BUSI pairs, d≤8 candidates visually adjudicated | C4 full rehash; 12,056,505 pairs; two cross-set candidates both visibly distinct; §5 | Followed as a finite pHash screen; “no training contamination” is broader than the screen proves |
| (h): reader ≥4a, case normalization, saved predictions after primary, no threshold change | `birads_comparison.py` reads saved CSV/XLSX; commit 5afac64 after external-v1; metrics and cross-tabs reproduce | Followed numerically; stray `c` reader column is misstated in protocol but excluded image makes it vacuous |
| (i): M1–M3 only saved CSV inputs, recovered logit, isolated v2 outputs | `v2_recalib_curve.py:84`, `:164`; no raw-image/model access in computation; external files unchanged | Followed |
| (j): per cohort, grid with k<n/2, 500 natural-prevalence draws, seed42, degenerate fallback without redraw | `run_curve:155`; RNG `[seed, cohort_index, k, draw]`; 11000 draws; BrEaST unique case assertion | Followed; no patient guarantee for other three cohorts |
| (k): M1 highest sens≥.90 threshold; M2 temperature refit both thresholds; M3 unconstrained Platt | `draw_metrics:108`, `fit_platt:99`, `pick_threshold:36` | Named historical routine used consistently, but **highest-threshold text is false** (§2.5) |
| (k): M2 failures | `except RuntimeError` → frozen-pipeline fallback, `fit_failed` column | **Not preregistered**; .072–.204 of k=10 fits affected; disclosed after implementation, not cured by later disclosure |
| (l): n−k complement evaluation, recovery compares same held-out rows to full-cohort oracle | `run_curve:173–194`; independently rerun | Followed; oracle is deliberately in-sample and recovery can exceed 1 or be negative |
| (l): median, **IQR and 5th–95th percentile band** | `summarize:199–216` saves median plus **2.5th/97.5th** percentiles, no IQR or 5th/95th columns | **New minor deviation** missed previously. Wider 95% draw bands do not soften a success criterion, but the registered requested summaries were replaced/omitted |
| (m): k* recovery median≥.80 and sens median≥.85, full grid, unchanged v1 | `k_star:220`; all grid cells; exact criterion reproduced | Followed; k_reliable separately post-hoc, not a registered success test |
| (n): exactly keep-list member persistence, same pipeline, per-image equality-or-abort, no metric computation | `v2_dump_members.py:33–88`; saved 10 columns and comparison; all 2454 means checked | Followed at saved-artifact level; no new external pass in this audit |
| (o): U_std, U_range, negative margin only | `v2_disagreement.py:59`, saved columns | Followed |
| (p): error target, per-cohort AUROC/bootstrap, q=5/10/20/30, stable ID tie break, retained metrics/enrichment | `error_auroc_ci:83`, `abstention_rows:114`; all cells rerun | Followed; two figures substituted for registered one, disclosed minor presentation deviation |
| (q): all three AND clauses, no sensitivity reduction at q=.10 | `verdict:147`; all four fail; no rounded comparison | Followed; post-hoc dissatisfaction with criterion does not change registered 0/4 |
| (r): no internal ensemble OOF disagreement | Correctly stated in protocol; code has only one held-out model per internal image | **Violated in documentation**, which labels internal AUC as ensemble (§2.1), not by running in-fold disagreement |
| (s): v2 isolation, LOCO held cohort excluded from train/val/calibration/early stop, v1 final | `v2_data.py:74–105`; all v2 tests; `v2_loco.py` separated stages/artifacts | Followed in inspected code/data; later removal of licensed-image figures means literal “no file” permanence has a documented nonnumeric exception |
| (t): four runs, BUS-BRA1–4+external85% train, BUS-BRA5+external15% val, seed42 | v2 data frame construction; 5 v2 tests | Followed; external slices are image-level, contradicting global README/CLAUDE “never split by image” |
| (t): early stop, fit T/threshold on training-side validation, same hyperparameters, one-model TTA | config lineage and `v2_loco.py:80`, `:164` | Broadly followed; patience **7** was not prespecified; threshold implementation defect persists |
| (t): single-shot held-out scoring | `v2_loco.py:211` asserts output absent, `--confirm`, existing artifacts | Guard exists here, unlike v1; Python `-O` disables this assertion |
| (t): comparator m5 orig/flip from saved members, separate ensemble reference, OR success branches | `v2_loco.py:247`, `v2_loco_report.py`; all metrics and CIs rerun | Followed; A4/4, B2/4; question about specificity is broader than the AUC-only passing branch |
| (u): same BUS-BRA protocol with backbone substitution, coverage before training, freeze before one external run | load/export reports, configs, timestamps, public weights; Q2 saved metrics rerun | Traceable; no new training to test original optimization trajectory |
| (u): USFM cannot cleanly map→BiomedCLIP, declare before training, no third option | 27 unusable tensors/absent pos_embed documented; substitution note before export/smoke; 150/150 BiomedCLIP coverage recorded | Consistent with code and artifacts; USFM native architecture was not tested, so conclusion must stay about substitute |
| (u): ≥3/4 AUC gains .01 OR ≥30% benign shift reduction | all four external comparisons and own OOF medians recomputed | Followed strictly; SYSUCC .7001626175 fails, not rounded to .70 |
| (v): Q2 first, failed Q2→v1 ImageNet ViT for LOCO | Q2 verdict/tag precedes all LOCO runs; configs/weights identify v1 backbone | Followed |
| Substitution note: exact resulting filenames and no third candidate | models/checksum manifest, scripts and saved artifacts | Followed in inspected history |

**New major enforcement gap:** `external_val.main:206–230` checks only whether `--confirm` was supplied. `run_external:133–160` writes `external_*_preds.csv` unconditionally with `to_csv`. Repeating the documented command would overwrite the supposedly single-shot outputs; the code does not abort first. I did not execute that destructive counterexample. The source is sufficient evidence. PROVENANCE:161 incorrectly describes this as a “single-run guard.” Absence of a guard is not proof that an overwrite happened; history shows none committed.

No rule was found to have been quietly rounded to rescue the abstention/BiomedCLIP/LOCO verdicts. The actual residual issues are a falsely described threshold algorithm, omitted registered interval summaries, runtime fallback and early-stopping choices outside the original specification, subjective CoarseDropout adjudication, and overbroad single-shot assurances.

**Verdict: FAIL — major.** Most operational rules and all three negative-result outcomes are followed, but literal compliance is not complete. The threshold rule and v1 single-run enforcement fail; additional declared and newly found deviations must remain explicit.

## 5. Data-audit reproduction

C4 independently opened raw images with PIL, computed 64-bit `imagehash.phash(..., hash_size=8)`, compared XOR bit counts for every pair, and reconstructed connected components without using the repository's dedup output as a decision oracle. It then compared generated filenames/labels to the committed keep-lists. This is a raw-data re-audit, not merely a recount of the saved hit CSV.

| Set | Raw lesion images | d≤8 within-set pairs | Removed as duplicate | Removed in label-conflict components | Final keep-list | Exact saved filename/label match |
|---|---|---|---|---|---|---|
| BUSI | 386 | 7 | 7 | 0 | **379** | True |
| GDPH | 846 | 37 | 34 | 2 | **810** | True |
| SYSUCC | 1559 | 583 | 507 | 39 | **1013** | True |

The registered full sweep uses BUSI's **379 kept images**, plus 1875 BUS-BRA, 252 BrEaST, 846 raw GDPH and 1559 raw SYSUCC: **12,056,505 unordered pairs**. The extra raw-BUSI dedup calculation is separate, so it is not silently included in that headline count. SYSUCC removal fraction is 546/1559 = 35.02%; this includes 39 conflict images, not 546 independently verified duplicate pairs. The distinction is stated in the detailed report but the “35% duplicates” shorthand is imprecise.

Cross-set candidates at d≤8:

```text
BUS-BRA bus_0999-l.png ↔ SYSUCC malignant(81).png: d=8
GDPH benign(784).png ↔ SYSUCC malignant(23).png: d=8
all other cross-set pair minima: >=10
```

I viewed the four original images directly. Both pairs are structurally different scans/lesions, so I agree with the visual refutations. Repeating the screen on **final keep-lists** leaves only the BUS-BRA↔SYSUCC candidate; the other candidate loses an endpoint through within-set dedup. Therefore **zero adjudicated cross-set near-duplicates** is supported; **zero pHash hits** is false. A supplemental pixel check independently confirms **88/154 zero-pHash-distance GDPH/SYSUCC pairs are pixel-identical** (C12). The within-BUS-BRA `bus_0012-l/-r` d=8 pair is the same patient in the same official fold and creates no cross-fold leak.

`BUSI_WHU` is absent from `external_val.DATASETS`, v2 executable cohort registries, `data/splits` and every tracked result/prediction CSV/JSON inspected (C11: `WHU_TRACKED_RESULT_HITS []`). Exclusion discussions in protocol/report are appropriately present. This audit did not attempt to recover WHU labels or score it.

The inference allowed by this evidence is finite: no visually confirmed d≤8 cross-set duplicate among these files. It does not exclude the same patient appearing with a different frame, crop, view, scanner encoding, or pHash distance >8. There are no cross-institution patient IDs to test that stronger proposition. I would have expected an unrefuted low-distance cross-set pair, mismatching connected-component keep-lists, class-conflict rows retained, or WHU paths in the registry if the reported audit had failed. None was found.

**Verdict: PASS — minor scope limitation.** The requested keep-list counts and zero **adjudicated cross-set** duplicates reproduce. Unqualified “no near-duplicate”/“no training contamination” language still exceeds the exact screen, as listed in §8.

## 6. Pipeline parity

### Source/AST and preprocessing

C5 compared source bytes and ASTs (with docstrings stripped, attributes excluded), and traced callers rather than relying on names:

```text
src/inference.py vs deploy/inference.py: byte-identical; 21 function ASTs identical
app/app.py vs deploy/app.py: byte-identical; 7 function ASTs identical
evaluate.predict_hflip_pair vs inference.predict_hflip_pair: AST-identical
```

`calibrate.probs_to_logits` and `inference.probs_to_logits` differ syntactically in where epsilon is supplied, but use the same clipping/logit convention. External/eval loaders share `BusDataset` and validation transforms. App/deploy call the single-image inference helper and integer resize; external validation calls the loader helper and albumentations/OpenCV resize. These are **two preprocessing routes**, not one shared function. Historical external-v1 was produced at 2cbe1da before `src/inference.py` was created at 2859043; current shared code is a later refactor validated by self-check.

Raw-image resize sweep, independently rerun on every relevant image:

| Set | Images with differing resized pixels | Maximum grayscale difference |
|---|---|---|
| BUS-BRA, 713 distinct original sizes | **0/1875** | 0 |
| BrEaST | **99/252** | 1 |
| BUSI | **0/379** | 0 |
| GDPH | **309/810** | 1 |
| SYSUCC | **1013/1013** | 1 |

`src/inference.py:14–24` correctly explains size-dependent OpenCV KleidiCV dispatch and the integer port. The saved `reports/v2_resize_dispatch_impact.csv` contains 1421 affected external rows, maximum calibrated difference **.007227**, median about **.000770** and two boundary flips: BrEaST **case038 .268900→.267838**, **case200 .269467→.265696**, both benign. I recomputed resize pixels and audited that saved impact CSV; I did **not** rerun external model inference to regenerate the probability-impact file. The four bundled BUS-BRA example tensors are bitwise identical across the two preprocessing routes.

Reduction order matters too. The member persistence guarantee reproduces all **2454** saved raw probabilities bitwise when using the original float32 **mean of each orig/flip pair, then mean of five pairs**. A flat float32 mean of ten entries can differ by up to 2.384e-7. The previous audit's literal “mean of 10 == bitwise” phrasing omits that detail; the actual implementation is reproducible with its original grouping.

### Self-check: actual output

The `external_val.self_check` function was executed with mounted BUS-BRA and fold-5 weights. Only paths and `num_workers=0` were adapted in memory for the read-only/inline execution; the successful run retained **MPS**. C9 produced output byte-identical to the committed file:

```text
self-check: fold-5 val, fold-5 ckpt only, n = 383
  expected (reports/tta_vit_summary.csv): 0.9234433158791244
  reproduced by external_val code path:   0.9234433158791243
  calibration (T=2.3644) + threshold (0.2683) applied: 156/383 flagged malignant
  SELF-CHECK PASSED — AUC reproduced digit-for-digit.
```

The expected CSV differs by one final floating-point digit from the reproduced AUC; the committed text already shows that distinction. A CPU run also passed with AUC **0.9234433158791243**, but flagged **155/383**, not 156. An AUC-only assertion cannot guarantee identical threshold decisions across hardware/batching. README's byte-identical-output promise needs the tested device/runtime qualification; the actual frozen MPS path reproduces.

### Live Space: four bundled examples

C6 queried the live [Space](https://huggingface.co/spaces/happytommy/breast-us-cad) using its `/predict` API and only the four bundled images, with downloaded output files disabled and no restart. Local app-path probabilities reproduced `reports/app_example_probs.json` bitwise. The remote source and image bytes were also inspected: app.py and requirements match deploy; all four example files match app/deploy; live inference.py and README lag current deploy (Space last push P1, before later HF-path/P6 changes), so full live-tree equality is **not** asserted.

| Example | Recorded/local raw p | Recorded/local calibrated p | Live calibrated p | Max absolute raw/cal difference |
|---|---|---|---|---|
| benign bus_0186-r | .02160569845000282 | .16621921816972768 | .1662192214946394 | 3.3249117e-9 |
| benign bus_0223-l | .05352667083934648 | .22882983989439548 | .22882984950348725 | 9.6090918e-9 |
| malignant bus_0328-r | .8696613252162934 | .6905586133554297 | .6905585278113721 | **1.0728836064e-7** |
| malignant bus_0663-l | .8794305622577667 | .698551227617947 | .6985512526504918 | 2.9802322e-8 |

All four decisions match. API wall times were approximately **31.54, 4.57, 4.897, 4.789 seconds**; median after the first call ≈**4.789 s**. The first request is not a controlled cold-start benchmark. The historical 6–9 s range traces to earlier measurements, but is not a guaranteed live latency interval. No mathematical discrepancy is implied by normal service timing variation.

Had parity been broken, I would have expected mismatching AST bodies, incorrect weight/calibration IDs, differing BUS-BRA tensors, a failed self-check, or demo probabilities outside the recorded numerical floor. None occurred for the tested frozen path/examples. The all-external resize discrepancy is real, already documented in code, and remains a deployment-parity limitation.

**Verdict: CONCERN — minor.** Core inference, MPS self-check and live example probabilities pass. Preprocessing is not universally shared, CPU decisions are not byte-identical, and the live tree is behind current deploy. The code docstrings accurately describe resize dispatch; README's strict .007 bound does not.

## 7. Statistical claims

**Bootstrap units are implemented as registered, but they do not solve selection bias or missing patient identity.** `pick_threshold.patient_bootstrap_ci:67` resamples complete BUS-BRA patient clusters; `external_val.patient_bootstrap:105` groups by case IDs for BrEaST and filename IDs for other cohorts. `v2_disagreement.error_auroc_ci:83` uses the corresponding row/case units. `v2_loco_report.paired_delta_ci:93` uses the same sampled rows for both models in each paired difference. All intervals in the external, disagreement and LOCO result tables were recomputed (§2). Recalibration ribbons are **empirical variation over overlapping train/holdout draws**, not bootstrap CIs for a population parameter, and are not independent replication studies.

The internal sensitivity/specificity CI holds the data-selected threshold fixed while resampling the same OOF set. It is conditional on those scores and that threshold; it omits checkpoint, TTA-adoption and threshold-selection uncertainty. “In-sample” is now correctly stated, but the P2 rows are not a complete repair:

- For outer fold k, P2 chooses a threshold from OOF predictions on folds j≠k. Each checkpoint producing those scores was trained on fold k. The threshold therefore depends indirectly on the outer evaluation data/labels through fitted checkpoints. The checkpoint evaluated on k was also selected using k's validation AUC. There is no outer-training/inner-calibration retraining loop. This is a leave-fold-out threshold sensitivity calculation on a fixed, already selected OOF artifact, not fully nested out-of-sample model evaluation.
- The common positive frozen T is also fitted using all OOF labels, but a strictly increasing common transform cancels from exact rank-threshold decisions. I do **not** claim that T alone causes additional decision leakage here; the checkpoint-selection and overlapping-training dependencies are the substantive defects.
- Epoch 18 is chosen using the same five validation histories. The .0111 difference quantifies sensitivity to choosing this post-hoc epoch, not the unknown amount of optimism relative to an independent development/test protocol. No unbiased corrected ensemble AUC was produced.

| Language/claim | Independent evidence | Assessment |
|---|---|---|
| U_std beats margin on BUSI/GDPH with nonoverlapping CIs | BUSI [.811859,.894995] vs [.694308,.808911]; GDPH [.771759,.832895] vs [.682380,.751755] | Supported by the stated conservative nonoverlap criterion; no multiplicity-adjusted inference claimed |
| BrEaST disagreement superiority | U_std [.710037,.831430] vs margin [.594550,.728604] overlap | Point estimate higher, not established by nonoverlap. Current report:132 correctly says overlap |
| “3/4 … 優於 margin” in report:166 | Three higher AUROC point estimates, only two nonoverlapping CI pairs | Must say **point estimates**; “signal is real” is too strong if read as three demonstrated comparative improvements |
| LOCO AUC criterion MET | Exact preregistered point rule: four ΔAUC≥.01 vs single model | Correct as a decision-rule result, not proof of positive population ΔAUC on all four |
| BrEaST/SYSUCC LOCO ΔAUC gain | Paired CIs [−.015532,.055033], [−.009895,.034638] | **Cross zero.** Current headline now discloses this correctly |
| BUSI/GDPH LOCO ΔAUC gain | Paired CIs [.007374,.054995], [.037068,.074350] | Exclude zero; GDPH unpaired AUC CIs also do not overlap |
| All LOCO Δspec CIs exclude zero | §2 table | True, but thresholds differ and sensitivity is lower on three cohorts; this is not a fixed-operating-point causal training-effect estimate |
| BiomedCLIP lower on all four | Four negative AUC point differences, overlapping unpaired intervals | Correct only as point estimates; report:135 now qualifies this |
| GDPH Reader1 “significantly” higher | No model-vs-reader paired test/CI | Original unsupported significance wording removed; current report:120 says point estimate and untested |
| ConvNeXt and ViT “統計平手” (report:97) | Similar CV means, correlated folds, no equivalence/noninferiority test | Unsupported statistical equivalence claim. Also Δ means = **.0002309698**, which rounds .0002 correctly |
| 10–20 labels recover specificity | Median criterion passes; k=10 sens .830/.846/.866/.909; broad draw bands | Supported only with median/sensitivity/reliability qualifications now mostly present |
| k_reliable “reliability” | Post-hoc 2.5th percentile of recovery ≥.5 | This metric alone contains **no sensitivity floor**. It is not a guarantee of clinical reliability or a prospective sample-size requirement |
| Internal versus external calibration shift | Different predictor distributions (§2.1); benign medians and fixed-threshold operating points | Descriptive finding supported; model calibration transport failure as an isolated mechanism not identified |
| “位置而非尺度” logit shift (report:160) | Temperature-only adaptation underperforms | Does not exclude combined location/scale changes, shape changes, prevalence effects or predictor mismatch; no direct shift-model comparison |
| All external mistakes are FPs, not missed cancers | 76 FNs in saved confusion matrices | False; sensitivity .918–.971 is not 1.0 |

The registered failure criteria were not softened to turn any of the three negatives into a positive. CoarseDropout's subjective visual gate cannot be “mechanical” in the same sense as arithmetic tests; its actual discard is recorded. AUC is not a calibration metric, confidence-interval overlap does not establish equivalence, and a fixed-threshold specificity increase is expected whenever a higher threshold is transported. These elementary distinctions remain blurred in the discussion.

**Verdict: FAIL — major.** CI resampling units and numerical intervals pass, and the prominent old CI errors were corrected. The asserted nested bias correction, statistical tie, monotonic equivalence, mechanistic calibration conclusion and no-missed-cancers language are not supported.


## 8. Sentence-level overstatement scan

The following are the current sentences/passages a hostile reviewer can reasonably challenge. Quotes are copied directly from HEAD, including complete source lines where several sentences share a line; the reason distinguishes the problematic part from adjacent correct material. Repeated claims are identified at each location. Qualitative, future-plan and literal-wording concerns are not being presented as proof of falsified results.

### O1. README.md:18–25

> Malignant-class Grad-CAM localizes to the lesion body and margin
> (bottom row). Benign-class Grad-CAM is diffuse by nature — 'benign
> evidence' is the absence of a suspicious focus — and can include the
> burned-in annotation text present in every BUS-BRA image (top row). This
> is a documented limitation of the training set; the border-occlusion probe
> in RESULTS.md shows its effect on predicted probabilities is small. CAMs
> are single-model (fold-5) visualizations; decisions come from the 5-model
> ensemble.

**Reason:** The caption generalizes from selected CAMs to what benign evidence is “by nature.” The border probe measured malignant fold-5 images only, not these benign examples; median ViT drop .00469 coexists with 20/121 drops >.2 and maximum .86569. It cannot show the causal effect of annotation text is small. The single-model CAM/ensemble-decision distinction itself is correct.

### O2. README.md:33–34

> | Internal (b): 5-ckpt ensemble + hflip TTA + T, pooled OOF (n=1875) | 0.9254 | 0.903 † | 0.771 † |
> | Internal (b′, POST-HOC): nested operating point, threshold from the other 4 folds, evaluated per fold | — | 0.900 ± 0.057 | 0.780 ± 0.132 |

**Reason:** Row (b) is one held-out checkpoint per image plus flip, not a five-checkpoint ensemble. Row (b′) is threshold-row holdout on already selected OOF predictions, not a fully nested model evaluation (§2.1, §7).

### O3. README.md:40–49

> Rows (a) and (b) are two different predictors. In (a) each fold's AUC is
> that of the checkpoint saved at the epoch with the best AUC on the same
> held-out fold (`best_epoch` in `reports/cv_vit_summary.csv`), so the CV
> mean is optimistically biased; row (a′) quantifies it post hoc from the
> wandb histories (optimism 0.011 ± 0.007 AUC). The pooled OOF predictions
> in (b) inherit those checkpoints. † Sens/spec in (b) are **in-sample**:
> the frozen threshold 0.2683 (sens ≥ 0.90 rule) was fitted on the same
> calibrated pooled-OOF predictions it is evaluated on; row (b′) is the
> nested out-of-sample estimate (thresholds 0.236–0.311; held-out sens
> misses 0.90 on 3/5 folds). Both POST-HOC rows: RESULTS.md "Post-hoc

**Reason:** Separating predictors is appropriate, but the .011 difference is a post-hoc epoch sensitivity comparison, not an unbiased quantified correction. The claimed nested out-of-sample estimate retains training/selection dependencies. The lowest threshold rounds .235, not .236.

### O4. README.md:53–59

> **Key finding:** discrimination transfers reasonably (BUSI and GDPH
> within or near the internal range; BrEaST and SYSUCC lower, both CIs
> below the internal 0.9254); the operating point does not. The frozen
> threshold held sensitivity 0.918–0.971 on every external cohort while
> specificity dropped from 0.771 (in-sample internal) to 0.409–0.630, i.e.
> the errors are false positives, not missed cancers — whether that
> direction is "safe" was not assessed (no harm analysis). Follow-up

**Reason:** The relative AUC qualification and harm-analysis caveat are welcome, but the categorical ‘not missed cancers’ claim is false: 76 external false negatives. Sensitivity below 1 cannot support it. The internal/external comparison also changes predictor.

### O5. README.md:77–78

> - **Patient-level splits only**, enforced by tests (need the raw datasets
>   on disk)

**Reason:** False globally: v2 external training/validation slices are image-level, explicitly authorized at protocol:430. Tests enforce that design, not patient independence on unidentified external patients.

### O6. README.md:86–90

> - **Dataset auditing**: 12.06M-pair perceptual-hash sweep (all
>   within- and cross-set pairs among 1875/252/379/846/1559 images; 35%
>   duplicates found in one public cohort, incl. the same image released
>   under both class labels). Result: no near-duplicate at pHash d ≤ 8 —
>   not "zero overlap"

**Reason:** 12.06M is correct. The conclusion does not specify cross-set/adjudicated scope: one same-patient within-BUS-BRA d=8 pair and one refuted final-keep-list cross-set d=8 hit remain. The finite screen cannot exclude different images from the same patient. The 35% removal includes label-conflict components.

### O7. README.md:95–99

> - Every number in RESULTS.md recomputes from committed CSV/JSON
>   artifacts; the deployed demo shares the same model, calibration and
>   inference module as the validation pipeline, while its preprocessing
>   differs from the validation path on non-BUS-BRA image sizes
>   (documented; impact on calibrated probability ≤ 7×10⁻³)

**Reason:** Universal CSV/JSON reproducibility is false (historical timing, subjective CAM outcomes, no CoarseDropout OOF CSV, private wandb inputs). Sharing core inference does not share preprocessing. The strict ≤.007 bound is exceeded by .007227.

### O8. README.md:123–131

> any file whose sha256 differs from the official one. All checkpoints
> (`models/*.pt`) are git-ignored and published at
> [happytommy/breast-us-cad-weights](https://huggingface.co/happytommy/breast-us-cad-weights):
> the five frozen-v1 classifiers, the segmentation model and the two JSONs
> at the repo root (the only files the demo uses), and the v2 research
> artifacts under `v2/` (`v2_biomedclip_fold{1-5}.pt`,
> `v2_loco_{breast,busi,gdph,sysucc}.pt`, their JSONs, and the exported
> BiomedCLIP init). `models/CHECKSUMS.txt` holds the sha256 of every
> published file; `inference.resolve_weight` resolves both layouts.

**Reason:** The listed published v1/v2 release artifacts are verified, but ‘All checkpoints (models/*.pt)’ is broader than the release: earlier EfficientNet, ConvNeXt and CoarseDropout checkpoints on the original disk are not all published. Restrict the quantifier to the listed release artifacts.

### O9. README.md:150–156

> This re-runs the frozen pipeline on BUS-BRA fold 5 and must reproduce
> AUC 0.9234433158791243 digit-for-digit (output byte-identical to
> `reports/external_selfcheck.txt`). If `models/cv_vit_fold5.pt` is
> absent, `inference.resolve_weight` **downloads it from the HF weights
> repo automatically** (network access; default local HF cache) — there
> is no offline switch. `pytest` runs the 8 split tests (all five raw
> datasets required).

**Reason:** Byte-identical self-check output is verified on MPS with the tested runtime; CPU gives the same AUC but 155 versus 156 flagged images. There are now 15 tests, not eight. Automatic download/no-offline-switch disclosure is accurate.

### O10. README.md:182–186

> - On 2026-08-31 the identity of all prior commits was rewritten with
>   `git filter-branch` (hostname-derived email → the author's email);
>   author/committer dates were preserved, tags carried over, and the
>   `frozen-v1` annotated tag re-created at the same date (commit 2c24500
>   message). Pre-rewrite hashes therefore do not map to current commits.

**Reason:** The rewrite changed hashes, but PROVENANCE now supplies a verified complete 30-pair mapping. ‘Do not map’ misstates the current evidence; distinguish absent old objects in a clone from a documented mapping.

### O11. README.md:187–192

> - The repository was **private at audit time (2026-09-06)** and is public
>   since 2026-09-07 (https://github.com/thomasf45566/breast-us-cad); the only public artifact created during the
>   study, the HF weights repo (2026-08-30), postdates the external-v1
>   commit (2026-08-29). The freeze-before-test ordering therefore rests on
>   local, rewritten, self-assigned commit timestamps; the Zenodo DOI above
>   is the first external timestamp, post hoc for external-v1.

**Reason:** The self-certified chronology limitation is correctly described, but the HF weights were not the only public artifact during the study: the Space also existed. Exact historic privacy is not established by today's public API. Zenodo is post-hoc for v1, correctly admitted.

### O12. README.md:203–206

>   Done (P4, 2026-09-07): [docs/PROVENANCE.md](docs/PROVENANCE.md) — tag
>   timeline, identity-rewrite evidence (full pre→post commit mapping by tree
>   hash), HF timestamps, forward OSF commitment; CITATION.cff and
>   .zenodo.json prepared; Zenodo DOI pending upload.

**Reason:** The pending-upload statement is stale and contradicts README:244 and the live published DOI.

### O13. README.md:224–229

> All datasets are public research releases; none are redistributed in
> this repo. Download links, citations, and license notes for all five
> cohorts in [data/README.md](data/README.md): BUS-BRA, BrEaST, and
> Curated BUSI are CC BY 4.0; GDPH and SYSUCC come from the HoVer-Trans
> release (Mo et al., *IEEE TMI* 2023, DOI 10.1109/TMI.2023.3236011) —
> no license file accompanies that release, so check its original terms.

**Reason:** ‘None are redistributed’ is ambiguous and conflicts with the explicitly redistributed BUS-BRA example images and dataset-derived figures/partition. ‘Complete raw datasets are not redistributed’ would be precise. This audit does not make a legal licensing judgment.

### O14. docs/report.md:10–12

> > responding to the 2026-09-06 audit」)。每一事實句皆對應 RESULTS.md、data/external_protocol.md 或 reports/、
> > models/ 下已提交之 CSV/JSON;無來源之句已刪除,或明確標示為「未記錄之
> > session 內觀察」。逐項對照表見 docs/AUDIT_RESPONSE_2026-09-06.md。

**Reason:** The universal sourcing assurance is contradicted by the false internal predictor label, temperature subtraction, fold ranking and reader-study information claim below.

### O15. docs/report.md:18

> **背景:** 台灣女性乳房緻密比例高,超音波為第一線工具,但判讀者間變異顯著。深度學習模型之跨機構可攜性——尤其 operating point 層級——鮮少被嚴謹評估。

**Reason:** High density has a traceable external source, but broad first-line-use and rare-rigorous-evaluation claims are not established by the repository's experiments. They need appropriately scoped clinical and literature citations, not RESULTS.md as a substitute.

### O16. docs/report.md:20

> **方法:** 以 BUS-BRA(1,875 張 / 1,064 位病人,病理確診)依官方 patient-level 五折訓練 ViT-B/16,經 hflip TTA 與 temperature scaling,於 pooled out-of-fold 預測上以 sensitivity ≥ 0.90 規則預定 operating point 後凍結,再依 pre-registered protocol 於四個外部世代(BrEaST、去重 BUSI、GDPH、SYSUCC;共 2,454 張)單次驗證。外部世代與訓練集之污染檢查為全對感知雜湊比對(五個集合 1,875/252/379/846/1,559 張,集合內與集合間全部無序配對共 12,056,505 對;d ≤ 8 之候選經目視裁決),結果為「pHash d ≤ 8 無近重複」,而非「零重疊」。其後以四個 protocol amendments 進行部署導向後續研究:site-specific recalibration 學習曲線、跨場域閾值轉移(post-hoc)、ensemble disagreement 之 abstention 分析、以及領域預訓練骨幹(BiomedCLIP 替代)與多來源訓練(leave-one-cohort-out, LOCO)之 pre-registered 比較。

**Reason:** Dataset sizes and 12,056,505 pairs reproduce. The unqualified ‘pHash d ≤ 8 無近重複’ needs cross-set, visually adjudicated and finite-screen scope. Same-patient overlap beyond the pHash cutoff remains untestable.

### O17. docs/report.md:22

> **結果:** 內部指標分為兩個不同的預測器:(a) 單模型、無 TTA 之五折 CV AUC 0.9307 ± 0.0161,各折取該折 validation AUC 最佳之 epoch 之檢查點(best-epoch on the reported fold;POST-HOC 固定 epoch 下 0.9195 ± 0.0164,樂觀量 0.011 ± 0.007);(b) 五模型 ensemble + TTA + T 之 pooled OOF AUC 0.9254。凍結 operating point(閾值 0.2683)之 sens 0.903 / spec 0.771 係於同一批 OOF 預測上擬合閾值後之 in-sample 值(POST-HOC 嵌套 out-of-sample 估計 0.900 ± 0.057 / 0.780 ± 0.132)。外部 AUC 0.838–0.934,sensitivity 皆 ≥ 0.918,specificity 降至 0.409–0.630;伴隨良性校準後機率中位數由 0.06 右移至 0.17–0.31(描述性)。判讀者比較:GDPH 模型於 sensitivity 與 specificity 兩軸皆低於兩位判讀者;SYSUCC 模型居兩位判讀者之間;κ(0.22 / 0.51)為判讀者間一致性,不涉及模型。後續研究:10–20 例本地標註即可於中位數上恢復接近 oracle 之 specificity(pre-registered k*),但 k=10 之中位 sensitivity 為 0.83–0.91;post-hoc 之 draw-level 可靠性指標 k_reliable 僅 GDPH 於 k=200 達標,其餘三世代於各自 pre-registered 網格內未達;單調機率校準接本地閾值重選與直接重定閾值為結構性等價(實跑曲線於 k ≤ 30 因擬合失敗回退與同分而有小差異);外部世代彼此借用閾值之 specificity 皆高於訓練集閾值(post-hoc、in-sample 閾值、描述性;GDPH 閾值使跨場域 sens 降至 0.80–0.87);BiomedCLIP 替代骨幹依 pre-registered 判準 **NOT CLAIMED**(分支 A 0/4、分支 B 1/4;外部 AUC 點估計於四世代皆較低);LOCO 多來源訓練達成 pre-registered 判準(分支 A:ΔAUC ≥ +0.01 vs v1 fold-5 單模型於 4/4),但 paired ΔAUC CI 於 BrEaST、SYSUCC 含零,分支 B 2/4 未達,對照為單模型(相對已部署之 ensemble 僅 2/4 世代 ≥ +0.01),held-out sensitivity 0.80–0.96。

**Reason:** The ensemble OOF label is false; the post-hoc nested estimate is not fully independent; structural equivalence omits actual negative Platt slopes. Other listed external numbers, k* values, negative verdicts and newly adjacent LOCO caveats reproduce (§2).

### O18. docs/report.md:24

> **結論:** 判別力可跨洲際遷移而 operating point 不可(四世代單發驗證)。外部誤判方向為假陽性增加而非漏診;此方向是否「安全」未經危害分析。訓練端修正中,多來源訓練達成其 pre-registered 判準(附上述保留),領域預訓練替代品未達;部署端之本地閾值重選於中位數有效但小樣本下 draw-level 不可靠。**部署之最後一哩為以本地資料設定 operating point**,此為本研究最有支撐之結論。

**Reason:** ‘而非漏診’ is contradicted by 76 false negatives. Local threshold-setting is a supported research direction; calling it the final deployment step overstates completeness in the absence of clinical validation. Higher local thresholds can reduce sensitivity.

### O19. docs/report.md:30

> ### 1.1 臨床脈絡

**Reason:** The >80% dense-mammogram figure for Taiwanese women under 55 is supported by the cited Chang study. The broad ‘significantly superior sensitivity’/first-line inference is not established by that density study or this repo and needs a task-specific clinical source. κ is descriptive inter-reader agreement, not a general-purpose need assessment.

### O20. docs/report.md:34

> 既有文獻多以單一資料集之影像層級隨機切分報告 AUC 0.93–0.98,存在 patient-level leakage 與資料品質問題(BUSI 之重複影像已見諸文獻);外部驗證研究普遍顯示 AUC 降至 0.85–0.88,而 operating point(閾值層級)之可攜性、及其部署端與訓練端修正策略之系統性比較,均鮮少報告。

**Reason:** The literature-wide AUC ranges .93–.98 and .85–.88, frequency of image splitting/leakage and novelty claim have no systematic supporting source list or extraction in the repo. Cannot be reproduced from the study's prediction CSVs.

### O21. docs/report.md:43

> ### 2.1 訓練資料

**Reason:** Dataset counts/folds are verified, but ‘不在版本庫內’ is now false: P3 committed the exact official partition.

### O22. docs/report.md:49

> TTA 為原圖與水平翻轉之機率平均;**TTA 之採用未經 pre-registration**——係於看到 pooled OOF AUC +0.0023 後決定(凍結前之研究者自由度,作用於其後用以擬合 T 與閾值之同一批 OOF 資料)。Temperature 於 pooled OOF logits 以 LBFGS 擬合;operating point 於校準後 OOF 上取 sensitivity ≥ 0.90 之最高閾值,patient-level bootstrap(2,000 次)估 CI。最終管線(五模型 ensemble + TTA + T + 閾值)以 `frozen-v1` 凍結。

**Reason:** The highest-threshold assertion is contradicted by the default roc_curve pruning (§2.5). The other preprocessing/TTA-adoption disclosures in this paragraph are traceable.

### O23. docs/report.md:83

> Git tags:`baseline-v1`→`cv-v1`→`frozen-v1`→`external-v1`→`v2-biomedclip`→`v2-loco`。RESULTS.md 每一節列出產生腳本與 reports/、models/ 下之 CSV/JSON 產物,所有表列數值可由該等已提交檔案重算。**不在版本庫內者:** 所有模型檢查點(models/*.pt,.gitignore)與 BUS-BRA fold 檔(data/raw/busbra/5-fold-cv.csv)。v1 之五個 cv_vit 檢查點、分割模型與兩個 JSON 公開於 HF 權重庫 happytommy/breast-us-cad-weights;v2 之九個檢查點(v2_biomedclip_fold{1-5}、v2_loco_{breast,busi,gdph,sysucc})與 BiomedCLIP 匯出權重尚未公開,Q1/Q2 目前僅能自已存預測 CSV 層級重現。MPS 訓練無 bitwise 決定性(未啟用 deterministic algorithms、DataLoader workers=4),檢查點本身無法逐位重現;推論碼每次變更重跑 self-check。

**Reason:** P3 published all listed v2 release weights and committed the official fold file. The statements that they remain unavailable and Q1/Q2 can only be reproduced at CSV level are false. The universal table-recomputation claim also exceeds committed artifacts.

### O24. docs/report.md:88–90

> - 稽核當時(2026-09-06)GitHub 版本庫為私有;唯一公開產物為 HF 權重庫(2026-08-30 建立),其時間晚於 external-v1 commit(2026-08-29)。凍結先於外部評分之時序,目前僅有本機、已重寫、自行指定之 commit 時間戳與檔案 mtime 為據,無第三方紀錄。
> - Amendment 1 於登錄時已含其所登錄之去重計數與 pHash 結果(對模型指標而言仍在先)。
> - 公開時間戳(Zenodo/OSF)列於後續工作。

**Reason:** The historic lack of a pre-run public anchor remains true; ‘only public artifact’ omits the Space. Public timestamping is no longer merely future work: a post-hoc Zenodo deposition exists.

### O25. docs/report.md:97

> 五折 CV(單模型、無 TTA、各折 best-epoch 檢查點,見 §2.2):EfficientNet-B0 0.891 ± 0.027;ConvNeXt-Small 0.9304 ± 0.0173 與 ViT-B/16 0.9307 ± 0.0161 統計平手(Δ 0.0002),ViT 因 CPU 推論快 9 倍(40 vs 376 ms)與 Grad-CAM 可用性獲選。可解釋性稽核:ViT 末層 CAM 空白為飽和 logit 之方法 artifact(blocks[-2].norm1 可用);所有檢視影像含燒錄標註;occlusion 中位降幅 < 0.01(伴重尾,~16% 影像降幅 > 0.2,受病灶延伸至邊界混淆)。校準前 ensemble 明顯 overconfident(T = 2.36);校準後中段機率仍殘餘輕度 overconfidence(reliability diagram)。CoarseDropout:CV 0.9329 ± 0.0199、pooled OOF 0.9139 以 0.0008 壓線過 gate 1、gate 2 未過(1/3 改善、1 不變、1 熱圖改善但預測 0.75→0.28)→ 依規則棄用;gate 2 為三張影像之主觀目視判斷。**POST-HOC 量化(2026-09-07,RESULTS.md「Post-hoc analyses」§1、§3):** 五折 best-epoch AUC 0.9307 ± 0.0161 對固定 epoch 18(五個最佳 epoch 之中位數)之 0.9195 ± 0.0164,樂觀量 0.0111 ± 0.0073(對最終 epoch 30 為 0.0071 ± 0.0059),折間排序不變;fold-5 121 張惡性中,校準前 TTA 機率落於 [0.6, 0.95] 者 32 張(26.4%),> 0.95 者 61 張——先前「僅 3 張」之數字無來源且不被重現。

**Reason:** No equivalence test establishes ‘統計平手’; T is fitted on single-model OOF, not an internally evaluated ensemble. The fixed-epoch fold ranking DOES change (folds 3 and 5 swap). The best-minus-fixed difference is not an unbiased optimism estimate. The heavy-tail occlusion qualification is appropriately present here, unlike the README caption.

### O26. docs/report.md:100

> TTA:pooled OOF 0.9231→0.9254(採納,未 pre-registered)。T = 2.3644;ECE(15 bins)0.0716→0.0401;NLL 0.4411→0.3237;AUC 不變(assert)。Operating point 0.2683:sens 0.9028(0.874–0.928)/ spec 0.7713(0.745–0.798)/ PPV 0.654 / NPV 0.943;混淆 548/290/59/978。**此 sens/spec 為 in-sample:閾值於同一批 pooled OOF 預測上擬合**(Amendment 2 (l) 對同一運算之稱謂為「in-sample oracle」)。**POST-HOC 嵌套估計(2026-09-07,RESULTS.md「Post-hoc analyses」§2):** 以其餘四折選閾值、於第 k 折評估,五個閾值 0.236–0.311(0.2753 ± 0.0354),out-of-sample sens 0.9003 ± 0.0569 / spec 0.7802 ± 0.1315(pooled 決策 0.9012 / 0.7784)——平均值與 in-sample 相近,但 held-out sens 於 3/5 折低於 0.90 設計下限(0.849–0.976),spec 介於 0.550–0.878。註:ensemble 無無偏內部估計(每張影像對 5 成員中 4 者為 in-fold),外部驗證為其首次考試。

**Reason:** Threshold-row holdout is called nested out-of-sample despite checkpoint-selection and cross-fold training dependencies; the .236 minimum is double-rounded. The final admission that the ensemble has no unbiased internal estimate directly contradicts abstract/table labeling.

### O27. docs/report.md:112

> | 內部(pooled OOF,ensemble+TTA+T;sens/spec 為 in-sample)| 0.925 | 0.903 | 0.771 |

**Reason:** The internal table's ‘ensemble+TTA+T’ label is false; it is a pooled single-held-out-checkpoint OOF estimate.

### O28. docs/report.md:123

> 內部 TP 熱區於病灶本體與邊緣;內部 FP 3/4 聚焦真實可疑結構、1/4 顯示殘餘標註敏感;外部 FP 7/8 熱區落於病灶本體(SYSUCC 四張皆低回音分葉狀良性)——與 appearance-driven 之解讀相符,但 CAM 不能量化之。註(2026-09-07):該 GDPH/SYSUCC 外部 FP 圖庫因資料集無明確授權,不隨版本庫散布(僅作者本機保留);版本庫內改附 BrEaST 良性假陽性前 8 例之同設計圖庫(reports/gradcam_external_fp_breast.png,CC BY 4.0),其閱讀未納入上述 7/8 之陳述。

**Reason:** The 3/4 internal FP and 7/8 external FP CAM readings are selected-image qualitative tallies, not localization accuracy or causal evidence. The report correctly notes that the original external gallery is not distributed; consequently 7/8 is not independently reviewable from the clone's replacement BrEaST gallery. It must remain a trace-only qualitative observation, not a verified numerical headline.

### O29. docs/report.md:126

> Sanity:k=0 逐位重現 external-v1;oracle 閾值 BrEaST 0.376 / BUSI 0.385 / GDPH 0.448 / SYSUCC 0.331,對應 frozen→oracle specificity:0.409→0.623 / 0.630→0.819 / 0.453→0.775 / 0.474→0.568。Pre-registered k*:GDPH、SYSUCC = 10;BrEaST、BUSI = 20;中位恢復比例近 1.0(GDPH spec 0.453 → k=10 中位 0.827,oracle 0.775)。**Sensitivity 代價:** k=10 之中位 sens 為 BrEaST 0.830 / BUSI 0.846 / GDPH 0.866 / SYSUCC 0.909,較凍結管線之外部 sens(≥ 0.918)為低;判準容許至 0.85。單類別抽樣僅見於 k=10(BrEaST 0.8% / BUSI 0.6% / SYSUCC 2.4% / GDPH 0)。k=10–30 之 95% specificity 帶涵蓋約 0–0.97;**post-hoc k_reliable(2.5 百分位恢復 ≥ 0.5)僅 GDPH 於 k=200 達標,BrEaST、BUSI(網格至 k=100)與 SYSUCC(至 k=200)皆未達**——於 pre-registered 網格內,無任何方法使小樣本重校準達到 draw-level 可靠。方法比較:M2b/M3 與 M1 為結構性等價(單調變換 + 排序空間閾值),實跑曲線於 k ≤ 30 僅因 M2 擬合失敗回退與 Platt 同分而異,k* 亦有差異(BrEaST M1 20 / M2b 10;GDPH M1 10 / M2b 20;SYSUCC M1 10 / M2b 20);M2a sens 最高(0.92–0.95)、帶最窄,但恢復不足(GDPH 0.657 vs oracle 0.775),k* 未達;T 擬合失敗率 k=10 為 7–20%、k ≥ 100 為 0。

**Reason:** SYSUCC oracle specificity rounds .567, not .568. M3 is not invariably order preserving: 35 negative-slope k=10 fits were reproduced, so fallback/ties are not the only explanation of departures from M1. The median/sensitivity/reliability caveats otherwise reflect the saved draws.

### O30. docs/report.md:129

> 內部參考列逐位重現(閾值 0.2683、T 2.3644)。四個外部閾值 0.3755(BrEaST)/ 0.3851(BUSI)/ 0.4484(GDPH)/ 0.3314(SYSUCC)均為各世代全資料之 in-sample 擬合。任一外部閾值於所有其他世代之 specificity 皆高於內部閾值(無 CI);GDPH 之 0.448 換得最高 spec(0.71–0.89)但跨場域 sens 降至 0.80–0.87。M2 temperature 轉移效果甚微(BrEaST T 2.27 / SYSUCC T 2.23,與內部差 0.03–0.04)。

**Reason:** The temperature difference .03–.04 is arithmetically false: .09265/.13905. Higher thresholds increase specificity by construction; without a sensitivity-constrained comparison this is not general superior portability.

### O31. docs/report.md:150

> **判準:分支 A 4/4 → MET(via branch A)。** 同段保留事項:(i) paired ΔAUC CI 僅 BUSI、GDPH 排除零,BrEaST、SYSUCC 含零;(ii) 分支 B 2/4 未達(BUSI 0.847、SYSUCC 0.800 低於 0.85);(iii) 判準之對照為 v1 fold-5 **單模型**;相對已部署之 v1 **ensemble**(external-v1),LOCO ΔAUC 為 +0.0115 / +0.0033 / +0.0301 / +0.0064,僅 GDPH、BrEaST 兩世代 ≥ +0.01;(iv) 註冊之問題為 specificity 崩落,而通過之分支為 AUC;Δspec(+0.27 至 +0.30)四世代 paired CI 皆排除零,但係於各模型不同閾值下比較(閾值定位混淆)。機制(描述性):BrEaST 以閾值定位為主;BUSI 良性中位降至 0.051(全案最低);GDPH 判別力提升最明確(AUC CI 不重疊、唯一 LOCO sens 高於 v1-single 之世代);SYSUCC 良性下移同時拖累惡性(sens 0.800)。Sens 可攜性:val ≈ 0.90 → held-out 0.80–0.96,對照 v1 凍結閾值之全域 ≥ 0.918。

**Reason:** The registered AUC verdict and adjacent CI/comparator caveats are correct. The sentence assigning BrEaST gains mainly to threshold placement is a causal/quantitative allocation unsupported by a controlled threshold/model ablation. Label it a hypothesis, not an established mechanism.

### O32. docs/report.md:160

> 外部驗證確立 AUC 與 operating point 可攜性之分離,伴隨良性分佈之位置偏移(M2a 顯示僅重估 T 無法恢復,漂移為 logit 空間之位置而非尺度)。後續研究:(1) 部署端——10–20 例本地標註即可於中位數恢復 specificity(以 k=10 中位 sens 0.83–0.91 為代價),但 draw-level 可靠性於 pre-registered 網格內僅 GDPH k=200 達標;本地標籤之價值在閾值定位(單調校準 + 本地閾值 ≡ 閾值重定之結構性等價);(2) 跨場域(post-hoc、描述性)——任一外部閾值轉移至其他世代之 specificity 皆高於內部閾值;*未檢定之詮釋:BUS-BRA 之良性影像相對「乾淨」致閾值系統性偏低*;(3) 訓練端——BiomedCLIP 替代骨幹依判準 NOT CLAIMED;多來源訓練(LOCO)達成分支 A(ΔAUC +0.013 至 +0.056 vs 單模型,2/4 CI 含零)並於各自閾值下 Δspec +0.27–0.30,惟增益之大宗仍為「多場域 validation 產出更佳之起始閾值」,且 sensitivity 可攜性由 v1 之全域 ≥ 0.918 退為 0.80–0.96。**部署之最後一哩仍為以本地資料設定 operating point。**

**Reason:** Failure of temperature-only adaptation does not establish ‘location rather than scale’; Platt order reversal also invalidates universal equivalence. ‘Most gain’ from improved starting threshold is not quantified by an ablation. Final-mile deployment rhetoric exceeds a small-label retrospective simulation.

### O33. docs/report.md:163

> 所有外部世代 sensitivity ≥ 0.918(v1 凍結閾值),specificity 0.41–0.63:外部誤判為假陽性(37–59% 良性被呼叫)而非漏診。此方向是否「可接受」未經任何危害分析;且 LOCO 模型於其自身閾值下 held-out sens 降至 0.80。與判讀者比較:GDPH 模型於兩軸皆低於兩位判讀者,SYSUCC 居兩者之間;κ 0.22–0.52 為判讀者間一致性,說明一致性工具之需求,不涉及模型表現。*未檢定之詮釋:外部 FP 之 Grad-CAM 熱區落於病灶本體,與「非典型良性外觀」相符;POST-HOC 逐影像交叉表(§3.5)顯示 GDPH reader1 僅對模型 14% 之假陽性評為 ≥ 4a,故「模型與判讀者被同一批影像誤導」對該判讀者不成立,對 reader2 僅部分成立;未做任何檢定。* 此比較存在資訊不對等(判讀者見完整檢查,模型僅見單張截圖)。

**Reason:** Two direct falsehoods: ‘rather than missed diagnoses’ contradicts saved FNs; ‘readers saw a complete examination’ contradicts the source HoVer-Trans paper, which says each reader assessed one image at a time (§8 source check). κ alone does not establish a clinical need or model utility.

### O34. docs/report.md:166

> (1) CoarseDropout:gate 2 未過即棄、不迭代(gate 2 為三張影像之主觀判斷,未測任何 robustness 指標);(2) abstention 判準:0/4,依規定不放寬,*事後詮釋*為信號真實(AUROC 0.77–0.86,3/4 世代優於 margin)而判準將轉介計為 sensitivity 損失——修正之途徑為新的 pre-registration;(3) BiomedCLIP 替代骨幹:0/4 與 1/4,NOT CLAIMED。三者皆依註冊判準機械式套用、未於 RESULTS.md 放寬。

**Reason:** Three negatives are now faithfully stated, but the subjective CoarseDropout visual judgment is not mechanical in the arithmetic sense. ‘Signal real’ and superiority on 3/4 should be confined to point estimates: BrEaST comparative intervals overlap. The post-hoc interpretation label appropriately prevents rewriting the registered failure.

### O35. docs/report.md:172

> 單一國家單一機構之訓練資料;無台灣/NTUH 資料、無 IRB 臨床驗證、非醫療器材;**內部 CV AUC 為各折 best-epoch 值(POST-HOC 量化:固定 epoch 下 0.9195 ± 0.0164,樂觀量 ≈ 0.011),內部 sens/spec 為 in-sample(POST-HOC 嵌套估計 0.900 ± 0.057 / 0.780 ± 0.132,折間變異大)**;TTA 採用未 pre-registered;pre-registration 為內部版本控制、無外部時間戳、歷史曾重寫身分(§2.10);燒錄標註為潛在捷徑(無標註影像之表現預期較低);三世代無病人 ID(image-level bootstrap 與 LOCO val 切分之潛在樂觀);PPV 不可轉移;ensemble 無無偏內部估計;判讀者比較之資訊不對等與評分脈絡未知;中段機率殘餘 overconfidence;USFM 未於原生骨架受測;LOCO 為單模型(對照亦為單模型),early-stop patience 未預先指定;v2 檢查點與 fold 檔未公開;展示系統於非 BUS-BRA 尺寸存在 resize 分派差異(校準機率最大 7×10⁻³、中位 8×10⁻⁴,2/1,421 邊界翻轉)。

**Reason:** The v2 weights/fold-file unavailability claim is obsolete. The nested correction remains overinterpreted. Worse performance on images without annotations is an untested expectation, not a measured result. Reader scoring context is not wholly unknown: the primary paper explicitly describes single-image reading.

### O36. docs/report.md:175

> (1) NTUH 回溯性研究(IRB 規劃中),第一階段約 100 例:不動模型、僅設 operating point(起始候選為四個外部 in-sample 閾值 0.331 / 0.376 / 0.385 / 0.448 之任一,須於本地資料上重選)並驗證台灣族群判別力;第二階段 BI-RADS 3/4A 追蹤 upgrade 預測;第三階段超音波腋下淋巴結轉移預測以支持腋下手術降階。(2) Abstention 判準之重新 pre-registration(workflow 層級 sensitivity 與轉介率)。(3) USFM 於原生 BEiT 骨架之比較;LOCO × 領域預訓練之組合。(4) BUSI_WHU 內容層級標籤復原。(5) 多視角/多模態(影像+報告)以縮小與判讀者之資訊差。(6) Provenance:v2 檢查點與 fold 檔公開、Zenodo/OSF 外部時間戳、(plan.md P3–P4;epoch 選擇偏差與 in-sample operating point 之 post-hoc 量化已於 P2 完成,見 §3.1、§3.2)。

**Reason:** The provenance tasks are partly already completed (folds, v2 weights, Zenodo). The proposed ~100-case NTUH study is a future plan, not a justified sample size or validated clinical pathway; no precision/power rationale is supplied.

### O37. docs/report.md:180

> 在 patient-level 切分、內部 pre-registration 與凍結式單發驗證之紀律下,單一機構訓練之乳房超音波分類器於四個外部世代展現 AUC 0.84–0.93 之判別力,其 operating point 則不可攜(specificity 0.77 → 0.41–0.63),需本地設定。訓練端修正中,多來源訓練達成其 pre-registered 判準(僅 AUC 分支,2/4 CI 含零),領域預訓練替代品未達;部署端之本地閾值重選於中位數有效但小樣本下不可靠。外部誤判方向為假陽性;其臨床可接受性未經檢驗。本研究同時提供公開資料集稽核之實務範式,及可直接落地 NTUH 之分階段研究路徑。

**Reason:** AUC and operating-point ranges reproduce, with predictor-change and in-sample internal caveats. False positives are not the only external errors. ‘Directly implementable NTUH pathway’ is a proposal unsupported by local clinical, workflow or prospective validation.

### Primary-source checks and the three original failures

The Taiwanese density statement has a traceable primary source: [Chang et al., Age-Specific Breast Density Changes in Taiwanese Women](https://pubmed.ncbi.nlm.nih.gov/32375295/). Its result supports the density statistic, not every separate clinical-superiority claim attached to it. No systematic literature extraction supporting the report's broad AUC ranges was supplied or reproduced.

The reader-information assertion is more than an uncited possibility: [HoVer-Trans, §V-C](https://arxiv.org/html/2205.08390v2) explicitly says “readers assess only one image each time” and describes the lack of additional information. That is the opposite of report:163's complete-examination assertion. The source discusses two experienced sonographers; the repository should accurately identify the reader columns and reading setup instead of inventing an information disadvantage for the model. I did not import the source paper's own pooled performance results into this repository's cohort comparisons.

The negative outcomes themselves now retain their original criteria:

| Experiment | Original rule and trace | Current outcome | Audit conclusion |
|---|---|---|---|
| CoarseDropout | `plan.md` adoption gate: pooled OOF AUC ≥ plain−.01 **AND** visible improvement in caliper-adjacent CAM heat; RESULTS:61–82 | Gate 1 reported pass by .0008, gate 2 fail on the selected three-image visual review; discarded | Faithfully reported as a **subjective visual-gate failure**. The .9139 OOF score lacks a committed per-image CSV for this audit; no claim of independently measured robustness |
| Ensemble-disagreement abstention | Protocol:349–362: U_std AUROC≥.65 **AND** at q=.10 spec improves≥.05 with sensitivity not reduced **AND** AUROC exceeds margin | **0/4**, independently reproduced | Faithful. No tolerance introduced to rescue lost sensitivity; interpretive objections are now explicitly post-hoc |
| BiomedCLIP substitute | Protocol:486–493: AUC≥v1+.01 on ≥3/4 **OR** benign-shift ratio≤.70 on ≥3/4 | A0/4, B1/4, **NOT CLAIMED**; SYSUCC .7001626175 fails | Faithful and strict. It is a substitute-backbone result, not a test of USFM in its native architecture |

**Verdict: FAIL — major.** Numerous important corrections are real, especially the negative-result wording and LOCO caveats. Remaining literal falsehoods include the ensemble OOF label, no-missed-cancers language, reader information, temperature subtraction, unchanged fold ranking and unpublished-artifact claims; mechanistic and deployment assertions also remain too strong.

## 9. Reproducibility from scratch

**The literal fresh-clone recipe is not fully verified by this re-audit.** The instruction to write nothing except this report precluded creating a venv, installing files, symlinking mounted datasets or allowing result-producing script mains to overwrite committed outputs. I report those deviations rather than borrowing the prior audit's fresh-environment PASS. The environment used here is a pre-existing Python 3.12.13 venv in the original working copy; all code and committed predictions came from this clone.

| README step / requirement | What I actually ran or inspected | Outcome and deviation |
|---|---|---|
| `uv venv --python 3.12 && source .venv/bin/activate` | Inspected existing `/Users/thomaswang/projects/breast-us-cad/.venv/bin/python` | **Not executed from scratch**; would create other files. No clean-build claim |
| `uv pip install -r requirements.txt` | Parsed every non-comment requirement with `packaging.requirements.Requirement`, compared `importlib.metadata.version` against specifiers | **176 requirements, missing [], mismatch []**. This establishes installed top-level compatibility, not resolver/install reproducibility on a clean machine |
| Runtime stack | Python 3.12.13; torch 2.13.0; numpy 2.5.2; pandas 3.0.5; timm 1.0.28; OpenCV-headless 5.0.0.93; albumentations 2.0.8; pytest 9.1.1; openpyxl 3.1.5; open_clip_torch 3.3.0 | Available stack successfully imports/runs tested paths. The two unpinned packages can drift; MPS-specific configuration is not portable without changes |
| Dataset layout | README:138–144 now lists exact directories/XLSX filenames | Documentation defect resolved. This clone has no `data/raw`; `$DATA_ROOT` does not automatically configure all scripts. Reads redirected in memory to mounted raw data |
| Weights | Used `$MODELS_ROOT`, verified published checksum manifest | All 28 listed files match local and remote hashes; no need to download binaries to disk. Default loaders otherwise fetch missing files from HF and cache them |
| `python src/external_val.py --self-check` | Invoked the actual self-check function after mounted path adaptation; worker count 0 for inline Python, MPS preserved | **Pass**, AUC .9234433158791243; output byte-identical on MPS. CPU also passes AUC but one classification differs (§6) |
| `pytest` | C2 runs the entire suite; disables cache/bytecode, substitutes memory-backed tmp_path/save/load, maps raw reads | Bare clone **7 passed, 8 errors** (missing data). Mounted run **15 passed in 1.11 s**. README still says eight tests |
| Full frozen-v1 training chain | Traced `cross_validate.py → tta_eval.py → calibrate.py → pick_threshold.py` and original configs/logs | Recipe now names the necessary chain. **Not trained**, as instructed. No exact checkpoint regeneration or convergence claim |
| wandb | Read configs/metadata/history; no logging/session initialization | README now documents `WANDB_MODE=offline`. Logs used for historical reproduction are not committed or deposited. `wandb`'s datastore reader normally opens `r+b`; read-only mount required replacing that open with `rb` in the audit adapter |
| v2 artifact availability | HF API/LFS and file-byte checks, all 28 manifest entries | Previously unavailable v2 checkpoints/export are public. Full v2 new-image inference was not run. Saved-prediction metrics/CIs reproduce |
| CLAUDE eval command | `evaluate.py` parser inspected | Now valid `--ckpt models/cv_vit_fold5.pt --split val [--tta]`; brackets denote optional argument, not literal shell syntax. Old `models/best.pt --split test` defect corrected |
| `scripts/consistency_sweep.py` | Actual `main` with output sink in memory | Output equals committed check byte-for-byte (§2.6); no semantic guarantee |
| pHash scripts | Independent all-pairs implementation with identical hashing rule | Counts/keep-lists reproduce; avoided mains that write CSVs/figures |
| Network access | Approved read-only HTTP for GitHub/HF/Zenodo, plus authorized demo calls | Initial sandbox DNS failed, retry succeeded. No new model inference on external raw cohorts |

The new full training instructions fix the prior single-fold recipe, but cannot promise bitwise training reproduction: MPS nondeterminism, epoch selection and augmentation RNG behavior remain. README appropriately calls training regeneration approximate. Conversely, saying *all inference* is exact is too broad: the self-check verifies AUC, preprocessing differs for external dimensions, and device/batch reductions can change borderline decisions.

A truly verified clean build would require an independently created venv, full installation log, package resolution on the target platform, normal disk-backed pytest fixtures and the documented dataset layout. Those are the missing evidence I would require before assigning a from-scratch PASS. This audit supplies successful mounted-runtime checks, not that stronger result.

**Verdict: CONCERN — major verification limit.** The mounted environment passes the full suite and MPS self-check; published artifact availability is substantially repaired. Clean environment construction and retraining were not performed under the explicit read-only constraints, so reproducibility from scratch cannot be certified here.

## 10. Code-quality risks

Search commands included `rg -n 'train_folds|val_folds|Dataset|DataLoader|build_master_df|DATASETS' src configs tests`, `rg -n 'except|pass$|warnings|random|seed|determin|MPS_FALLBACK' src scripts`, `rg -n 'data/real|/Users/|https?://|hf_hub_download|wandb|read_csv|read_excel|torch.load' src scripts app deploy configs`, and AST comparison of the inference/evaluation copies. Findings are distinguished from bugs shown to alter the headline numbers.

| Risk | Exact source/evidence | Consequence |
|---|---|---|
| Threshold extraction uses a plotting simplification | `src/pick_threshold.py:38` default ROC pruning | **Major, demonstrated:** wrong highest-threshold selection on OOF/oracles and 2432 M1 draws (§2.5) |
| v1 outputs can be overwritten | `external_val.py:133–160`, `:206–230` | **Major enforcement gap:** `--confirm` is not a single-shot guard; a repeated command writes the same CSV paths |
| “Nested” calculation lacks outer retraining | `posthoc_nested_threshold.py:53–68`; `tta_eval.py:44–91` | **Major interpretation risk:** name/docstring suggests independent nested evaluation that dependencies do not support |
| Platt negative slopes and suppressed convergence warnings | `v2_recalib_curve.py:99–105` | Demonstrated negative slopes reverse ranking; suppressed `ConvergenceWarning` conceals fit diagnostics. Returned parameters/iterations are not all persisted, making method-equivalence claims hard to audit |
| Temperature fitter unconstrained; broad fallback catch | `calibrate.fit_temperature:41–59`; `v2_recalib_curve.py:130–138` | Nonpositive T causes runtime fallback; catch handles any `RuntimeError`, not exclusively the intended invalid-T condition. No explicit finite-T validation. No claim that observed stored metrics are corrupted by NaNs |
| Hardcoded paths; environment roots inconsistently supported | `data.py:28`, `external_val.py:42–44`, `phash_sweep.py:40–52`, `dedup_busi.py:32`, `birads_comparison.py:20`, model constants in evaluation scripts | Minor portability risk. Contradicts CLAUDE rule 5; no user-specific `/Users/...` source path found. README now admits the layout |
| Incomplete raw-data cardinality guard | `data.py:63` inner merge, `:81` checks first five image paths | `validate='one_to_one'` does not ensure all metadata IDs have a fold row. Missing later images fail only on loading. Actual mounted IDs/files checked here match; latent defensive-programming gap |
| Existing checkpoint configs are stale | `models/cv_vit_fold*.pt`; RESULTS:1210 | Future `checkpoint_payload` fixed/tested; published frozen weights still need external run metadata to identify actual folds |
| Training nondeterminism | `train.py:25–28`, `data.get_transforms`, configs workers=4 | Seeds Python/NumPy/torch but no deterministic-algorithm enforcement or explicit loader generator/worker-init policy. Albumentations Compose is not explicitly seeded in this code. MPS fallback is assumed in shell, not configured by the program. Analysis RNGs do reproduce |
| Missing-weight network fallback is mutable/unpinned | `inference.resolve_weight:76–86` | HF default branch is used; supplied `CHECKSUMS.txt` is not automatically checked by loader. No offline switch. File identity passed this audit, but future downloads are not cryptographically enforced at inference time |
| Deserializing full checkpoints | `torch.load(..., weights_only=False)` in inference/eval | Trusts the published checkpoint source; avoid presenting arbitrary downloaded checkpoints as inert data. No malicious content found or alleged |
| Drift-prone duplicate helpers | `evaluate.predict_hflip_pair`, `inference.predict_hflip_pair`; logit and CAM helper copies | AST matches today where tested; future changes can drift. External-v1 predates the shared module, so name sharing alone is insufficient proof |
| Deploy helper mutates frozen source for a different owner | `scripts/deploy_hf.py:90–118`, `patch_default_repo` | No-op when current owner/repo matches; otherwise rewrites `src/inference.py`. Author's narrow no-op observation is correct, but does not eliminate the deployment-script design risk |
| Repeated analysis mains append/overwrite | `compare_backbones.py`, result-writing mains, `v2_loco.py:276–277` | Readers rerunning scripts can mutate the chronological ledger or output tables. This audit called pure computation functions/output sinks instead |
| Assert-based guards | e.g. `v2_loco.py:211`, reproduction asserts | `python -O` removes assertions; guard guarantees should not be described as unconditional under all runtimes |
| Stale or confusing interfaces | `evaluate.py` future-external-loader docstring, val-only `--split`; `cross_validate.py` default prefix `cv_effb0`; `app/app.py:25` cwd change at import | Minor reproducibility/embedding confusion; corrected CLAUDE command itself is now valid |
| Plotting and optional network at import | pyplot imports in `explain.py`/`birads_comparison.py`; albumentations version check unless disabled | Headless/cache/network side effects matter for read-only/offline use. The audit set `MPLBACKEND=Agg`, `NO_ALBUMENTATIONS_UPDATE=1`, no-bytecode and cache adapters |
| Live verification tooling | `scripts/verify_space.py`, retry handlers | Retries/logs failures rather than bare silent pass. A printed mismatch is not a substitute for an explicit failing process status in CI; scope the claimed automation guarantee accordingly |
| Other explicit network operations | HF upload/deploy/export scripts; wandb in train/train_seg/v2_loco; pretrained backbone resolution | Expected functions, not concealed external-data ingestion. Do not run them merely to reproduce saved metrics |
| `data/real` and notebook | No matching `data/real` read path; EDA notebook uses BUS-BRA | No hidden real-patient dataset consumption found |

There is no broad `except: pass` pattern hiding evaluation failures. The meaningful concerns are narrower: swallowed Platt convergence warnings, conflation of runtime failure causes, and scientific guarantees expressed through docstrings/assertions without sufficient defensive checks. No critical malicious or unexplained external-data path was identified. I would expect a hidden loader/network fetch feeding v1 training, unseeded analysis resampling, or silently skipped prediction rows if such a defect were present; the inspected code and reproduced row counts did not show them.

**Verdict: CONCERN — major.** Most hygiene issues are minor and pre-existing, but highest-threshold extraction and v1 overwrite behavior have direct scientific/reproducibility consequences. The prior conclusion that nothing in code affects a number is no longer defensible.

## 11. Disposition of every previous-audit item

Status meanings: **RESOLVED** = the specific original defect is corrected, or a prior positive finding is independently reverified; it does not mean the whole study is problem-free. **STILL OPEN** = the underlying problem persists, including cases where disclosure improved but independence cannot be repaired retrospectively. **DISPUTED-BY-AUTHOR-CORRECTLY / INCORRECTLY** refer to actual rebuttals, not to ordinary fixes. Some original items bundled separate issues; those are split explicitly below rather than assigning an ambiguous blanket closure. Evidence pointers refer to the current sections above and current file:line.

### 11.1 Previous §1, §3–§6 and additional findings

| Previous item | Status | Current evidence |
|---|---|---|
| §1 official fold/patient integrity, no mixed patient pathology, OOF assignments | **RESOLVED** (reverified positive finding) | §1: 1875 exact matches, 1064 patients, zero spanning folds; official bytes equal |
| §1 no external data in v1 training | **RESOLVED** (reverified) | v1 dataset construction + five actual wandb fold configs; §1 |
| §1 fold-5 involvement in model screening / epoch-selection bias | **STILL OPEN** | Selection remains in frozen artifacts. README:40–49 discloses it; P2 fixed-epoch sensitivity analysis is not unbiased correction (§2, §7) |
| §1 false fold metadata: future saves | **RESOLVED** | `train.checkpoint_payload:94`, 3 passing checkpoint-config tests |
| §1 false fold metadata: existing frozen checkpoints | **STILL OPEN** | All five still carry raw YAML; local wandb supplies correct folds; RESULTS:1210 honestly explains |
| §1 official partition missing from repo | **RESOLVED** | committed `data/splits/busbra_official_5fold.csv`, official SHA match, tamper/fallback tests |
| §1 misleading report text about partition | **STILL OPEN** | report:43/:83/:172 still say absent/unpublished |
| §1 eight passing tests | **RESOLVED** (expanded suite reverified) | Current full suite 15 passes with mounted raw data; README test count stale |
| §3 freeze→protocol→external artifact chronology | **RESOLVED** (internally reverified) | Exact commits/tags, one adding commit per CSV, original mtimes; §3 |
| §3 amendment2/3/4 precede computations and committed alone | **RESOLVED** (reverified) | 610e87f/78f3991/3faa70b each protocol-only, before first outputs |
| §3 Amendment 1 already contains data-audit outcomes | **STILL OPEN** as prospective registration limitation | Now explicitly disclosed; PROVENANCE:20/:154 still overgeneralize |
| §3 prior hashes inaccessible in clone; mapping unavailable | **RESOLVED** for documented mapping; source-object limitation remains | 30/30 mappings checked on original read-only git object store; old objects still absent from clone/ZIP |
| §3 dates preserved through identity rewrite | **RESOLVED** (independently demonstrated) | Equal trees, author dates, committer dates and messages for all 30 pairs; §3 |
| §3 repo private/no public snapshot | **RESOLVED** for current availability | GitHub public, Zenodo deposition verified; exact historic privacy transition not proved |
| §3 no independent pre-run timestamp | **STILL OPEN** | Zenodo/HF all postdate external-v1. Response I.1 correctly calls gap permanent, but “resolved-by-P4” can mean disclosure only |
| §3 no committed external prediction overwrites, RESULTS append-only | **RESOLVED** (reverified bounded claim) | `git log -p` only additions; zero removed RESULTS/protocol lines. Never overwritten in any uncommitted state remains unprovable |
| §4 v1 label mappings/exclusions/keep-lists/metrics | **RESOLVED** (reverified) | All rule rows in §4; raw IDs and output counts match |
| §4 M2 invalid-T fallback not preregistered | **STILL OPEN** | Runtime rule persists, affects up to 20.4% at k=10; now disclosed at report:75/RESULTS:499 |
| §4 early-stop patience7 not preregistered | **STILL OPEN** | `configs/v2_loco.yaml:17`, report:79 disclosure cannot retro-register |
| §4 TTA adoption not preregistered | **STILL OPEN** as development freedom | Truthfully disclosed README:195/report:49; frozen OOF still used for adoption/T/threshold |
| §4 one abstention figure split into two | **STILL OPEN** as literal minor deviation | Declared at report:77/RESULTS:608, no numerical impact |
| §4 Q1 wording exceeds AUC branch | **RESOLVED** for original headline omission | README:65–72/report:150 now state A-only, 2 zero-crossing CIs, single-model comparator and B failures |
| §4 Q2 fallback, backbone order, exact criterion | **RESOLVED** (reverified) | Substitution before smoke; Q2 fail→v1 LOCO; SYSUCC ratio remains fail |
| §4 CoarseDropout gate2 subjective | **STILL OPEN** as design limitation | Now explicitly called visual gate on three images; no objective robustness assessment |
| §5 SYSUCC/GDPH/BUSI dedup counts, full hash sweep | **RESOLVED** (reverified) | Raw recomputation and exact final keep-list equality (§5) |
| §5 bus_0012-l/-r low-distance pair / author E.8 rebuttal | **DISPUTED-BY-AUTHOR-CORRECTLY** on leakage implication | Same patient, same fold; no leakage correction needed. New broad README no-near-duplicate wording still needs scope |
| §5 cross-set candidates refuted; WHU absent | **RESOLVED** (reverified) | Viewed four raw images; two refuted candidates; no WHU registered/results rows |
| §6 AST/shared inference parity and v1 self-check | **RESOLVED** (reverified within stated scope) | 21/7 ASTs, actual MPS self-check; §6. Does not imply universally identical preprocessing |
| §6 v2 BiomedCLIP self-check previous positive result | **RESOLVED** (independently reverified) | Supplemental C12 runs the actual internal self-check on CPU: AUC .9230647908649297, 196/383 flagged, PASS; no external inference |
| §6 unshared resize-dispatch route | **STILL OPEN** as actual parity limitation | Every raw image re-resized; 1421 affected, max1gray. Documentation substantially corrected, strict README bound still wrong |
| §6 example encoding difference / author E.9 rebuttal | **DISPUTED-BY-AUTHOR-CORRECTLY** | Encoding difference did not imply probability error; current app/deploy/remote example bytes now equal and all probabilities match within floor |
| §6 live probabilities vs recorded values | **RESOLVED** (reverified) | Four authorized live calls, max1.0728836064e-7; §6 |
| §6 historical warm latency 6.15 vs6.8 / E.10 | **RESOLVED** as differing measurements, not mathematical contradiction | P6 gives historical range; this audit's warm median4.789s illustrates variability, not falsification |

### 11.2 Previous §2: all eight discrepancy items, including subitems

| Previous item | Status | Current evidence |
|---|---|---|
| 2.1 9.7M pair count | **RESOLVED** | README:86/report:20/:69 give12,056,505; independently enumerated |
| 2.2 sensitivity≥.92 everywhere | **RESOLVED** | .918 now in current headline/app/plan; RESULTS:1220 errata explicitly correct old immutable lines485/1058; BrEaST90/98 |
| 2.3 first-three disagreement CIs nonoverlapping | **RESOLVED** | report:132 names BUSI/GDPH only and quotes BrEaST overlap |
| 2.4 resize≤.001 | **RESOLVED** for original order-of-magnitude error | report:172 now approximately .007; exact code docstring .0072. New README strict≤.007 defect remains separately open |
| 2.5 1879 BUS-BRA images | **RESOLVED** | RESULTS erratum corrects duplicate-counted examples; raw1875/713sizes |
| 2.6a only3/121 malignant probabilities in band | **RESOLVED** | Saved OOF independently gives32/121 TTA,27plain; P2 CSV and report:97 |
| 2.6b six checks / written GO | **RESOLVED** as unsupported factual assertion removed | report:71 explicitly labels it unrecorded session observation, not evidence |
| 2.6c BUS-CoT exclusion story | **RESOLVED** | Removed from current report; no replacement invented provenance |
| 2.6d +.01 fold5 screening gate | **RESOLVED** | report:47 no longer asserts an unrecorded numeric screening rule |
| 2.7 k_reliable≈100–200 generalized to cohorts | **RESOLVED** | README:63/report:22/:126 say only GDPH at 200, others not reached, explicitly post-hoc |
| 2.8 GDPH model within readers' range | **RESOLVED** | report:22/:120 states below both on both axes; raw reader recomputation confirms |

Every positive numerical row in the previous §2 was reconsidered, not assumed: current §2 tables cover CV/backbones, per-fold/plain/TTA/pooled AUC, T/ECE/NLL/threshold/confusion/PPV/NPV/CIs, all four external metrics/medians, member reproduction, oracle values, BI-RADS, M1/all-method summaries/k*/k_reliable, cross-site, disagreement, BiomedCLIP, LOCO, resize and live examples. These prior positive items are **RESOLVED (reverified)** except: (i) the five-member internal OOF interpretation is **STILL OPEN and incorrect**; (ii) flat-mean-of-ten bitwise equality needs the original reduction order; (iii) SYSUCC oracle .568 is a **STILL OPEN** double-rounding error; (iv) CoarseDropout pooled OOF .9139 and qualitative/historical values are **STILL OPEN** for independent committed-CSV reproduction; (v) all M2/M3 fits were not rerun, so only their saved-draw summaries plus targeted Platt refits are independently verified. The previous blanket “every number in RESULTS reproduces” PASS was too broad.

### 11.3 Previous §7: every statistical-language row

| Previous item | Status | Current evidence |
|---|---|---|
| 7.1 BUSI/GDPH disagreement CI nonoverlap | **RESOLVED** (supported claim reverified) | §7 exact intervals |
| 7.2 BrEaST included in nonoverlap claim | **RESOLVED** | report:132 corrected |
| 7.3 reader significantly above ROC | **RESOLVED** | report:120 says point estimate, no test |
| 7.4 GDPH LOCO AUC CIs do not overlap | **RESOLVED** (reverified) | .930–.960 vs .867–.913 and positive paired delta |
| 7.5 LOCO headline omits zero-crossing CIs/single comparator | **RESOLVED** | README:65–72/report:150 adjacent caveats |
| 7.6 Δspec CIs positive at differing thresholds | **RESOLVED** (claim/caveat reverified) | §2 paired intervals; report:150 admits threshold confounding |
| 7.7 BiomedCLIP lower AUC presented as population effect | **RESOLVED** for original claim | report:135 explicitly point estimates and overlapping CIs; NOT CLAIMED retained |
| 7.8 internal sens/spec not labeled in-sample | **RESOLVED** for disclosure; **STILL OPEN** for bias-removal claim | README:45 labels original correctly, then overclaims P2 nested estimate (§7) |
| 7.9 cross-site “better” lacks descriptive/post-hoc/sensitivity context | **RESOLVED** for omissions | report:22/:129 includes context/cost; new temperature arithmetic/causal interpretation errors remain |
| 7.10 small-label median claim drops sensitivity/reliability caveats | **RESOLVED** | README:60–64/report:22/:126 now includes both |
| Bootstrap units correct, patient/image limitation | **RESOLVED** (reverified) | Code and all numerical CI checks; no patient bootstrap falsely claimed for unidentified external patients |

### 11.4 Previous §8: all 29 quoted overstatement items

| Old item | Status | Current disposition and exact evidence |
|---|---|---|
| 8.1 mixed internal result row | **STILL OPEN** | Row split fixed one confusion, but README:33/report:22 still call single-model OOF a five-model ensemble; P2 label adds another independence overclaim |
| 8.2 unqualified discrimination transfers | **RESOLVED** for original README omission | README:53–55 explicitly names lower BrEaST/SYSUCC and CIs; broader discussion qualifications remain needed |
| 8.3 “fails in safe direction” | **STILL OPEN** in revised form | Safety is now unassessed, correctly, but README:58/report:24/:163 still say not missed cancers despite 76 FNs |
| 8.4 k_reliable≈100–200 | **RESOLVED** | Only GDPH at 200, post-hoc, not reached elsewhere |
| 8.5 LOCO caveats absent | **RESOLVED** | All four major caveats now present adjacent to claim |
| 8.6 preregistered before results | **STILL OPEN** as independent-timeline claim | Local chronology verified; Amendment1 post-audit and no pre-run public anchor persist and are disclosed |
| 8.7 9.7M pair count | **RESOLVED** | Exact 12,056,505 |
| 8.8 two vs three negatives | **RESOLVED** | README:91/report:165–166 agree on three |
| 8.9 every number traces/exact shared pipeline | **STILL OPEN** | Core sharing/discrepancy disclosed, but universal CSV claim and strict resize bound false; §2.6 token check inadequate |
| 8.10 fold file and checkpoints shown as committed | **RESOLVED** for README/map/public release; **STILL OPEN** for stale report | README:106–131 now accurate for listed artifacts; report:43/:83/:172 contradicts P3 |
| 8.11 reproduce block trains one model | **RESOLVED** for recipe | README:158–173 supplies full chain and approximate-training caveat; this re-audit did not train |
| 8.12 9.7M and “zero overlap” | **RESOLVED** for original language; **STILL OPEN** for unqualified new pHash scope | Count corrected; report:69 gives cutoff limitation; README:89 still omits cross-set/adjudicated scope |
| 8.13 GDPH model between readers / κ implies model | **RESOLVED** | report:22/:120 corrects both |
| 8.14 local labels reliability/sensitivity | **RESOLVED** | Actual medians/only GDPH at 200 now stated |
| 8.15 all monotonic calibrations give identical decisions | **STILL OPEN** | Exact-as-run claim softened, but structural-equivalence explanation still ignores negative Platt slopes (35 k=10 draws); §2.5 |
| 8.16 cross-site thresholds universally superior | **RESOLVED** for original scope omissions | Now post-hoc, specificity-only, in-sample sources, sens cost stated; temperature subtraction newly false |
| 8.17 BiomedCLIP positive finding despite failed criterion | **RESOLVED** | NOT CLAIMED in abstract/discussion/conclusion; exact ratio test rerun |
| 8.18 LOCO criterion statement without caveats | **RESOLVED** | Same evidence as8.5 |
| 8.19 written GO record | **RESOLVED** | Not presented as recorded fact at report:71 |
| 8.20 BUS-CoT exclusion | **RESOLVED** | Unsupported story deleted |
| 8.21 3/121 confidence-band count | **RESOLVED** | P2 replaces with32/121 and27plain |
| 8.22 specificity entirely caused by style/benign shift | **RESOLVED** for original §3.4 sentence; **STILL OPEN** for related §4.1 mechanism | report:117 now labels hypothesis; report:160 still asserts location-not-scale and dominant threshold mechanism without direct test |
| 8.23 reader significance | **RESOLVED** | No-test caveat added |
| 8.24 first-three CI nonoverlap | **RESOLVED** | BrEaST overlap stated |
| 8.25 reinterpret abstention failure | **RESOLVED** for mislabeling | Registered0/4 remains first; objection explicitly post-hoc; no criterion relaxation |
| 8.26 readers/model misled by same images | **RESOLVED** for untested concordance assertion | Actual per-image P2 cross-tab now exists; GDPHreader1 agrees on only 14% of modelFPs. New complete-examination excuse is false (§8) |
| 8.27 resize≤.001 | **RESOLVED** for old magnitude; **STILL OPEN** for strict .007 upper bound | Actual .007227; docstrings accurate .0072 |
| 8.28 threshold consensus .33–.38 omitsGDPH | **RESOLVED** | report:175 lists all four: .331/.376/.385/.448 and requires local re-selection |
| 8.29 every number to commit / unavailablev2 | **STILL OPEN** for universal claim/stale report; availability itself **RESOLVED** | HF28/28 verified, but report:83 still says unpublished and universalCSVreproduction untrue |
| §8 closing CoarseDropout criterion-faithfulness check | **RESOLVED** for truthful reporting; visual-design limitation **STILL OPEN** | §8 negative-result table |
| §8 closing abstention criterion-faithfulness check | **RESOLVED** | Exact 0/4 and post-hoc interpretation labeling |
| §8 closing BiomedCLIP contradiction | **RESOLVED** | Exact NOT CLAIMED throughout main current findings |

### 11.5 Previous §9: all eight reproducibility rows

| Old item | Status | Independent evidence |
|---|---|---|
| 9.1 venv build succeeds | **STILL OPEN** for this re-audit's clean-build verification | No new files allowed; existing Python3.12.13 used. Prior PASS not adopted |
| 9.2 dependency install succeeds | **STILL OPEN** for fresh-install verification | All 176 current requirements installed/satisfy specifiers; no new resolver/install run |
| 9.3 missing explicit raw layout | **RESOLVED** | README:138–144 lists exact paths/files |
| 9.4 one-fold recipe/wandb/nondeterminism | **RESOLVED** for documentation; nondeterminism **STILL OPEN** by design | Full chain/offline setting/approximate training now explicit; no retraining |
| 9.5 self-check requires data/automatic download undocumented | **RESOLVED** | README:150–155 now describes download; actual MPS self-check passes |
| 9.6 tests require datasets | **RESOLVED** for disclosure; stale count **STILL OPEN** | Full 15 passes with mount; README says8 |
| 9.7 unavailablev2 checkpoints/export | **RESOLVED** | All listed public artifacts and hashes verified; report text still stale (§8) |
| 9.8 brokenCLAUDE evaluate command | **RESOLVED** | Correct existing checkpoint name and val-only parser argument |

### 11.6 Previous §10: every code-quality bullet and author rebuttal

| Original risk / response | Status | Evidence |
|---|---|---|
| Hardcoded raw/model paths vsCLAUDErule5 | **STILL OPEN** | Same source literals; README admits them; §10 |
| No user-specific absolute paths | **RESOLVED** (reverified positive finding) | Source search found no `/Users/...` path |
| No bare silent exception handling | **RESOLVED** narrowly | No `except: pass`; however convergence-warning suppression/broad RuntimeError fallback remain concerns |
| Training seeds/MPS/fallback/loader nondeterminism | **STILL OPEN** | Documented, not eliminated; configs/helpers unchanged in relevant respects |
| Seeded statistical analysis | **RESOLVED** (reverified) | Bootstraps and11kM1draws reproduced |
| Misleading checkpoint configs | **RESOLVED** for future code; **STILL OPEN** for published originals | §1,3passingtests, originalconfigsunchanged |
| Duplicate inference/logit/CAM helpers | **STILL OPEN** as drift risk | ASTmatches today; duplication persists |
| evaluate stale docstring/val-only split | **STILL OPEN** for confusing interface; broken command **RESOLVED** | Current parser/docstring vs correctedCLAUDE |
| cross_validate defaultcv_effb0 | **STILL OPEN** | Default persists; documented fullVitcommand suppliesprefix |
| compare_backbones appendsRESULTS on repeat | **STILL OPEN** | Unchanged write behavior |
| deploy_hf in-place source patch, narrow same-ID no-op rebuttal | **DISPUTED-BY-AUTHOR-CORRECTLY** for present-ID behavior | `patch_default_repo` is indeed no-op when target matches |
| deploy_hf no-op used to dismiss general frozen-source mutation risk | **DISPUTED-BY-AUTHOR-INCORRECTLY** as full closure | Different valid deploy owner still mutates source; risk not removed by the same-ID observation |
| app import changescwd | **STILL OPEN** | `app/app.py:25` |
| pyplot without explicitAgg in some scripts | **STILL OPEN** as environment dependency | Audit neededheadless/cachecontrols; no claim all environments fail |
| Missing-weightHFdownload implicit | **RESOLVED** for disclosure; offline/revision/checksum enforcement **STILL OPEN** | READMEexplicit; loaderstilldefaultmain/nooffline/noshaenforcement |
| wandb training / pretrainedexport network | **STILL OPEN** as reproducibility dependency | Offline mode nowdocumented; no hiddennewdatafeedfound |
| Nothing readsdata/real | **RESOLVED** (reverified) | Searchnegative |
| EDA notebook fiveoutputcells/internalonly | **RESOLVED** (prior harmlessscope retained) | Noexternaltrainingpathfound; notusedasnumericproof |
| E.11 blanket deferredcodequality response | **STILL OPEN** | P2/P3fix onlysomeitems; residualriskstableabove |

### 11.7 Author response's later completion claims

- **P2 “resolved” bias:** **STILL OPEN** beyond disclosure/sensitivity analysis. It neither removes validation-selected checkpoints nor builds a nested outer model; it also added the false unchanged-fold-ranking sentence. The new probability-band and reader-concordance tables themselves reproduce and are **RESOLVED**.
- **P3 folds/weights/checksums/HFpaths:** **RESOLVED** for the actual artifacts and tests. The report's contrary availability statements remain **STILL OPEN**; fixing a file does not fix every document claiming it absent.
- **P4 rewrite mapping/deposition:** **RESOLVED** for all 30 mappings and current tag/archive/remote identity. **STILL OPEN** for pre-run external timestamping, which the author correctly admits cannot be repaired. Embedded archive provenance is stale; future OSF commitments are not achieved registrations.
- **P6 retry/latency/consistency:** historical latency framing/retry change **RESOLVED** for that narrow issue; semantic consistency **STILL OPEN**. The exact generated output reproduces, but arbitrary numeric-token matches certify unrelated claims and miss demonstrable errors (§2.6).
- **Response's new caption (“honest composite”): STILL OPEN.** Single-modelCAM/ensembledecision disclosure is correct, but the benign-evidence explanation and small-effect causal inference overreach a malignant-only, heavy-tailed border probe (README:18–25; O1).

The previous executive summary's three positives were not simply accepted: split integrity and deduplication were independently reverified; external saved numerical evaluation was reverified, but its claim of “exactly what it says” must now be limited by missing single-shot enforcement and the internal comparator mislabel. The three old major themes—optimistic internal assessment, document overstatement, self-certified provenance—remain material even after real artifact/publication improvements. New findings add the threshold-selection defect, rank-reversing Platt fits, wrong reader-study information, incomplete registered interval summaries and weak token-based consistency assurance.

### 11.8 Exact command appendix

The following commands are the actual in-memory audit programs used for the numerical and artifact checks, with outputs summarized in §§1–10. Paths point to this session's mounted read-only original data/models/environment. They are evidence of what ran, not instructions to rerun inference on new external data. C1's ordinary import preceded the stricter cache harness; subsequent commands prevent authored temporary artifacts. C5 completed its AST/CPU checks before an audit-harness argument-order error in its final example loop; the corrected MPS/local-example command C9 supplies the successful end-to-end evidence. No failed harness invocation is presented as a successful repository test.

#### C1 — split joins, internal/external metrics, threshold counterexample and consistency sweep

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /Users/thomaswang/projects/breast-us-cad/.venv/bin/python -B -u - <<'PY'
import sys,os,io,json,hashlib,importlib.util,difflib
from pathlib import Path
sys.path.insert(0,'src')
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,roc_curve
from calibrate import fit_temperature,probs_to_logits,nll,ece,bin_stats
from pick_threshold import pick_operating_point,confusion_counts,point_metrics,patient_bootstrap_ci
from external_val import patient_bootstrap
root=Path(os.environ['DATA_ROOT'])/'raw/busbra'
m=pd.read_csv(root/'bus_data.csv'); f=pd.read_csv(root/'5-fold-cv.csv'); d=m.merge(f[['ID','kFold']],on='ID',validate='1:1'); o=pd.read_csv('reports/oof_vit_preds.csv'); o['ID']=o.image_path.map(lambda p:Path(p).stem); j=o.merge(d[['ID','Case','Pathology','kFold']],on='ID',validate='1:1')
print('SPLITS',len(m),m.Case.nunique(),d.groupby('Case').kFold.nunique().gt(1).sum(),d.groupby('kFold').size().to_dict(),(j.fold==j.kFold).sum(),(j.patient_id==j.Case).sum(),d.groupby('Case').Pathology.first().value_counts().to_dict()); print('FOLD_BYTES',Path('data/splits/busbra_official_5fold.csv').read_bytes()==(root/'5-fold-cv.csv').read_bytes(),hashlib.sha256((root/'5-fold-cv.csv').read_bytes()).hexdigest())
for p in Path('reports').glob('*summary.csv'):
 if 'cv_' in p.name or '_cv_' in p.name:
  s=pd.read_csv(p); s=s[pd.to_numeric(s.fold,errors='coerce').notna()] if 'fold' in s else s
  if 'auc' in s: print('CV',p.name,len(s),s.auc.mean(),s.auc.std(ddof=1))
y=o.y_true.to_numpy(); z=probs_to_logits(o.y_prob_tta.to_numpy()); T=fit_temperature(z,y); p=1/(1+np.exp(-z/T)); thr=pick_operating_point(y,p)
print('OOF',roc_auc_score(y,o.y_prob_plain),roc_auc_score(y,o.y_prob_tta),'T',T,'thr',thr,'counts',confusion_counts(y,p,thr)); print('INTERNAL',point_metrics(*confusion_counts(y,p,thr)),patient_bootstrap_ci(o.assign(prob_cal=p),thr)); print('ECE_NLL',ece(bin_stats(y,o.y_prob_tta.to_numpy()),len(o)),ece(bin_stats(y,p),len(o)),nll(y,o.y_prob_tta.to_numpy()),nll(y,p)); print('MEDIANS',np.median(p[y==0]),np.median(p[y==1]))
print('HIGHEST_THR_EXACT',np.sort(p[y==1])[::-1][int(np.ceil(.9*(y==1).sum()))-1],thr)
for k,g in o.groupby('fold'): print('FOLD_AUC',k,roc_auc_score(g.y_true,g.y_prob_plain),roc_auc_score(g.y_true,g.y_prob_tta))
for c in ['breast','busi','gdph','sysucc']:
 e=pd.read_csv(f'reports/external_{c}_preds.csv'); yy=e.y_true.to_numpy(); pp=e.y_prob_calibrated.to_numpy(); cc=confusion_counts(yy,pp,thr); ci=patient_bootstrap(e.rename(columns={'y_true':'label'}),pp,thr)
 print('EXTERNAL',c,len(e),roc_auc_score(yy,pp),cc,point_metrics(*cc),ci,'medians',np.median(pp[yy==0]),np.median(pp[yy==1]),'pred_mismatch',int(((pp>=thr)!=e.y_pred).sum())); print('ORACLE',c,pick_operating_point(yy,pp),point_metrics(*confusion_counts(yy,pp,pick_operating_point(yy,pp))))
print('NESTED')
for k in range(1,6):
 fit=o.fold.to_numpy()!=k; ev=~fit; th=pick_operating_point(y[fit],p[fit]); print(k,th,point_metrics(*confusion_counts(y[ev],p[ev],th)))
spec=importlib.util.spec_from_file_location('sweep','scripts/consistency_sweep.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
class MemoryOutput:
 def write_text(self,s,**kw): self.s=s
 def __str__(self): return '<in-memory output; no file written>'
mod.OUT=MemoryOutput(); mod.main(); original=Path('docs/consistency_check.md').read_text(); print('SWEEP_BYTE_IDENTICAL',original==mod.OUT.s,'original_sha256',hashlib.sha256(original.encode()).hexdigest(),'rerun_sha256',hashlib.sha256(mod.OUT.s.encode()).hexdigest()); print(''.join(difflib.unified_diff(original.splitlines(True),mod.OUT.s.splitlines(True)))[:5000])
PY
```

#### C2 — full pytest suite, bare clone and mounted read-only harness

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NO_ALBUMENTATIONS_UPDATE=1 MPLBACKEND=Agg /Users/thomaswang/projects/breast-us-cad/.venv/bin/python -B -u - <<'PY'
import os
os.environ['MPLCONFIGDIR']='/Users/thomaswang/.matplotlib'
_real_access=os.access
os.access=lambda p,m,*a,**kw: True if str(p)=='/Users/thomaswang/.matplotlib' else _real_access(p,m,*a,**kw)
import tempfile
tempfile.tempdir='/tmp'
os.environ['TORCHINDUCTOR_CACHE_DIR']='/__audit_temp__/torch'
_makedirs=os.makedirs
os.makedirs=lambda p,*a,**kw: None if str(p).startswith('/__audit_temp__') else _makedirs(p,*a,**kw)
class NoDiskTemp:
 def __init__(self,*a,**kw): self.name='/__audit_temp__/unused'
 def __enter__(self): return self.name
 def __exit__(self,*a): pass
 def cleanup(self): pass
tempfile.TemporaryDirectory=NoDiskTemp
import sys,os,io
from pathlib import Path,PosixPath
sys.path.insert(0,'src')
import pytest,pandas as pd,cv2,torch
# Keep the two filesystem-oriented tests entirely in memory.
mem={}
class MemPath(PosixPath):
 def exists(self): return str(self) in mem
 def mkdir(self,*a,**kw): mem[str(self)]=None
 def write_bytes(self,b): mem[str(self)]=b; return len(b)
 def read_bytes(self): return mem[str(self)]
_pexists,_pread=Path.exists,Path.read_bytes
Path.exists=lambda p: str(p) in mem if str(p).startswith('/__audit_memory__') else _pexists(p)
Path.read_bytes=lambda p: mem[str(p)] if str(p).startswith('/__audit_memory__') else _pread(p)
class AuditFixtures:
 @pytest.fixture
 def tmp_path(self,request): return MemPath('/__audit_memory__')/request.node.name
save,load=torch.save,torch.load
def ms(obj,f,*a,**kw):
 if isinstance(f,MemPath):
  b=io.BytesIO(); save(obj,b,*a,**kw); mem[str(f)]=b.getvalue(); return
 return save(obj,f,*a,**kw)
def ml(f,*a,**kw): return load(io.BytesIO(mem[str(f)]) if isinstance(f,MemPath) else f,*a,**kw)
torch.save,torch.load=ms,ml
# Reject all actual writes, including cache writes.
def guard(event,args):
 if event=='open' and args[0] != '/dev/null' and isinstance(args[2],int) and args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND): raise PermissionError('AUDIT read-only: '+str(args[0]))
 if event in ('os.mkdir','os.remove','os.rename','os.rmdir'): raise PermissionError('AUDIT read-only: '+str(args))
sys.addaudithook(guard)
print('BARE CLONE TEST SUITE (in-memory temp fixtures; no mounts)')
r=pytest.main(['-q','--capture=sys','--tb=short','-p','no:cacheprovider','tests'],plugins=[AuditFixtures()]); print('BARE_EXIT',r)
# Only physical data accesses redirected; source, configs, keep-lists and reports stay in clone.
def mapped(p):
 s=str(p); prefix='data/raw/'
 return str(Path(os.environ['DATA_ROOT'])/'raw'/s[len(prefix):]) if s.startswith(prefix) else p
rc,re,im,exists=pd.read_csv,pd.read_excel,cv2.imread,Path.exists
pd.read_csv=lambda p,*a,**kw: rc(mapped(p),*a,**kw)
pd.read_excel=lambda p,*a,**kw: re(mapped(p),*a,**kw)
cv2.imread=lambda p,*a,**kw: im(str(mapped(p)),*a,**kw)
Path.exists=lambda p: exists(Path(mapped(p))) if str(p).startswith('data/raw/') else exists(p)
_mounted_read=Path.read_bytes
Path.read_bytes=lambda p: _pread(Path(mapped(p))) if str(p).startswith('data/raw/') else _mounted_read(p)
print('MOUNTED SUITE (same source/tests; in-memory temp fixtures; read redirection only)')
r=pytest.main(['-v','--capture=sys','--tb=short','-p','no:cacheprovider','tests'],plugins=[AuditFixtures()]); print('MOUNTED_EXIT',r)
PY
```

#### C3 — checkpoint/wandb provenance, epoch histories and identity-rewrite mapping

```bash
PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 MPLBACKEND=Agg /Users/thomaswang/projects/breast-us-cad/.venv/bin/python -B -u - <<'PY'
import os
os.environ['MPLCONFIGDIR']='/Users/thomaswang/.matplotlib'
_real_access=os.access
os.access=lambda p,m,*a,**kw: True if str(p)=='/Users/thomaswang/.matplotlib' else _real_access(p,m,*a,**kw)
import tempfile
tempfile.tempdir='/tmp'
os.environ['TORCHINDUCTOR_CACHE_DIR']='/__audit_temp__/torch'
_makedirs=os.makedirs
os.makedirs=lambda p,*a,**kw: None if str(p).startswith('/__audit_temp__') else _makedirs(p,*a,**kw)
class NoDiskTemp:
 def __init__(self,*a,**kw): self.name='/__audit_temp__/unused'
 def __enter__(self): return self.name
 def __exit__(self,*a): pass
 def cleanup(self): pass
tempfile.TemporaryDirectory=NoDiskTemp
import sys,hashlib,subprocess,json,re
from pathlib import Path
sys.path.insert(0,'src')
import posthoc_epoch_selection as ep,torch,yaml,pandas as pd
orig=Path(os.environ['DATA_ROOT']).parent
ep.WANDB_DIR=orig/'wandb'
runs=ep.find_cv_vit_runs()
for k,r in runs.items():
 m=json.loads((r/'files/wandb-metadata.json').read_text()); c=yaml.safe_load((r/'files/config.yaml').read_text())
 ck=torch.load(Path(os.environ['MODELS_ROOT'])/f'cv_vit_fold{k}.pt',map_location='cpu',weights_only=False)
 print('RUN',k,m['startedAt'],m['git']['commit'],c['train_folds']['value'],c['val_folds']['value'],'CKPT',ck['config']['data']['train_folds'],ck['config']['data']['val_folds'],ck['epoch'],ck['val_auc'])
import builtins
_realopen=builtins.open
ep.datastore.open=lambda fname,mode='r',*a,**kw: _realopen(fname,'rb' if mode=='r+b' else mode,*a,**kw)
table,k=ep.tabulate({f:ep.read_history(r) for f,r in runs.items()}); ep.assert_reproduces_summary(table); saved=pd.read_csv('reports/posthoc_epoch_selection.csv'); print('EPOCH_TABLE_MAX_DIFF',(table.select_dtypes('number')-saved.select_dtypes('number')).abs().max().max()); print(table.to_string(index=False)); print('EPOCH_MEAN_SD',table.select_dtypes('number').agg(['mean','std']).to_string())
def g(*a,original=False): return subprocess.check_output(['git','-C',str(orig if original else Path.cwd()),*a],text=True).strip()
rows=[]
for line in Path('docs/PROVENANCE.md').read_text().splitlines():
 mm=re.match(r'\| ([0-9a-f]{7}) \| (.*?) \| (.*?) \| ([0-9a-f]{7}) \|',line)
 if mm:
  old,date,email,new=mm.groups()
  try:
   a=g('show','-s','--format=%T%n%aI%n%cI%n%B',old,original=True); b=g('show','-s','--format=%T%n%aI%n%cI%n%B',new); rows.append((old,new,a==b))
  except subprocess.CalledProcessError: rows.append((old,new,'missing'))
print('REWRITE_MAPPING',rows)
print('FSCK_CLONE',g('fsck','--unreachable','--no-reflogs'))
print('FSCK_ORIGINAL_UNREACHABLE_COMMITS',sum('unreachable commit' in l for l in g('fsck','--unreachable','--no-reflogs',original=True).splitlines()))
for path in ['RESULTS.md','data/external_protocol.md']:
 diff=g('diff','2cbe1da','HEAD','--',path); print('REMOVED_LINES',path,sum(l.startswith('-') and not l.startswith('---') for l in diff.splitlines()))
for sha in ['436ab56','610e87f','78f3991','3faa70b','4fc3fb2']:
 print('AMENDMENT_FILES',sha,g('show','--format=','--name-only',sha))
print('LOCAL_CHECKSUMS')
for l in Path('models/CHECKSUMS.txt').read_text().splitlines():
 if not l or l.startswith('#'): continue
 sha,p=l.split(maxsplit=1); p=p.lstrip('*'); file=Path(p)
 if not file.exists(): file=Path(os.environ['MODELS_ROOT'])/p.removeprefix('models/')
 if file.exists():
  h=hashlib.sha256()
  with file.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  print(p,h.hexdigest()==sha)
 else: print(p,'MISSING')
PY
```

#### C4 — independent all-pairs pHash, dedup keep-lists and every-image resize sweep

```bash
PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 MPLBACKEND=Agg /Users/thomaswang/projects/breast-us-cad/.venv/bin/python -B -u - <<'PY'
import os
os.environ['MPLCONFIGDIR']='/Users/thomaswang/.matplotlib'
_real_access=os.access
os.access=lambda p,m,*a,**kw: True if str(p)=='/Users/thomaswang/.matplotlib' else _real_access(p,m,*a,**kw)
import tempfile
tempfile.tempdir='/tmp'
os.environ['TORCHINDUCTOR_CACHE_DIR']='/__audit_temp__/torch'
_makedirs=os.makedirs
os.makedirs=lambda p,*a,**kw: None if str(p).startswith('/__audit_temp__') else _makedirs(p,*a,**kw)
class NoDiskTemp:
 def __init__(self,*a,**kw): self.name='/__audit_temp__/unused'
 def __enter__(self): return self.name
 def __exit__(self,*a): pass
 def cleanup(self): pass
tempfile.TemporaryDirectory=NoDiskTemp
import sys,itertools,hashlib,io
from pathlib import Path
sys.path.insert(0,'src')
import numpy as np,pandas as pd,cv2,imagehash
from PIL import Image
from inference import resize_bilinear_frozen
raw=Path(os.environ['DATA_ROOT'])/'raw'
meta=pd.read_csv(raw/'busbra/bus_data.csv')
br=pd.read_excel(raw/'breast_poland/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx')
sets={'busbra':[raw/'busbra/Images'/f'{id}.png' for id in meta.ID],
'breast':[raw/'breast_poland'/f for f in br.loc[br.Classification!='normal','Image_filename']],
'busi':[Path(os.environ['DATA_ROOT'])/p.removeprefix('data/') for p in pd.read_csv('data/splits/busi_clean.csv').image_path],
'gdph':sorted((raw/'gdph_sysucc/GDPH').glob('*.png')),
'sysucc':sorted((raw/'gdph_sysucc/SYSUCC').glob('*.png')),
'busi_raw':sorted(p for p in (raw/'busi/images').glob('*.png') if not p.name.startswith('normal'))}
hashes={}
for c,files in sets.items():
 hashes[c]=[int(str(imagehash.phash(Image.open(p).convert('L'),hash_size=8)),16) for p in files]; print('HASHED',c,len(files))
hits={}; pairtotal=0
for ia,a in enumerate(list(sets)[:5]):
 for b in list(sets)[ia:5]:
  found=[]; minimum=64; count=0
  for i,ha in enumerate(hashes[a]):
   for j in range(i+1 if a==b else 0,len(hashes[b])):
    d=(ha^hashes[b][j]).bit_count(); minimum=min(minimum,d); count+=1
    if d<=8:found.append((i,j,d))
  pairtotal+=count; hits[a,b]=found; print('PAIRS',a,b,count,minimum,len(found))
  if a!=b: print('CROSS_CANDIDATES',[(sets[a][i].name,sets[b][j].name,d) for i,j,d in found])
print('TOTAL_PAIRS',pairtotal)
for c in ['gdph','sysucc','busi_raw']:
 files=sets[c]; n=len(files); parent=list(range(n))
 def find(i):
  while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
  return i
 hh=hits.get((c,c))
 if hh is None: hh=[(i,j,(hashes[c][i]^hashes[c][j]).bit_count()) for i in range(n) for j in range(i+1,n) if (hashes[c][i]^hashes[c][j]).bit_count()<=8]
 for i,j,d in hh: parent[find(j)]=find(i)
 groups={}
 for i in range(n):groups.setdefault(find(i),[]).append(files[i].name)
 keep=[]; conflicts=[]; dup=[]
 for g in groups.values():
  cl={s.split('(')[0].split('_')[0] for s in g}
  if len(cl)>1: conflicts+=g
  else: keep.append(sorted(g)[0]);dup+=sorted(g)[1:]
 cohort=c.replace('_raw',''); saved=pd.read_csv(f'data/splits/{cohort}_clean.csv')
 print('DEDUP',c,n,len(keep),'dups',len(dup),'conflicts',len(conflicts),'matches',sorted(keep)==sorted(saved.filename),'hits',len(hh))
 for column in ['label','class']: print('LABEL_MATCH',c,column,all((r.label==int(r.filename.startswith('malignant'))) and r['class']==r.filename.split('(')[0].split('_')[0] for _,r in saved.iterrows()))
# Recheck distances using final keep-lists, not just original raw sets.
for c in ['gdph','sysucc']:
 keep=set(pd.read_csv(f'data/splits/{c}_clean.csv').filename); indices=[i for i,p in enumerate(sets[c]) if p.name in keep]; sets[c]=[sets[c][i] for i in indices]; hashes[c]=[hashes[c][i] for i in indices]
for ia,a in enumerate(list(sets)[:5]):
 for b in list(sets)[ia:5]:
  hh=[(sets[a][i].name,sets[b][j].name,(ha^hb).bit_count()) for i,ha in enumerate(hashes[a]) for j,hb in enumerate(hashes[b]) if (a!=b or i<j) and (ha^hb).bit_count()<=8]
  if hh:print('FINAL_KEEPLIST_HITS',a,b,hh)
for c in list(sets)[:5]:
 n_diff=0; mx=0; shapes=set()
 for f in sets[c]:
  gray=cv2.imread(str(f),0); rgb=np.repeat(gray[:,:,None],3,axis=2); shapes.add(gray.shape)
  d=np.abs(cv2.resize(rgb,(224,224),interpolation=cv2.INTER_LINEAR).astype(int)-resize_bilinear_frozen(rgb).astype(int)).max()
  n_diff+=int(d>0);mx=max(mx,int(d))
 print('RESIZE',c,n_diff,len(sets[c]),mx,'shapes',len(shapes))
PY
```

#### C5 — AST/source parity and initial CPU self-check

```bash
PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 MPLBACKEND=Agg /Users/thomaswang/projects/breast-us-cad/.venv/bin/python -B -u - <<'PY'
import os
os.environ['MPLCONFIGDIR']='/Users/thomaswang/.matplotlib'
_real_access=os.access
os.access=lambda p,m,*a,**kw: True if str(p)=='/Users/thomaswang/.matplotlib' else _real_access(p,m,*a,**kw)
import tempfile
tempfile.tempdir='/tmp'
os.environ['TORCHINDUCTOR_CACHE_DIR']='/__audit_temp__/torch'
_makedirs=os.makedirs
os.makedirs=lambda p,*a,**kw: None if str(p).startswith('/__audit_temp__') else _makedirs(p,*a,**kw)
class NoDiskTemp:
 def __init__(self,*a,**kw): self.name='/__audit_temp__/unused'
 def __enter__(self): return self.name
 def __exit__(self,*a): pass
 def cleanup(self): pass
tempfile.TemporaryDirectory=NoDiskTemp
import sys,json,ast,hashlib,io,contextlib
from pathlib import Path
sys.path.insert(0,'src')
os.environ['BREAST_US_CAD_MODELS_DIR']=os.environ['MODELS_ROOT']
import torch,cv2,yaml,numpy as np
import inference,external_val
torch.set_num_threads(4)
def funcs(p):
 t=ast.parse(Path(p).read_text())
 for node in ast.walk(t):
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
 return {n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
for a,b in [('src/inference.py','deploy/inference.py'),('app/app.py','deploy/app.py'),('src/evaluate.py','src/inference.py'),('src/calibrate.py','src/inference.py')]:
 x,y=funcs(a),funcs(b); shared=set(x)&set(y);print('AST',a,b,'byte_equal',Path(a).read_bytes()==Path(b).read_bytes(),'identical',sorted(n for n in shared if x[n]==y[n]),'different',sorted(n for n in shared if x[n]!=y[n]))
cfg=yaml.safe_load(Path('configs/vit.yaml').read_text())
try: external_val.self_check(cfg)
except Exception as e: print('LITERAL_SELF_CHECK',type(e).__name__,str(e))
cfg['data']['root']=str(Path(os.environ['DATA_ROOT'])/'raw/busbra'); cfg['data']['num_workers']=0
print('MPS_AVAILABLE',torch.backends.mps.is_available())
if not torch.backends.mps.is_available(): cfg['train']['device']='cpu'
buf=io.StringIO()
with contextlib.redirect_stdout(buf): external_val.self_check(cfg)
print(buf.getvalue());print('SELF_CHECK_BYTE_MATCH',buf.getvalue()==Path('reports/external_selfcheck.txt').read_text())
models=[inference.load_classifier(f,'cpu') for f in inference.CKPT_FILES]; T=inference.load_calibration()
expected=json.loads(Path('reports/app_example_probs.json').read_text())
from data import get_transforms
for p in sorted(Path('app/examples').glob('*.png')):
 g=cv2.imread(str(p),0); x=inference.preprocess_gray(g); val=get_transforms('val')(image=np.repeat(g[:,:,None],3,axis=2))['image']
 raw=inference.ensemble_prob(models,x); cal=float(inference.calibrate_probs(np.array([raw]),T)[0]); print('EXAMPLE',p.name,'tensor_equal',torch.equal(x.squeeze(0),val),'raw',repr(raw),'cal',repr(cal),'delta',max(abs(raw-float(expected[p.name]['raw'])),abs(cal-float(expected[p.name]['cal']))))
PY
```

#### C6 — live Space, four bundled examples only

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1 /Users/thomaswang/projects/breast-us-cad/.venv/bin/python -B -u - <<'PY'
import json,re,time,statistics,urllib.request,hashlib
from pathlib import Path
from gradio_client import Client,handle_file
expected=json.loads(Path('reports/app_example_probs.json').read_text())
client=Client('happytommy/breast-us-cad',verbose=False,download_files=False)
times=[]
for p in sorted(Path('app/examples').glob('*.png')):
 url='https://huggingface.co/spaces/happytommy/breast-us-cad/resolve/main/examples/'+p.name
 with urllib.request.urlopen(url,timeout=40) as r:b=r.read()
 print('EXAMPLE_FILE',p.name,'space_vs_app_bytes',b==p.read_bytes(),'space_vs_deploy',b==Path('deploy/examples',p.name).read_bytes(),flush=True)
 for attempt in range(3):
  t=time.perf_counter()
  try:
   out=client.predict(handle_file(str(p)),api_name='/predict')
   dt=time.perf_counter()-t; m=re.search(r'<!-- raw=([\d.e-]+) cal=([\d.e-]+) -->',out[2])
   if not m: raise ValueError('no numeric comment')
   raw,cal=map(float,m.groups());delta=max(abs(raw-float(expected[p.name]['raw'])),abs(cal-float(expected[p.name]['cal'])));times.append(dt)
   print('LIVE',p.name,repr(raw),repr(cal),'max_delta',delta,'seconds',dt,flush=True);break
  except Exception as e:print('LIVE_ATTEMPT',attempt+1,type(e).__name__,str(e)[:400],flush=True)
print('LIVE_TIME_MEDIAN',statistics.median(times) if times else None)
PY
```

#### C7 — full M1 rerun and exhaustive-threshold counterfactual

```bash
PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 MPLBACKEND=Agg /Users/thomaswang/projects/breast-us-cad/.venv/bin/python -B -u - <<'PY'
import os
os.environ['MPLCONFIGDIR']='/Users/thomaswang/.matplotlib'
_real_access=os.access
os.access=lambda p,m,*a,**kw: True if str(p)=='/Users/thomaswang/.matplotlib' else _real_access(p,m,*a,**kw)
import tempfile
tempfile.tempdir='/tmp'
os.environ['TORCHINDUCTOR_CACHE_DIR']='/__audit_temp__/torch'
_makedirs=os.makedirs
os.makedirs=lambda p,*a,**kw: None if str(p).startswith('/__audit_temp__') else _makedirs(p,*a,**kw)
class NoDiskTemp:
 def __init__(self,*a,**kw): self.name='/__audit_temp__/unused'
 def __enter__(self): return self.name
 def __exit__(self,*a): pass
 def cleanup(self): pass
tempfile.TemporaryDirectory=NoDiskTemp
import sys,json,contextlib,io
from pathlib import Path
sys.path.insert(0,'src')
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
import v2_recalib_curve as r
from calibrate import probs_to_logits
from pick_threshold import confusion_counts,point_metrics
allrows=[]
for c in r.COHORTS:
 a=r.run_curve(c,'M1',r.K_GRID,500,42); allrows.append(a); old=pd.read_csv('reports/v2_recalib_M1_draws.csv'); old=old[old.cohort==c].reset_index(drop=True)
 common=a.select_dtypes('number').columns.intersection(old.select_dtypes('number').columns)
 print('M1_REPRO',c,len(a),'maxdiff',(a[common]-old[common]).abs().max().max(),'degenerate',a.groupby('k').degenerate.mean().to_dict())
m1=r.summarize(pd.concat(allrows)); methods=pd.read_csv('reports/v2_recalib_methods_draws.csv'); summary=r.summarize(methods); oldsum=pd.read_csv('reports/v2_recalib_methods_summary.csv')
print('ALL_METHODS_SUMMARY_MAX_DIFF',(summary.select_dtypes('number')-oldsum.select_dtypes('number')).abs().max().max())
for c in r.COHORTS:
 print('KSTAR_KRELIABLE',c,[(m,r.k_star(summary,c,m),r.k_reliable(summary,c,m)) for m in r.METHODS])
 print('K10',summary[(summary.cohort==c)&(summary.k==10)][['method','sens_median','spec_median','recovery_median','frac_degenerate','frac_fit_failed']].to_dict('records'))
# independent rule: highest observed score retaining at least ceil(0.9*n_pos) positives
def exact(y,p):return float(np.sort(p[y==1])[::-1][int(np.ceil(.9*(y==1).sum()))-1])
o=pd.read_csv('reports/oof_vit_preds.csv');T=json.loads(Path('models/calibration.json').read_text())['temperature']; p=1/(1+np.exp(-probs_to_logits(o.y_prob_tta.to_numpy())/T));y=o.y_true.to_numpy()
for name,yy,pp in [('OOF',y,p)]+[(c,pd.read_csv(r.PREDS_CSV[c]).y_true.to_numpy(),pd.read_csv(r.PREDS_CSV[c]).y_prob_calibrated.to_numpy()) for c in r.COHORTS]:
 old=r.pick_operating_point(yy,pp); new=exact(yy,pp); print('EXACT_RULE',name,old,new,confusion_counts(yy,pp,old),confusion_counts(yy,pp,new))
r.pick_operating_point=exact
for c,old in zip(r.COHORTS,allrows):
 new=r.run_curve(c,'M1',r.K_GRID,500,42); changes=(old.threshold!=new.threshold); s=r.summarize(new)
 print('EXACT_RULE_M1_IMPACT',c,'changed_draws',changes.sum(),'max_delta_sens',abs(old.sens-new.sens).max(),'max_delta_spec',abs(old.spec-new.spec).max(),'kstar',r.k_star(s,c,'M1'))
x=pd.read_csv('reports/v2_resize_dispatch_impact.csv');print('RESIZE_SAVED',x.columns.tolist());print(x.select_dtypes('number').agg(['max','median']).to_string());print(x[x.decision_flip.astype(bool)].to_string(index=False))
PY
```

#### C8 — disagreement, BiomedCLIP and LOCO metrics/intervals

```bash
PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 MPLBACKEND=Agg /Users/thomaswang/projects/breast-us-cad/.venv/bin/python -B -u - <<'PY'
import os
os.environ['MPLCONFIGDIR']='/Users/thomaswang/.matplotlib'
_real_access=os.access
os.access=lambda p,m,*a,**kw: True if str(p)=='/Users/thomaswang/.matplotlib' else _real_access(p,m,*a,**kw)
import tempfile
tempfile.tempdir='/tmp'
os.environ['TORCHINDUCTOR_CACHE_DIR']='/__audit_temp__/torch'
_makedirs=os.makedirs
os.makedirs=lambda p,*a,**kw: None if str(p).startswith('/__audit_temp__') else _makedirs(p,*a,**kw)
class NoDiskTemp:
 def __init__(self,*a,**kw): self.name='/__audit_temp__/unused'
 def __enter__(self): return self.name
 def __exit__(self,*a): pass
 def cleanup(self): pass
tempfile.TemporaryDirectory=NoDiskTemp
import sys,json,io,contextlib
from pathlib import Path
sys.path.insert(0,'src')
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
import v2_disagreement as dis,v2_loco_report as lr,birads_comparison as br
from external_val import patient_bootstrap
from inference import calibrate_probs
from calibrate import fit_temperature,probs_to_logits
from pick_threshold import pick_operating_point,confusion_counts,point_metrics,patient_bootstrap_ci
thr=json.loads(Path('models/operating_point.json').read_text())['threshold']
oldmet=pd.read_csv('reports/v2_abstention_metrics.csv');oldcur=pd.read_csv('reports/v2_abstention_curves.csv');oldtable=pd.read_csv('reports/v2_loco_q1c_table.csv')
for c in dis.COHORTS:
 df=dis.load_cohort(c,thr); m=[]; curves=[]
 mem=pd.read_csv(f'reports/v2_members_{c}.csv');pred=pd.read_csv(f'reports/external_{c}_preds.csv')
 P=mem[dis.MEMBER_COLS].to_numpy(dtype=np.float32)
 pairmean=((P[:,0::2]+P[:,1::2])/2).mean(axis=1)
 print('MEMBER_PAIRMEAN_BITWISE',c,np.array_equal(pairmean,pred.y_prob_raw.to_numpy(dtype=np.float32)),'FLATMEAN_DIFF',abs(P.mean(axis=1)-pred.y_prob_raw.to_numpy(dtype=np.float32)).max())
 for sig in dis.SIGNALS:
  out=dis.error_auroc_ci(df,sig); m.append({'cohort':c,'signal':sig,**out});curves+=dis.abstention_rows(df,c,sig)
  ref=oldmet[(oldmet.cohort==c)&(oldmet.signal==sig)].iloc[0]; print('DISAGREEMENT',c,sig,out,'maxdiff',max(abs(out[k]-ref[k]) for k in out))
 mt=pd.DataFrame(m);ct=pd.DataFrame(curves); ref=oldcur[oldcur.cohort==c].reset_index(drop=True);cols=ct.select_dtypes('number').columns.intersection(ref.select_dtypes('number').columns)
 print('ABSTENTION_CURVE_MAX_DIFF',c,(ct[cols]-ref[cols]).abs().max().max(),'VERDICT',dis.verdict(mt,ct,c))
 d=lr.load_cohort(c); rows={}
 for model,col,t in [('loco','p_loco',d['thr_loco']),('v1_single','p_v1s',lr.V1_THR),('v1_ensemble','p_ens',lr.V1_THR)]:
  out=lr.model_metrics(d['df'],col,t);rows[model]=out; ref=oldtable[(oldtable.cohort==c)&(oldtable.model==model)].iloc[0];print('LOCO_METRICS',c,model,out,'maxdiff',max(abs(out[k]-ref[k]) for k in out))
 delta=lr.paired_delta_ci(d['df'],d['thr_loco']);print('LOCO_DELTA',c,rows['loco']['auc']-rows['v1_single']['auc'],delta,'vs_ensemble',rows['loco']['auc']-rows['v1_ensemble']['auc'])
br.XLSX=Path(os.environ['DATA_ROOT'])/'raw/gdph_sysucc/BIRADS&FOLD.xlsx'
meta=pd.read_excel(br.XLSX)
print('XLSX',len(meta),meta.columns.tolist(),meta.ID.nunique())
for reader in br.READERS: print('STRAY',reader,meta[~meta[reader].map(br.normalize).isin(br.VALID)][['ID',reader]].to_dict('records'))
for c in br.COHORTS:
 d=br.load_cohort(c);br.cohort_table(c,d)
 for reader in br.READERS:
  b=d[d.y_true==0];fp=b.y_pred==1;rp=b[reader].isin(br.POSITIVE)
  print('CONCORDANCE',c,reader,int((fp&rp).sum()),int(fp.sum()),int((~fp&rp).sum()),int((~fp).sum()))
v2=pd.read_csv('reports/v2_biomedclip_oof_preds.csv')
print('V2_COLS',v2.columns.tolist())
y=v2.y_true.to_numpy();p=v2.y_prob_tta.to_numpy();T=fit_temperature(probs_to_logits(p),y);pc=calibrate_probs(p,T);t=pick_operating_point(y,pc);base=float(np.median(pc[y==0]))
print('BIOMED_OOF',roc_auc_score(y,p),T,t,base,point_metrics(*confusion_counts(y,pc,t)),patient_bootstrap_ci(v2.assign(prob_cal=pc),t))
for c in dis.COHORTS:
 d=pd.read_csv(f'reports/v2_biomedclip_external_{c}_preds.csv');y=d.y_true.to_numpy();p=d.y_prob_calibrated.to_numpy();print('BIOMED_EXTERNAL',c,roc_auc_score(y,p),point_metrics(*confusion_counts(y,p,t)),patient_bootstrap(d.rename(columns={'y_true':'label'}),p,t),'benign_med',np.median(p[y==0]))
PY
```

#### C9 — successful MPS self-check and local four-example reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 MPLBACKEND=Agg /Users/thomaswang/projects/breast-us-cad/.venv/bin/python -B -u - <<'PY'
import os
os.environ['MPLCONFIGDIR']='/Users/thomaswang/.matplotlib'
_real_access=os.access
os.access=lambda p,m,*a,**kw: True if str(p)=='/Users/thomaswang/.matplotlib' else _real_access(p,m,*a,**kw)
import tempfile
tempfile.tempdir='/tmp'
os.environ['TORCHINDUCTOR_CACHE_DIR']='/__audit_temp__/torch'
_makedirs=os.makedirs
os.makedirs=lambda p,*a,**kw: None if str(p).startswith('/__audit_temp__') else _makedirs(p,*a,**kw)
class NoDiskTemp:
 def __init__(self,*a,**kw): self.name='/__audit_temp__/unused'
 def __enter__(self): return self.name
 def __exit__(self,*a): pass
 def cleanup(self): pass
tempfile.TemporaryDirectory=NoDiskTemp
import sys,json,contextlib,io,difflib
from pathlib import Path
sys.path.insert(0,'src')
os.environ['BREAST_US_CAD_MODELS_DIR']=os.environ['MODELS_ROOT']
import torch,cv2,yaml,numpy as np,pandas as pd
import inference,external_val
torch.set_num_threads(4)
cfg=yaml.safe_load(Path('configs/vit.yaml').read_text());cfg['data']['root']=str(Path(os.environ['DATA_ROOT'])/'raw/busbra');cfg['data']['num_workers']=0
print('MPS_AVAILABLE',torch.backends.mps.is_available())
buf=io.StringIO()
with contextlib.redirect_stdout(buf):external_val.self_check(cfg)
print(buf.getvalue());print('SELF_CHECK_BYTE_MATCH',buf.getvalue()==Path('reports/external_selfcheck.txt').read_text())
models=[inference.load_classifier(f,'cpu') for f in inference.CKPT_FILES];T=inference.load_calibration();expected=json.loads(Path('reports/app_example_probs.json').read_text())
from data import get_transforms
for p in sorted(Path('app/examples').glob('*.png')):
 g=cv2.imread(str(p),0);x=inference.preprocess_gray(g);val=get_transforms('val')(image=np.repeat(g[:,:,None],3,axis=2))['image']
 raw=inference.ensemble_prob(x,models);cal=float(inference.calibrate_probs(np.array([raw]),T)[0]);print('EXAMPLE',p.name,'tensor_equal',torch.equal(x.squeeze(0),val),'raw',repr(raw),'cal',repr(cal),'delta',max(abs(raw-float(expected[p.name]['raw'])),abs(cal-float(expected[p.name]['cal']))))
PY
```

#### C10 — remote archive bytes, HF history/checksums and Space source checks

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/thomaswang/projects/breast-us-cad/.venv/bin/python -B -u - <<'PY'
import urllib.request,json,hashlib,subprocess,concurrent.futures,http.client
from pathlib import Path
def get(u):
 with urllib.request.urlopen(u,timeout=40) as r:return r.read()
url='https://zenodo.org/api/records/22630912/files/breast-us-cad-v1.0.zip/content'
try:
 with urllib.request.urlopen(url,timeout=60) as r:
  b=bytearray()
  while True:
   try:x=r.read(1024*1024)
   except http.client.IncompleteRead as e:b.extend(e.partial);raise
   if not x:break
   b.extend(x)
  a=subprocess.check_output(['git','archive','--format=zip','--prefix=breast-us-cad-v1.0/','v1.0-audited'])
  print('ZENODO_ARCHIVE',len(b),hashlib.sha256(b).hexdigest(),hashlib.md5(b).hexdigest(),'BYTE_EQUAL',bytes(b)==a,flush=True)
except Exception as e:print('ARCHIVE_ERROR',repr(e),len(b),flush=True)
for kind,name in [('models','happytommy/breast-us-cad-weights'),('spaces','happytommy/breast-us-cad')]:
 try:
  meta=json.loads(get(f'https://huggingface.co/api/{kind}/{name}'))
  print('HF_INFO',kind,{k:meta.get(k) for k in ['sha','createdAt','lastModified','runtime']})
  commits=json.loads(get(f'https://huggingface.co/api/{kind}/{name}/commits/main'))
  print('HF_COMMITS',kind,json.dumps(commits))
 except Exception as e: print('HF_ERROR',repr(e))
meta=json.loads(get('https://huggingface.co/api/models/happytommy/breast-us-cad-weights?blobs=true'))
siblings={f['rfilename']:f for f in meta['siblings']}
def check(l):
 sha,p=l.split(maxsplit=1); name=('v2/'+p if p.startswith(('v2_','pretrained/')) else p); f=siblings[name]
 actual=f['lfs']['sha256'] if 'lfs' in f else hashlib.sha256(get('https://huggingface.co/happytommy/breast-us-cad-weights/resolve/main/'+name)).hexdigest()
 return name,sha==actual
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: print('HF_CHECKSUMS',list(ex.map(check,[l for l in Path('models/CHECKSUMS.txt').read_text().splitlines() if l and not l.startswith('#')])))
for f in ['app.py','inference.py','requirements.txt','README.md']:
 try:print('SPACE_BYTES',f,get('https://huggingface.co/spaces/happytommy/breast-us-cad/resolve/main/'+f)==Path('deploy',f).read_bytes())
 except Exception as e:print('SPACE_ERROR',f,repr(e))
PY
```

#### C11 — negative Platt slopes, XLSX join, WHU absence and artifact histories

```bash
PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 MPLBACKEND=Agg /Users/thomaswang/projects/breast-us-cad/.venv/bin/python -B -u - <<'PY'
import os
os.environ['MPLCONFIGDIR']='/Users/thomaswang/.matplotlib'
_real_access=os.access
os.access=lambda p,m,*a,**kw: True if str(p)=='/Users/thomaswang/.matplotlib' else _real_access(p,m,*a,**kw)
import tempfile
tempfile.tempdir='/tmp'
os.environ['TORCHINDUCTOR_CACHE_DIR']='/__audit_temp__/torch'
_makedirs=os.makedirs
os.makedirs=lambda p,*a,**kw: None if str(p).startswith('/__audit_temp__') else _makedirs(p,*a,**kw)
class NoDiskTemp:
 def __init__(self,*a,**kw): self.name='/__audit_temp__/unused'
 def __enter__(self): return self.name
 def __exit__(self,*a): pass
 def cleanup(self): pass
tempfile.TemporaryDirectory=NoDiskTemp
import sys,json,warnings,subprocess
from pathlib import Path
sys.path.insert(0,'src')
import numpy as np,pandas as pd
import v2_recalib_curve as r
from calibrate import probs_to_logits
warnings.filterwarnings('ignore',category=FutureWarning)
for ci,c in enumerate(r.COHORTS):
 d=r.load_cohort(c); y=d.y_true.to_numpy();z=probs_to_logits(d.y_prob_raw.to_numpy()); negative=[];zero=0
 for draw in range(500):
  ix=np.random.default_rng([42,ci,10,draw]).choice(len(d),size=10,replace=False)
  if np.unique(y[ix]).size<2:continue
  model=r.fit_platt(z[ix],y[ix]);a=float(model.coef_[0,0])
  if a<0:negative.append((draw,a))
  zero+=a==0
 print('PLATT_NEGATIVE_SLOPES_K10',c,len(negative),'zero',zero,'examples',negative[:4])
# bytes/label joins, raw label inventory and scope
meta=pd.read_excel(Path(os.environ['DATA_ROOT'])/'raw/gdph_sysucc/BIRADS&FOLD.xlsx')
rawids=[]
for c in ['GDPH','SYSUCC']:
 rawids.extend(p.stem for p in (Path(os.environ['DATA_ROOT'])/'raw/gdph_sysucc'/c).glob('*.png'))
print('GDPH_SYSUCC_XLSX_JOIN',len(rawids),len(set(rawids)),len(meta),set(rawids)==set(meta.ID))
paths=subprocess.check_output(['git','ls-files','reports','data/splits'],text=True).splitlines()
hits=[]
for f in paths:
 if f.endswith(('.csv','.json')) and 'busi_whu' in Path(f).read_text().lower():hits.append(f)
print('WHU_TRACKED_RESULT_HITS',hits)
# inspect all prediction/calibration/keep-list histories and timestamps without printing data rows
for pattern in ['reports/external_*_preds.csv','reports/oof_vit_preds.csv','models/calibration.json','models/operating_point.json','data/splits/*.csv','reports/v2_members_*.csv','reports/v2_loco_*_preds.csv']:
 files=[str(p) for p in Path('.').glob(pattern)]
 for f in files:
  log=subprocess.check_output(['git','log','--format=%h %aI %cI','--name-status','--',f],text=True).strip().replace('\n',' | ');print('ARTIFACT_HISTORY',f,log)
for f in ['reports/external_breast_preds.csv','reports/external_busi_preds.csv','reports/external_gdph_preds.csv','reports/external_sysucc_preds.csv','reports/v2_recalib_M1_draws.csv','reports/v2_members_breast.csv','src/v2_data.py']:
 p=Path(os.environ['DATA_ROOT']).parent/f
 import datetime
 print('ORIGINAL_MTIME',f,datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone(datetime.timedelta(hours=8))).isoformat())
PY
```

#### C12 — supplemental v2 internal self-check and zero-distance pixel comparison

```bash
PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 MPLBACKEND=Agg /Users/thomaswang/projects/breast-us-cad/.venv/bin/python -B -u - <<'PY'
import os
os.environ['MPLCONFIGDIR']='/Users/thomaswang/.matplotlib'
_real_access=os.access
os.access=lambda p,m,*a,**kw: True if str(p)=='/Users/thomaswang/.matplotlib' else _real_access(p,m,*a,**kw)
import tempfile
tempfile.tempdir='/tmp'
os.environ['TORCHINDUCTOR_CACHE_DIR']='/__audit_temp__/torch'
_makedirs=os.makedirs
os.makedirs=lambda p,*a,**kw: None if str(p).startswith('/__audit_temp__') else _makedirs(p,*a,**kw)
class NoDiskTemp:
 def __init__(self,*a,**kw): self.name='/__audit_temp__/unused'
 def __enter__(self): return self.name
 def __exit__(self,*a): pass
 def cleanup(self): pass
tempfile.TemporaryDirectory=NoDiskTemp

import sys
from pathlib import Path
sys.path.insert(0,'src')
os.environ['BREAST_US_CAD_MODELS_DIR']=os.environ['MODELS_ROOT']
import torch,yaml,pandas as pd,numpy as np
import v2_biomedclip_external as v
torch.set_num_threads(4)
cfg=yaml.safe_load(Path(v.CONFIG).read_text())
cfg['data']['root']=str(Path(os.environ['DATA_ROOT'])/'raw/busbra')
cfg['data']['num_workers']=0
cfg['train']['device']='cpu'
v.self_check(cfg)
from PIL import Image
hits=pd.read_csv('reports/phash_sweep_hits.csv')
d0=hits[(hits.distance==0)&(hits.set_a==hits.set_b)&hits.set_a.isin(['gdph','sysucc'])]
same=0
for r in d0.itertuples():
 root=Path(os.environ['DATA_ROOT'])/'raw/gdph_sysucc'/r.set_a.upper()
 a=np.asarray(Image.open(root/r.image_a));b=np.asarray(Image.open(root/r.image_b))
 same+=np.array_equal(a,b)
print('D0_PIXEL_IDENTICAL',len(d0),same)
PY
```

Actual output:

```text
self-check: fold-5 val, v2 fold-5 ckpt only, n = 383
  expected (reports/v2_biomedclip_tta_summary.csv):          0.9230647908649297
  reproduced by external code path:     0.9230647908649297
  calibration (T=2.8645) + threshold (0.2007) applied: 196/383 flagged malignant
  SELF-CHECK PASSED — AUC reproduced digit-for-digit.
D0_PIXEL_IDENTICAL 154 88
```

#### C13 — dependency inventory, saved segmentation aggregates and cross-backbone shift ratios

```bash
PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 MPLBACKEND=Agg /Users/thomaswang/projects/breast-us-cad/.venv/bin/python -B -u - <<'PY'
import os
os.environ['MPLCONFIGDIR']='/Users/thomaswang/.matplotlib'
_real_access=os.access
os.access=lambda p,m,*a,**kw: True if str(p)=='/Users/thomaswang/.matplotlib' else _real_access(p,m,*a,**kw)
import tempfile
tempfile.tempdir='/tmp'
os.environ['TORCHINDUCTOR_CACHE_DIR']='/__audit_temp__/torch'
_makedirs=os.makedirs
os.makedirs=lambda p,*a,**kw: None if str(p).startswith('/__audit_temp__') else _makedirs(p,*a,**kw)
class NoDiskTemp:
 def __init__(self,*a,**kw): self.name='/__audit_temp__/unused'
 def __enter__(self): return self.name
 def __exit__(self,*a): pass
 def cleanup(self): pass
tempfile.TemporaryDirectory=NoDiskTemp
import sys,json,subprocess,importlib.metadata as md
from pathlib import Path
sys.path.insert(0,'src')
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
import v2_cross_site as cs
from calibrate import fit_temperature,probs_to_logits
from pick_threshold import pick_operating_point,confusion_counts,point_metrics
T=json.loads(Path('models/calibration.json').read_text())['temperature']
print('DEPENDENCIES')
from packaging.requirements import Requirement
missing=[];mismatch=[];n=0
for line in Path('requirements.txt').read_text().splitlines():
 if not line.strip() or line.startswith('#'):continue
 r=Requirement(line);n+=1
 try:v=md.version(r.name)
 except md.PackageNotFoundError:missing.append(r.name);continue
 if r.specifier and v not in r.specifier:mismatch.append((r.name,str(r.specifier),v))
print('REQ_COUNT',n,'MISSING',missing,'MISMATCH',mismatch)
print('SEG',pd.read_csv('reports/seg_metrics_seg_unet_effb0.csv').select_dtypes('number').agg(['mean','median']).to_string())
o=pd.read_csv('reports/oof_vit_preds.csv'); v=pd.read_csv('reports/v2_biomedclip_oof_preds.csv');med=lambda df,t:np.median((1/(1+np.exp(-probs_to_logits(df.y_prob_tta.to_numpy())/t)))[df.y_true.to_numpy()==0]);a=med(o,T);T2=json.loads(Path('models/v2_biomedclip_calibration.json').read_text())['temperature'];b=med(v,T2)
for c in cs.COHORTS:
 x=pd.read_csv(f'reports/external_{c}_preds.csv');z=pd.read_csv(f'reports/v2_biomedclip_external_{c}_preds.csv');shift1=x.loc[x.y_true==0,'y_prob_calibrated'].median()-a;shift2=z.loc[z.y_true==0,'y_prob_calibrated'].median()-b;print('SHIFT_RATIO',c,shift1,shift2,abs(shift2)/abs(shift1))
print('COMMIT_COUNT',subprocess.check_output(['git','rev-list','--count','HEAD'],text=True).strip(),'BEFORE_P4',subprocess.check_output(['git','rev-list','--count','1f196a9^'],text=True).strip())
PY
```

The separate full cross-site matrix refit was also executed in the earlier part of this same program (C13's source task, before the dependency tail): for each source, `pick_operating_point(y, sigmoid(z/T))` and `fit_temperature(z,y)`, followed by `v2_cross_site.sens_spec` for all five targets. Output: source thresholds/T listed in §2.4; maximum difference across the 20 saved rows **2.220446049250313e-16**. A later audit-only occlusion-summary line mistakenly grouped a wide CSV by nonexistent column `model`; that wrapper error was corrected by reading `drop_vit` and `drop_cn` directly (121 rows, medians .004689932/.007868886, maxima .86568594/.9946281, counts >.2 =20/19). It was not a repository failure.

**Verdict for §11: FAIL — major.** The author's corrections genuinely resolve many original wording and availability defects. They do not resolve the underlying internal-validation bias, permanent lack of a pre-run external timestamp, or all scientific/code risks. The “resolved-by-P2/P4/P6” labels must be read narrowly; new empirical counterexamples prevent a blanket closure.

## Executive summary — one page

**Overall verdict: FAIL — major for the manuscript as written.** The repository supports reproducible retrospective external performance and several carefully recorded negative findings. It does not yet support its full account of internal ensemble validation, bias correction, protocol enforcement or causal/clinical interpretation. I found no demonstrated cross-fold patient leak or fabricated external score. That is materially narrower than certifying every claim.

**The three most serious findings**

1. **The internal headline describes the wrong predictor and overstates the correction for validation reuse.** The .9254 OOF AUC comes from one held-out checkpoint per image plus flip, not a five-checkpoint ensemble. Temperature/threshold fitted on that distribution are applied to an ensemble externally. Best-epoch selection uses the reported folds; the post-hoc “nested” threshold analysis does not retrain independent outer models, and epoch18 was selected from the same validation histories. These facts undermine the claimed unbiased correction and a clean same-predictor explanation of calibration transport failure (README:33–49; report:22/:100/:112; `tta_eval.py:44–91`).

2. **The stated threshold rule is not literally implemented, and protocol permanence is not enforced.** ROC pruning discards valid thresholds: the actual highest internal sens≥.90 threshold is .2687816532, not .2683212035. An exhaustive-rule counterfactual changes 2432/11000 M1 thresholds and can substantially change held-out decisions, although the reported M1 k* conclusions survive. The v1 external command also overwrites existing CSVs if repeated; `--confirm` is not a single-run guard. History contains no committed overwrite, but the stronger enforcement claim is false (`pick_threshold.py:38`; `external_val.py:133–160`).

3. **The revised paper still contains factual and mathematical errors that the consistency check cannot detect.** It says external errors are not missed cancers despite 76 false negatives; says readers saw complete examinations when the primary paper explicitly says one image at a time; omits negative Platt slopes from its equivalence argument; misstates temperature differences and fixed-epoch fold ranking; and still calls published v2 weights/folds unavailable. The byte-reproducible “0 unverified” sweep matches arbitrary number tokens, not claims. These are not cured by a global sourcing assurance.

**The three strongest verified claims**

1. **Official BUS-BRA splits and inspected v1 training provenance are consistent:** 1875/1875 OOF assignments match the official partition, no patient spans folds, all five wandb configs use the correct complements, and all 15 current tests pass with mounted data.

2. **The four external saved evaluations reproduce:** AUCs .854161/.933879/.915387/.838030, their sens/spec and bootstrap intervals, the M1 learning curves, disagreement 0/4 failure, BiomedCLIP NOT CLAIMED, and the qualified LOCO A-only verdict all reproduce at the reported precision.

3. **The data audit and published release are independently verifiable:** GDPH 846→810 and SYSUCC 1559→1013 keep-lists reproduce; both cross-set pHash candidates are visually refuted; all 28 published checksum entries match. The live Zenodo ZIP is byte-identical to tag fa193a8 and has SHA-256 `bb45b685fa807fa14e83f01271f5256ab4cc933d5c43f53b176dd19b5b182c1e`. All 30 identity-rewrite mappings preserve trees, dates and messages.

The Zenodo timestamp remains post-hoc for external-v1, and its embedded provenance still names the superseded archive. The mounted v1/v2 internal self-checks pass; a clean environment build was not performed under the one-output-file constraint. The appropriate publishable conclusion is narrower: these fixed retrospective datasets show preserved discrimination of varying strength and poor specificity at the frozen operating point, with limited small-label adaptation reliability. Claims of an internally validated ensemble, resolved optimism, general clinical readiness, or fully enforced preregistration require correction or genuinely independent new studies.

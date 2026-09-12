# External validation protocol — frozen-v1

Written 2026-08-29, BEFORE any model inference on external images.
Research prototype — not for diagnostic use. The frozen model is
tag `frozen-v1` (5× cv_vit ckpts + hflip TTA + T=2.3644 + thr=0.2683);
nothing in this protocol may be changed after external scores are seen,
and the external run happens exactly ONCE.

## Task definition

The frozen task is benign-vs-malignant classification of a lesion-containing
ultrasound image (BUS-BRA contains no normal class). Images without a lesion
("normal") are therefore OUT OF SCOPE and excluded from both external sets.

## (a) BrEaST (data/raw/breast_poland)

- **Label source:** `Classification` column of
  `BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx` (one row per case;
  `Image_filename` gives the image, 256 unique).
- **Mapping:** benign → 0 (n=154), malignant → 1 (n=98).
- **Exclusions:** `Classification == "normal"` (n=4: cases 45, 61, 209, 213;
  all BIRADS 1, verification "not applicable") — no lesion, out of scope.
- **Images used:** the main `caseXXX.png` per case only. `*_tumor.png` and
  `*_otherN.png` are segmentation masks (see `Mask_tumor_filename` /
  `Mask_other_filename` columns) — never fed to the classifier.
- **Included set:** 252 images / 252 cases (prevalence 98/252 = 38.9%).
- **Patient unit:** one case = one image = one patient (no separate patient
  column exists; the dataset provides 256 single-image cases), so
  case-level bootstrap is identical to image-level.

## (b) BUSI (data/raw/busi)

- **Label source:** filename class prefix in `images/`
  (`benign_id_*.png`, `malignant_id_*.png`, `normal_id_*.png`).
- **Mapping:** benign → 0 (n=222 raw), malignant → 1 (n=164 raw).
- **Exclusions:** the `normal` class entirely (n=64), plus images dropped by
  the deduplication step below.
- **Dedup:** BUSI is known to contain duplicate/near-duplicate images. The
  frozen keep-list is `data/splits/busi_clean.csv`, produced by
  `src/dedup_busi.py` (perceptual hash; threshold and distance histogram
  reported there) BEFORE any scoring. Only keep-list images are evaluated.
- **Masks** in `masks/` are never fed to the classifier.
- **Patient unit:** BUSI publishes NO patient identifiers. Bootstrap is
  therefore IMAGE-LEVEL, and this limitation is stated wherever BUSI CIs are
  reported (per-patient correlation may make the CIs optimistically narrow).

## (c) Preprocessing — exactly the frozen validation path

Identical to BUS-BRA validation, no per-dataset adjustment of any kind:

1. `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` (BrEaST RGBA and BUSI grayscale
   PNGs both collapse to single-channel here, same as training data).
2. Replicate grayscale to 3 channels.
3. `get_transforms("val", 224)`: Resize 224×224 → Normalize (ImageNet
   mean/std) → tensor. No masks, no cropping, no intensity rescaling, no
   caliper/text removal.

Inference = frozen pipeline: mean sigmoid prob over 5 checkpoints ×
{original, hflip} (10 forward passes/image), temperature T from
`models/calibration.json` applied to logit of the averaged prob, decision
threshold from `models/operating_point.json`.

## (d) Metrics per dataset (each reported separately, never pooled)

- **AUC** on calibrated probabilities, with 95% percentile bootstrap CI,
  2000 iterations, seed 42: patient-level resampling where patient IDs
  exist (BrEaST: case-level ≡ image-level), image-level for BUSI (stated
  explicitly next to the number).
- **Sensitivity and specificity at the frozen threshold 0.2683** (calibrated
  prob), same bootstrap CIs.
- **Prevalence** of malignancy in the evaluated set, stated alongside PPV/NPV
  if quoted (BrEaST 38.9%; BUSI after dedup, from busi_clean.csv).
- Outputs: per-image predictions CSV, ROC + confusion matrix PNGs to
  reports/, one RESULTS.md section per dataset. Results are recorded
  regardless of outcome; no re-tuning, no second run.

---

# Amendment 1 — 2026-08-29 (before any external inference)

Verified before amending: reports/ contained no external prediction files
(no external_*_preds.csv, no external ROC/CM); external_selfcheck.txt is the
internal BUS-BRA fold-5 check only. Two candidate datasets were added
(data/raw/gdph_sysucc, data/raw/busi_whu); decisions below.

## (e) GDPH and SYSUCC — two additional cohorts (reported separately)

Source: data/raw/gdph_sysucc, folders GDPH/ (Guangdong Provincial People's
Hospital) and SYSUCC/ (Sun Yat-sen University Cancer Center). Always two
separate cohorts in all reporting; never pooled with each other or anything
else.

- **Label source:** filename prefix `benign(N).png` / `malignant(N).png`,
  verified to match the `ID` column of BIRADS&FOLD.xlsx 1:1 (2405 rows =
  2405 files, zero mismatches, zero cross-folder collisions; SYSUCC holds
  benign 0–442 + malignant 0–1115, GDPH benign 443–885 + malignant
  1116–1518). The xlsx `fold` column is the source paper's CV split and is
  ignored.
- **No patient IDs** exist → IMAGE-level bootstrap, stated wherever CIs are
  reported (same caveat as BUSI).
- **Dedup (required — heavy duplication found):** frozen keep-lists
  data/splits/gdph_clean.csv and data/splits/sysucc_clean.csv from
  src/phash_sweep.py, same rule as BUSI (64-bit pHash, d ≤ 8, keep
  lexicographically first per group, label-conflict groups dropped
  entirely):
  - GDPH: 846 → 810 (34 dups, 2 label-conflict)
  - SYSUCC: 1559 → 1013 (507 dups, 39 label-conflict — ~35% of the release;
    88 of 154 d=0 pairs are pixel-identical, and several exact duplicates
    appear under BOTH class labels)
  - Verification: d ≤ 8 within-set flags sample-confirmed as true
    duplicates (reports/phash_within_d8_sample.png).
- **Preprocessing/metrics:** unchanged from (c)/(d); RGB PNGs collapse via
  the same grayscale imread. Prevalence: GDPH 375/810 = 46.3%, SYSUCC
  724/1013 = 71.5% (cancer-center case mix — PPV/NPV caveat applies).

## (f) BUSI_WHU — EXCLUDED (label mapping unverifiable)

Pre-registered rule: verify the label mapping or exclude. Verification
attempted 2026-08-29, time-boxed, and FAILED:

- On-disk copy (data/raw/busi_whu) has no metadata/readme of any kind;
  filenames are bare `0NNNN.bmp` / `1NNNN-k.bmp`. The leading-digit
  hypothesis gives 756/171, contradicting the published counts of 560
  benign / 367 malignant (DSATNet paper, Med Phys 2025; Mendeley record
  k6cpmwybk3 confirms 927 total).
- Mendeley record (data.mendeley.com/datasets/k6cpmwybk3/1): description
  only, no per-file labels, no readme; file-listing API unavailable.
- DSATNet repo (github.com/Skylanding/DSATNet, util/dataset.py): the
  dataloader is segmentation-only — it pairs img/*.bmp with gt/*_anno.bmp
  and contains NO benign/malignant reading logic. It confirms our directory
  layout but cannot label it.
- Authors' HuggingFace re-release (huangjin520/busi-whu-seg) carries labels
  in filenames, but files are renumbered (benign_0001...) and re-split
  (70/15/15 vs our 60/20/20) — not mappable to our copy by name. (EMGANet
  README states the labels were added to filenames for that re-release.)

**Decision: EXCLUDED from external validation.** May be revisited only via
content-level matching against the labeled re-release, as a separate
pre-registered amendment BEFORE any scoring of it.

## (g) Full cross-set pHash sweep (contamination check)

src/phash_sweep.py, all pairs among BUS-BRA (1875), BrEaST (252), BUSI
keep-list (379), GDPH (846), SYSUCC (1559); hits at d ≤ 8 are CANDIDATES
requiring visual adjudication (reports/phash_sweep_hits.csv):

- BUS-BRA vs each external set: one candidate (bus_0999-l ↔ SYSUCC
  malignant(81), d=8) — visually REFUTED, different scans
  (reports/phash_cross_pairs.png). **No training contamination.**
- Between external sets: one candidate (GDPH benign(784) ↔ SYSUCC
  malignant(23), d=8) — visually REFUTED, different scans.
- All other set pairs: minimum distance ≥ 10, no candidates.

No cross-set exclusions required; keep-lists stand as generated.

## (h) Pre-registered SECONDARY analysis — BI-RADS reader comparison

On GDPH and SYSUCC (keep-list images only, per cohort), compare the frozen
model's operating point against each of the two BI-RADS reader columns in
BIRADS&FOLD.xlsx. Registered BEFORE any external inference; to be run only
AFTER the primary single-shot validation, using the primary run's saved
predictions (no second inference pass).

- Reader positive call: normalized BI-RADS ∈ {4a, 4b, 4c, 5}, i.e. ≥ 4a.
  Normalization: lowercase the letter suffix (4A → 4a); numeric 2/3/5 as-is.
- The single reader1 row with stray value 'c' is excluded from reader
  comparisons only (the image itself stays in the primary analysis).
- Report per cohort per reader: sens/spec of the reader vs pathology label,
  alongside model sens/spec at the frozen threshold on the same images.
  Descriptive comparison only — no threshold adjustment in response.

---

# Amendment 2 — Site-specific recalibration study (v2)

Written 2026-08-31, BEFORE any v2 computation. Verified at time of writing:
reports/ contains no v2_* files and RESULTS.md has no v2 section; the only
external artifacts are the external-v1 outputs (external_{breast,busi,gdph,
sysucc}_preds.csv with 252/379/810/1013 rows, matching (a)/(b)/(e)).

This is a strictly POST-HOC secondary study on external-v1's saved
predictions. It answers one question: how many locally labeled images (k)
would a new site need to recover specificity lost to domain shift, while
keeping sensitivity? Nothing here alters, re-runs, or reinterprets any
external-v1 number; the external-v1 tables in RESULTS.md stand as recorded.

## (i) Inputs and outputs

- **Inputs:** the saved prediction CSVs ONLY —
  reports/external_{breast,busi,gdph,sysucc}_preds.csv. No inference, no
  training, no model or checkpoint changes, no touching of raw images.
- **Logit recovery:** the CSVs store y_prob_raw and y_prob_calibrated but no
  logit column. The frozen pipeline defines calibration as
  p_cal = sigmoid(logit(p_raw)/T), so the pre-calibration ensemble logit is
  recovered deterministically as z = log(p_raw / (1 − p_raw)) from
  y_prob_raw. M2/M3 operate on z; no new forward passes.
- **Outputs:** files under reports/v2_* only, plus a single new RESULTS.md
  section titled "v2: Site-specific recalibration". External-v1 files and
  sections are never edited.

## (j) Design

- **Per cohort, never pooled** (BrEaST, BUSI, GDPH, SYSUCC — same keep-lists
  as external-v1, i.e. exactly the rows of each preds CSV).
- **k grid:** k ∈ {10, 20, 30, 50, 100, 200}, dropping any k ≥ n/2. With
  the known cohort sizes this fixes: BrEaST (n=252) and BUSI (n=379):
  k ∈ {10, 20, 30, 50, 100}; GDPH (n=810): k ∈ {10, 20, 30, 50, 100, 200};
  SYSUCC (n=1013): full grid.
- **R = 500 random draws per (cohort, k), seed 42** (one master seed; draw
  seeds derived deterministically from it).
- **Sampling at natural prevalence** — simple random sampling of the
  cohort's rows, NO stratification by label. BrEaST: sampling is
  PATIENT-level on the patient_id column (per (a), one case = one image =
  one patient, so this coincides with image-level — stated for the record).
  BUSI, GDPH, SYSUCC: image-level sampling, because no patient IDs exist;
  this is a LIMITATION stated wherever v2 results are reported (correlated
  images from one patient can appear split across the k-set and the
  held-out set, which flatters the learning curve).
- **Degenerate draws** (pre-registered handling): if a draw's k-set contains
  zero positives or zero negatives, threshold selection (M1, M2b, M3) is
  undefined → that draw falls back to the frozen threshold 0.2683 for the
  affected method, and the fraction of such draws is reported per
  (cohort, k). No draw is discarded or redrawn.

## (k) Methods

- **M1 (primary) — local threshold re-selection:** on the k local CALIBRATED
  probabilities (y_prob_calibrated as saved; frozen T unchanged), re-select
  the decision threshold with the frozen rule: highest threshold achieving
  sens ≥ 0.90 on the k images (same rule as pick_threshold.py).
- **M2 (secondary) — local temperature refit:** refit a single temperature
  T_local on the k local recovered logits z (NLL minimization, same
  procedure as calibrate.py), then apply (a) the FROZEN internal threshold
  0.2683 to the re-calibrated probs, and (b) the local sens ≥ 0.90 rule on
  the re-calibrated probs.
- **M3 (secondary) — Platt scaling:** fit logistic (a, b) on the k local
  logits z (p = sigmoid(a·z + b)), then apply the local sens ≥ 0.90 rule.

## (l) Evaluation

- Always on the n − k HELD-OUT images of the same cohort (the draw's
  complement); the k-set is never scored.
- **Metrics per (cohort, k, method, draw):** sensitivity and specificity on
  the held-out set, plus
  **recovery fraction = (spec_k − spec_frozen) / (spec_oracle − spec_frozen)**
  where, all evaluated on the SAME held-out n − k images:
  - spec_k: specificity of the method's threshold/calibration from the k-set;
  - spec_frozen: specificity of the frozen pipeline as-is (threshold 0.2683
    on y_prob_calibrated) — the external-v1 operating point;
  - spec_oracle: specificity of the threshold chosen by the frozen
    sens ≥ 0.90 rule using ALL n images of the cohort (in-sample oracle,
    computed once per cohort on calibrated probs).
- Summaries: median and IQR (and 5th–95th percentile band) over the R = 500
  draws, per (cohort, k, method).

## (m) Pre-registered decision metric

- **k\* = the smallest k with median recovery fraction ≥ 0.80 AND median
  sensitivity ≥ 0.85** (both on the held-out sets, per cohort, primary
  method M1). Reported per cohort; "not reached" is a valid outcome.
- ALL k values on the grid are reported for all methods — no post-hoc
  selection of k values, methods, or cohorts, regardless of outcome.
- This study produces NO change to the frozen model, calibration, or
  operating point. Any future deployment-style threshold adaptation would
  be a new study, not a revision of frozen-v1 or external-v1.

---

# Amendment 3 — Ensemble disagreement as an abstention signal (v2)

Written 2026-08-31, BEFORE any computation for this amendment. Verified at
time of writing: reports/ contains no v2_members_* files and no abstention
analysis exists anywhere in reports/, RESULTS.md, or src/; the working tree
is clean. This amendment is committed ALONE, before any code or output it
describes.

This is a v2 secondary study. It asks one question: does disagreement among
the 10 frozen ensemble members (5 checkpoints × {original, hflip}) predict
the frozen pipeline's errors on the external cohorts well enough to serve
as an abstention ("defer to human") signal? Nothing here alters, re-runs,
or reinterprets any external-v1 metric; the external-v1 tables in
RESULTS.md stand as recorded.

## (n) Authorized re-run — member-probability persistence ONLY

External-v1 saved only the ensemble-mean probabilities; the 10 per-member
probabilities per image were not persisted. A re-run of the frozen
inference pipeline is therefore authorized, tightly constrained:

- **Scope:** the frozen 5-checkpoint × hflip pipeline on the four external
  cohorts — BrEaST, BUSI, GDPH, SYSUCC — with EXACTLY the external-v1
  keep-lists (the rows of reports/external_{cohort}_preds.csv) and EXACTLY
  the frozen preprocessing of (c). No model, calibration, threshold, or
  transform change of any kind.
- **Sole purpose:** persist the 10 per-member sigmoid probabilities per
  image to reports/v2_members_{cohort}.csv (one row per image: image id,
  label, 10 member columns, recomputed mean).
- **Hard reproduction guarantee, asserted in code:** for EVERY image, the
  recomputed ensemble mean must reproduce the saved y_prob_raw column of
  the external-v1 preds CSV bitwise, or to within 1e-9 absolute. On ANY
  mismatch the script ABORTS, writes nothing further, and the failure is
  reported; no analysis proceeds until the discrepancy is explained.
- **No metrics are computed in the re-run script** — no AUC, no sens/spec,
  no error flags. It writes the member CSVs and the reproduction check
  result, nothing else. This keeps the re-run a pure persistence step, not
  a second evaluation.

## (o) Uncertainty signals (pre-registered)

Per image, from the 10 member probabilities p_1..p_10 (raw, pre-temperature):

- **U_std** = standard deviation of the 10 member probabilities (primary).
- **U_range** = max(p_i) − min(p_i).
- **Baseline M** (margin to threshold) = |p_calibrated − 0.2683|, computed
  from the SAVED y_prob_calibrated column — deliberately a signal that
  needs no member probabilities, so it costs nothing extra. LOW margin =
  high uncertainty; for comparability, the abstention/AUROC direction is
  −M (abstain on smallest margins).

No other signals; no signal combinations; no tuning of any signal on
external data.

## (p) Analyses — per cohort, never pooled

Target variable: **error** = misclassified at the frozen operating point
(prediction = [y_prob_calibrated ≥ 0.2683] vs label), exactly the
external-v1 decisions.

- **(a) Error-prediction AUROC** of each signal (U_std, U_range, −M)
  against the error indicator, per cohort. 95% percentile bootstrap CI,
  2000 iterations, seed 42, same resampling unit as (d): case-level for
  BrEaST, image-level for BUSI/GDPH/SYSUCC (limitation stated as usual).
- **(b) Abstention curves:** for each signal and each abstention fraction
  q ∈ {5, 10, 20, 30}%, abstain the top-q% most uncertain images of the
  cohort (ties broken by stable sort on image id). Report, per (cohort,
  signal, q): sensitivity and specificity on the RETAINED set at the
  frozen threshold, retained-set n and prevalence, and the **error
  enrichment ratio** = (error rate in the abstained set) / (error rate in
  the full cohort).
- **(c)** The same curves for the margin baseline −M — the comparison that
  decides whether member disagreement adds anything over what the single
  saved calibrated probability already provides.

Outputs: reports/v2_abstention_{metrics,curves}.csv +
reports/v2_abstention.png, and ONE new RESULTS.md section titled
"v2: Ensemble disagreement as an abstention signal". External-v1 and
earlier v2 files/sections are never edited.

## (q) Pre-registered success criterion

"Disagreement is a useful abstention signal" is claimed for a cohort ONLY
if ALL three hold:

1. error-AUROC(U_std) ≥ 0.65;
2. at q = 10%, retained-set specificity ≥ (cohort's frozen external-v1
   specificity + 0.05) AND retained-set sensitivity ≥ the cohort's frozen
   external-v1 sensitivity;
3. error-AUROC(U_std) > error-AUROC(−M) on that cohort.

All cohorts and all cells of (p) are reported regardless of outcome — no
post-hoc selection of cohorts, signals, or q values. Failure on any
criterion is recorded as "not useful on this cohort".

## (r) No internal OOF reference — stated limitation

An internal (BUS-BRA OOF) version of this analysis is NOT available: each
OOF image is scored by only its ONE held-out fold's checkpoint, so no
5-member (let alone 10-member) disagreement exists for internal data
without re-scoring OOF images with in-fold checkpoints, which would be
leakage. The abstention analysis is therefore external-only, with no
internal reference value for the error-AUROC; this is stated wherever
Amendment 3 results are reported.

---

# Amendment 4 — Multi-source training and domain-pretrained backbone (v2)

Written 2026-08-31, BEFORE any code or computation for this amendment.
Verified at time of writing: the working tree is clean; models/ and
reports/ contain no LOCO, USFM, or BiomedCLIP artifacts and no models/v2_*
files; RESULTS.md contains no multi-source or domain-pretraining section.
The only v2 artifacts are those of Amendments 2–3 (reports/v2_recalib_*,
v2_members_*, v2_abstention_*, v2_disagreement_*, v2_cross_site*).

This amendment authorizes TRAINING for the first time outside the
frozen-v1 line, and therefore supersedes rule (b)'s "external-only"
restriction FOR v2 MODELS ONLY, under the scope rules of (s). It asks two
questions:

- **Q1:** does multi-source training (adding external cohorts to the
  training pool) reduce the domain-shift specificity collapse observed in
  external-v1?
- **Q2:** does a domain-pretrained backbone (USFM) reduce that shift
  relative to the ImageNet-pretrained v1 backbone, under the v1 protocol
  unchanged?

## (s) Scope — external-v1 stands; v2 training authorization

- **external-v1 (tag `external-v1`) is FINAL and untouched.** No file,
  number, table, or section of frozen-v1 or external-v1 is altered,
  re-run, or reinterpreted by this amendment. The frozen-v1 model,
  calibration, threshold, and the demo app do not change.
- External cohorts (BrEaST, BUSI, GDPH, SYSUCC — always their frozen
  dedup keep-lists: the 252-image BrEaST set of (a),
  data/splits/{busi,gdph,sysucc}_clean.csv) MAY enter TRAINING for v2
  models only, as specified in (t).
- **LOCO guarantee:** in each leave-one-cohort-out run, the held-out
  cohort is never seen by that run in ANY form — not in training, not in
  validation, not in early stopping, calibration, or threshold selection.
  Its evaluation is SINGLE-SHOT: one scoring pass after the run's model,
  temperature, and threshold are fixed; results recorded regardless of
  outcome, no re-tuning, no second run. The cross-set pHash sweep of (g)
  (all set pairs clean at d ≤ 8 after visual adjudication) is the
  contamination check that makes training on three cohorts while holding
  out the fourth defensible.
- **Artifact isolation:** every v2-line-4 artifact lives in models/v2_*,
  reports/v2_*, and dedicated RESULTS.md sections
  ("v2: Domain-pretrained backbone (Q2)" and
  "v2: Multi-source LOCO training (Q1)"). External-v1 and earlier v2
  files/sections are never edited.

## (t) Q1 — leave-one-cohort-out (LOCO) multi-source training

- **Four runs**, holding out in turn: BrEaST, BUSI, GDPH, SYSUCC.
- **Training pool per run:** BUS-BRA official folds 1–4 (patient-level,
  per golden rule) + 85% of EACH of the other three external cohorts
  (keep-list rows).
- **Training-side validation per run:** BUS-BRA official fold 5
  (patient-level) + the remaining 15% of each training external cohort.
  The 15% splits are IMAGE-level (no patient IDs exist for BUSI, GDPH,
  SYSUCC; BrEaST case-level ≡ image-level per (a)), drawn once with
  seed 42 and reused across runs. LIMITATION, stated wherever Q1 results
  are reported: image-level external splits can place correlated images
  on both sides, flattering validation metrics; and the validation
  mixture's prevalence matches no single deployment site.
- **Validation roles (all decided on validation only, before the
  held-out cohort is touched):** early stopping (on validation AUC),
  temperature fit (NLL, same procedure as calibrate.py, on the pooled
  validation set), and threshold = frozen rule (highest threshold with
  sens ≥ 0.90 on pooled calibrated validation probs).
- **Model:** ONE model per run (no ensemble) + hflip TTA. Backbone per
  rule (v), decided by Q2's outcome before any Q1 training. All other
  hyperparameters identical to v1 (configs/baseline.yaml lineage: same
  augmentation, optimizer, schedule, epochs, img_size 224).
- **Held-out cohort metrics (single-shot, per run):** AUC on calibrated
  probs with 95% percentile bootstrap CI, 2000 iterations, seed 42, same
  resampling units as (d) (case-level BrEaST, image-level otherwise,
  limitation stated); sensitivity and specificity at that run's
  training-side threshold; median calibrated probability on benign
  images (the (d)/Amendment-2 shift marker).
- **Comparator — v1 fold-5 single model + hflip TTA,** computed from the
  SAVED reports/v2_members_{cohort}.csv only (no v1 re-run): columns
  m5_orig/m5_flip are the fold-5 checkpoint (models/cv_vit_fold5.pt;
  ordering fixed by inference.CKPT_FILES, verified at time of writing),
  so p_raw = (m5_orig + m5_flip)/2, p_cal = sigmoid(logit(p_raw)/2.3644),
  decisions at 0.2683. This matches Q1's single-model + TTA form and its
  BUS-BRA folds 1–4 training data. The v1 full 10-member ensemble
  (external-v1 as recorded) is reported alongside as reference only.
- **Pre-registered criterion — "multi-source training reduces shift"
  is claimed ONLY if:** ΔAUC ≥ +0.01 vs v1-single on ≥ 3/4 held-out
  cohorts, OR Δspec ≥ +0.10 (at the respective thresholds) on ≥ 3/4
  held-out cohorts with that run's held-out sensitivity ≥ 0.85.
  All four runs are reported regardless of outcome.

## (u) Q2 — domain-pretrained backbone (USFM), v1 protocol replicated

Replicate the v1 protocol EXACTLY with only the backbone replaced:

- **Pipeline:** BUS-BRA official 5-fold CV (patient-level folds,
  data/splits/*.csv), same augmentation/optimizer/schedule/epochs as v1,
  hflip TTA, ONE temperature fit on pooled OOF TTA probs, threshold =
  sens ≥ 0.90 rule on calibrated pooled OOF, then FREEZE
  (models/v2_usfm_fold{1-5}.pt + v2_usfm_calibration.json +
  v2_usfm_operating_point.json), then exactly ONE evaluation on the four
  external cohorts with the frozen keep-lists and preprocessing (c).
  Single-shot; recorded regardless of outcome.
- **Backbone:** USFM (openmedlab/USFM) pretrained weights loaded into a
  ViT-B/16 skeleton (224 input, matching v1's vit_base_patch16_224
  geometry). Loading is verified by a stated weight-coverage check
  (fraction of backbone tensors loaded from the USFM checkpoint) before
  training.
- **Pre-registered fallback:** if USFM cannot be loaded cleanly within
  one working session, substitute the BiomedCLIP ViT-B/16 image encoder
  (open_clip), and STATE the substitution in the protocol and RESULTS.md
  BEFORE any training with it. No third option.
- **Pre-registered criterion — "domain pretraining reduces shift" is
  claimed ONLY if:** external AUC ≥ v1 (full-ensemble external-v1) AUC
  + 0.01 on ≥ 3/4 cohorts, OR the benign median calibrated-probability
  shift shrinks by ≥ 30% on ≥ 3/4 cohorts, where per cohort
  shift = median p_cal(benign, external cohort) − median p_cal(benign,
  own pooled OOF), v1's value computed from the saved
  reports/oof_vit_preds.csv + external_*_preds.csv, and "shrinks ≥ 30%"
  means |shift_v2| ≤ 0.70 · |shift_v1|.

## (v) Order and backbone rule — fixed now

Q2 runs FIRST. Then: **Q1's backbone = USFM if Q2's criterion in (u) is
met, else the v1 ImageNet vit_base_patch16_224** (same backbone as v1,
giving the cleanest multi-source-vs-v1 comparison). This decision rule is
fixed before any training; whichever branch fires, it is recorded in
RESULTS.md with the Q2 verdict that triggered it. No other backbone may
be introduced under this amendment.

## Amendment 4 — substitution note (2026-09-01, declared pre-training)

Executed per the pre-registered fallback in (u). The USFM release
(openmedlab/USFM, USFM_latest.pth) is a BEiT-style ViT-B/16: it contains
no absolute position embedding — positional information lives solely in a
shared relative-position-bias table added inside every attention block —
plus per-block LayerScale parameters, none of which have any
representation in the v1 vit_base_patch16_224 skeleton. Coverage report
(src/v2_load_usfm.py): pos_embed UNFILLED, 27/188 checkpoint tensors
UNUSED (the entire positional mechanism + LayerScale). Loading is
therefore NOT CLEAN under (u)'s coverage check.

Accordingly, **Q2's backbone is the BiomedCLIP ViT-B/16 image encoder**
(open_clip, hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224),
whose vision tower is exactly a timm vit_base_patch16_224. Everything
else in (u) is unchanged; Q2 artifact names become
models/v2_biomedclip_fold{1-5}.pt +
v2_biomedclip_{calibration,operating_point}.json, and rule (v) reads
"BiomedCLIP" wherever it says "USFM". No third option remains. Declared
BEFORE any v2 training (no training run, smoke or otherwise, has been
launched at the time of this note).

---

# Post-hoc notes on protocol compliance (appended 2026-09-12; append-only, no line above edited)

Recorded after the two independent re-audits of 2026-09-11
(docs/AUDIT2_claude_2026-09-11.md, docs/AUDIT2_astra_2026-09-11.md;
response in docs/AUDIT2_RESPONSE.md). None of these notes changes any
registered rule, criterion, result or artifact; each states a discrepancy
between the text above and what was implemented, enforced or distributed.

1. **"Highest threshold achieving sens ≥ 0.90" — base protocol, (k), (t),
   pick_threshold.py.** As implemented, the candidate thresholds are the
   operating points returned by sklearn.metrics.roc_curve with its default
   drop_intermediate=True, which drops collinear ROC points; the selected
   threshold is the highest sens ≥ 0.90 point among those. Exhaustively,
   the highest qualifying score on the pooled OOF is 0.26878 (one image
   different, identical specificity) vs the frozen 0.26832, which is
   retained as frozen. On the four full-cohort oracles the two rules differ
   on BrEaST (0.37553 vs 0.37983, one image) and SYSUCC (0.33136 vs
   0.33504, four images) and coincide on BUSI/GDPH. Re-running the 11,000
   M1 draws with the exhaustive rule changes 2,432 selected thresholds
   (259/372/604/1,197 per cohort, none at k=10) with k* unchanged
   (20/20/10/10). Wherever "the frozen rule" is cited in results it means
   the routine as implemented (src/posthoc_threshold_rule.py; RESULTS.md
   POST-HOC 6).
2. **(l) summaries.** Registered: "median and IQR (and 5th–95th percentile
   band)". Implemented: median and 2.5th/97.5th percentiles (a 95% band,
   wider than registered), no IQR column. Presentational deviation; no
   criterion depends on the band.
3. **(k) M3.** Platt scaling is order-preserving only for a positive slope;
   the unconstrained fits produced negative slopes on 12/4/3/16 draws
   (BrEaST/BUSI/GDPH/SYSUCC) at k=10 and 1 (SYSUCC) at k=20, 0 at k ≥ 30.
   The "structural equivalence" of M2b/M3 with M1 stated in RESULTS.md holds
   for positive-slope transforms only.
4. **(k) M2 fallback.** The non-positive-T fallback to the frozen pipeline
   (fit_failed) is a run-time addition not registered above (already
   declared in RESULTS.md; 7–20% of k=10 draws).
5. **(t) LOCO.** Early-stopping patience 7 was not specified above (declared
   in RESULTS.md and the report). The 85/15 external slices are image-level
   as stated in (t); they are not persisted in any artifact and are
   reproducible only by re-running the seeded code (src/v2_data.py). The
   hold-out-GDPH training was launched twice at the same commit (wandb
   k896krp6 aborted at epoch 3; 3jn4cicy 19 min later produced the
   checkpoint); the held-out cohort is untouched during --train, so this is
   a restart, not a peek, but it was unrecorded until 2026-09-12.
6. **(d) and (p) presentation.** "One RESULTS.md section per dataset" was
   implemented as one section with a four-row table; the single registered
   Amendment 3 figure was split into two PNGs (already declared).
7. **(h) stray value.** The stray 'c' sits in the reader2 column (SYSUCC
   benign(274)), not reader1 as written above; the image is dedup-dropped,
   so the exclusion is vacuous (already recorded in RESULTS.md).
8. **(g) adjudication figure.** reports/phash_cross_pairs.png, cited above
   as the evidence, is no longer distributed (P4c, 2026-09-07: it embeds
   GDPH/SYSUCC images released without an explicit licence); the
   distributed evidence is reports/phash_cross_pairs_table.csv +
   reports/phash_cross_pairs_busbra_thumb.png. One re-auditor repeated the
   visual adjudication from the raw images with the same conclusion.
   "No training contamination" above means: no visually confirmed
   cross-set near-duplicate at pHash d ≤ 8 — a finite screen that cannot
   exclude same-patient re-scans or crops at d > 8. "35% duplicates"
   (SYSUCC) counts 507 duplicate images plus 39 label-conflict images.
9. **(d) single run — enforcement.** src/external_val.py's --confirm was an
   intent flag only: it authorized execution and never refused an existing
   output, so a repeated command would have overwritten the CSVs in place.
   The four v1 prediction CSVs have one adding commit each and were never
   modified (git history), which is the evidence for single-shot. A refusal
   of existing outputs (--overwrite required) was added on 2026-09-12 as
   post-hoc enforcement; it did not exist at the time of the run.
10. **(u) "patient-level folds, data/splits/*.csv".** When Amendment 4 was
    written the BUS-BRA folds were not in data/splits (they were in
    git-ignored data/raw); the official file was committed there on
    2026-09-07 (data/splits/busbra_official_5fold.csv, sha256-verified).
11. **(r) and the internal reference.** The pooled OOF used for T, the
    threshold and every "internal" reference number is ONE held-out
    checkpoint per image + hflip TTA, not the deployed ensemble (as (r)
    states); T and the threshold were therefore fitted on one predictor
    and applied to another externally. RESULTS.md POST-HOC 5 separates the
    predictor-change and cohort-change components of the internal→external
    shift (cohort component ≫ predictor component).
12. **(u) v2 BiomedCLIP operating point.** The frozen threshold
    0.20070531964302063 applied to the committed
    reports/v2_biomedclip_oof_preds.csv (float32) gives 546/358/61/910,
    sens 0.8995; the freeze script picked it on in-memory float64
    probabilities where the confusion is 547/358/60/910, sens 0.9012. One
    image at the float32 round-trip boundary; the JSON and the NOT CLAIMED
    verdict are unchanged.
13. **Amendment 1 registration.** Amendment 1 was committed together with
    the keep-lists and pHash artifacts it registers (436ab56, 10 files) —
    after the data audit it describes, before any model metric on
    GDPH/SYSUCC. Amendments 2/3/4 were each committed alone before the
    computation they govern.

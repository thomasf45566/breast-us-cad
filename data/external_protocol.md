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

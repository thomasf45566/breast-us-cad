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

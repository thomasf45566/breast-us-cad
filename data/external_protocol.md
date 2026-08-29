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

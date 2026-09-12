# Results Log

| Date | Run | Model | Train/Val | AUC | Sens | Spec | Thr | Ckpt | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-26 | baseline_effb0_fold5 | effnet_b0 | folds 1-4 / 5 | 0.8823 | 0.769 | 0.840 | 0.505 (Youden) | baseline_effb0_fold5.pt | first full run, 30 ep, wandb t41mf6ys, tag baseline-v1 |
| 2026-08-27 | cv_effb0 (5-fold) | effnet_b0 | official 5-fold CV | 0.891 ± 0.027 | 0.769 ± 0.080 | 0.864 ± 0.064 | per-fold Youden | cv_effb0_fold{1-5}.pt | 30 ep/fold, wandb group cv_effb0, details in reports/cv_summary.csv |
| 2026-08-27 | convnext_small_fold5 | convnext_small | folds 1-4 / 5 | 0.9257 | 0.876 | 0.870 | 0.307 (Youden) | convnext_small_fold5.pt | screening, 30 ep, lr 1e-4, best ep 22 |
| 2026-08-27 | vit_b16_fold5 | vit_base_patch16_224 | folds 1-4 / 5 | 0.9293 | 0.884 | 0.847 | 0.788 (Youden) | vit_b16_fold5.pt | screening, 30 ep, lr 1e-4, warmup 3, best ep 17 |
| 2026-08-28 | cv_convnext (5-fold) | convnext_small | official 5-fold CV | 0.930 ± 0.017 | 0.852 ± 0.028 | 0.886 ± 0.047 | per-fold Youden | cv_convnext_fold{1-5}.pt | 30 ep/fold, lr 1e-4, wandb group cv_convnext, per-fold AUC 0.954/0.940/0.922/0.909/0.926, details in reports/cv_convnext_summary.csv |
| 2026-08-28 | cv_vit (5-fold) | vit_base_patch16_224 | official 5-fold CV | 0.931 ± 0.016 | 0.881 ± 0.025 | 0.868 ± 0.047 | per-fold Youden | cv_vit_fold{1-5}.pt | 30 ep/fold, lr 1e-4, warmup 3, wandb group cv_vit, per-fold AUC 0.953/0.941/0.919/0.913/0.927, details in reports/cv_vit_summary.csv |
| 2026-08-29 | cv_vit_cdrop (5-fold) | vit_base_patch16_224 | official 5-fold CV | 0.933 ± 0.020 | 0.853 ± 0.025 | 0.874 ± 0.036 | per-fold Youden | cv_vit_cdrop_fold{1-5}.pt | pre-registered CoarseDropout experiment (configs/vit_cdrop.yaml), per-fold AUC 0.956/0.943/0.930/0.902/0.934, DISCARDED per adoption rule (below), kept as audit trail |
| 2026-08-29 | tta_vit (hflip TTA) | vit_base_patch16_224 | official 5-fold CV, eval-only | 0.932 ± 0.016 | 0.844 ± 0.020 | 0.895 ± 0.024 | per-fold Youden | cv_vit_fold{1-5}.pt (unchanged) | no retraining; sigmoid probs averaged over original + hflip; src/tta_eval.py, details in reports/tta_vit_summary.csv, TTA section below |

## Backbone comparison (5-fold CV, BUS-BRA official folds)

| | effnet_b0 | convnext_small | vit_b16 |
|---|---|---|---|
| Fold 1 AUC | 0.9388 | 0.9541 | 0.9526 |
| Fold 2 AUC | 0.8711 | 0.9404 | 0.9411 |
| Fold 3 AUC | 0.8817 | 0.9221 | 0.9193 |
| Fold 4 AUC | 0.8839 | 0.9093 | 0.9132 |
| Fold 5 AUC | 0.8794 | 0.9263 | 0.9272 |
| **AUC mean ± sd** | 0.891 ± 0.027 | 0.9304 ± 0.0173 | 0.9307 ± 0.0161 |
| Per-fold AUC range | 0.8711–0.9388 | 0.9093–0.9541 | 0.9132–0.9526 |
| Sens mean ± sd | 0.769 ± 0.080 | 0.852 ± 0.028 | 0.881 ± 0.025 |
| Spec mean ± sd | 0.864 ± 0.064 | 0.886 ± 0.047 | 0.868 ± 0.047 |
| Params (M) | 4.0 | 49.5 | 85.8 |
| Ckpt size | 16 MB | 189 MB | 327 MB |
| Epoch time (MPS) | fastest | ~145 s | ~180 s |
| CPU inference (1×224×224, median of 20) | 79 ms | 376 ms | 40 ms |

convnext_small and vit_b16 are statistically indistinguishable on AUC (Δmean
0.0002, well below both models' fold SD); both clearly beat effb0. Per-fold
Youden thresholds are unstable (e.g. 0.015–0.92 across convnext folds) — do not
interpret sens/spec too literally before calibration + pooled-OOF threshold.
ViT is ~9× faster than ConvNeXt on CPU despite more params: its dense matmuls
vectorize well, while 7×7 depthwise convs are slow in PyTorch's CPU path.
Winner decision: USER (per plan.md).

## Interpretability audit (fold-5 CV ckpts, 2026-08-28)

Grids: reports/gradcam_check{,2}_*.png · raw-image audit: reports/artifact_audit_raw.png
· probe: reports/occlusion_border15.csv · script: src/backbone_diagnostic.py

- Both models show caliper-adjacent heat on malignant TPs (ConvNeXt most
  visibly); calipers bracket lesions in both classes, so they are a spatial
  shortcut rather than a class-discriminative cue. Update (adoption eval,
  2026-08-29): at blocks[-2].norm1 the plain ViT localizes all 4 saturated
  malignant TPs well — caliper-adjacent heat is a localized observation on 3
  moderate-confidence cases, and external validation is the definitive test.
- ViT attention rollout concentrates on the bottom burned-in-text band, but
  rollout is class-agnostic — treat as secondary evidence only.
- Border-occlusion probe (outer 15%, image-mean fill): median prob drop < 0.01
  for both models, with a heavy tail (~16% of images with drop > 0.2) for both
  — confounded by lesions extending to the border.

Method note: ViT's blank TP CAMs in check 1 were a method artifact (saturated
last-block target), confirmed resolved at blocks[-2].norm1 by the adoption
eval grid (reports/gradcam_adoption_vit_cdrop.png).

## Robustness experiment: CoarseDropout ViT (pre-registered, 2026-08-29)

Adoption rule frozen before results (plan.md): adopt only if pooled OOF AUC
>= (plain ViT pooled OOF - 0.01) AND visibly reduced caliper-adjacent heat on
malignant TPs. Script: src/adoption_eval.py · grid:
reports/gradcam_adoption_vit_cdrop.png

| | plain cv_vit | cv_vit_cdrop |
|---|---|---|
| Pooled OOF AUC (1875 imgs) | 0.9231 | 0.9139 |
| Mean fold AUC ± sd | 0.9307 ± 0.0161 | 0.9329 ± 0.0199 |
| Gate 1 (>= 0.9131) | — | PASS (margin 0.0008) |
| Gate 2 (visibly reduced caliper heat) | — | FAIL (1/3 improved, 1 unchanged, 1 improved map but flipped a malignant 0.75→0.28) |

**Verdict: KEEP plain cv_vit; cdrop discarded per the pre-registered AND
rule, no iteration.** Gate 1 passed by only 0.0008 and the pooled ordering
actually reversed the mean-of-folds ordering.

Metric note: pooled OOF AUC (0.9231) sits below mean-of-folds (0.9307)
because pooling concatenates five uncalibrated per-fold probability scales.
The headline internal metric is mean-of-folds 0.9307 ± 0.0161; pooled OOF is
the operational basis for calibration and thresholding.

## TTA — horizontal flip only (eval-time, 2026-08-29)

Sigmoid probabilities averaged over the original and horizontally flipped
image; no retraining, checkpoints unchanged (models/cv_vit_fold{1-5}.pt).
Vertical flip deliberately excluded (would break ultrasound depth
orientation). Script: src/tta_eval.py; the same TTA is available as
`--tta` on src/evaluate.py. Per-fold plain AUCs exactly reproduce the
cv_vit row above (pipeline sanity check).

| Held-out fold | n | AUC plain | AUC + hflip TTA |
|---|---|---|---|
| 1 | 376 | 0.9526 | 0.9571 |
| 2 | 385 | 0.9411 | 0.9397 |
| 3 | 366 | 0.9193 | 0.9238 |
| 4 | 365 | 0.9132 | 0.9174 |
| 5 | 383 | 0.9272 | 0.9234 |
| **Mean ± sd** | — | 0.9307 ± 0.0161 | 0.9323 ± 0.0162 |
| **Pooled OOF** | 1875 | 0.9231 | 0.9254 |

TTA improves 3/5 folds and the pooled OOF AUC (+0.0023) at 2× inference
cost. Pooled OOF predictions for both settings are saved to
reports/oof_vit_preds.csv (input to the calibration step); pooled ROC +
confusion matrices in reports/{roc,cm}_oof_vit_{plain,tta}.png.
**Decision (USER, 2026-08-29): TTA ADOPTED** — hflip TTA + 5-model ensemble
is the frozen inference pipeline for all downstream steps (plan.md standing
decisions).

## Calibration — temperature scaling (pooled OOF, hflip TTA, 2026-08-29)

Single scalar T fit by LBFGS on NLL over the 1875 pooled OOF TTA
probabilities (reports/oof_vit_preds.csv, y_prob_tta). Logit convention:
logit of the TTA-averaged prob, log(p/(1-p)) with p clipped to
[1e-6, 1-1e-6] — not the average of per-flip logits. Script:
src/calibrate.py · params: models/calibration.json · diagram:
reports/reliability_oof_vit_tta.png

| | Before | After (T = 2.3644) |
|---|---|---|
| ECE (15 equal-width bins) | 0.0716 | 0.0401 |
| NLL | 0.4411 | 0.3237 |
| AUC | 0.9254 | 0.9254 (unchanged — monotone, asserted) |

T ≈ 2.36 means the ensemble's raw probabilities were substantially
overconfident. Residual miscalibration after scaling is mild
overconfidence in the mid-probability bins (see diagram); a single
temperature cannot fix bin-shape effects, which is acceptable for the
sens ≥ 0.90 thresholding step that follows.

## Operating point — sens ≥ 0.90, max spec (calibrated OOF, 2026-08-29)

Chosen on the calibrated pooled OOF probabilities (n = 1875 images, 1064
patients). Script: src/pick_threshold.py · params:
models/operating_point.json · slide asset: reports/roc_oof_operating_point.png
· 95% CIs: patient-level bootstrap (resample patient_ids, 2000 iterations,
seed 42).

| | Value | 95% CI |
|---|---|---|
| Threshold (calibrated prob) | 0.2683 | — |
| Sensitivity | 0.9028 | 0.874–0.928 |
| Specificity | 0.7713 | 0.745–0.798 |
| PPV | 0.6539 | — |
| NPV | 0.9431 | — |
| Confusion (tp/fp/fn/tn) | 548 / 290 / 59 / 978 | — |

PPV reflects BUS-BRA's ~32% malignancy prevalence; at screening prevalence
it would be far lower — a talking point, not a deployment claim.

## FROZEN MODEL (frozen-v1, 2026-08-29)

No model, calibration, or threshold changes after this tag. External
validation runs against exactly this artifact set, once.

- **Checkpoints:** models/cv_vit_fold{1-5}.pt — vit_base_patch16_224,
  one per official BUS-BRA CV fold (30 ep, lr 1e-4, warmup 3, seed 42)
- **Inference:** 5-model ensemble + hflip TTA — mean of the 10 sigmoid
  probs (5 ckpts × {original, hflipped}); vertical flip excluded
- **Calibration:** models/calibration.json — temperature T = 2.3644 applied
  to the logit of the ensemble-averaged prob (p clipped to [1e-6, 1-1e-6])
- **Operating point:** models/operating_point.json — threshold 0.2683 on
  calibrated prob (rule: sens ≥ 0.90 with max spec on pooled OOF)
- **Final internal metrics (BUS-BRA):** mean-of-folds AUC 0.9307 ± 0.0161
  (headline) · pooled OOF AUC 0.9254 with TTA · ECE (15 bins) 0.0401 after
  calibration · sens 0.9028 / spec 0.7713 at the frozen threshold

Note: per-fold CV numbers above use each fold's single checkpoint; the
frozen ensemble itself has no unbiased internal estimate (every image is
in-fold for 4 of 5 members) — its honest test is the external validation.

## External validation (single-shot, frozen-v1)

Run 2026-08-29 per data/external_protocol.md (incl. Amendment 1), tag
external-v1. ONE run of `python src/external_val.py --dataset all --confirm`
against the frozen artifact set (5× cv_vit ckpts + hflip TTA + T=2.3644 +
threshold 0.2683). No adjustments of any kind; cohorts never pooled.
Bootstrap: 2000 iterations, seed 42 — patient-level for BrEaST (1 case =
1 image = 1 patient, so case-level ≡ image-level); IMAGE-level for BUSI,
GDPH, SYSUCC (no patient IDs published — CIs may be optimistically narrow).

| Cohort | n images | Prevalence | AUC (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) |
|---|---|---|---|---|---|
| BrEaST | 252 | 0.389 | 0.8542 (0.8019–0.9024) | 0.9184 (0.8605–0.9678) | 0.4091 (0.3333–0.4897) |
| BUSI (clean) | 379 | 0.430 | 0.9339 (0.9062–0.9580) | 0.9571 (0.9250–0.9868) | 0.6296 (0.5603–0.6927) |
| GDPH | 810 | 0.463 | 0.9154 (0.8949–0.9343) | 0.9707 (0.9529–0.9866) | 0.4529 (0.4060–0.5000) |
| SYSUCC | 1013 | 0.715 | 0.8380 (0.8098–0.8655) | 0.9309 (0.9119–0.9489) | 0.4740 (0.4169–0.5318) |

Internal reference (BUS-BRA pooled OOF): AUC 0.9254, sens 0.9028 /
spec 0.7713 at the same frozen threshold.

Reading: discrimination transfers reasonably (AUC 0.84–0.93; BUSI and GDPH
within or near the internal range, BrEaST and SYSUCC lower). The operating
point does NOT transfer: sensitivity stays ≥ 0.90 on all four cohorts, but
specificity collapses from 0.77 internally to 0.41–0.63 externally — the
calibrated probabilities shift upward under domain shift, so the frozen
threshold over-calls malignancy. PPV/NPV per cohort in the run log /
per-image CSVs; SYSUCC is a cancer-center case mix (prevalence 0.715), so
its PPV 0.816 / NPV 0.733 are not comparable to the other cohorts.
Confusions (tp/fp/fn/tn): BrEaST 90/91/8/63 · BUSI 156/80/7/136 ·
GDPH 364/238/11/197 · SYSUCC 674/152/50/137.

Artifacts per cohort: reports/external_{breast,busi,gdph,sysucc}_preds.csv
(per-image y_true, raw + calibrated prob, decision),
reports/roc_external_*.png, reports/cm_external_*.png.

Recorded as-is per protocol: no re-tuning, no threshold change, no second
run. Next pre-registered step: BI-RADS reader comparison (protocol §h) on
GDPH/SYSUCC from these saved predictions.

## Probability shift under domain shift (descriptive, post-hoc)

Descriptive analysis of the SAVED single-shot predictions only — no model,
no inference, no threshold changes. Script: src/plot_prob_shift.py · slide
asset: reports/prob_shift_external.png (calibrated-probability histograms by
class, internal vs each external cohort, frozen threshold dashed).

| Cohort | median p (benign) | median p (malignant) | frac benign ≥ thr (=1−spec) |
|---|---|---|---|
| Internal (OOF) | 0.0625 | 0.8533 | 0.2287 |
| BrEaST | 0.3061 | 0.7561 | 0.5909 |
| BUSI | 0.1739 | 0.8309 | 0.3704 |
| GDPH | 0.2894 | 0.7698 | 0.5471 |
| SYSUCC | 0.2869 | 0.6804 | 0.5260 |

The specificity collapse is a benign-distribution shift: median benign
calibrated prob rises from 0.06 internally to 0.17–0.31 externally,
pushing 37–59% of benign images over the frozen 0.2683 threshold, while
malignant medians stay high (0.68–0.83) — hence sensitivity holds. The
frozen calibration/threshold encode BUS-BRA's benign appearance; external
benigns look "more suspicious" to the model. Descriptive observation only;
per protocol the threshold stays frozen.

## Pre-registered secondary analysis: model vs radiologist BI-RADS

Protocol §h, run 2026-08-29 from the SAVED single-shot predictions only
(no new inference). Script: src/birads_comparison.py · figures:
reports/birads_comparison_{gdph,sysucc}.png. Reader positive call:
normalized BI-RADS ≥ 4a (4A→4a case-fold). The pre-registered stray-'c'
exclusion turned out vacuous on the analyzed images: the affected row
(SYSUCC benign(274); the stray value sits in the reader2 column, not
reader1 as the protocol text guessed) was already removed by the dedup
keep-list, so n is unchanged. Reader agreement is on the binarized
(≥ 4a) calls. Descriptive only; the frozen threshold is unchanged.

### GDPH (n = 810)

| Rater | Sensitivity | Specificity |
|---|---|---|
| model (thr 0.2683) | 0.9707 | 0.4529 |
| BIRADS-reader1 | 0.9760 | 0.8943 |
| BIRADS-reader2 | 0.9787 | 0.5126 |

Reader1 vs reader2 agreement 0.7593, Cohen's κ 0.5149. On GDPH the model
matches both readers' sensitivity but reader1 achieves far higher
specificity (0.89 vs the model's 0.45); reader2 sits close to the model's
ROC curve.

### SYSUCC (n = 1013)

| Rater | Sensitivity | Specificity |
|---|---|---|
| model (thr 0.2683) | 0.9309 | 0.4740 |
| BIRADS-reader1 | 0.9130 | 0.6505 |
| BIRADS-reader2 | 0.9931 | 0.1315 |

Reader1 vs reader2 agreement 0.7887, Cohen's κ 0.2152. On SYSUCC the model
operating point lies between the two readers, who themselves diverge widely
(reader2 calls 87% of benigns ≥ 4a; κ 0.22).

## Grad-CAM gallery (frozen-v1, 2026-08-29)

Script: src/explain.py · figures: reports/gradcam_gallery_internal.png,
reports/gradcam_external_fp.png. CAMs use the fold-5 checkpoint only
(Grad-CAM at blocks[-2].norm1, the frozen saliency method), targeting the
predicted class — single-model visualizations, while decisions come from
the 5-ckpt hflip-TTA ensemble. Internal picks are drawn from the pooled
OOF calibrated preds at the frozen threshold but restricted to fold-5
images, so the CAM checkpoint is held-out for everything shown; within
each cell, calibrated probs in 0.3–0.9 are preferred over saturated ones.
External FPs are the 4 highest-calibrated-prob benign false positives
from SYSUCC and GDPH each.

Where the heat lands (qualitative, single-model):

- **TP:** heat sits on the lesion body and its margins in all 4; the
  moderate-confidence TPs (p 0.49–0.63) are the tightest, a compact blob
  on the hypoechoic mass with little spill.
- **TN / FN** (benign-target CAMs): diffuse and non-lesional — heat
  spreads over superficial parenchyma bands, posterior/deep regions, and
  in two TNs partly over burned-in annotation text. "Benign evidence" is
  apparently the absence of a suspicious focus rather than a localized
  structure, so these maps are the least informative of the four cells.
- **FP:** 3/4 fixate on genuinely suspicious-looking structure (irregular
  hypoechoic regions with posterior shadowing); the lowest-prob FP
  (bus_0625-l, p 0.35) instead lights a small echogenic focus adjacent to
  a caliper mark — residual annotation sensitivity consistent with the
  earlier interpretability audit.
- **External FPs (domain shift):** heat lands squarely on the lesion body
  in 7/8 — all 4 SYSUCC cases are markedly hypoechoic, lobulated benign
  masses the model reads as malignant on appearance, not artifact. GDPH
  is messier: 2/4 show heat spilling across broad superficial bands, and
  benign(755) adds heat over a column of reverberation-like bright dots.
  Net: the external specificity collapse looks appearance-driven
  (atypical benign morphology + acquisition style), not text/caliper
  driven.

Caveat: CAMs are qualitative, single-checkpoint, original-orientation
only; they support the domain-shift reading but do not quantify it.

## Segmentation — U-Net effb0 (demo feature, 2026-08-30)

Script: src/train_seg.py, src/eval_seg.py · figure:
reports/seg_examples.png · per-image metrics:
reports/seg_metrics_seg_unet_effb0.csv · wandb run seg_unet_effb0
(ybo2j891) · checkpoint models/seg_unet_effb0.pt.

Deliberately minimal scope (bundled demo feature for the app): ONE
model, ONE training run, no CV, no ensemble, no external evaluation.
smp U-Net with efficientnet-b0 encoder (imagenet init), input 224,
Dice + BCE loss (equal weight), AdamW lr 3e-4, cosine, 40 epochs,
batch 16, mps. Trained on BUS-BRA official folds 1-4 (1492 images,
all with masks), validated on fold 5 (383 images); best checkpoint
by mean per-image val Dice at threshold 0.5 (epoch 37).

| Split | n | Mean Dice | Median Dice | Mean IoU | Median IoU |
|---|---|---|---|---|---|
| BUS-BRA fold 5 | 383 | **0.9016** | 0.9325 | **0.8326** | 0.8736 |

Distribution: 280/383 images (73%) above Dice 0.9; only 5/383 below
Dice 0.5. The example grid (4 good / 4 median / 4 worst) shows the
failure modes are the expected ones: heavily shadowed lesions, low
contrast fields where the model fragments the mask or grabs a
different hypoechoic region, and one case where a large ill-defined
GT lesion is only partially covered. Median cases (~Dice 0.93) are
already visually tight.

Metric note: Dice/IoU computed per image at 224×224 (masks resized
with nearest interpolation), thresholding sigmoid probs at 0.5;
mean is over images, not pixels pooled.

## Gradio demo app (manual Step 11, 2026-08-30)

Script: app/app.py (`python app/app.py`, CPU) · examples bundled in
app/examples/.

Frozen-v1 inference verbatim: 5x cv_vit_fold{1-5} ckpts x {orig, hflip}
sigmoid probs averaged, T and threshold read from models/calibration.json
and models/operating_point.json (nothing hardcoded). Three panels per
image: U-Net (seg_unet_effb0) lesion contour on the original, fold-5
Grad-CAM @ blocks[-2].norm1 (predicted-class target, single-model caveat
shown in the card), and a result card with the calibrated probability
bar + operating-point call ("threshold set for sensitivity >= 0.90 on
internal validation"). Fixed banner top and bottom: research prototype,
not a medical device. About section covers data, external validation,
and the domain-shift specificity caveat.

End-to-end CPU latency (ensemble+TTA+seg+CAM, 1 image, warm, M4):
**0.53 s** (target < 3 s), printed at startup.

Bundled examples (BUS-BRA fold 5, moderate difficulty by OOF calibrated
prob, all classified correctly by the app): benign bus_0186-r (16.6%),
benign bus_0223-l (22.9%), malignant bus_0328-r (69.1%), malignant
bus_0663-l (69.9%). App probabilities are legitimately higher/lower than
the OOF picks' (~0.46 mal / ~0.24 ben): OOF used only the held-out fold-5
ckpt, while the app's 5-ckpt ensemble includes four members trained on
fold 5 — expected, and why example difficulty was chosen on OOF.

## Hugging Face Spaces deployment (Step 12, 2026-08-30)

Live demo: **https://huggingface.co/spaces/happytommy/breast-us-cad**
(public Gradio Space, cpu-basic) · weights:
https://huggingface.co/happytommy/breast-us-cad-weights (5x cv_vit fold
ckpts, seg_unet_effb0.pt, calibration.json, operating_point.json).

Structure: src/inference.py is the complete frozen-v1 inference path
(no wandb / albumentations / training imports); app/app.py and
src/external_val.py both import it. deploy/ bundles app.py +
inference.py + the 4 examples and is the Space repo verbatim
(scripts/deploy_hf.py rebuilds and pushes it; scripts/verify_space.py
checks the live Space against reports/app_example_probs.json).
Requirements are pinned CPU wheels; albumentations was dropped — the val
transform is re-implemented in cv2+numpy, verified bitwise-identical to
the albumentations pipeline on all BUS-BRA images.

Refactor verification (local, after every change): external_val.py
--self-check reproduces fold-5 TTA AUC 0.9234433158791243
digit-for-digit (byte-identical output to reports/external_selfcheck.txt)
and the four bundled examples reproduce their recorded probabilities
bitwise (reports/app_example_probs.json).

**Cross-platform determinism finding.** The first deploy produced
probabilities off by up to ~1e-3 (16.9% vs 16.6% on bus_0186-r).
Root cause: cv2.resize(INTER_LINEAR, uint8) is not platform-stable —
the local Apple-Silicon OpenCV 5.0 build routes it to the Arm KleidiCV
HAL, x86 builds do not, and no runtime flag reconciles them (decode and
normalize were verified identical via checksums; only the resized pixels
differed). Fix: inference.resize_bilinear_frozen, a pure-integer numpy
port of the KleidiCV bilinear kernel (16-bit fixed-point center-aligned
coordinates, 8-bit fractions, vertical-then-horizontal lerp with
round-half-up), verified bitwise-identical to local cv2.resize on all
1,879 BUS-BRA images (713 distinct sizes) + the 4 examples, and
deterministic across platforms. Local numbers are unchanged (all
verifications re-passed bitwise).

Live verification (scripts/verify_space.py): all four example
probabilities agree with local to max |delta| = 1.07e-07 — the measured
cross-architecture BLAS floor (identical model on identical input
tensors differs by ~1e-7 in prob space between Apple Accelerate and
x86 BLAS; measured with a fixed synthetic input). Bitwise float
equality across CPU architectures is unattainable; at any display
precision the live Space and local app are identical, and all four
examples classify correctly. Latency (2 vCPU): cold start
(restart -> first prediction) **9.7 s** (first-ever boot additionally
downloads 1.7 GB of weights, ~minutes); warm median **6.8 s**/image
(local M4: 0.54 s).

## v2: Site-specific recalibration

Secondary post-hoc study per data/external_protocol.md Amendment 2
(committed 2026-08-31 BEFORE any v2 computation). Question: how many
locally labeled images k does a new site need to recover the specificity
lost to domain shift? Inputs: the SAVED external-v1 prediction CSVs only —
no inference, no training, no model changes; every external-v1 number
above stands untouched. Script: src/v2_recalib_curve.py.

Pre-registered sanity checks (all passed, printed by the script before the
run): (a) k=0 at the frozen threshold reproduces external-v1
confusions/sens/spec digit-for-digit on all four cohorts, and recomputed
decisions match the saved y_pred column exactly; (b) the all-n oracle
threshold satisfies sens >= 0.90 in-sample by construction; (c) calibration
and evaluation index sets verified disjoint.

### M1 (primary): local threshold re-selection

Per cohort: R = 500 draws of k images at natural prevalence (seed 42;
BrEaST patient-level ≡ image-level; BUSI/GDPH/SYSUCC image-level —
LIMITATION: correlated same-patient images can split across the k-set and
holdout, flattering the curve). Threshold re-selected on the k calibrated
probs with the frozen rule (highest thr with sens >= 0.90); evaluated on
the held-out n−k. Recovery = (spec_k − spec_frozen)/(spec_oracle −
spec_frozen), all on the same holdout. Degenerate single-class draws fall
back to the frozen threshold (only at k=10: 0.8% BrEaST, 0.6% BUSI, 2.4%
SYSUCC). Medians [2.5–97.5 pct] over draws; full grid in
reports/v2_recalib_M1_summary.csv, per-draw rows in
reports/v2_recalib_M1_draws.csv, figure reports/v2_recalib_M1_curves.png.

| Cohort | frozen→oracle spec | k | sens median [95%] | spec median [95%] | recovery median |
|---|---|---|---|---|---|
| BrEaST | 0.409 → 0.623 | 10 | 0.830 [0.250–1.000] | 0.764 [0.000–0.970] | 1.64 |
| | | 20 | 0.897 [0.602–1.000] | 0.639 [0.000–0.895] | 1.10 |
| | | 30 | 0.870 [0.604–0.989] | 0.699 [0.008–0.887] | 1.36 |
| | | 50 | 0.895 [0.698–1.000] | 0.640 [0.185–0.853] | 1.07 |
| | | 100 | 0.897 [0.748–0.984] | 0.636 [0.280–0.814] | 1.00 |
| BUSI | 0.630 → 0.819 | 10 | 0.846 [0.318–0.997] | 0.900 [0.223–0.995] | 1.43 |
| | | 20 | 0.889 [0.586–0.994] | 0.824 [0.276–0.971] | 1.04 |
| | | 30 | 0.880 [0.658–0.993] | 0.828 [0.297–0.965] | 1.05 |
| | | 50 | 0.894 [0.715–0.985] | 0.818 [0.480–0.961] | 1.00 |
| | | 100 | 0.904 [0.771–0.971] | 0.802 [0.627–0.944] | 0.90 |
| GDPH | 0.453 → 0.775 | 10 | 0.866 [0.290–0.995] | 0.827 [0.174–0.993] | 1.16 |
| | | 20 | 0.888 [0.598–0.995] | 0.798 [0.177–0.957] | 1.07 |
| | | 30 | 0.887 [0.642–0.989] | 0.800 [0.213–0.952] | 1.08 |
| | | 50 | 0.897 [0.746–0.983] | 0.779 [0.340–0.909] | 1.02 |
| | | 100 | 0.896 [0.780–0.969] | 0.781 [0.528–0.888] | 1.02 |
| | | 200 | 0.903 [0.822–0.958] | 0.771 [0.644–0.850] | 0.98 |
| SYSUCC | 0.474 → 0.568 | 10 | 0.909 [0.576–0.999] | 0.554 [0.046–0.878] | 0.85 |
| | | 20 | 0.904 [0.713–0.992] | 0.563 [0.175–0.788] | 1.00 |
| | | 30 | 0.904 [0.753–0.981] | 0.562 [0.282–0.764] | 1.00 |
| | | 50 | 0.913 [0.799–0.979] | 0.535 [0.303–0.716] | 0.69 |
| | | 100 | 0.903 [0.825–0.967] | 0.564 [0.386–0.692] | 1.00 |
| | | 200 | 0.905 [0.846–0.950] | 0.561 [0.445–0.663] | 1.00 |

**Pre-registered k\*** (smallest k with median recovery >= 0.80 AND median
sens >= 0.85): **GDPH 10, SYSUCC 10, BrEaST 20, BUSI 20.**

Reading: in the MEDIAN, a handful of local labels (10–20) already moves
the threshold to near-oracle specificity — the shift is mostly a location
problem, and re-selecting the threshold locally fixes most of it. Two
honest caveats. (1) The median hides brutal draw-to-draw variance: at
k = 10–30 the 95% specificity bands span roughly 0→0.97, i.e. an
individual site recalibrating on 10 images can land anywhere; bands only
become usable at k ≈ 100–200. (2) Median sensitivity sits at 0.83–0.90 —
re-selecting on k samples trades away some of the frozen pipeline's
external sensitivity (≥ 0.92 everywhere in external-v1), and the
pre-registered criterion tolerates that down to 0.85. SYSUCC's recovery
ratio is noisy because its frozen→oracle gap is small (0.474→0.568).
All k on the pre-registered grid are reported; no post-hoc selection.
The frozen model, calibration, and operating point remain unchanged.

### M2/M3 (secondary): recalibration methods comparison

Same harness, draws, and seed as M1 (all methods see identical k-sets).
Logits z = log(p/(1−p)) of the raw pre-temperature TTA probability
(calibrate.probs_to_logits convention, recovered from y_prob_raw per
Amendment 2 (i)). M2: temperature refit on the k local logits with the
calibrate.py LBFGS fitter, applied with (a) the frozen threshold 0.2683
and (b) the local sens ≥ 0.90 rule. M3: unregularized Platt scaling
p = sigmoid(a·z + b) + local rule. Fallback rule (run-time necessity,
same spirit as the degenerate-draw rule): an M2 fit returning a
non-positive temperature (ill-posed local NLL, e.g. a locally
anti-correlated k-set) reverts that draw to the frozen pipeline, flagged
fit_failed — 7–20% of draws at k=10, ≤ 1% by k=30, 0% at k ≥ 100
(per-cell fractions in the summary CSV). Files:
reports/v2_recalib_methods_{draws,summary}.csv, figure
reports/v2_recalib_methods.png (its band-width row is POST-HOC, see
below).

k_reliable = smallest k with 2.5th-percentile recovery ≥ 0.5. **POST-HOC
metric** — defined 2026-08-31 after the M1 results were seen, NOT part of
pre-registered Amendment 2; reported for transparency alongside the
pre-registered k*. Medians at k=30/k=100:

| Cohort | Method | k* (pre-reg) | k_reliable (POST-HOC) | k=30 spec / sens | k=100 spec / sens |
|---|---|---|---|---|---|
| BrEaST | M1 | 20 | not reached | 0.699 / 0.870 | 0.636 / 0.897 |
| | M2a | not reached | not reached | 0.448 / 0.920 | 0.424 / 0.923 |
| | M2b | 10 | not reached | 0.696 / 0.872 | 0.636 / 0.897 |
| | M3 | 20 | not reached | 0.699 / 0.870 | 0.636 / 0.897 |
| BUSI | M1 | 20 | not reached | 0.828 / 0.880 | 0.802 / 0.904 |
| | M2a | not reached | not reached | 0.754 / 0.920 | 0.748 / 0.924 |
| | M2b | 20 | not reached | 0.826 / 0.883 | 0.802 / 0.904 |
| | M3 | 20 | not reached | 0.830 / 0.879 | 0.802 / 0.904 |
| GDPH | M1 | 10 | 200 | 0.800 / 0.887 | 0.781 / 0.896 |
| | M2a | not reached | not reached | 0.657 / 0.941 | 0.654 / 0.940 |
| | M2b | 20 | 200 | 0.788 / 0.894 | 0.781 / 0.896 |
| | M3 | 10 | 200 | 0.800 / 0.887 | 0.781 / 0.896 |
| SYSUCC | M1 | 10 | not reached | 0.562 / 0.904 | 0.564 / 0.903 |
| | M2a | not reached | not reached | 0.505 / 0.921 | 0.490 / 0.926 |
| | M2b | 20 | not reached | 0.561 / 0.905 | 0.564 / 0.903 |
| | M3 | 10 | not reached | 0.562 / 0.904 | 0.564 / 0.903 |

Reading — three results, one of them structural:

1. **M2b and M3 are M1 in disguise.** Temperature and Platt scaling are
   monotone transforms of the score, and the local sens ≥ 0.90 rule picks
   the same rank-space boundary regardless of the monotone map — so once
   the threshold is re-selected locally, the recalibration map is
   irrelevant. Their curves coincide with M1 everywhere (differences at
   k ≤ 30 are only the M2 fit-failure fallbacks and Platt ties). The k
   local labels' entire value is in placing the threshold, not reshaping
   the probabilities.
2. **M2a shows the shift is not just scale.** Refitting only T while
   keeping the frozen threshold holds sensitivity highest of all methods
   (0.92–0.95) but recovers only part of the specificity gap (e.g. GDPH
   0.657 vs oracle 0.775; BrEaST barely moves, 0.42–0.45 vs frozen 0.41)
   — k* is never reached. The external benign-shift is a location
   problem in logit space, which a symmetric-around-p=0.5 temperature
   cannot fix at a 0.2683 threshold.
3. **Stability trade-off (band-width panel, POST-HOC):** M2a has clearly
   the narrowest 95% specificity bands at every k (the frozen threshold
   doesn't jump draw-to-draw; only T varies) — more reliable, less
   recovery. M1/M2b/M3 are identically unstable, so the answer to
   "are M2/M3 more stable than M1 at small k?" is NO for the local-rule
   variants and YES only for M2a, bought with less specificity.
   k_reliable is reached only on GDPH (k=200, local-rule methods): at a
   draw-level reliability bar, no method makes small-k recalibration
   trustworthy inside the pre-registered grid.

All k on the pre-registered grid reported for every method; no post-hoc
selection of k, methods, or cohorts. M1 draws re-verified to reproduce the
committed first run before the extension (printed determinism check).
The frozen model, calibration, and operating point remain unchanged.

### POST-HOC descriptive: cross-site calibration transfer

NOT pre-registered (Amendment 2 covers within-cohort curves only);
descriptive, saved predictions only, frozen pipeline unchanged. Script:
src/v2_cross_site.py · matrix reports/v2_cross_site_matrix.png · values
reports/v2_cross_site.csv. Each source's FULL data calibrates M1
(re-selected threshold, transferred in calibrated-prob space) and M2
(refit temperature, transferred with the frozen 0.2683 threshold; a
source-selected threshold under refit T gives identical decisions to M1 —
monotone map — so it is not a separate variant). Diagonals are in-sample.
Source calibrations: Internal thr 0.2683 / T 2.3644 (reproduces the
frozen pipeline exactly, asserted), BrEaST 0.3755 / 2.2717, BUSI 0.3851 /
1.5804, GDPH 0.4484 / 1.3778, SYSUCC 0.3314 / 2.2253.

Per-row reading (factual):

- **Internal (OOF):** reproduces external-v1 on all four targets in both
  panels (spec 0.41/0.63/0.45/0.47, sens 0.92–0.97).
- **BrEaST-calibrated:** M1 lifts every target's specificity to 0.62–0.80
  at sens 0.87–0.93; its M2 temperature (2.27 ≈ frozen 2.36) leaves the
  M2 row within 0.03 of the internal row.
- **BUSI-calibrated:** M1 gives 0.62–0.69 specificity off-diagonal at
  sens 0.87–0.93; its sharper T = 1.58 raises M2 specificity to
  0.57–0.60 off-diagonal at sens 0.90–0.95.
- **GDPH-calibrated:** the highest threshold (0.4484) yields the largest
  off-diagonal M1 specificity (0.71–0.72 on BrEaST/SYSUCC, 0.89 on BUSI)
  but drops sensitivity to 0.80–0.87 on all three other targets, below
  the 0.90 design floor.
- **SYSUCC-calibrated:** the smallest threshold shift (0.3314) keeps sens
  at 0.92–0.95 everywhere with moderate M1 specificity gains (0.56–0.74);
  its M2 row (T = 2.23) sits within 0.04 of the internal row.

Across rows: every externally re-selected M1 threshold gives higher
specificity than the internal threshold on every target, and M2 rows move
specificity less than M1 rows at matched sensitivity throughout.

## v2: Ensemble disagreement as an abstention signal

Pre-registered: data/external_protocol.md Amendment 3 (committed 78f3991
BEFORE any computation). Run 2026-08-31. Scripts: src/v2_dump_members.py
(member persistence) + src/v2_disagreement.py (analyses). Outputs:
reports/v2_members_{cohort}.csv, reports/v2_abstention_{metrics,curves}.csv,
reports/v2_disagreement_auroc.png, reports/v2_abstention_curves.png.
(Presentational deviation from (p): the single registered figure
v2_abstention.png was split into the two PNGs above; analyses unchanged.)
External-v1 numbers untouched; frozen model/calibration/threshold unchanged.

**Member persistence (n) — reproduction check PASSED, bitwise.** The
constrained re-run of the frozen 5-ckpt × hflip pipeline reproduced the
saved y_prob_raw column bitwise at the stored float32 precision on ALL
images of all four cohorts (252/252, 379/379, 810/810, 1013/1013; max
float64 delta vs the decimal parse 3.0e-08 = shortest-repr gap; the
registered 1e-9 fallback was not needed). No metrics were computed in the
re-run script.

### (a) Error-prediction AUROC (target = misclassified at thr 0.2683)

95% percentile bootstrap ×2000, seed 42; case-level BrEaST (≡ image-level),
IMAGE-level BUSI/GDPH/SYSUCC (no patient IDs — CIs may be optimistically
narrow).

| cohort | errors | U_std | U_range | margin baseline |
|---|---|---|---|---|
| BrEaST | 99/252 | **0.774** (0.710–0.831) | 0.772 (0.708–0.829) | 0.664 (0.595–0.729) |
| BUSI | 87/379 | **0.856** (0.812–0.895) | 0.851 (0.806–0.891) | 0.754 (0.694–0.809) |
| GDPH | 249/810 | **0.802** (0.772–0.833) | 0.800 (0.770–0.831) | 0.717 (0.682–0.752) |
| SYSUCC | 202/1013 | 0.626 (0.589–0.665) | 0.626 (0.589–0.665) | **0.747** (0.710–0.784) |

U_std ≈ U_range everywhere (Δ ≤ 0.005). U_std beats the margin baseline
clearly on BrEaST/BUSI/GDPH (+0.09 to +0.11, non-overlapping CIs on
BUSI/GDPH) but LOSES to it on SYSUCC (0.626 vs 0.747) — on the
highest-prevalence cancer-center cohort, member disagreement is the
weakest error signal of the three.

### (b)/(c) Abstention curves (retained-set sens/spec at the frozen threshold)

Full grid q ∈ {5,10,20,30}% in reports/v2_abstention_curves.csv +
v2_abstention_curves.png. At the pre-registered decision point q = 10%
(frozen → retained; enrich = abstained-set error rate / full-cohort rate):

| cohort | signal | sens | spec | enrich |
|---|---|---|---|---|
| BrEaST | U_std | 0.9184 → 0.9091 | 0.4091 → 0.4532 | ×1.53 |
| BrEaST | margin | 0.9184 → 0.9474 | 0.4091 → 0.3939 | ×1.43 |
| BUSI | U_std | 0.9571 → 0.9527 | 0.6296 → 0.7047 | ×2.64 |
| BUSI | margin | 0.9571 → 0.9557 | 0.6296 → 0.6612 | ×2.06 |
| GDPH | U_std | 0.9707 → 0.9675 | 0.4529 → 0.5038 | ×1.77 |
| GDPH | margin | 0.9707 → 0.9756 | 0.4529 → 0.4389 | ×1.53 |
| SYSUCC | U_std | 0.9309 → 0.9226 | 0.4740 → 0.5150 | ×1.14 |
| SYSUCC | margin | 0.9309 → 0.9615 | 0.4740 → 0.4703 | ×2.53 |

Consistent pattern across all q: abstaining by U_std raises retained
specificity monotonically (BUSI reaches 0.88 at q=30%) at the cost of a
small monotone sensitivity decline; abstaining by margin does the
opposite — retained sensitivity rises (near-threshold cases are
disproportionately positives at this low threshold) while specificity
stays flat or falls. The two signals abstain different images.

### Pre-registered verdict (q) — applied mechanically

Criteria: (1) error-AUROC(U_std) ≥ 0.65; (2) at q=10%, retained spec ≥
frozen + 0.05 AND retained sens ≥ frozen; (3) U_std > margin on
error-AUROC.

| cohort | c1 ≥0.65 | c2 q=10% | c3 beats margin | verdict |
|---|---|---|---|---|
| BrEaST | PASS (0.774) | FAIL (spec +0.044 < 0.05; sens 0.9184→0.9091) | PASS | **NOT USEFUL** |
| BUSI | PASS (0.856) | FAIL (spec +0.075 ✓; sens 0.9571→0.9527) | PASS | **NOT USEFUL** |
| GDPH | PASS (0.802) | FAIL (spec +0.051 ✓; sens 0.9707→0.9675) | PASS | **NOT USEFUL** |
| SYSUCC | FAIL (0.626) | FAIL (spec +0.041; sens 0.9309→0.9226) | FAIL (0.626 < 0.747) | **NOT USEFUL** |

**Verdict: 0/4 cohorts meet the pre-registered bar** — in every case
because criterion 2's sensitivity condition (retained sens not below the
frozen full-set value) fails: U_std-ranked abstention always removes a
few true positives along with the errors, costing 0.3–0.9 points of
sensitivity at q=10%. Recorded as registered; no criterion is relaxed
post-hoc. Factual reading alongside the verdict: disagreement IS strongly
error-informative on 3/4 cohorts (AUROC 0.77–0.86, error enrichment up to
×2.6, spec +4.4 to +7.5 points at q=10%), so the pre-registered bar
failed on the strict sensitivity-preservation clause, not on signal
quality — except on SYSUCC, where disagreement genuinely underperforms
the free margin baseline. Any softer criterion (e.g. "sens within CI of
frozen") would be a NEW pre-registration, not a re-read of this one.

**Limitation (r):** no internal OOF reference exists — each OOF image was
scored by only its one held-out fold's checkpoint, so a 10-member
disagreement signal cannot be computed internally without leakage. This
analysis is external-only, with no internal error-AUROC to compare
against.

## v2: Domain-pretrained backbone (Q2)

Protocol: data/external_protocol.md Amendment 4 (u), committed pre-code
at 3faa70b. Pre-registered fallback rule applied below.

### Q2a — USFM loading spike (2026-09-01): NOT CLEAN → BiomedCLIP substitution

**Verdict: USFM cannot be loaded cleanly into the v1 ViT skeleton. Per
the pre-registered fallback in (u), Q2's backbone is substituted with the
BiomedCLIP ViT-B/16 image encoder (open_clip). Declared here and in the
protocol BEFORE any v2 training.**

Evidence (src/v2_load_usfm.py; checkpoint models/pretrained/
USFM_latest.pth, the openmedlab/USFM Google Drive release, 327 MB,
188 tensors):

- USFM is a BEiT-style ViT-B/16, not a vanilla ViT: the checkpoint
  contains NO absolute pos_embed at all — position is encoded solely by a
  shared relative-position-bias table
  (rel_pos_bias.relative_position_bias_table, 732×12) added inside every
  attention block, an architecture component vanilla timm
  vit_base_patch16_224 has no parameter slot for. It also carries
  per-block LayerScale (gamma_1/gamma_2 × 12) and BEiT split q/v biases
  (no k bias).
- Best-effort mapping: 149/150 skeleton backbone tensors filled (99.3%) —
  patch embed, cls token, all 12 blocks (qkv bias fused as
  q_bias ⊕ 0 ⊕ v_bias), final norm all LOADED. pos_embed UNFILLED
  (nothing in the checkpoint to load), and 27 checkpoint tensors UNUSED:
  24 LayerScale gammas, the rel-pos-bias table + index, mask_token.
- Forward passes with the mapped weights: CPU and MPS both clean (no
  NaNs, max |MPS−CPU| 8.1e-06); CLS-feature cosine vs an
  ImageNet-initialized ViT −0.03, i.e. the USFM weights genuinely loaded.
- Why NOT CLEAN despite 99.3% tensor coverage: the unused tensors are the
  model's ENTIRE positional mechanism and its residual-branch scaling. A
  vanilla-ViT forward with these weights computes a different function
  from USFM — a randomly initialized pos_embed stands in for trained
  relative position biases and residual branches lose their 0.1-scale
  gammas. The pre-registered coverage check ("patch/pos embeddings,
  blocks, and norm all loaded") fails on pos_embed.
- Not pursued: loading USFM into a timm BEiT skeleton would be clean but
  changes the architecture, violating (u)'s "matching v1's
  vit_base_patch16_224 geometry" and rule (v)'s no-other-backbone clause.

### Substitution (declared pre-training, per (u))

Q2's backbone = BiomedCLIP ViT-B/16 image encoder
(hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 via
open_clip). Its vision tower is exactly a timm vit_base_patch16_224, so
v1's architecture, preprocessing, and training protocol carry over with
only the initialization replaced. Q2 artifact names become
models/v2_biomedclip_fold{1-5}.pt +
v2_biomedclip_{calibration,operating_point}.json; rule (v) reads
"BiomedCLIP" where it says "USFM". Limitation, stated up front:
BiomedCLIP pretraining used CLIP normalization statistics while the v1
pipeline (kept unchanged per (u)) normalizes with ImageNet statistics;
fine-tuning must absorb that difference.

### BiomedCLIP setup + smoke test (2026-09-01)

- Export (src/v2_export_biomedclip.py, reading visual.trunk.* directly
  from open_clip_pytorch_model.bin): **CLEAN — 150/150** backbone tensors,
  zero unfilled / zero unused, all five components LOADED; forward passes
  no NaNs on CPU and MPS (max |MPS−CPU| 1.9e-03 at feature std 2.59);
  CLS-feature cosine vs ImageNet init 0.048 → weights genuinely loaded.
  Saved to models/pretrained/biomedclip_vitb16_timm.pt (consumed by
  train.py via model.init_state_dict, which requires exactly head.* to
  remain randomly initialized).
- One-epoch fold-5 smoke (configs/v2_biomedclip.yaml, --epochs 1,
  checkpoint kept out of models/): **epoch time 142.7 s (MPS, M4),
  val AUC 0.8434**, train_loss 0.827 — training is numerically healthy
  from the BiomedCLIP init. wandb run v2_biomedclip_fold5_smoke
  (830ejnrt). Q2b (full 5-fold CV) NOT launched.

### Q2b — 5-fold CV training runs (2026-09-02, wandb group cv_biomedclip)

configs/v2_biomedclip.yaml (v1 protocol, BiomedCLIP init), 30 epochs/fold
on MPS (~1.5–2 h/fold under caffeinate), init_state_dict validation
confirmed in-log for every fold (150 trunk tensors loaded, head random).
Checkpoints models/v2_biomedclip_fold{1-5}.pt; per-fold metrics at each
fold's own best-val-AUC epoch (reports/v2_biomedclip_cv_summary.csv):

| fold | AUC | v1 ViT AUC | Δ |
|---|---|---|---|
| 1 | 0.9438 | 0.9526 | −0.009 |
| 2 | 0.9229 | 0.9411 | −0.018 |
| 3 | 0.9119 | 0.9193 | −0.007 |
| 4 | 0.8835 | 0.9132 | −0.030 |
| 5 | 0.9230 | 0.9272 | −0.004 |
| **mean ± sd** | **0.9170 ± 0.0220** | 0.9307 ± 0.0161 | −0.014 |

Internal CV observation (descriptive only): the BiomedCLIP init trails
the v1 ImageNet init on all five folds. This does NOT decide Q2 — the
pre-registered criterion in (u) is about external shift reduction and is
evaluated only after the freeze, single-shot. Remaining Q2b steps
pending: hflip TTA on the OOF pool, one temperature fit, sens ≥ 0.90
threshold, then FREEZE (v2_biomedclip_{calibration,operating_point}.json).

### Q2b — TTA, calibration, operating point, FREEZE (2026-09-02)

src/v2_biomedclip_freeze.py (v1 pipeline replicated: tta_eval →
calibrate → pick_threshold conventions, imported not re-implemented;
v1 artifacts asserted byte-untouched). All on BUS-BRA OOF only — no
external data touched before the freeze.

- Pooled OOF hflip-TTA AUC **0.9109** (plain 0.9092; v1 reference
  0.9254). Per-fold TTA: 0.9461 / 0.9267 / 0.9150 / 0.8832 / 0.9231;
  per-fold plain AUCs reproduce reports/v2_biomedclip_cv_summary.csv
  exactly. reports/v2_biomedclip_{tta_summary,oof_preds}.csv.
- Temperature **T = 2.8645** on pooled OOF TTA (v1: 2.3644): NLL
  0.5479 → 0.3464, ECE(15) 0.0976 → 0.0296, AUC unchanged (asserted).
  models/v2_biomedclip_calibration.json,
  reports/v2_biomedclip_reliability.png.
- Operating point (sens ≥ 0.90, max spec, calibrated OOF): threshold
  **0.2007** — sens 0.9012 (CI 0.8715–0.9294), spec 0.7177 (CI
  0.6887–0.7481), PPV 0.6044, NPV 0.9381 (patient-level bootstrap
  ×2000, seed 42). models/v2_biomedclip_operating_point.json,
  reports/v2_biomedclip_roc_oof_operating_point.png.
- **FREEZE (v2_biomedclip):** 5 ckpts + hflip TTA + T=2.8645 +
  thr=0.2007. No model changes after this commit; next step is the
  single-shot Q2c external run.

Internal comparison (descriptive): v2 trails v1 on pooled OOF AUC
(−0.0145) and on internal specificity at the respective operating
points (0.7177 vs 0.7713).

### Q2c — single-shot external evaluation (2026-09-02)

src/v2_biomedclip_external.py. Self-check first: the external code path
(inference.ensemble_tta_probs_from_loader) reproduced the freeze
script's fold-5 TTA AUC digit-for-digit (0.9230647908649297) —
PASSED. Then exactly ONE --confirm run on the four frozen keep-lists;
recorded as-is, no re-tuning. The v1 side of the table is recomputed
from the SAVED reports/external_*_preds.csv (no v1 inference) and
asserted against the recorded external-v1 AUCs before use. Each
pipeline is scored at its own frozen threshold (v1: T=2.3644,
thr=0.2683; v2: T=2.8645, thr=0.2007). Benign shift = median
p_cal(benign, cohort) − median p_cal(benign, own pooled OOF); own-OOF
benign medians: v1 0.0625, v2 0.0808.

| cohort | n | AUC v1 (CI) | AUC v2 (CI) | ΔAUC | sens v1→v2 | spec v1→v2 | benign med v1→v2 | shift v1→v2 | ratio |
|---|---|---|---|---|---|---|---|---|---|
| BrEaST | 252 | 0.8542 (0.802–0.902) | 0.8441 (0.789–0.894) | −0.0101 | 0.918→0.929 | 0.409→0.325 | 0.306→0.298 | +0.244→+0.217 | 0.892 |
| BUSI | 379 | 0.9339 (0.906–0.958) | 0.9102 (0.875–0.942) | −0.0237 | 0.957→0.926 | 0.630→0.602 | 0.174→0.158 | +0.111→+0.077 | 0.689 |
| GDPH | 810 | 0.9154 (0.895–0.934) | 0.8821 (0.858–0.904) | −0.0333 | 0.971→0.968 | 0.453→0.345 | 0.289→0.299 | +0.227→+0.218 | 0.962 |
| SYSUCC | 1013 | 0.8380 (0.810–0.866) | 0.8308 (0.801–0.858) | −0.0072 | 0.931→0.946 | 0.474→0.443 | 0.287→0.238 | +0.224→+0.157 | 0.700 |

reports/v2_biomedclip_external_comparison.csv +
v2_biomedclip_external_{cohort}_preds.csv + per-cohort ROC/CM pngs.
Bootstrap ×2000 seed 42, case-level BrEaST / image-level otherwise
(limitation as in (d)).

### Q2 verdict (Amendment 4 (u), applied mechanically) and rule (v) decision

- **Branch A** (v2 AUC ≥ v1 + 0.01 on ≥ 3/4 cohorts): **0/4** — v2's
  external AUC is LOWER on all four cohorts. NOT MET.
- **Branch B** (|shift_v2| ≤ 0.70·|shift_v1| on ≥ 3/4 cohorts): **1/4**
  (BUSI 0.689). SYSUCC's exact ratio is 0.700163 — it misses the ≤ 0.70
  cut by 0.0002 and is counted as a fail, per the mechanical rule, no
  rounding. NOT MET.
- **Q2 VERDICT: "domain pretraining reduces shift" is NOT CLAIMED.**
  The BiomedCLIP init reduced the benign probability shift somewhat on
  all four cohorts (ratios 0.69–0.96) but paid for it with uniformly
  lower external AUC, and neither pre-registered branch fires. Sens
  held ≥ 0.926 everywhere at v2's own threshold; specificity was worse
  than v1 on all four cohorts.
- **Rule (v) decision — Q1 LOCO backbone = v1 ImageNet
  vit_base_patch16_224** (same backbone as v1, cleanest
  multi-source-vs-v1 comparison). No other backbone may be introduced
  under Amendment 4.
- Interpretation note (descriptive): BiomedCLIP's PMC-figure pretraining
  is not ultrasound-specific, and the CLIP→ImageNet normalization
  mismatch was absorbed by fine-tuning as required by (u); the USFM
  weights that motivated Q2 could not be used (Q2a). The Q2 result
  therefore speaks to THIS substitute init, not to ultrasound MIM
  pretraining in general.

## v2: Multi-source LOCO training (Q1)

Protocol: Amendment 4 (t); backbone fixed by rule (v) after the Q2
verdict: **v1 ImageNet vit_base_patch16_224** (identical to v1).
Pipeline: src/v2_loco.py over v2_data.build_multisource_df — one model
per run, v1 hyperparameters + early stop (patience 7) on pooled val AUC;
the run's temperature and sens ≥ 0.90 threshold are fit on the pooled
training-side validation only (hflip TTA), then the held-out cohort is
scored ONCE (script refuses a second run). Comparator per (t): v1 fold-5
single model + TTA recomputed from the SAVED reports/v2_members_*.csv
(m5 columns, T=2.3644, thr 0.2683) — no v1 inference. LIMITATION stated
per (t): external val slices are image-level (correlated images may sit
on both sides, flattering val metrics) and the pooled val mixture's
prevalence matches no single site.

Pre-registered criterion (t), decided over all four runs: claim only if
ΔAUC ≥ +0.01 vs v1-single on ≥ 3/4 held-out cohorts, OR Δspec ≥ +0.10
on ≥ 3/4 with that run's held-out sens ≥ 0.85.

### Run 1 — hold_out = BrEaST (2026-09-03, wandb v2_loco_breast)

Train 3,363 / val 714 (pos_weight 1.163); early stop at epoch 11, best
epoch 4, pooled val TTA AUC 0.9388; T = 1.2216, thr = 0.4738 (val sens
0.9045 / spec 0.8350). models/v2_loco_breast.pt + run JSONs frozen
before the held-out pass.

| n=252, case-level bootstrap | AUC (95% CI) | sens | spec | benign median p_cal |
|---|---|---|---|---|
| LOCO (thr 0.4738) | **0.8657** (0.817–0.910) | 0.8571 | 0.7532 | 0.2230 |
| v1-single (thr 0.2683) | 0.8467 (0.794–0.895) | 0.9388 | 0.4740 | 0.2789 |

ΔAUC **+0.0190**, Δspec **+0.2792** at held-out sens 0.8571 (≥ 0.85) —
this cohort satisfies both branches of the (t) criterion (verdict is
taken only after all four runs). Note the operating-point trade: the
LOCO threshold trades v1-single's sensitivity (0.939 → 0.857) for a
large specificity gain; the val-mixture threshold transfers far better
to BrEaST than v1's internal threshold did. v1 full ensemble (reference
only, external-v1): AUC 0.8542, sens 0.9184, spec 0.4091.
reports/v2_loco_breast_preds.csv, v2_loco_{roc,cm}_breast.png,
summary row in reports/v2_loco_summary.csv.

### Run 2 — hold_out = BUSI (2026-09-03, wandb v2_loco_busi)

Train 3,255 / val 695 (pos_weight 1.161); early stop at epoch 24, best
epoch 17, pooled val TTA AUC 0.9533; T = 2.4800, thr = 0.4116 (val sens
0.9027 / spec 0.8741). models/v2_loco_busi.pt + run JSONs frozen before
the held-out pass.

| n=379, image-level bootstrap | AUC (95% CI) | sens | spec | benign median p_cal |
|---|---|---|---|---|
| LOCO (thr 0.4116) | **0.9371** (0.912–0.959) | 0.8466 | 0.8796 | 0.0512 |
| v1-single (thr 0.2683) | 0.9069 (0.873–0.937) | 0.9387 | 0.6065 | 0.1501 |

ΔAUC **+0.0302** (branch A satisfied on this cohort), Δspec **+0.2731**
but held-out sens 0.8466 falls below the 0.85 clause — branch B is NOT
satisfied on this cohort (verdict is taken only after all four runs).
Same trade as Run 1, slightly steeper: the LOCO threshold gives up
sensitivity (0.939 → 0.847) for specificity (+0.27); here the sens cost
crosses the pre-registered floor. Benign median p_cal 0.0512 — the
lowest benign shift of any external eval so far. v1 full ensemble
(reference only, external-v1): AUC 0.9339, sens 0.9571, spec 0.6296.
reports/v2_loco_busi_preds.csv, v2_loco_{roc,cm}_busi.png,
summary row in reports/v2_loco_summary.csv.

### Run 3 — hold_out = GDPH (2026-09-03, wandb v2_loco_gdph)

Train 2,889 / val 630 (pos_weight 1.190); early stop at epoch 11, best
epoch 4, pooled val TTA AUC 0.9276; T = 0.9545, thr = 0.5410 (val sens
0.9011 / spec 0.8011). models/v2_loco_gdph.pt + run JSONs frozen before
the held-out pass.

| n=810, image-level bootstrap | AUC (95% CI) | sens | spec | benign median p_cal |
|---|---|---|---|---|
| LOCO (thr 0.5410) | **0.9454** (0.930–0.960) | 0.9600 | 0.7126 | 0.2783 |
| v1-single (thr 0.2683) | 0.8899 (0.867–0.913) | 0.9520 | 0.4115 | 0.3393 |

ΔAUC **+0.0556**, Δspec **+0.3011** at held-out sens 0.9600 (≥ 0.85) —
this cohort satisfies both branches of the (t) criterion (verdict is
taken only after all four runs), and it is the only run so far where
the LOCO model's held-out sensitivity also EXCEEDS v1-single's
(0.9600 vs 0.9520): no sens-for-spec trade was needed. Largest ΔAUC of
the three runs, and the CIs do not overlap. Note T = 0.9545 < 1 — the
pooled-val fit sharpens rather than softens this run's probabilities.
v1 full ensemble (reference only, external-v1): AUC 0.9154, sens
0.9707, spec 0.4529. reports/v2_loco_gdph_preds.csv,
v2_loco_{roc,cm}_gdph.png, summary row in reports/v2_loco_summary.csv.

### Run 4 — hold_out = SYSUCC (2026-09-03, wandb v2_loco_sysucc)

Train 2,716 / val 600 (pos_weight 1.655); early stop at epoch 9, best
epoch 2, pooled val TTA AUC 0.9327; T = 0.8761, thr = 0.4968 (val sens
0.9045 / spec 0.8263). models/v2_loco_sysucc.pt + run JSONs frozen
before the held-out pass.

| n=1013, image-level bootstrap | AUC (95% CI) | sens | spec | benign median p_cal |
|---|---|---|---|---|
| LOCO (thr 0.4968) | **0.8444** (0.817–0.870) | 0.7997 | 0.7163 | 0.1706 |
| v1-single (thr 0.2683) | 0.8319 (0.804–0.859) | 0.9337 | 0.4498 | 0.3178 |

ΔAUC **+0.0126** (branch A satisfied on this cohort); Δspec **+0.2664**
but held-out sens 0.7997 falls well below the 0.85 clause — branch B is
NOT satisfied on this cohort (verdict is taken only after all four
runs; this completes the four). The hardest cohort for every model so
far, and the steepest operating-point trade: the LOCO threshold gives
up 0.134 sensitivity for +0.27 specificity. Best epoch 2 with early
stop at 9 — the multi-source pool saturates on SYSUCC-adjacent signal
almost immediately. v1 full ensemble (reference only, external-v1):
AUC 0.8380, sens 0.9309, spec 0.4740.
reports/v2_loco_sysucc_preds.csv, v2_loco_{roc,cm}_sysucc.png,
summary row in reports/v2_loco_summary.csv.

### Q1c — summary table + paired delta CIs (2026-09-04, src/v2_loco_report.py)

Post-hoc reporting from SAVED artifacts only (LOCO preds CSVs, members
CSVs for v1-single, external-v1 preds for the v1-ensemble reference) — no
inference. Before writing anything the script asserts every point value
reproduces reports/v2_loco_summary.csv (tolerance 1e-6, the float32 CSV
round-trip floor) and the v1-ensemble AUCs reproduce the external-v1
table. Delta CIs are PAIRED bootstrap (the same resamples scored under
both models), protocol (d) units — case-level BrEaST, image-level
otherwise (limitation as before) — 2000 iterations, seed 42.

| Held-out cohort | Model | AUC (95% CI) | Sens | Spec | Benign median p_cal | ΔAUC vs v1-single (95% CI) | Δspec (95% CI) |
|---|---|---|---|---|---|---|---|
| BrEaST (n=252) | v1-single | 0.8467 (0.794–0.895) | 0.9388 | 0.4740 | 0.2789 | | |
| | v1-ensemble (ref) | 0.8542 (0.802–0.902) | 0.9184 | 0.4091 | 0.3061 | | |
| | v2-LOCO | **0.8657** (0.817–0.910) | 0.8571 | 0.7532 | 0.2230 | **+0.0190** (−0.0155 to +0.0550) | **+0.2792** (+0.1961 to +0.3618) |
| BUSI (n=379) | v1-single | 0.9069 (0.873–0.937) | 0.9387 | 0.6065 | 0.1501 | | |
| | v1-ensemble (ref) | 0.9339 (0.906–0.958) | 0.9571 | 0.6296 | 0.1739 | | |
| | v2-LOCO | **0.9371** (0.912–0.959) | 0.8466 | 0.8796 | 0.0512 | **+0.0302** (+0.0074 to +0.0550) | **+0.2731** (+0.2087 to +0.3378) |
| GDPH (n=810) | v1-single | 0.8899 (0.867–0.913) | 0.9520 | 0.4115 | 0.3393 | | |
| | v1-ensemble (ref) | 0.9154 (0.895–0.934) | 0.9707 | 0.4529 | 0.2894 | | |
| | v2-LOCO | **0.9454** (0.930–0.960) | 0.9600 | 0.7126 | 0.2783 | **+0.0556** (+0.0371 to +0.0743) | **+0.3011** (+0.2484 to +0.3573) |
| SYSUCC (n=1013) | v1-single | 0.8319 (0.804–0.859) | 0.9337 | 0.4498 | 0.3178 | | |
| | v1-ensemble (ref) | 0.8380 (0.810–0.866) | 0.9309 | 0.4740 | 0.2869 | | |
| | v2-LOCO | **0.8444** (0.817–0.870) | 0.7997 | 0.7163 | 0.1706 | **+0.0126** (−0.0099 to +0.0346) | **+0.2664** (+0.2122 to +0.3206) |

Thresholds: v1 rows at the frozen 0.2683 (T=2.3644); LOCO rows at each
run's val-fit threshold (0.4738 / 0.4116 / 0.5410 / 0.4968). v1-ensemble
is reference only per (t); the comparator for the criterion is v1-single.
Descriptive, not part of the criterion: the paired ΔAUC CI excludes 0 on
BUSI and GDPH but crosses 0 on BrEaST and SYSUCC; the paired Δspec CI
excludes 0 on all four (lower bounds +0.20 to +0.25 on three).
Figure: reports/v2_loco_summary.png — per cohort, ROC overlay (v1-single
vs LOCO, both operating points marked) over the benign calibrated-
probability distributions with both thresholds. Table CSV:
reports/v2_loco_q1c_table.csv.

### Q1 verdict (Amendment 4 (t), applied mechanically)

| Held-out | Branch A: ΔAUC ≥ +0.01 | Branch B: Δspec ≥ +0.10 AND sens ≥ 0.85 |
|---|---|---|
| BrEaST | PASS (+0.0190) | PASS (+0.2792, sens 0.8571) |
| BUSI | PASS (+0.0302) | fail (+0.2731, but sens 0.8466 < 0.85) |
| GDPH | PASS (+0.0556) | PASS (+0.3011, sens 0.9600) |
| SYSUCC | PASS (+0.0126) | fail (+0.2664, but sens 0.7997 < 0.85) |

Branch A: **4/4** (≥ 3/4 required) → **criterion MET via branch A**.
Branch B: 2/4 — not met on its own. Per (t) the claim is therefore made:
**multi-source training reduces the domain-shift specificity collapse**,
carried by the AUC branch; the specificity branch fails only through its
sensitivity clause (Δspec ≥ +0.10 itself holds on 4/4, paired CIs
excluding zero).

**Mechanism, per cohort (benign-median deltas, LOCO − v1-single):**
BrEaST −0.056, BUSI −0.099, GDPH −0.061, SYSUCC −0.147.
- **BrEaST:** benign distribution barely moves (0.279 → 0.223) while the
  operating threshold moves +0.21 — the spec gain (+0.28) is mostly
  threshold placement: the val-mixture threshold lands above the BrEaST
  benign mass where v1's internal threshold sat inside it (v1 benign
  median 0.2789 > thr 0.2683).
- **BUSI:** genuine distribution-shift reduction — benign median drops
  0.150 → 0.051 (the lowest of any external eval in the project) on top
  of ΔAUC +0.030 (CI excludes 0); threshold placement adds the rest.
- **GDPH:** the largest separability gain (ΔAUC +0.056, CI +0.037 to
  +0.074, LOCO/v1 AUC CIs disjoint); benign median stays high (0.278),
  so the spec gain needs both the better ROC and the higher threshold.
  Only cohort where LOCO sens also exceeds v1-single's (0.960 vs 0.952).
- **SYSUCC:** the largest benign-median drop (0.318 → 0.171, nearly
  halved) yet the smallest ΔAUC (+0.013, CI crosses 0) — the downshift
  is not benign-specific: malignant probabilities move down with it,
  and held-out sens falls to 0.7997 at the run threshold. The shift
  marker improves without a matching separability gain.

**Sensitivity portability (stated factually):** every LOCO run hit
sens ≈ 0.90 on its pooled validation, but held-out sensitivity landed
at 0.857 / 0.847 / 0.960 / 0.800 — 0.80–0.86 on 3/4 cohorts, below the
0.85 floor on two — whereas v1's frozen threshold held sens ≥ 0.92 on
all four (single and ensemble). The multi-source threshold buys its
+0.27 specificity at a sensitivity cost that itself does not port
reliably; the val-sens ≥ 0.90 rule does not guarantee held-out
sens ≥ 0.85.

**Descriptive anomalies (recorded, not interpreted):** (i) the SYSUCC
run's temperature is 0.876 — the smallest of the four and the only one
materially below 1, i.e. the sole clearly underconfident case (GDPH's
0.9545 sits just under unity; every other temperature in the project is
> 1); (ii) the same run converged at best epoch 2 (early stop at 9) —
the multi-source pool saturates on SYSUCC-adjacent signal almost
immediately.

Tag: v2-loco. Artifacts: src/v2_loco_report.py,
reports/v2_loco_q1c_table.csv, reports/v2_loco_summary.png (point
values asserted against reports/v2_loco_summary.csv and the external-v1
records at run time).

## Reproducibility notes

**Resize-kernel dispatch (2026-09-01, read-only diagnostic —
src/v2_resize_impact.py, reports/v2_resize_dispatch_impact.csv):**
cv2.resize INTER_LINEAR dispatches the KleidiCV kernel (which
inference.resize_bilinear_frozen ports) only for some image sizes, so the
port's bitwise equality with local cv2 holds on the BUS-BRA size envelope
(all 713 sizes) but not on 1,421/2,454 external keep-list images (BrEaST
99/252, BUSI 0/379, GDPH 309/810, SYSUCC 1013/1013), where the two resizes
differ by ≤ 1 uint8 gray level; run through the full frozen ensemble this
moves calibrated probabilities by max 0.0072 / median 0.0008 and flips
2/1,421 decisions at the frozen threshold (case038, case200 — BrEaST cases
within 0.002 of 0.2683). External-v1 numbers are untouched (canonical
albumentations path; re-verified here to ≤ 3e-08, the MPS batch-composition
floor); the deployed Space's port remains the defined deployment resize.

## Post-hoc analyses responding to the 2026-09-06 audit

**POST-HOC — not pre-registered.** Run 2026-09-07 (plan.md P2) from
committed artifacts and local wandb run histories only: no retraining, no
new external inference, no change to any frozen model, calibration or
threshold. Each analysis answers one item of docs/AUDIT_2026-09-06.md; the
per-row values are in reports/posthoc_*.csv. All reproduction asserts named
below are executed by the scripts before anything is written.

### POST-HOC 1 — Epoch-selection sensitivity (audit §1)

Script: src/posthoc_epoch_selection.py · CSV: reports/posthoc_epoch_selection.csv
· inputs: the five cv_vit wandb datastores (wandb/run-20260828_{084412,100807,
113529,151003,164240}-*/run-*.wandb, matched by args `--config configs/vit.yaml
--prefix cv_vit`; per-run val_folds read from the wandb config).
train.py saves the checkpoint at the epoch with the best AUC on the held-out
fold and cross_validate.py reports that fold's AUC, so every per-fold CV AUC
in this file (0.9307 ± 0.0161 headline) is a max over 30 epochs on the fold
it is reported on. Fixed epoch = median of the five best epochs
{29, 10, 18, 16, 19} = 18. Assert: the best-epoch column reproduces
reports/cv_vit_summary.csv (auc max |Δ| 1.1e-16; best_epoch identical).

| fold | best epoch | AUC @ best epoch (reported) | AUC @ epoch 30 | AUC @ epoch 18 (fixed) | optimism best − fixed |
|---|---|---|---|---|---|
| 1 | 29 | 0.9526 | 0.9523 | 0.9419 | +0.0106 |
| 2 | 10 | 0.9411 | 0.9306 | 0.9283 | +0.0128 |
| 3 | 18 | 0.9193 | 0.9129 | 0.9193 | +0.0000 (fixed epoch = its best epoch) |
| 4 | 16 | 0.9132 | 0.9098 | 0.9012 | +0.0120 |
| 5 | 19 | 0.9272 | 0.9120 | 0.9069 | +0.0203 |
| **mean ± SD** | — | **0.9307 ± 0.0161** | 0.9235 ± 0.0181 | **0.9195 ± 0.0164** | **0.0111 ± 0.0073** |

Optimism best − final epoch: 0.0071 ± 0.0059. Reading (descriptive): the
reported CV mean is about 0.011 AUC above a fixed-epoch protocol and about
0.007 above the last epoch; the fold ranking is unchanged. The pooled OOF
predictions used for T and the frozen threshold come from these best-epoch
checkpoints and inherit the same selection. Because fold 3's best epoch is
the median, its optimism is zero by construction and the mean is, if
anything, slightly understated.

### POST-HOC 2 — Nested (out-of-sample) internal operating point (audit §7)

Script: src/posthoc_nested_threshold.py · CSV: reports/posthoc_nested_threshold.csv
· inputs: reports/oof_vit_preds.csv (y_prob_tta), models/calibration.json
(frozen T = 2.3644), the frozen rule from pick_threshold.py (highest
threshold with sens ≥ 0.90). For each fold k the threshold is picked on the
other four folds' calibrated OOF probabilities and evaluated on fold k.
Assert: the in-sample recomputation reproduces models/operating_point.json
(threshold 0.2683, sens 0.9028, spec 0.7713).

| fold k | n | threshold from folds ≠ k | sens on k | spec on k |
|---|---|---|---|---|
| 1 | 376 | 0.2450 | 0.9426 | 0.8031 |
| 2 | 385 | 0.2355 | 0.9760 | 0.5500 |
| 3 | 366 | 0.3108 | 0.8500 | 0.8415 |
| 4 | 365 | 0.3108 | 0.8487 | 0.8780 |
| 5 | 383 | 0.2743 | 0.8843 | 0.8282 |
| **mean ± SD** | — | 0.2753 ± 0.0354 | **0.9003 ± 0.0569** | **0.7802 ± 0.1315** |
| pooled out-of-sample decisions | 1875 | (per-fold) | 0.9012 | 0.7784 |
| in-sample (frozen 0.2683, reference) | 1875 | 0.2683 | 0.9028 | 0.7713 |

Reading (descriptive): on average the out-of-sample operating point lands
where the in-sample one does (sens 0.900 / spec 0.780 vs 0.903 / 0.771), so
the in-sample optimism of the headline sens/spec is small in the mean. The
per-fold spread is the real finding: the re-selected threshold moves
between 0.236 and 0.311, held-out sensitivity ranges 0.849–0.976 and
misses the 0.90 design floor on 3/5 folds, and specificity ranges
0.550–0.878. The frozen threshold's "sens ≥ 0.90" is a property of the
fitting set, not a guarantee on new patients — consistent with the
external-v1 and Amendment 2 findings.

### POST-HOC 3 — Overconfidence count (audit §2 item 6 / report §3.1)

Script: src/posthoc_overconfidence.py · CSV: reports/posthoc_overconfidence.csv
· input: reports/oof_vit_preds.csv. Malignant OOF images whose RAW
(pre-temperature) probability lies in [0.6, 0.95]:

| scope | column | n malignant | in [0.6, 0.95] | > 0.95 | < 0.6 |
|---|---|---|---|---|---|
| fold 5 | y_prob_tta (frozen path) | 121 | **32 (26.4%)** | 61 | 28 |
| fold 5 | y_prob_plain | 121 | 27 (22.3%) | 69 | 25 |
| all folds | y_prob_tta | 607 | 116 (19.1%) | 358 | 133 |
| all folds | y_prob_plain | 607 | 97 (16.0%) | 383 | 127 |

The sentence formerly in docs/report.md §3.1 ("121 malignant, only 3 with
probability in 0.6–0.95") is NOT reproduced by any committed artifact: the
count is 32/121 (TTA) or 27/121 (plain). It was deleted in P1; this table is
the sourced replacement. About half of the malignant OOF probabilities
exceed 0.95 before temperature scaling (T = 2.36 then pulls them toward the
middle), which is the calibration-side overconfidence already described in
the Calibration section.

### POST-HOC 4 — Model false positives vs reader BI-RADS calls (audit §8 item 26)

Script: src/posthoc_reader_concordance.py · CSV: reports/posthoc_reader_concordance.csv
· inputs: reports/external_{gdph,sysucc}_preds.csv joined 1:1 with the two
reader columns of BIRADS&FOLD.xlsx exactly as in birads_comparison.py
(≥ 4a positive; rows with a non-BI-RADS reader value excluded from that
reader's column only). Benign images only; no per-image concordance was
pre-registered and nothing is tested.

| cohort | reader | n benign | model FP | model FP also ≥ 4a by reader | model TN called ≥ 4a by reader | reader ≥ 4a on all benign |
|---|---|---|---|---|---|---|
| GDPH | reader1 | 435 | 238 | 34/238 = 0.143 | 12/197 = 0.061 | 0.106 |
| GDPH | reader2 | 435 | 238 | 152/238 = 0.639 | 60/197 = 0.305 | 0.487 |
| SYSUCC | reader1 | 289 | 152 | 81/152 = 0.533 | 20/137 = 0.146 | 0.349 |
| SYSUCC | reader2 | 289 | 152 | 143/152 = 0.941 | 108/137 = 0.788 | 0.869 |

Reading (descriptive): in all four reader × cohort cells the model's false
positives are called ≥ 4a by the reader more often than its true negatives
are, so the model's over-calls are enriched for images a radiologist also
found suspicious. But the enrichment is far from agreement: on GDPH the
high-specificity reader1 called only 14% of the model's 238 false positives
≥ 4a, i.e. 86% of the model's GDPH over-calls are benigns reader1 rated
BI-RADS ≤ 3. The report's former claim that model and readers were "misled
by the same atypical benigns in the same way" is therefore not supported
for reader1 and only loosely for reader2; it stays deleted.

### train.py checkpoint metadata (audit §1, code fix in the same commit)

train.checkpoint_payload now stores the folds actually used in each run
(config.data.train_folds/val_folds as resolved by cross_validate.py);
tests/test_checkpoint_config.py covers it. Existing checkpoints
(models/cv_vit_fold{1-5}.pt, cv_convnext_*, cv_effb0_*, cv_vit_cdrop_*,
v2_biomedclip_fold{1-5}.pt) still carry the raw YAML
(train_folds=[1,2,3,4], val_folds=[5]) and are NOT rewritten — the folds
really used are in the wandb run configs, verified by the audit.

## Errata (2026-09-07)

Appended after the independent audit (docs/AUDIT_2026-09-06.md). No line
above this section has been edited since it was written; corrections are
recorded here only, so that `git diff 2cbe1da HEAD -- RESULTS.md` continues
to remove zero lines.

- line 403: '1,879 BUS-BRA images (713 distinct sizes) + the 4 examples' →
  '1,875 BUS-BRA images (713 distinct sizes), including the 4 bundled
  examples' — reason: BUS-BRA has 1,875 images (models/operating_point.json
  n_images = 1875); the four bundled examples are BUS-BRA images and were
  counted a second time.
- line 485: 'external sensitivity (≥ 0.92 everywhere in external-v1)' →
  'external sensitivity (≥ 0.918 everywhere in external-v1)' — reason:
  BrEaST sensitivity is 90/98 = 0.9184 (line 185, confusion line 201).
- line 1058: "whereas v1's frozen threshold held sens ≥ 0.92 on all four
  (single and ensemble)" → "held sens ≥ 0.918 on all four (ensemble; the
  fold-5 single model ≥ 0.9337)" — reason: same BrEaST ensemble value
  0.9184; single-model values 0.9388/0.9387/0.9520/0.9337 (Q1c table).
- lines 1028–1033: 'Per (t) the claim is therefore made: **multi-source
  training reduces the domain-shift specificity collapse**, carried by the
  AUC branch; the specificity branch fails only through its sensitivity
  clause (Δspec ≥ +0.10 itself holds on 4/4, paired CIs excluding zero).' →
  'Per (t) the registered claim is made on branch A alone (ΔAUC ≥ +0.01 vs
  the v1 fold-5 SINGLE model on 4/4). Stated in the same paragraph: the
  paired ΔAUC CI includes zero on 2/4 cohorts (BrEaST −0.0155…+0.0550,
  SYSUCC −0.0099…+0.0346); branch B fails 2/4 (sens 0.8466, 0.7997 < 0.85);
  against the deployed v1 ensemble ΔAUC is +0.0115 / +0.0033 / +0.0301 /
  +0.0064, only 2/4 ≥ +0.01; the registered question (protocol line 390)
  concerned the specificity collapse, which the fired AUC branch does not
  test directly; Δspec ≥ +0.10 holds on 4/4 with paired CIs excluding zero
  but at different thresholds per model.' — reason: audit §4 "softening",
  §7 and §8 item 5 — the wording exceeded what the fired branch tests.
- lines 165–167 and 3–12 (context, not a numeric error): the per-fold CV
  AUCs and the 0.9307 ± 0.0161 headline are best-epoch-on-the-reported-fold
  values (POST-HOC 1 above: fixed-epoch mean 0.9195 ± 0.0164); the
  sens 0.9028 / spec 0.7713 at the frozen threshold are in-sample
  (POST-HOC 2: nested mean 0.9003 ± 0.0569 / 0.7802 ± 0.1315).
- 2026-09-07 (P4c, redistribution ruling): three figures referenced above
  embed raw GDPH/SYSUCC images, a release without an explicit licence, and
  are therefore RETAINED LOCALLY BUT EXCLUDED FROM DISTRIBUTION (untracked;
  listed in .gitignore): reports/gradcam_external_fp.png (Grad-CAM gallery
  section, line 275), reports/phash_cross_pairs.png and
  reports/phash_within_d8_sample.png (protocol (g)/(e), lines 149/111 of
  data/external_protocol.md). Tracked replacements built from saved
  artifacts only: reports/gradcam_external_fp_breast.png (same gallery
  design on the 8 highest-calibrated-prob BrEaST benign false positives,
  CC BY 4.0; src/explain_breast_fp.py), reports/phash_cross_pairs_table.csv
  + reports/phash_cross_pairs_busbra_thumb.png (the two d = 8 cross-set
  candidates with their protocol (g) adjudication; BUS-BRA-side thumbnail
  only), reports/phash_within_d8_table.csv (the four sampled SYSUCC
  within-set pairs, distances and dedup action; src/phash_tables.py). The
  qualitative reading of the GDPH/SYSUCC FP CAMs (lines 300–307) stands as
  recorded; it can no longer be checked from the distributed repository,
  only from the author's local copy. GDPH/SYSUCC filenames in tracked CSVs
  are retained as identifiers (THIRD_PARTY_DATA.md §4).
- 2026-09-07 (P6, Space warm latency): line 416 records a single warm median
  of 6.8 s (2026-08-30). Two further measurements exist: 6.15 s (independent
  audit, 2026-09-06, docs/AUDIT_2026-09-06.md §6) and 8.62 s (median of 6
  warm calls by scripts/verify_space.py on 2026-09-07 after the P1 re-push,
  cpu-basic). All documents now report the warm latency as a range,
  6–9 s across three measurements (6.15 / 6.8 / 8.6), instead of a single
  value; the 2026-08-30 number stands as recorded above.

## Post-hoc analyses responding to the 2026-09-11 re-audits

**POST-HOC — not pre-registered.** Run 2026-09-12 (plan.md P8') from
committed artifacts only: no inference, no retraining, no change to any
frozen model, calibration, threshold or committed result file. The two
independent re-audits are archived unmodified at
docs/AUDIT2_claude_2026-09-11.md and docs/AUDIT2_astra_2026-09-11.md; the
item-by-item response is docs/AUDIT2_RESPONSE.md. Reproduction asserts
named below are executed by the scripts before anything is written.

### POST-HOC 5 — Predictor change vs cohort change in the internal→external shift (Astra §2.1, §7)

Script: src/posthoc_predictor_shift.py · CSVs:
reports/posthoc_predictor_shift.csv, reports/posthoc_predictor_shift_members.csv
· inputs: reports/oof_vit_preds.csv, reports/v2_members_{cohort}.csv
(Amendment 3 (n) member probabilities), reports/external_{cohort}_preds.csv,
models/calibration.json, models/operating_point.json — frozen T and
threshold applied unchanged. Asserts: the pooled-OOF row reproduces
models/operating_point.json (548/290/59/978, AUC 0.9254); the ensemble
rebuilt from the members with the frozen reduction order reproduces the
external-v1 decisions on 252/379/810/1013 images (max raw gap 3e-8); the
fold-5 single-model rows reproduce reports/v2_loco_summary.csv.

Why: every "internal reference" number in this file (pooled OOF AUC 0.9254,
benign median 0.0625, sens 0.9028 / spec 0.7713 at 0.2683) describes ONE
HELD-OUT CHECKPOINT PER IMAGE + hflip TTA + T — not the deployed
5-checkpoint ensemble, which has four in-fold members for every internal
image and therefore no unbiased internal estimate (line 169–171). T and
the threshold were fitted on that single-checkpoint distribution and
applied to the ensemble externally, so an internal→external comparison
changes the predictor as well as the cohort. This table separates the two.

| cohort | predictor | n | AUC | benign median p_cal | sens | spec |
|---|---|---|---|---|---|---|
| internal (BUS-BRA, pooled OOF) | one held-out ckpt per image + TTA (the file's internal reference) | 1875 | 0.9254 | 0.0625 | 0.9028 | 0.7713 |
| internal (BUS-BRA fold 5 OOF) | fold-5 single ckpt + TTA | 383 | 0.9234 | 0.0273 | 0.8926 | 0.8168 |
| BrEaST | fold-5 single ckpt + TTA | 252 | 0.8467 | 0.2789 | 0.9388 | 0.4740 |
| BrEaST | 5-ckpt ensemble + TTA (deployed; external-v1) | 252 | 0.8542 | 0.3061 | 0.9184 | 0.4091 |
| BUSI | fold-5 single ckpt + TTA | 379 | 0.9069 | 0.1501 | 0.9387 | 0.6065 |
| BUSI | 5-ckpt ensemble + TTA (deployed; external-v1) | 379 | 0.9339 | 0.1739 | 0.9571 | 0.6296 |
| GDPH | fold-5 single ckpt + TTA | 810 | 0.8899 | 0.3393 | 0.9520 | 0.4115 |
| GDPH | 5-ckpt ensemble + TTA (deployed; external-v1) | 810 | 0.9154 | 0.2894 | 0.9707 | 0.4529 |
| SYSUCC | fold-5 single ckpt + TTA | 1013 | 0.8319 | 0.3178 | 0.9337 | 0.4498 |
| SYSUCC | 5-ckpt ensemble + TTA (deployed; external-v1) | 1013 | 0.8380 | 0.2869 | 0.9309 | 0.4740 |

Per-member specificity / benign median at 0.2683 (m1 … m5, then the
ensemble), reports/posthoc_predictor_shift_members.csv: BrEaST spec
0.584 / 0.494 / 0.526 / 0.578 / 0.474 → 0.409, benign median 0.208 / 0.272
/ 0.210 / 0.229 / 0.279 → 0.306; BUSI 0.833 / 0.713 / 0.787 / 0.787 / 0.606
→ 0.630, 0.046 / 0.153 / 0.039 / 0.098 / 0.150 → 0.174; GDPH 0.644 / 0.524
/ 0.729 / 0.605 / 0.411 → 0.453, 0.125 / 0.258 / 0.082 / 0.206 / 0.339 →
0.289; SYSUCC 0.668 / 0.543 / 0.592 / 0.633 / 0.450 → 0.474, 0.109 / 0.237
/ 0.132 / 0.174 / 0.318 → 0.287.

Reading (factual): with the predictor held fixed (fold-5 single checkpoint
+ TTA), moving from internal fold-5 OOF to the external cohorts raises the
benign median calibrated probability from 0.027 to 0.150–0.339 and lowers
specificity from 0.817 to 0.41–0.61 (cohort-change component: benign
median +0.12 to +0.31, spec −0.21 to −0.41). On the same external cohort,
replacing that single checkpoint by the deployed 5-checkpoint ensemble
moves the benign median by −0.050 to +0.027 and specificity by −0.065 to
+0.041 (predictor-change component; the ensemble is less specific than
fold 5 on BrEaST and more specific on BUSI/GDPH/SYSUCC). The cohort
component is an order of magnitude larger than the predictor component on
every cohort, so the qualitative external-v1 reading (specificity collapse
under domain shift) survives the predictor change — but the internal
reference row (benign median 0.0625, spec 0.7713) is neither the ensemble
nor the fold-5 model and should not be read as a same-predictor baseline
for any external number. Two further facts: fold 5 is the least specific of
the five single checkpoints on all four external cohorts, so the Amendment
4 (t) comparator is the most pessimistic single member; and the ensemble's
specificity lies at or below the least specific member on BrEaST and within
0.02–0.04 of it elsewhere (averaging five members does not average their
specificities at a fixed threshold).

### POST-HOC 6 — Threshold rule as implemented; M1 counterfactual; Platt slopes (Astra §2.5)

Script: src/posthoc_threshold_rule.py · CSVs: reports/posthoc_threshold_rule.csv,
reports/posthoc_threshold_rule_M1.csv, reports/posthoc_platt_slopes.csv ·
inputs: reports/oof_vit_preds.csv, reports/external_*_preds.csv,
reports/v2_recalib_M1_draws.csv (read only), models/*.json.

**6a — the rule.** pick_threshold.pick_operating_point takes "the highest
threshold with sens ≥ 0.90" AMONG THE OPERATING POINTS RETURNED BY
sklearn.metrics.roc_curve WITH ITS DEFAULT drop_intermediate=True, which
drops collinear ROC points. The exhaustive rule — the highest observed
score with sens ≥ 0.90, i.e. the ⌈0.9·n_pos⌉-th largest positive score
(cross-checked against roc_curve(drop_intermediate=False)) — differs on the
pooled OOF and on two of the four external oracles. Specificity is
identical in every case; the difference is which borderline positive(s)
sit above the threshold. The frozen 0.26832 is RETAINED as frozen; every
"frozen rule" citation in this file means the routine as implemented.

| data | thr routine (frozen) | thr exhaustive | tp/fp/fn/tn routine → exhaustive | sens routine → exhaustive | spec | images differing |
|---|---|---|---|---|---|---|
| internal pooled OOF (n=1875) | 0.26832 | 0.26878 | 548/290/59/978 → 547/290/60/978 | 0.9028 → 0.9012 | 0.7713 | 1 |
| BrEaST full-cohort oracle | 0.37553 | 0.37983 | 90/58/8/96 → 89/58/9/96 | 0.9184 → 0.9082 | 0.6234 | 1 |
| BUSI full-cohort oracle | 0.38508 | 0.38508 | 147/39/16/177 (same) | 0.9018 | 0.8194 | 0 |
| GDPH full-cohort oracle | 0.44840 | 0.44840 | 338/98/37/337 (same) | 0.9013 | 0.7747 | 0 |
| SYSUCC full-cohort oracle | 0.33136 | 0.33504 | 656/125/68/164 → 652/125/72/164 | 0.9061 → 0.9006 | 0.5675 | 4 |

**6b — Amendment 2 M1 counterfactual.** The 11,000 committed M1 draws
(same seeds, same k-sets) re-run with the exhaustive rule; the committed
reports/v2_recalib_M1_{draws,summary}.csv are unchanged. Thresholds change
on 2,432/11,000 draws — BrEaST 259, BUSI 372, GDPH 604, SYSUCC 1,197 — and
on none of the k=10 draws (at k=10 roc_curve drops nothing). Largest
single-draw held-out change: spec 0.393 / 0.213 / 0.272 / 0.521, sens
0.284 / 0.281 / 0.290 / 0.234 (BrEaST / BUSI / GDPH / SYSUCC; maxima across
draws, not typical effects). Medians move by ≤ 0.013 spec and ≤ 0.019 sens
on BrEaST/BUSI/GDPH and by up to +0.041 spec / −0.019 sens on SYSUCC
(k=20). **k\* is unchanged under the exhaustive rule: GDPH 10, SYSUCC 10,
BrEaST 20, BUSI 20.** Per-(cohort, k) counts and medians under both rules
are in reports/posthoc_threshold_rule_M1.csv.

**6c — M3 Platt slopes.** The "structural equivalence" of M2b/M3 with M1
(lines 535–542) holds for POSITIVE-slope monotone transforms only.
Refitting every k-set's unconstrained Platt fit (same draws): negative
slope — a rank-reversing map — on 12 / 4 / 3 / 16 draws at k=10 (BrEaST /
BUSI / GDPH / SYSUCC), 1 draw at k=20 (SYSUCC), 0 at k ≥ 30. Those draws
are not explained by fit-failure fallback or ties; they are the third
source of M3–M1 departures at small k.

## Errata (2026-09-12)

Appended after the two re-audits of 2026-09-11. No line above the
"Errata (2026-09-07)" heading has been edited; `git diff 2cbe1da HEAD --
RESULTS.md` still removes zero lines. Each entry quotes the line as it
stands and states what it should say.

- lines 32–33: 'convnext_small and vit_b16 are statistically
  indistinguishable on AUC' → 'the CV means differ by 0.0002; no
  equivalence test was performed' (Astra §7).
- line 126: "T ≈ 2.36 means the ensemble's raw probabilities were
  substantially overconfident" → the pooled-OOF probabilities that T was
  fitted on are ONE held-out checkpoint per image + TTA, not the
  ensemble's (POST-HOC 5).
- lines 165–167, 190–191, 221 (and the same numbers wherever "internal
  reference" appears): pooled OOF AUC 0.9254, benign median 0.0625, sens
  0.9028 / spec 0.7713 describe one held-out checkpoint per image + hflip
  TTA + T. The deployed 5-checkpoint ensemble has no unbiased internal
  estimate (lines 169–171 say so); T and the threshold were fitted on the
  single-checkpoint distribution and applied to the ensemble externally,
  so the internal→external operating-point comparison changes the
  predictor as well as the cohort — POST-HOC 5 separates the components.
  models/operating_point.json `probability_space` ("hflip-TTA ensemble
  prob") and the title of reports/roc_oof_operating_point.png carry the
  same mislabel; both are frozen artifacts and are not edited.
- lines 142, 163–164, 441, 1136–1137 and every "highest threshold with
  sens ≥ 0.90": as implemented, the highest such point AMONG sklearn
  roc_curve(drop_intermediate=True) operating points; the exhaustive
  highest qualifying score is 0.26878 (one image; identical specificity);
  the frozen 0.26832 stands (POST-HOC 6a). src/pick_threshold.py's
  docstring now says so; the JSON is unchanged.
- lines 441–447 (M1) and Amendment 2 (k): the "frozen rule" used inside
  the 11,000 draws is the routine above; the exhaustive rule would change
  2,432 selected thresholds with k\* unchanged (POST-HOC 6b).
- line 483: 'bands only become usable at k ≈ 100–200' → unsupported as
  written: the POST-HOC draw-level bar (k_reliable, 2.5th-percentile
  recovery ≥ 0.5, lines 509–512) is reached only on GDPH at k=200 and not
  reached on BrEaST/BUSI (grid to k=100) or SYSUCC (to k=200), lines
  516–531. The sentence should read 'the POST-HOC k_reliable bar is met
  only on GDPH at k=200' (Claude §2 item 2.7, new item 6).
- lines 467 and 487: '0.568' → 0.567 (164/289 = 0.5675, rounds down;
  Astra §2.5 item 4).
- lines 535–542: 'M2b and M3 are M1 in disguise … monotone transforms …
  differences at k ≤ 30 are only the M2 fit-failure fallbacks and Platt
  ties' → holds for POSITIVE-slope transforms; unconstrained Platt fits
  had negative slope (rank reversal) on 12/4/3/16 draws at k=10 and 1 at
  SYSUCC k=20 (POST-HOC 6c), a third source of departure.
- lines 584–585 and 595: 'within 0.03 / 0.04 of the internal row' is the
  M2-row SPECIFICITY difference; the temperatures 2.27 (BrEaST) and 2.23
  (SYSUCC) differ from the internal 2.3644 by 0.09 and 0.14. docs/report.md
  §3.8 had mis-stated the latter as 0.03–0.04 (corrected).
- line 1126: 'the fold ranking is unchanged' → FALSE: at the best epoch
  the ranking is 1 > 2 > 5 > 3 > 4 (0.9526 / 0.9411 / 0.9272 / 0.9193 /
  0.9132); at epoch 18 it is 1 > 2 > 3 > 5 > 4 (0.9419 / 0.9283 / 0.9193 /
  0.9069 / 0.9012) — folds 3 and 5 swap (table lines 1117–1121; Astra
  §2.2). lines 1128–1130 'if anything, slightly understated' → not
  established: fold 3's zero is by construction and says nothing about
  the direction of the mean.
- lines 1102–1130 (POST-HOC 1) and the word 'optimism' in lines 1115,
  1122, 1124: the 0.0111 ± 0.0073 is the sensitivity of the reported CV
  AUC to epoch selection, with the fixed epoch (18) chosen post hoc as
  the median of the five best epochs from the same validation histories;
  it is not an unbiased estimate of optimism (Astra §2.2, §7).
- lines 1132–1161 (POST-HOC 2, 'Nested (out-of-sample) internal operating
  point'): → 'leave-fold-out threshold sensitivity on the fixed OOF
  artifact'. Only the threshold-fitting rows are held out; the OOF scores
  on folds ≠ k come from checkpoints TRAINED on fold k, and the fold-k
  checkpoint was selected at its best-val-AUC epoch on fold k; there is
  no outer retraining loop, so this is not a nested out-of-sample model
  evaluation (Astra §2.2, §7). src/posthoc_nested_threshold.py's
  docstring now says so; the file and CSV names are kept for continuity.
  line 1157: '0.236' → 0.235 (0.23545, rounds down).
- lines 379–381: 'deploy/ … is the Space repo verbatim' → true on
  2026-08-30 and again since the 2026-09-12 re-push (Space sha 7d41f22);
  false between P3 (2026-09-07 12:12) and 2026-09-12: the live Space ran
  the P1 inference.py (no hf_repo_path, the "1879" docstring) and the P1
  README (no licence sentence) because the P3/P4b changes were never
  pushed (Claude §6(e)). verify_space.py after the re-push: max |Δ|
  1.07e-07 on the four examples, warm median 6.27 s.
- line 403 (already errata'd 2026-09-07): src/inference.py:153 and
  deploy/inference.py:153 carried the same '1879' until 2026-09-12 (now
  1875); the live Space shipped it until the same re-push.
- line 416 and Errata 2026-09-07 (warm latency 'range 6–9 s across three
  measurements'): three further warm medians exist — 7.18 s (Claude
  re-audit, 2026-09-11), 4.79 s (Astra re-audit, 2026-09-11, post-first-call
  median of three) and 6.27 s (P8' re-push, 2026-09-12) — six measured
  medians 4.8–8.6 s; documents now say 'about 5–9 s'.
- lines 807–811 (Q2b operating point 'sens 0.9012 … PPV 0.6044'): the
  stored threshold 0.20070531964302063 applied to the committed
  reports/v2_biomedclip_oof_preds.csv (float32 probabilities) gives
  546/358/61/910, sens 0.8995; the freeze script picked it on in-memory
  float64 probabilities where the confusion is 547/358/60/910. One image
  at the float32 round-trip boundary; models/v2_biomedclip_operating_point.json
  is a frozen artifact and is not edited; the Q2 verdict (NOT CLAIMED) is
  unaffected (Claude §2).
- lines 935–940 (LOCO run 3, GDPH): the training was launched twice at
  commit eeca3fc — wandb k896krp6 aborted at epoch 3 (val AUC 0.8948, no
  summary) and 3jn4cicy 19 min later produced the checkpoint. The
  held-out cohort is untouched during --train (restart, not a peek), but
  the aborted attempt was unrecorded (Claude §1(d)).
- lines 875, 882–884 (LOCO): the 85/15 external slices are image-level
  (stated) and are NOT persisted in any artifact; they are reproducible
  only by re-running the seeded code (src/v2_data.py; tests assert
  determinism) (Claude §1(e)).
- lines 176–178, 208–210 ('ONE run … no second run'): true by git
  history (one adding commit per CSV, never modified). The script's
  --confirm was an intent flag and did not refuse existing outputs at the
  time; an output-exists refusal (--overwrite required) was added on
  2026-09-12 as post-hoc enforcement (Astra §4).
- lines 300–307 (external FP CAM tally '7/8'): a qualitative tally on the
  eight selected images, not a localisation measurement; the figure is
  not distributed (Errata 2026-09-07), so the tally cannot be checked
  from the repository. Stands as recorded, labelled as such.
- lines 193–194 ('BUSI and GDPH within or near the internal range'): the
  'internal range' was undefined; the pooled OOF 0.9254 is a point
  estimate of a different predictor. Descriptive: the BrEaST and SYSUCC
  external AUC CIs lie below 0.9254, the BUSI and GDPH CIs contain it
  (Claude §7).
- lines 195–198 and 227–233 (external-v1 reading): 'sensitivity stays ≥
  0.90' is correct (0.918–0.971; 76 false negatives across the four
  cohorts: 8/7/11/50), and the errors are predominantly, not exclusively,
  false positives (561). The report/README sentence 'false positives, not
  missed cancers' was wrong and is corrected (Astra §2.3, §7).
- data/external_protocol.md (h) and line 240–244: already recorded — the
  stray 'c' is in the reader2 column.
- Reader information (docs/report.md §4.2, not this file): the sentence
  'readers had the full examination' had no source; the HoVer-Trans
  release does not document whether the reader columns come from
  clinical reads or the paper's single-image reader study (Mo et al.
  §IV-C, §V-C) — corrected in the report.
- lines 387–389 (self-check 'byte-identical output'): holds on the MPS path
  the artifacts were produced with. A CPU run reproduces the AUC
  0.9234433158791243 but flags 155/383 images at the threshold instead of
  156 (docs/AUDIT2_astra_2026-09-11.md §6): device/batch reductions can
  move one borderline decision, so 'byte-identical' is a statement about
  the tested device, not about every runtime.

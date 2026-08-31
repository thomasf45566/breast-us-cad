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

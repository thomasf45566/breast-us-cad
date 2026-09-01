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

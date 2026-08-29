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

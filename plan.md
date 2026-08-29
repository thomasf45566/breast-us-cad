# plan.md — BreastUS-CAD task state
Read this at session start. Update the checkboxes when a task completes.

## Current phase: Step 8 (model improvement) → freeze → external validation

## Done
- [x] Env, repo, data (BUS-BRA / BrEaST / BUSI downloaded)
- [x] data.py + patient-level fold checks + EDA notebook
- [x] Baseline effb0 fold-5: AUC 0.8823 (tag: baseline-v1)
- [x] 5-fold CV effb0: AUC 0.891 ± 0.027 (tag: cv-v1)
- [x] Backbone screening fold-5: convnext_small 0.9257, vit_b16 0.9293
- [x] 5-fold CV convnext_small: AUC 0.930 ± 0.017 (summary in reports/cv_convnext_summary.csv)
- [x] 5-fold CV vit_b16: AUC 0.931 ± 0.016 (comparison table in RESULTS.md)
- [x] Backbone comparison table + interpretability audit (RESULTS.md)
- [x] Backbone decision: vit_base_patch16_224 (see standing decisions)
- [x] Pre-registered CoarseDropout ViT CV: cdrop discarded per pre-registered
      rule; plain cv_vit is the frozen backbone artifact set
      (models/cv_vit_fold{1-5}.pt). Details in RESULTS.md.
- [x] TTA (hflip only) on plain cv_vit CV ckpts: pooled OOF AUC 0.9231 →
      0.9254 with TTA (mean fold 0.9307 → 0.9323, 3/5 folds improved).
      src/tta_eval.py, `--tta` on evaluate.py, table in RESULTS.md, OOF
      preds (both settings) in reports/oof_vit_preds.csv.

## Next (strict order)
1. calibrate.py: pooled OOF preds → temperature scaling
2. pick_threshold.py: sens ≥ 0.90 operating point on calibrated OOF
3. FREEZE: tag frozen-v1 (5 ckpts + TTA setting + calibration.json +
   operating_point.json). After this tag, NO model changes.
4. External validation (ONE run only): dedup_busi.py → external_val.py
   on BrEaST + cleaned BUSI, ensemble of 5 ckpts, frozen threshold.
   Results go to RESULTS.md regardless of outcome. Never re-tune after.
5. Grad-CAM gallery (TP/TN/FP/FN × 4)
6. Segmentation: U-Net on BUS-BRA masks (fallback: MedSAM2 zero-shot)
7. Gradio app → HF Spaces
8. README / slides / one-pager / rehearsal

## Standing decisions (do not relitigate)
- Patient-level splits only; BUS-BRA official folds
- BrEaST + BUSI are external-only, single-shot evaluation
- AUC is the primary metric; operating point from pooled OOF, frozen
- Numbers live in RESULTS.md; every result maps to a commit
- Winner backbone: vit_base_patch16_224 (AUC tied with convnext, 9x faster CPU
  inference, saliency usable at blocks[-2].norm1)
- Saliency method for demo/slides: Grad-CAM at blocks[-2].norm1. Attention
  rollout is analysis-only, never in the demo.
- Caliper/text inpainting: rejected for this project scope → future work
  (NTUH data collection will export annotation-free images)
- ONE pre-registered robustness experiment before TTA: CoarseDropout ViT CV.
  Adoption rule, decided BEFORE seeing results: adopt only if pooled OOF AUC
  >= (current ViT OOF AUC - 0.01) AND the saliency check shows visibly reduced
  caliper-adjacent heat on malignant TPs. Otherwise discard, no iteration,
  no second variant. Either way, next step is TTA on whichever ViT wins.
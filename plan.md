# plan.md — BreastUS-CAD task state
Read this at session start. Update the checkboxes when a task completes.

## Current phase: Step 8 (model improvement) → freeze → external validation

## Done
- [x] Env, repo, data (BUS-BRA / BrEaST / BUSI downloaded)
- [x] data.py + patient-level fold checks + EDA notebook
- [x] Baseline effb0 fold-5: AUC 0.8823 (tag: baseline-v1)
- [x] 5-fold CV effb0: AUC 0.891 ± 0.027 (tag: cv-v1)
- [x] Backbone screening fold-5: convnext_small 0.9257, vit_b16 0.9293
- [x] 5-fold CV convnext_small
- [x] 5-fold CV vit_b16: AUC 0.931 ± 0.016 (comparison table in RESULTS.md)

## Next (strict order)
1. Backbone comparison table → USER decides winner (not Claude)
2. TTA (hflip only) on winner's CV checkpoints, pooled OOF AUC
3. calibrate.py: pooled OOF preds → temperature scaling
4. pick_threshold.py: sens ≥ 0.90 operating point on calibrated OOF
5. FREEZE: tag frozen-v1 (5 ckpts + TTA setting + calibration.json +
   operating_point.json). After this tag, NO model changes.
6. External validation (ONE run only): dedup_busi.py → external_val.py
   on BrEaST + cleaned BUSI, ensemble of 5 ckpts, frozen threshold.
   Results go to RESULTS.md regardless of outcome. Never re-tune after.
7. Grad-CAM gallery (TP/TN/FP/FN × 4)
8. Segmentation: U-Net on BUS-BRA masks (fallback: MedSAM2 zero-shot)
9. Gradio app → HF Spaces
10. README / slides / one-pager / rehearsal

## Standing decisions (do not relitigate)
- Patient-level splits only; BUS-BRA official folds
- BrEaST + BUSI are external-only, single-shot evaluation
- AUC is the primary metric; operating point from pooled OOF, frozen
- Numbers live in RESULTS.md; every result maps to a commit
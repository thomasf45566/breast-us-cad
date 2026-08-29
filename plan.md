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
- [x] calibrate.py: temperature scaling on pooled OOF TTA preds. T = 2.3644,
      ECE 0.0716 → 0.0401 (15 bins), NLL 0.4411 → 0.3237, AUC unchanged
      0.9254. models/calibration.json, reports/reliability_oof_vit_tta.png.
- [x] pick_threshold.py: threshold 0.2683 on calibrated OOF (sens ≥ 0.90,
      max spec). Sens 0.9028 (CI 0.874–0.928), spec 0.7713 (CI 0.745–0.798),
      PPV 0.654, NPV 0.943; patient-level bootstrap ×2000.
      models/operating_point.json, reports/roc_oof_operating_point.png.
- [x] FREEZE: tag frozen-v1 — 5 ckpts + hflip TTA + calibration.json
      (T=2.3644) + operating_point.json. NO model changes after this tag.
      Artifact list in RESULTS.md "FROZEN MODEL" section.
- [x] External validation PRE-FLIGHT (no external inference run):
      data/external_protocol.md (labels, exclusions, preprocessing, metrics
      — written before any scoring); dedup_busi.py → busi_clean.csv
      (386 → 379: 7 near-dups dropped, pHash d ≤ 8, histogram + pair grid
      in reports/, no BUSI↔BrEaST cross-dups, min d = 10); external_val.py
      frozen pipeline self-check PASSED (fold-5 TTA AUC 0.9234 reproduced
      digit-for-digit, reports/external_selfcheck.txt). Run needs --confirm.

## Next (strict order)
1. External validation — THE RUN (ONE run only):
   `python src/external_val.py --dataset all --confirm` on BrEaST (252 imgs,
   4 normals excluded) + cleaned BUSI (379 imgs). Frozen ensemble +
   threshold. Results go to RESULTS.md regardless of outcome. Never
   re-tune after.
2. Grad-CAM gallery (TP/TN/FP/FN × 4)
3. Segmentation: U-Net on BUS-BRA masks (fallback: MedSAM2 zero-shot)
4. Gradio app → HF Spaces
5. README / slides / one-pager / rehearsal

## Standing decisions (do not relitigate)
- TTA ADOPTED (2026-08-29): hflip TTA + 5-model ensemble is the frozen
  inference pipeline — all downstream evaluation, thresholding, external
  validation, and the demo app use it. Numbers in RESULTS.md TTA section.
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
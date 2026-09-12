# BreastUS-CAD

Breast ultrasound benign/malignant classifier. Research prototype for a
residency interview. NOT a medical device — never imply diagnostic use
in code, comments, or UI text.

## Stack
- macOS Apple Silicon (M4), Python 3.12, uv, PyTorch with MPS backend
- timm (classification), segmentation-models-pytorch (U-Net), MONAI utils
- albumentations, scikit-learn, wandb, gradio
- device = "mps"; PYTORCH_ENABLE_MPS_FALLBACK=1 is set in the shell
  environment (not in code); no float64 tensors

## Golden rules
1. PATIENT-LEVEL SPLITS ONLY. Never split by image. BUS-BRA official
   folds are the single source of truth: data/splits/busbra_official_5fold.csv
   (verbatim copy of the dataset's 5-fold-cv.csv, CC BY 4.0, notice in
   data/splits/BUSBRA_LICENSE.txt), sha256-verified by data.resolve_fold_file,
   with data/raw/busbra/5-fold-cv.csv as the fallback. The other
   data/splits/*.csv are the frozen external keep-lists. Scope: this is
   the v1 line and every BUS-BRA split. The v2 LOCO training-side 85/15
   slices of the external cohorts are IMAGE-level (no patient IDs exist;
   Amendment 4 (t)) — never describe v2 as patient-level.
2. BrEaST, BUSI, GDPH and SYSUCC are EXTERNAL VALIDATION ONLY for the
   v1 line — never train or tune on them, never peek at their metrics
   before the model is frozen. v2 LOCO training on external cohorts is
   permitted only under data/external_protocol.md Amendment 4.
3. Every training run logs to wandb. Every eval prints AUC, sensitivity,
   specificity, and saves ROC + confusion matrix to reports/.
4. Before claiming any task is done: run the relevant script end-to-end
   and show me the actual output (metrics, file paths, screenshots).
5. Keep functions small; config in configs/*.yaml; no hardcoded paths.
6. Grayscale ultrasound → replicate to 3 channels for ImageNet backbones.
7. At session start, read plan.md. When a task in plan.md completes,
   update its checkbox in the same commit as the work itself.
8. src/external_val.py refuses to overwrite an existing
   reports/external_*_preds.csv (needs --overwrite). Never pass
   --overwrite for the v1 line; the four external-v1 files are final.
9. The internal reference (pooled OOF 0.9254, benign median 0.0625,
   sens 0.9028 / spec 0.7713) is ONE held-out checkpoint per image +
   hflip TTA, not the deployed 5-checkpoint ensemble — never call it the
   ensemble's internal estimate (RESULTS.md POST-HOC 5).

## Commands
- Train: python src/train.py --config configs/baseline.yaml
- Eval:  python src/evaluate.py --ckpt models/cv_vit_fold5.pt --split val [--tta]
         (--split accepts only `val`; external cohorts go through
         src/external_val.py)
- App:   python app/app.py
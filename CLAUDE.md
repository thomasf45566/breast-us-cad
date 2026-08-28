# BreastUS-CAD

Breast ultrasound benign/malignant classifier. Research prototype for a
residency interview. NOT a medical device — never imply diagnostic use
in code, comments, or UI text.

## Stack
- macOS Apple Silicon (M4), Python 3.12, uv, PyTorch with MPS backend
- timm (classification), segmentation-models-pytorch (U-Net), MONAI utils
- albumentations, scikit-learn, wandb, gradio
- device = "mps"; PYTORCH_ENABLE_MPS_FALLBACK=1 is set; no float64 tensors

## Golden rules
1. PATIENT-LEVEL SPLITS ONLY. Never split by image. BUS-BRA official
   folds are the single source of truth (data/splits/*.csv).
2. BrEaST and BUSI are EXTERNAL VALIDATION ONLY — never train or tune
   on them, never peek at their metrics before the model is frozen.
3. Every training run logs to wandb. Every eval prints AUC, sensitivity,
   specificity, and saves ROC + confusion matrix to reports/.
4. Before claiming any task is done: run the relevant script end-to-end
   and show me the actual output (metrics, file paths, screenshots).
5. Keep functions small; config in configs/*.yaml; no hardcoded paths.
6. Grayscale ultrasound → replicate to 3 channels for ImageNet backbones.
7. At session start, read plan.md. When a task in plan.md completes,
   update its checkbox in the same commit as the work itself.

## Commands
- Train: python src/train.py --config configs/baseline.yaml
- Eval:  python src/evaluate.py --ckpt models/best.pt --split test
- App:   python app/app.py
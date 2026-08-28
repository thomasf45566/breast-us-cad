# Results Log

| Date | Run | Model | Train/Val | AUC | Sens | Spec | Thr | Ckpt | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-26 | baseline_effb0_fold5 | effnet_b0 | folds 1-4 / 5 | 0.8823 | 0.769 | 0.840 | 0.505 (Youden) | baseline_effb0_fold5.pt | first full run, 30 ep, wandb t41mf6ys, tag baseline-v1 |
| 2026-08-27 | cv_effb0 (5-fold) | effnet_b0 | official 5-fold CV | 0.891 ± 0.027 | 0.769 ± 0.080 | 0.864 ± 0.064 | per-fold Youden | cv_effb0_fold{1-5}.pt | 30 ep/fold, wandb group cv_effb0, details in reports/cv_summary.csv |
| 2026-08-27 | convnext_small_fold5 | convnext_small | folds 1-4 / 5 | 0.9257 | 0.876 | 0.870 | 0.307 (Youden) | convnext_small_fold5.pt | screening, 30 ep, lr 1e-4, best ep 22 |
| 2026-08-27 | vit_b16_fold5 | vit_base_patch16_224 | folds 1-4 / 5 | 0.9293 | 0.884 | 0.847 | 0.788 (Youden) | vit_b16_fold5.pt | screening, 30 ep, lr 1e-4, warmup 3, best ep 17 |
| 2026-08-28 | cv_convnext (5-fold) | convnext_small | official 5-fold CV | 0.930 ± 0.017 | 0.852 ± 0.028 | 0.886 ± 0.047 | per-fold Youden | cv_convnext_fold{1-5}.pt | 30 ep/fold, lr 1e-4, wandb group cv_convnext, per-fold AUC 0.954/0.940/0.922/0.909/0.926, details in reports/cv_convnext_summary.csv |

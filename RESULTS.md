# Results Log

| Date | Run | Model | Train/Val | AUC | Sens | Spec | Thr | Ckpt | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-26 | baseline_effb0_fold5 | effnet_b0 | folds 1-4 / 5 | 0.8823 | 0.769 | 0.840 | 0.505 (Youden) | baseline_effb0_fold5.pt | first full run, 30 ep, wandb t41mf6ys, tag baseline-v1 |
| 2026-08-27 | cv_effb0 (5-fold) | effnet_b0 | official 5-fold CV | 0.891 ± 0.027 | 0.769 ± 0.080 | 0.864 ± 0.064 | per-fold Youden | cv_effb0_fold{1-5}.pt | 30 ep/fold, wandb group cv_effb0, details in reports/cv_summary.csv |

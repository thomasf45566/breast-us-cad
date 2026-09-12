---
title: BreastUS-CAD
emoji: 🔬
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# BreastUS-CAD — research prototype

> **Research prototype — not for clinical use. Not a medical device.**
> This demo exists for methods discussion only. It must not be used to
> inform any diagnostic or treatment decision.

Benign/malignant classifier for breast ultrasound images, trained on the
public **BUS-BRA** dataset (1,875 images / 1,064 patients, official
patient-level 5-fold splits) and externally validated single-shot on four
cohorts (BrEaST, BUSI, GDPH, SYSUCC; AUC 0.84–0.93).

**Pipeline (frozen-v1):** 5× ViT-B/16 fold checkpoints × {original,
horizontal flip} sigmoid probabilities averaged, temperature-scaled
(T fit on pooled out-of-fold predictions), decision threshold chosen for
sensitivity ≥ 0.90 on internal validation. The lesion contour comes from a
separate U-Net (orientation only) and the heatmap is a single-model
(fold-5) Grad-CAM visualization — not the ensemble decision path.

**Known limitations.** Calibrated probabilities shift under domain shift:
on external cohorts, sensitivity held ≥ 0.918 at the frozen threshold but
specificity dropped from 0.77 (internal, in-sample) to 0.41–0.63. The
operating point was fitted on internal out-of-fold predictions only and does
not transfer across scanners and populations. Preprocessing in this Space
differs from the validation pipeline on image sizes outside the BUS-BRA
range (documented; measured calibrated-probability impact max 0.0072,
median 0.0008, on the external validation images).

**Weights:** downloaded at startup from the companion model repo
([breast-us-cad-weights](https://huggingface.co/happytommy/breast-us-cad-weights)).

**Example image attribution:** the four bundled examples are from the
BUS-BRA dataset validation fold — W. Gómez-Flores, M. J. Gregorio-Calas,
W. Coelho de Albuquerque Pereira, "BUS-BRA: A Breast Ultrasound Dataset
for Assessing Computer-Aided Diagnosis Systems," *Medical Physics*, 2024.
Used under the dataset's research license (CC BY 4.0; the dataset's
permission notice is reproduced in the source repository at
`data/splits/BUSBRA_LICENSE.txt`). The code of this Space is Apache-2.0.

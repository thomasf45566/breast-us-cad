# Third-party data: licences, required citations, and derived files in this repository

Research prototype — not a medical device. No raw dataset is redistributed
here. This file lists, per dataset, the licence or policy under which the
data were obtained, the citation the data owner requires (verbatim from
data/README.md), and exactly which tracked files in this repository derive
from that dataset. The code in this repository is Apache-2.0 (LICENSE);
docs/ and the figures under reports/ authored here are CC BY 4.0 (README);
neither licence extends to the third-party material listed below, which
stays under its own terms.

Redistribution check of 2026-09-07 (audit-response phase P4b) and ruling
(P4c): every tracked file under data/, reports/, app/examples/,
deploy/examples/ and docs/assets/ was inspected. No tracked table
reproduces the per-image reader BI-RADS columns of GDPH&SYSUCC's
`BIRADS&FOLD.xlsx` or any column of BrEaST's clinical annotation table
beyond the binary benign/malignant label. The three figures that embedded
raw GDPH/SYSUCC images were untracked and replaced (§4); GDPH/SYSUCC
filename identifiers in tables are retained (§4). Details in
docs/AUDIT_RESPONSE_2026-09-06.md §J.

## 1. BUS-BRA (training + internal validation)

- **Source:** Zenodo record 8231412, https://zenodo.org/records/8231412 (downloaded 2026-08-26).
- **Licence:** CC BY 4.0 (Zenodo metadata). The bundled `LICENSE.txt`
  additionally grants the right to use, copy, modify, publish and
  distribute the dataset provided the notice is included and the paper
  below is cited; the notice is reproduced in `data/splits/BUSBRA_LICENSE.txt`.
- **Required citation (verbatim):**
  > Wilfrido Gómez-Flores, Maria Julia Gregorio-Calas, and Wagner Coelho de
  > Albuquerque Pereira, "BUS-BRA: A Breast Ultrasound Dataset for Assessing
  > Computer-aided Diagnosis Systems," *Medical Physics*, vol. 51,
  > pp. 3110–3123, 2024. DOI: 10.1002/mp.16812
- **Derived files tracked here (all permitted under CC BY 4.0 with the citation above):**
  - Verbatim partition file: `data/splits/busbra_official_5fold.csv` (= the dataset's `5-fold-cv.csv`), notice in `data/splits/BUSBRA_LICENSE.txt`.
  - Raw BUS-BRA images (4 of 1,875): `app/examples/*.png`, `deploy/examples/*.png` (same four), `docs/assets/demo.png` (screenshots of two of them).
  - Figures embedding BUS-BRA images: `reports/gradcam_gallery_internal.png`, `reports/gradcam_check_convnext.png`, `reports/gradcam_check_vit.png`, `reports/gradcam_check2_convnext.png`, `reports/gradcam_check2_vit.png`, `reports/gradcam_adoption_vit_cdrop.png`, `reports/artifact_audit_raw.png`, `reports/cdrop_sample.png`, `reports/seg_examples.png`, and one panel of `reports/phash_cross_pairs.png` (bus_0999-l).
  - Per-image tables carrying BUS-BRA image IDs, patient (Case) IDs and pathology labels: `reports/oof_vit_preds.csv`, `reports/v2_biomedclip_oof_preds.csv`, `reports/preds_baseline_effb0.csv`, `reports/preds_convnext_small_fold5.csv`, `reports/preds_vit_b16_fold5.csv`, `reports/preds_vit_b16_fold5_tta.csv`, `reports/seg_metrics_seg_unet_effb0.csv`, `reports/occlusion_border15.csv`, and the `busbra` rows of `reports/phash_sweep_hits.csv`.
  - Aggregate-only outputs (no per-image data): `reports/cv_*summary.csv`, `reports/tta_vit_summary.csv`, `reports/posthoc_*.csv`, all ROC/CM/reliability PNGs.

## 2. BrEaST — Breast-Lesions-USG (external validation #1)

- **Source:** The Cancer Imaging Archive, collection "BrEaST-Lesions-USG", version 1, https://www.cancerimagingarchive.net/collection/breast-lesions-usg/ (downloaded 2026-08-26).
- **Licence:** CC BY 4.0 (TCIA collection page); TCIA data-usage policy applies.
- **Required citation (verbatim):**
  > Pawłowska, A., Ćwierz-Pieńkowska, A., Domalik, A., Jaguś, D.,
  > Kasprzak, P., Matkowski, R., Fura, Ł., Nowicki, A., & Zolek, N. (2024).
  > *A Curated Benchmark Dataset for Ultrasound Based Breast Lesion Analysis
  > (Breast-Lesions-USG)* (Version 1) [Dataset]. The Cancer Imaging Archive.
  > DOI: 10.7937/9WKK-Q141
- **Derived files tracked here:**
  - No BrEaST image is tracked.
  - Per-case tables with case ID (`caseNNN`) and the binary label taken from the `Classification` column of the clinical table (benign/malignant only; no BI-RADS, shape, margin, echogenicity or any other annotation column): `reports/external_breast_preds.csv`, `reports/v2_biomedclip_external_breast_preds.csv`, `reports/v2_loco_breast_preds.csv`, `reports/v2_members_breast.csv`, the `breast` rows of `reports/v2_resize_dispatch_impact.csv`.
  - Aggregate-only: `reports/roc_external_breast.png`, `reports/cm_external_breast.png`, the v2 ROC/CM PNGs, `reports/v2_recalib_*`, `reports/v2_abstention_*`, `reports/v2_cross_site.csv`, `reports/v2_loco_*table/summary*.csv`.

## 3. Curated BUSI (external validation #2)

- **Source:** Zenodo record 19047974, "Curated BUSI dataset", version 1.0, https://zenodo.org/records/19047974 (downloaded 2026-08-26).
- **Licence:** CC BY 4.0.
- **Required citation (verbatim):**
  > Aumente-Maestro, C., Díez, J., & Remeseiro, B. (2026). *Curated BUSI
  > dataset — Curated Breast Ultrasound Images* (Version 1.0) [Dataset].
  > Zenodo. DOI: 10.5281/zenodo.19047974
- **Derived files tracked here (permitted under CC BY 4.0 with the citation above):**
  - Figure embedding 14 raw BUSI images (the seven near-duplicate pairs dropped by dedup): `reports/busi_dup_pairs.png`.
  - Keep-list with filename, class, label and perceptual hash: `data/splits/busi_clean.csv`.
  - Per-image tables (filename + label from the filename prefix): `reports/external_busi_preds.csv`, `reports/v2_biomedclip_external_busi_preds.csv`, `reports/v2_loco_busi_preds.csv`, `reports/v2_members_busi.csv`, the `busi` rows of `reports/phash_sweep_hits.csv`.
  - Aggregate-only: `reports/busi_phash_distances.png`, ROC/CM PNGs, v2 summaries.

## 4. GDPH & SYSUCC (external validation #3 and #4)

- **Source:** dataset released with the HoVer-Trans paper (Google-Drive style release: `GDPH/`, `SYSUCC/`, `BIRADS&FOLD.xlsx`); local copy `data/raw/gdph_sysucc`.
- **Licence / policy:** **no licence file accompanies the release**; the terms of the original release govern any reuse, and redistribution rights are not established. Treated here as "no redistribution of images or per-image annotations".
- **Required citation (verbatim):**
  > Mo, Y., Han, C., Liu, Y., Liu, M., Shi, Z., Lin, J., Zhao, B.,
  > Huang, C., Qiu, B., Cui, Y., Wu, L., Pan, X., Xu, Z., Huang, X.,
  > Li, Z., Liu, Z., Wang, Y., & Liang, C. (2023). "HoVer-Trans:
  > Anatomy-aware HoVer-Transformer for ROI-free Breast Cancer Diagnosis
  > in Ultrasound Images," *IEEE Transactions on Medical Imaging*.
  > DOI: 10.1109/TMI.2023.3236011
- **Derived files tracked here:**
  - **Figures that embedded raw GDPH/SYSUCC images — RULING (2026-09-07, P4c): excluded from distribution.**
    `reports/gradcam_external_fp.png`, `reports/phash_cross_pairs.png` and
    `reports/phash_within_d8_sample.png` were untracked (`git rm --cached`),
    added to `.gitignore`, and are retained only on the author's disk.
    Tracked replacements, built from saved artifacts: `reports/gradcam_external_fp_breast.png`
    (BrEaST false positives, CC BY 4.0), `reports/phash_cross_pairs_table.csv`
    + `reports/phash_cross_pairs_busbra_thumb.png` (BUS-BRA-side thumbnail only),
    `reports/phash_within_d8_table.csv`. Recorded in RESULTS.md Errata.
  - Per-image reader BI-RADS columns from `BIRADS&FOLD.xlsx`: **none tracked** (verified 2026-09-07 over every CSV/JSON header). `reports/posthoc_reader_concordance.csv` and `reports/birads_comparison_*.png` are aggregate only.
  - Per-image tables with the release's filenames and the class encoded in the filename (`benign(N)`/`malignant(N)`), plus model outputs: `data/splits/gdph_clean.csv`, `data/splits/sysucc_clean.csv`, `reports/external_gdph_preds.csv`, `reports/external_sysucc_preds.csv`, `reports/v2_biomedclip_external_{gdph,sysucc}_preds.csv`, `reports/v2_loco_{gdph,sysucc}_preds.csv`, `reports/v2_members_{gdph,sysucc}.csv`, the `gdph`/`sysucc` rows of `reports/v2_resize_dispatch_impact.csv` and `reports/phash_sweep_hits.csv`. **Ruling (2026-09-07, P4c): these identifiers are RETAINED for reproducibility.** They contain no image pixels and no content of the annotation table (`BIRADS&FOLD.xlsx`); the class token in each filename is part of the release's own file naming, not an added annotation. Anyone holding the release can join the tables back to the images; nobody can reconstruct any image or reader rating from them.
  - Aggregate-only: ROC/CM PNGs, `reports/prob_shift_external.png` (histograms), v2 summaries and curves.

## 5. Pretrained weights used as initialisation (not data)

- ImageNet-pretrained `vit_base_patch16_224` via timm (v1, LOCO) — timm's licence.
- BiomedCLIP ViT-B/16 image encoder (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) re-exported to `v2/pretrained/biomedclip_vitb16_timm.pt` on Hugging Face — subject to the BiomedCLIP licence.
- USFM (`openmedlab/USFM`, `USFM_latest.pth`): inspected only (loading spike), never trained on, not redistributed.

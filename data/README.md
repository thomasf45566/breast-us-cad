# Data Sources

This document records the provenance of all datasets used in this project.
All datasets were downloaded on **2026-08-26** and are distributed under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

Research use only. This project is a research prototype and NOT a medical
device; no dataset here is used for diagnostic purposes.

---

## 1. BUS-BRA (primary dataset — training)

- **Role:** Training and internal validation. The official patient-level
  folds (`data/splits/*.csv`) are the single source of truth for splits.
- **Source:** Zenodo record 8231412
- **URL:** https://zenodo.org/records/8231412
- **License:** CC BY 4.0
- **Downloaded:** 2026-08-26
- **Citation:**
  > Wilfrido Gómez-Flores, Maria Julia Gregorio-Calas, and Wagner Coelho de
  > Albuquerque Pereira, "BUS-BRA: A Breast Ultrasound Dataset for Assessing
  > Computer-aided Diagnosis Systems," *Medical Physics*, vol. 51,
  > pp. 3110–3123, 2024. DOI: [10.1002/mp.16812](https://doi.org/10.1002/mp.16812)

## 2. BrEaST (external validation #1)

- **Role:** External validation only — **never used for training or tuning**.
  Inference only; metrics are not inspected before the model is frozen.
- **Source:** The Cancer Imaging Archive (TCIA), collection
  "BrEaST-Lesions-USG" (Pawłowska et al., *Scientific Data*, 2024)
- **URL:** https://www.cancerimagingarchive.net/collection/breast-lesions-usg/
- **Version:** 1
- **License:** CC BY 4.0
- **Downloaded:** 2026-08-26
- **Citation:**
  > Pawłowska, A., Ćwierz-Pieńkowska, A., Domalik, A., Jaguś, D.,
  > Kasprzak, P., Matkowski, R., Fura, Ł., Nowicki, A., & Zolek, N. (2024).
  > *A Curated Benchmark Dataset for Ultrasound Based Breast Lesion Analysis
  > (Breast-Lesions-USG)* (Version 1) [Dataset]. The Cancer Imaging Archive.
  > DOI: [10.7937/9WKK-Q141](https://doi.org/10.7937/9WKK-Q141)

## 3. Curated BUSI (external validation #2)

- **Role:** External validation only — **never used for training or tuning**.
  The curated release is used because the original BUSI dataset contains
  duplicate/near-duplicate images that require deduplication.
- **Source:** Zenodo record 19047974 — "Curated BUSI dataset"
  (Aumente-Maestro et al.)
- **URL:** https://zenodo.org/records/19047974
- **Version:** 1.0
- **License:** CC BY 4.0
- **Downloaded:** 2026-08-26
- **Citation:**
  > Aumente-Maestro, C., Díez, J., & Remeseiro, B. (2026). *Curated BUSI
  > dataset — Curated Breast Ultrasound Images* (Version 1.0) [Dataset].
  > Zenodo. DOI: [10.5281/zenodo.19047974](https://doi.org/10.5281/zenodo.19047974)

---

## Usage rules (see CLAUDE.md)

1. Patient-level splits only; BUS-BRA official folds are authoritative.
2. BrEaST and Curated BUSI are strictly held out for external validation.

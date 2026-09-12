# BreastUS-CAD

> **Research prototype — not for clinical use. Not a medical device.**

Benign/malignant breast ultrasound classification with a frozen,
internally pre-registered, single-shot external validation across four
cohorts on three continents — plus deployment-oriented follow-up studies
(site-specific recalibration, multi-source training).

**[Live demo](https://huggingface.co/spaces/happytommy/breast-us-cad)** ·
[Weights](https://huggingface.co/happytommy/breast-us-cad-weights) ·
[Full report](docs/report.md) ·
[Independent audit](docs/AUDIT_2026-09-06.md) /
[response](docs/AUDIT_RESPONSE_2026-09-06.md) ·
Re-audits 2026-09-11 ([Claude](docs/AUDIT2_claude_2026-09-11.md),
[Astra](docs/AUDIT2_astra_2026-09-11.md)) /
[response](docs/AUDIT2_RESPONSE.md)

![demo screenshot](docs/assets/demo.png)

Malignant-class Grad-CAM localizes to the lesion body and margin
(bottom row). Benign-class Grad-CAM was diffuse in the examples inspected
— the model's 'benign evidence' appears to be the absence of a suspicious
focus — and can include the burned-in annotation text present in every
BUS-BRA image (top row). This is a documented limitation of the training
set; the border-occlusion probe in RESULTS.md (malignant fold-5 images
only) found a median probability drop below 0.01 with a heavy tail (about
16% of images above 0.2), so it bounds the effect rather than excluding it.
CAMs are single-model (fold-5) visualizations; decisions come from the
5-model ensemble.

## Headline results

| | AUC | Sens | Spec |
|---|---|---|---|
| Internal (a): single model, no TTA, 5-fold patient-level CV, best epoch on the reported fold | 0.9307 ± 0.0161 | — | — |
| Internal (a′, POST-HOC): same runs, val AUC at one fixed epoch (18) | 0.9195 ± 0.0164 | — | — |
| Internal (b): pooled OOF — one held-out checkpoint per image + hflip TTA + T (n=1875); the deployed 5-ckpt ensemble has no unbiased internal estimate | 0.9254 | 0.903 † | 0.771 † |
| Internal (b′, POST-HOC): leave-fold-out threshold sensitivity on the fixed OOF artifact — threshold from the other 4 folds, evaluated per fold (not a nested model evaluation) | — | 0.900 ± 0.057 | 0.780 ± 0.132 |
| BrEaST (Poland, n=252) | 0.854 | 0.918 | 0.409 |
| BUSI cleaned (Egypt, n=379) | 0.934 | 0.957 | 0.630 |
| GDPH (China, n=810) | 0.915 | 0.971 | 0.453 |
| SYSUCC (China, n=1013) | 0.838 | 0.931 | 0.474 |

Rows (a) and (b) are two different predictors, and neither is the
deployed ensemble. In (a) each fold's AUC is that of the checkpoint saved
at the epoch with the best AUC on the same held-out fold (`best_epoch` in
`reports/cv_vit_summary.csv`), so the CV mean is optimistically biased;
row (a′) is a post-hoc sensitivity of that mean to epoch selection (fixed
epoch 18 = the median of the five best epochs, chosen after seeing the
curves; 0.011 ± 0.007 AUC lower) — a sensitivity analysis, not an
unbiased optimism estimate. Row (b) pools, for each image, the prediction
of the ONE checkpoint that held that image out (+ hflip TTA); the deployed
5-checkpoint ensemble scores every internal image with four in-fold
members and therefore has no unbiased internal estimate. The frozen T and
the threshold 0.2683 were fitted on this single-held-out-checkpoint
distribution and are applied to the ensemble externally, so every
internal→external operating-point comparison below changes the predictor
as well as the cohort (RESULTS.md POST-HOC 5 separates the two
components: the cohort component dominates). † Sens/spec in (b) are
**in-sample**: the threshold (sens ≥ 0.90 rule as implemented — see
RESULTS.md POST-HOC 6) was fitted on the same calibrated pooled-OOF
predictions it is evaluated on. Row (b′) re-picks the threshold on the
other four folds and evaluates on the fifth (thresholds 0.235–0.311;
held-out sens misses 0.90 on 3/5 folds), but the scores it re-uses come
from checkpoints trained on the evaluated fold, so it is a
threshold-selection sensitivity analysis, not a nested out-of-sample
estimate. All POST-HOC rows: RESULTS.md "Post-hoc analyses". External
rows: single-shot at that frozen threshold, never re-tuned.

**Key finding:** discrimination transfers with cohort-dependent loss
(external AUC 0.838–0.934 against a pooled-OOF 0.9254 that belongs to a
different predictor; the BrEaST and SYSUCC CIs lie below 0.9254, the BUSI
and GDPH CIs contain it — a description, not a test); the operating point
does not transfer. The frozen threshold held sensitivity 0.918–0.971 on
every external cohort (76 false negatives among 1,360 malignant images:
8 / 7 / 11 / 50) while specificity dropped from 0.771 (in-sample
internal, single-held-out-checkpoint predictor) to 0.409–0.630 (561 false
positives, deployed ensemble): the shift adds false positives far more
than false negatives, but cancers are still missed — whether that error
profile is "safe" was not assessed (no harm analysis). Follow-up
studies: re-selecting only the threshold on 10–20 local labels recovers
near-oracle specificity **in the median** (pre-registered k*), at a
median sensitivity of 0.83–0.91 at k=10; the POST-HOC draw-level
reliability bar (k_reliable) was reached only on GDPH at k=200 and not
reached on the other three cohorts within the pre-registered grid.
Multi-source LOCO training met its pre-registered criterion via the AUC
branch (ΔAUC +0.013 to +0.056 vs the v1 fold-5 **single** model on 4/4
held-out cohorts), with these caveats stated alongside: 2/4 paired ΔAUC
CIs include zero (BrEaST, SYSUCC); the specificity branch failed 2/4
(held-out sens 0.847 / 0.800 < 0.85); against the deployed v1 ensemble
only 2/4 cohorts reach +0.01; the registered question concerned the
specificity collapse. Δspec +0.27 to +0.30 (paired CIs exclude zero, at
different thresholds per model — so not a fixed-operating-point training
effect), held-out sensitivity 0.80–0.96.
[Details in the report.](docs/report.md)

## Why this repo might be worth your time

- **Patient-level splits** for every BUS-BRA split (official folds,
  enforced by tests that need the raw datasets on disk); the v2 LOCO
  training-side 85/15 slices of the external cohorts are **image-level**
  (no patient IDs exist; Amendment 4 (t))
- **Internally pre-registered protocols** (external protocol + 4
  amendments) — Amendments 2/3/4 each committed alone before the
  computation they govern; Amendment 1 committed together with the
  keep-lists and pHash artifacts it registers (after the data audit it
  describes, before any model metric on those cohorts) — in a private,
  version-controlled history with no external timestamp; see
  "Provenance" below
- **Freeze-then-test**: single-shot external validation, results
  recorded regardless of outcome (`git tag external-v1`); each prediction
  CSV has exactly one adding commit
- **Dataset auditing**: 12.06M-pair perceptual-hash sweep (all
  within- and cross-set pairs among 1875/252/379/846/1559 images; 35%
  duplicates found in one public cohort, incl. the same image released
  under both class labels — the 35% counts 507 duplicates plus 39
  label-conflict images). Result: no visually confirmed cross-set
  near-duplicate at pHash d ≤ 8 (two candidates, both refuted) — a finite
  screen, not "zero overlap"
- **Three pre-registered negative results** (CoarseDropout — a visual
  gate on three images; ensemble-disagreement abstention 0/4;
  domain-pretrained BiomedCLIP backbone NOT CLAIMED), reported as
  registered
- The result tables in RESULTS.md — internal CV / TTA / calibration /
  operating point, the four external cohorts, the reader comparison,
  every Amendment 2/3/4 table and every post-hoc table — recompute from
  committed CSV/JSON artifacts (both 2026-09-11 re-audits did so). Not
  recomputable from the repository: the wandb-derived numbers (per-epoch
  CV AUCs behind POST-HOC 1, the smoke-test AUC, epoch times), the
  CoarseDropout pooled OOF AUC 0.9139 (no per-image CSV was committed),
  latencies, and the qualitative CAM tallies. The deployed demo shares
  the same model, calibration and inference module as the validation
  pipeline, while its preprocessing differs from the validation path on
  non-BUS-BRA image sizes (documented; measured impact on calibrated
  probability max 0.0072, median 0.0008)

## Repository map

    src/            training, evaluation, frozen inference, v2 studies
    configs/        one YAML per experiment
    data/external_protocol.md   pre-registered protocol + amendments
    data/splits/    busbra_official_5fold.csv (verbatim official BUS-BRA
                    partition, CC BY 4.0, notice in BUSBRA_LICENSE.txt) +
                    frozen external keep-lists (busi/gdph/sysucc_clean.csv)
    models/         calibration + operating-point JSONs and CHECKSUMS.txt
                    (sha256 of every published weight); the .pt checkpoints
                    are git-ignored and live on HF (see below)
    tests/          patient-level split guards (pytest)
    RESULTS.md      every number, chronological, commit-linked
    plan.md         task state
    reports/        figures & per-image predictions (CSV)
    app/ deploy/    Gradio demo (local / HF Space)
    docs/           report, independent audit + response

**Folds and weights.** The official BUS-BRA partition is committed as
`data/splits/busbra_official_5fold.csv` (byte-identical to the dataset's
`5-fold-cv.csv`, Zenodo 8231412, CC BY 4.0); `data.resolve_fold_file`
prefers it, falls back to `data/raw/busbra/5-fold-cv.csv`, and refuses
any file whose sha256 differs from the official one. Checkpoints
(`models/*.pt`) are git-ignored; the release artifacts are published at
[happytommy/breast-us-cad-weights](https://huggingface.co/happytommy/breast-us-cad-weights):
the five frozen-v1 classifiers, the segmentation model and the two JSONs
at the repo root (the only files the demo uses), and the v2 research
artifacts under `v2/` (`v2_biomedclip_fold{1-5}.pt`,
`v2_loco_{breast,busi,gdph,sysucc}.pt`, their JSONs, and the exported
BiomedCLIP init). `models/CHECKSUMS.txt` holds the sha256 of every
published file; `inference.resolve_weight` resolves both layouts. Earlier
development checkpoints (EfficientNet-B0, ConvNeXt and CoarseDropout CV
folds) exist only on the author's disk and are not published.

## Reproduce

    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install -r requirements.txt

**Dataset layout (hardcoded paths).** Download from the original sources
(see below) into exactly:

    data/raw/busbra/            bus_data.csv, 5-fold-cv.csv, Images/, Masks/
    data/raw/breast_poland/     caseXXX.png + BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx
    data/raw/busi/              images/, masks/  (Curated BUSI)
    data/raw/gdph_sysucc/       GDPH/, SYSUCC/, BIRADS&FOLD.xlsx

**Frozen-v1 inference (exact).** With BUS-BRA on disk:

    python src/external_val.py --self-check

This re-runs the frozen pipeline on BUS-BRA fold 5 and must reproduce
AUC 0.9234433158791243 digit-for-digit (output byte-identical to
`reports/external_selfcheck.txt` on the MPS path the artifacts were
produced with; a CPU run reproduces the AUC but flags 155 instead of 156
images at the threshold — device/batch reductions can move a borderline
decision). If `models/cv_vit_fold5.pt` is absent,
`inference.resolve_weight` **downloads it from the HF weights repo
automatically** (network access; default local HF cache) — there is no
offline switch. `pytest` runs 19 tests: the 10 split tests need all five
raw datasets on disk, the other 9 (checkpoint payload, HF paths,
single-shot guard) run without data. The external-validation script
refuses to overwrite an existing `reports/external_*_preds.csv` unless
`--overwrite` is passed (a post-hoc enforcement addition of 2026-09-12;
the v1 files' single-shot status rests on their git history, one adding
commit each).

**Regenerating frozen-v1 from scratch (approximate).** The frozen
artifacts came from the full chain, not from a single `train.py` call
(`python src/train.py --config configs/vit.yaml` trains ONE model,
folds 1–4 → fold 5, to `models/vit_b16_fold5.pt`):

    python src/cross_validate.py --config configs/vit.yaml --prefix cv_vit   # 5 ckpts + reports/cv_vit_summary.csv
    python src/tta_eval.py          # hflip TTA → reports/oof_vit_preds.csv, tta_vit_summary.csv
    python src/calibrate.py         # T → models/calibration.json
    python src/pick_threshold.py    # thr → models/operating_point.json

Training logs to wandb and `train.py` has no offline flag; to run without
an account set `WANDB_MODE=offline` (or run `wandb offline`) before
launching. **MPS training is not bitwise reproducible** (no deterministic
algorithms, 4 DataLoader workers), so regenerated checkpoints will not
match the published ones digit-for-digit; only inference on the
downloaded frozen weights is exact.

## Provenance and limitations of pre-registration

- The protocol and its four amendments are an **internal,
  version-controlled pre-registration** (`data/external_protocol.md`,
  append-only; Amendments 2/3/4 each committed alone before the
  computation they govern; Amendment 1 committed with its 10 data-audit
  artifacts; the base protocol committed after the freeze and before the
  external run). They were **not externally time-stamped** (no OSF/Zenodo
  registration at the time of the runs). The single-shot status of the
  external run rests on git history (one adding commit per prediction
  CSV, never modified), not on a code guard: `--confirm` only authorized
  execution and did not refuse existing outputs until 2026-09-12.
- On 2026-08-31 the identity of all prior commits was rewritten with
  `git filter-branch` (hostname-derived email → the author's email);
  author/committer dates were preserved, tags carried over, and the
  `frozen-v1` annotated tag re-created at the same date (commit 2c24500
  message). Pre-rewrite hashes (e.g. in wandb metadata) therefore differ
  from current ones; the complete 30-commit pre→post mapping by tree hash
  and author date is in [docs/PROVENANCE.md](docs/PROVENANCE.md) §3, and
  is verifiable only from the author's working copy.
- The repository was **private at audit time (2026-09-06)** and is public
  since 2026-09-07 (https://github.com/thomasf45566/breast-us-cad); the public artifacts created during the
  study — the HF weights repo (2026-08-30 04:25Z) and the HF Space
  (2026-08-30 09:24Z) — both postdate the external-v1 commit (2026-08-29
  15:14Z). The freeze-before-test ordering therefore rests on local,
  rewritten, self-assigned commit timestamps; the Zenodo DOI below is the
  first external timestamp, post hoc for external-v1.
- Amendment 1 already contained the dedup counts and pHash outcome it
  registers (it precedes all model metrics, not the data audit).
- TTA adoption (hflip) was not pre-registered: decided pre-freeze after
  seeing +0.0023 pooled OOF AUC, on the same OOF data later used to fit
  T and the threshold.
- Done (P2, 2026-09-07): post-hoc quantification of the epoch-selection
  and in-sample operating-point biases; `train.py` now stores the folds
  actually used in each checkpoint (existing checkpoints carry the raw
  YAML). Done (P3, 2026-09-07): v2 checkpoints published under `v2/` with
  `models/CHECKSUMS.txt`; official fold file committed and hash-verified.
  Done (P4, 2026-09-07): [docs/PROVENANCE.md](docs/PROVENANCE.md) — tag
  timeline, identity-rewrite evidence (full pre→post commit mapping by tree
  hash), HF timestamps, forward OSF commitment; Zenodo DOI
  10.5281/zenodo.22630912 published 2026-09-07 (tag `v1.0-audited`). Done
  (P8', 2026-09-12): two independent re-audits archived
  ([Claude](docs/AUDIT2_claude_2026-09-11.md), [Astra](docs/AUDIT2_astra_2026-09-11.md))
  and answered item by item in [docs/AUDIT2_RESPONSE.md](docs/AUDIT2_RESPONSE.md);
  tag `v1.1-audited`; the archive now carries its own commit hash
  (`.git-commit`, PROVENANCE §7).

## Licences

- **Code** (src/, scripts/, tests/, app/, deploy/, configs/): Apache-2.0 — see [LICENSE](LICENSE).
- **Documentation and figures authored here** (docs/, RESULTS.md, plan.md, the
  plots and galleries under reports/): CC BY 4.0. This does not cover the
  third-party material embedded in some figures (see below).
- **Third-party data**: none of the five datasets is relicensed. Per-dataset
  licence, required citation and the exact list of derived files tracked
  here are in [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).
- **Example images**: the four images in `app/examples/` and
  `deploy/examples/` (and the README screenshot) are BUS-BRA images,
  redistributed under the dataset's terms reproduced in
  [data/splits/BUSBRA_LICENSE.txt](data/splits/BUSBRA_LICENSE.txt).

## Data availability & licenses

All datasets are public research releases; the raw datasets are not
redistributed in this repo (what is redistributed, under CC BY 4.0: the
four BUS-BRA example images, BUS-BRA-derived figures and the official
BUS-BRA partition — see Licences above). Download links, citations, and license notes for all five
cohorts in [data/README.md](data/README.md): BUS-BRA, BrEaST, and
Curated BUSI are CC BY 4.0; GDPH and SYSUCC come from the HoVer-Trans
release (Mo et al., *IEEE TMI* 2023, DOI 10.1109/TMI.2023.3236011) —
no license file accompanies that release, so check its original terms.

## Limitations

Single-institution training data; no IRB'd clinical validation;
internal CV AUC is best-epoch-on-the-reported-fold and the internal
operating point is in-sample and belongs to a single-held-out-checkpoint
predictor, not the deployed ensemble; the threshold rule as implemented
selects among sklearn `roc_curve` operating points (exhaustive rule:
0.26878, one image apart; RESULTS.md POST-HOC 6); burned-in annotations
are a potential shortcut; external cohorts without patient IDs force
image-level bootstraps and image-level v2 training slices;
pre-registration is internal only (above). Full list in
[report §4.5](docs/report.md).

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22630912.svg)](https://doi.org/10.5281/zenodo.22630912)

Zenodo DOI: [10.5281/zenodo.22630912](https://doi.org/10.5281/zenodo.22630912) — deposition of tag
`v1.0-audited` (commit fa193a8; the checksum shown by Zenodo is
authoritative; full record in [docs/PROVENANCE.md](docs/PROVENANCE.md) §7,
including the known defect that the v1.0 archive's embedded provenance
text names the superseded pre-re-point commit). Concept DOI (all
versions): 10.5281/zenodo.22630911. Tag `v1.1-audited` (2026-09-12) is
prepared as a new version of the same record; its version DOI is recorded
in PROVENANCE §7 once published. Cite via [CITATION.cff](CITATION.cff).
Manuscript in preparation.

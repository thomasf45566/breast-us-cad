# BreastUS-CAD

> **Research prototype — not for clinical use. Not a medical device.**

Benign/malignant breast ultrasound classification with a frozen,
pre-registered, single-shot external validation across four cohorts on
three continents — plus deployment-oriented follow-up studies
(site-specific recalibration, multi-source training).

**[Live demo](https://huggingface.co/spaces/happytommy/breast-us-cad)** ·
[Weights](https://huggingface.co/happytommy/breast-us-cad-weights) ·
[Full report](docs/report.md)

![demo screenshot](docs/assets/demo.png)

## Headline results

| | AUC | Sens | Spec |
|---|---|---|---|
| Internal (BUS-BRA, 5-fold patient-level CV) | 0.931 ± 0.016 | 0.903 | 0.771 |
| BrEaST (Poland, n=252) | 0.854 | 0.918 | 0.409 |
| BUSI cleaned (Egypt, n=379) | 0.934 | 0.957 | 0.630 |
| GDPH (China, n=810) | 0.915 | 0.971 | 0.453 |
| SYSUCC (China, n=1013) | 0.838 | 0.931 | 0.474 |

Internal AUC is mean-of-folds; sens/spec at the frozen threshold
(sens ≥ 0.90 rule on calibrated pooled out-of-fold predictions).
External rows: single-shot at that same frozen threshold.

**Key finding:** discrimination transfers; calibration does not. The
frozen operating point held sensitivity 0.918–0.971 on every external
cohort while specificity collapsed from 0.771 to 0.409–0.630 — the
model fails in the safe direction. Follow-up studies: re-selecting only
the threshold on 10–20 local labels recovers most of the lost
specificity in the median (draw-level reliability needs ≈100–200), and
multi-source LOCO training met its pre-registered criterion (ΔAUC
+0.013 to +0.056 on 4/4 held-out cohorts; Δspec +0.27 to +0.30, at
held-out sensitivity 0.80–0.96), with much of the specificity gain
coming from better threshold placement rather than reduced shift.
[Details in the report.](docs/report.md)

## Why this repo might be worth your time

- **Patient-level splits only**, enforced by tests
- **Pre-registered protocols** (external protocol + 4 amendments)
  committed before results
- **Freeze-then-test**: single-shot external validation, results
  recorded regardless of outcome (`git tag external-v1`)
- **Dataset auditing**: ~9.7M-pair perceptual-hash sweep (35%
  duplicates found in one public cohort, incl. the same image released
  under both class labels)
- **Three pre-registered negative results** (CoarseDropout robustness,
  ensemble-disagreement abstention, domain-pretrained backbone),
  reported as registered
- Every number traces to a commit; the deployed demo shares the exact
  inference module with the validation pipeline

## Repository map

    src/            training, evaluation, frozen inference, v2 studies
    configs/        one YAML per experiment
    data/external_protocol.md   pre-registered protocol + amendments
    data/splits/    official BUS-BRA folds + frozen external keep-lists
    models/         checkpoints, calibration + operating-point JSONs
    tests/          patient-level split guards (pytest)
    RESULTS.md      every number, chronological, commit-linked
    plan.md         task state
    reports/        figures & per-image predictions
    app/ deploy/    Gradio demo (local / HF Space)

## Reproduce

    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install -r requirements.txt
    # datasets: download from original sources (see below), then:
    python src/train.py --config configs/vit.yaml
    python src/external_val.py --self-check

The self-check re-runs the frozen pipeline on BUS-BRA fold 5 and must
reproduce AUC 0.9234 digit-for-digit before any external scoring.

## Data availability & licenses

All datasets are public research releases; none are redistributed in
this repo. Download links, citations, and license notes for all five
cohorts in [data/README.md](data/README.md): BUS-BRA, BrEaST, and
Curated BUSI are CC BY 4.0; GDPH and SYSUCC come from the HoVer-Trans
release (Mo et al., *IEEE TMI* 2023, DOI 10.1109/TMI.2023.3236011) —
no license file accompanies that release, so check its original terms.

## Limitations

Single-institution training data; no IRB'd clinical validation;
burned-in annotations are a potential shortcut; external cohorts
without patient IDs force image-level bootstraps. Full list in
[report §4.5](docs/report.md).

## Citation

(bibtex placeholder — manuscript in preparation)

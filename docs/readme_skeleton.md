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
| Internal (5-fold patient-level CV) | 0.931 ± 0.016 | 0.903 | 0.771 |
| BrEaST (Poland, n=252) | 0.854 | 0.918 | 0.409 |
| BUSI cleaned (Egypt, n=379) | 0.934 | 0.957 | 0.630 |
| GDPH (China, n=810) | 0.915 | 0.971 | 0.453 |
| SYSUCC (China, n=1013) | 0.838 | 0.931 | 0.474 |

**Key finding:** discrimination transfers; calibration does not. The
frozen operating point (sens ≥ 0.90 by design) held sensitivity ≥ 0.92
on every external cohort while specificity collapsed — the model fails
in the safe direction. Follow-up studies show 10–20 local labels recover
most specificity in the median (~100 for reliability), and multi-source
training helps mainly by producing a better starting threshold.
[Details in the report.](docs/report.md)

## Why this repo might be worth your time
- **Patient-level splits only**, enforced by tests
- **Pre-registered protocols** (4 amendments) committed before results
- **Freeze-then-test**: single-shot external validation, results recorded
  regardless of outcome (`git tag external-v1`)
- **Dataset auditing**: ~9.7M-pair perceptual-hash sweep (35% duplicates
  found in one public cohort, incl. same image labeled both classes)
- **Two pre-registered negative results**, reported as registered
- Every number traces to a commit; the deployed demo shares the exact
  inference module with the validation pipeline

## Repository map
    src/            training, evaluation, frozen inference, v2 studies
    configs/        one YAML per experiment
    data/external_protocol.md   pre-registered protocol + amendments
    RESULTS.md      every number, chronological, commit-linked
    plan.md         task state
    reports/        figures & per-image predictions
    app/ deploy/    Gradio demo (local / HF Space)

## Reproduce
    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install -r requirements.txt
    # datasets: see data/README.md (BUS-BRA, BrEaST, BUSI, GDPH&SYSUCC —
    # licenses require download from original sources)
    python src/train.py --config configs/vit.yaml
    python src/external_val.py --self-check

## Data availability & licenses
All datasets are public; none are redistributed here. Download links,
versions, and license notes in data/README.md.

## Limitations
Single-institution training data; no IRB'd clinical validation; burned-in
annotations are a potential shortcut; see report §4.5 for the full list.

## Citation
(bibtex placeholder — manuscript in preparation)
"""Deploy the frozen-v1 demo to Hugging Face: weights repo + Gradio Space.

Research prototype — not for clinical use. Not a medical device.

Requires a logged-in Hugging Face account (`hf auth login`) with write scope.
Idempotent: re-running uploads only changed files.

Steps:
1. Resolve the username; patch the weights-repo id into src/inference.py
   (default of BREAST_US_CAD_WEIGHTS_REPO) and deploy/README.md.
2. Create <user>/breast-us-cad-weights (model repo) and upload the frozen
   artifacts: 5x cv_vit_fold{1-5}.pt, seg_unet_effb0.pt, calibration.json,
   operating_point.json + a minimal model card.
3. Rebuild deploy/ (bundle app/app.py + src/inference.py + examples) and
   push it to the public Gradio Space <user>/breast-us-cad (free CPU).

Usage: python scripts/deploy_hf.py
"""

import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
WEIGHT_FILES = [
    *[f"cv_vit_fold{k}.pt" for k in (1, 2, 3, 4, 5)],
    "seg_unet_effb0.pt",
    "calibration.json",
    "operating_point.json",
]

WEIGHTS_CARD = """\
---
license: other
license_name: research-only
license_link: https://huggingface.co/{repo}
tags:
  - medical-imaging-research
  - breast-ultrasound
---

# BreastUS-CAD frozen-v1 weights

> **Research prototype — not for clinical use. Not a medical device.**

Frozen artifacts of the BreastUS-CAD demo ([Space](https://huggingface.co/spaces/{space})):
five ViT-B/16 cross-validation fold checkpoints (BUS-BRA official
patient-level folds), a U-Net (efficientnet-b0 encoder) lesion
segmentation demo model, the temperature-scaling parameter, and the
frozen decision threshold. Trained on the public BUS-BRA dataset
(Gómez-Flores et al., Medical Physics 2024). These root-level files are
the ONLY artifacts the Space and the frozen-v1 external validation use.

## v2/ — research artifacts, not the deployed model

Checkpoints of the pre-registered follow-up studies (repository
`data/external_protocol.md`, Amendment 4; results in `RESULTS.md`):

- `v2/v2_biomedclip_fold{{1-5}}.pt` + `v2/v2_biomedclip_{{calibration,operating_point}}.json`
  — Q2: same v1 protocol with the backbone initialised from the BiomedCLIP
  ViT-B/16 image encoder. Pre-registered criterion NOT MET (verdict:
  "domain pretraining reduces shift" is NOT CLAIMED).
- `v2/v2_loco_{{breast,busi,gdph,sysucc}}.pt` + per-run
  `_calibration.json` / `_operating_point.json` — Q1: leave-one-cohort-out
  multi-source single models (BUS-BRA folds 1–4 + 85% of the three other
  external cohorts). Criterion met via the AUC branch only; see RESULTS.md
  for the caveats stated alongside.
- `v2/pretrained/biomedclip_vitb16_timm.pt` — the BiomedCLIP vision tower
  re-exported into timm `vit_base_patch16_224` layout (initialisation only;
  derived from microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224,
  subject to its licence).

None of the v2 files is used by the demo. They are published so that the
v2 results can be reproduced at inference level from the same frozen
keep-lists, which an independent audit (2026-09-06) found impossible
while they existed only on the author's disk.

## Integrity

`CHECKSUMS.txt` lists the SHA-256 of every file in this repo in
`sha256sum` format (paths as in the repository's `models/` directory;
`v2_*`/`pretrained/*` entries live under `v2/` here).
`inference.resolve_weight` in the repository resolves both layouts.
"""


def patch_default_repo(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text()
    new = re.sub(pattern, replacement, text)
    if new != text:
        path.write_text(new)
        print(f"patched {path}")


def build_deploy_dir() -> None:
    """deploy/ = the complete Space repo: bundled app, inference, examples."""
    shutil.copy2(ROOT / "app" / "app.py", DEPLOY / "app.py")
    shutil.copy2(ROOT / "src" / "inference.py", DEPLOY / "inference.py")
    (DEPLOY / "examples").mkdir(exist_ok=True)
    for p in sorted((ROOT / "app" / "examples").glob("*.png")):
        shutil.copy2(p, DEPLOY / "examples" / p.name)
    print(f"deploy/ rebuilt: {sorted(p.name for p in DEPLOY.iterdir())}")


def main() -> None:
    api = HfApi()
    user = api.whoami()["name"]
    weights_repo = f"{user}/breast-us-cad-weights"
    space_repo = f"{user}/breast-us-cad"

    patch_default_repo(
        ROOT / "src" / "inference.py",
        r'"BREAST_US_CAD_WEIGHTS_REPO", "[^"]+"',
        f'"BREAST_US_CAD_WEIGHTS_REPO", "{weights_repo}"',
    )
    patch_default_repo(
        DEPLOY / "README.md",
        r"(?:huggingface\.co/)+(?:[^)\s]*breast-us-cad-weights[^)\s]*|WEIGHTS_REPO_PLACEHOLDER)",
        f"huggingface.co/{weights_repo}",
    )

    api.create_repo(weights_repo, repo_type="model", exist_ok=True)
    card = DEPLOY.parent / "reports" / "_weights_card.md"
    card.write_text(WEIGHTS_CARD.format(repo=weights_repo, space=space_repo))
    api.upload_file(
        path_or_fileobj=card, path_in_repo="README.md",
        repo_id=weights_repo, repo_type="model",
    )
    card.unlink()
    for f in WEIGHT_FILES:
        print(f"uploading {f} ...")
        api.upload_file(
            path_or_fileobj=ROOT / "models" / f, path_in_repo=f,
            repo_id=weights_repo, repo_type="model",
        )
    print(f"weights: https://huggingface.co/{weights_repo}")

    build_deploy_dir()
    api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=DEPLOY, repo_id=space_repo, repo_type="space")
    print(f"space:   https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()

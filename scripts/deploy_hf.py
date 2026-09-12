"""Deploy the frozen-v1 demo to Hugging Face: weights repo + Gradio Space.

Research prototype — not for clinical use. Not a medical device.

Requires a logged-in Hugging Face account (`hf auth login`) with write scope.
Idempotent: re-running uploads only changed files.

This script NEVER writes under src/ (re-audit 2026-09-11 §10: the earlier
version patched the default weights-repo id into src/inference.py in place,
a no-op for this account but a mutation of the frozen inference source for
any other). deploy/ is a build product: app/app.py and src/inference.py are
copied into it verbatim and only the COPY is patched when the resolved
account differs from the default id in src/inference.py; for the author's
account the copies stay byte-identical to their sources (asserted).

Steps:
1. Resolve the username; derive <user>/breast-us-cad-weights and
   <user>/breast-us-cad.
2. (unless --space-only) Create the weights repo and upload the frozen
   artifacts: 5x cv_vit_fold{1-5}.pt, seg_unet_effb0.pt, calibration.json,
   operating_point.json + a minimal model card.
3. Rebuild deploy/ (bundle app/app.py + src/inference.py + examples, patch
   deploy/README.md's weights link) and push it to the public Gradio Space.

Usage: python scripts/deploy_hf.py [--space-only] [--no-push]
       --space-only  skip the weights repo entirely (the usual re-deploy)
       --no-push     rebuild deploy/ only; touch nothing on Hugging Face
"""

import argparse
import filecmp
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
DEFAULT_REPO_RE = r'"BREAST_US_CAD_WEIGHTS_REPO", "([^"]+)"'

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


def default_weights_repo() -> str:
    """The default id baked into src/inference.py (read-only)."""
    m = re.search(DEFAULT_REPO_RE, (ROOT / "src" / "inference.py").read_text())
    assert m, "src/inference.py: BREAST_US_CAD_WEIGHTS_REPO default not found"
    return m.group(1)


def patch_copy(path: Path, pattern: str, replacement: str) -> None:
    """Regex patch of a file INSIDE deploy/ only."""
    assert DEPLOY in path.resolve().parents, f"refusing to patch outside deploy/: {path}"
    text = path.read_text()
    new = re.sub(pattern, replacement, text)
    if new != text:
        path.write_text(new)
        print(f"patched {path.relative_to(ROOT)}")


def build_deploy_dir(weights_repo: str) -> None:
    """deploy/ = the complete Space repo: bundled app, inference, examples.

    src/ and app/ are read, never written. If weights_repo differs from the
    default in src/inference.py the deploy/ COPY is patched; otherwise the
    copies are asserted byte-identical to their sources.
    """
    shutil.copy2(ROOT / "app" / "app.py", DEPLOY / "app.py")
    shutil.copy2(ROOT / "src" / "inference.py", DEPLOY / "inference.py")
    (DEPLOY / "examples").mkdir(exist_ok=True)
    for p in sorted((ROOT / "app" / "examples").glob("*.png")):
        shutil.copy2(p, DEPLOY / "examples" / p.name)
    if weights_repo != default_weights_repo():
        patch_copy(DEPLOY / "inference.py", DEFAULT_REPO_RE,
                   f'"BREAST_US_CAD_WEIGHTS_REPO", "{weights_repo}"')
    else:
        assert filecmp.cmp(ROOT / "src" / "inference.py", DEPLOY / "inference.py", shallow=False)
        assert filecmp.cmp(ROOT / "app" / "app.py", DEPLOY / "app.py", shallow=False)
    patch_copy(
        DEPLOY / "README.md",
        r"(?:huggingface\.co/)+(?:[^)\s]*breast-us-cad-weights[^)\s]*|WEIGHTS_REPO_PLACEHOLDER)",
        f"huggingface.co/{weights_repo}",
    )
    print(f"deploy/ rebuilt: {sorted(p.name for p in DEPLOY.iterdir())}")


def upload_weights(api: HfApi, weights_repo: str, space_repo: str) -> None:
    api.create_repo(weights_repo, repo_type="model", exist_ok=True)
    card = ROOT / "reports" / "_weights_card.md"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-only", action="store_true", help="do not touch the weights repo")
    parser.add_argument("--no-push", action="store_true", help="rebuild deploy/ only")
    args = parser.parse_args()

    api = HfApi()
    user = default_weights_repo().split("/")[0] if args.no_push else api.whoami()["name"]
    weights_repo = f"{user}/breast-us-cad-weights"
    space_repo = f"{user}/breast-us-cad"

    if not args.space_only and not args.no_push:
        upload_weights(api, weights_repo, space_repo)

    build_deploy_dir(weights_repo)
    if args.no_push:
        print("--no-push: deploy/ rebuilt, nothing uploaded")
        return
    api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    info = api.upload_folder(folder_path=DEPLOY, repo_id=space_repo, repo_type="space")
    print(f"space:   https://huggingface.co/spaces/{space_repo}")
    print(f"space commit: {getattr(info, 'oid', info)}")


if __name__ == "__main__":
    main()

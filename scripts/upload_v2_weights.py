"""Publish the v2 research checkpoints to the HF weights repo under v2/ and
record SHA-256 checksums for every published artifact (v1 + v2).

Research prototype — not for clinical use. Not a medical device.

The v2 artifacts (BiomedCLIP-init 5-fold ensemble, four LOCO single models,
their calibration / operating-point JSONs, and the exported BiomedCLIP
ViT-B/16 init) are research artifacts of the pre-registered follow-up
studies (data/external_protocol.md Amendment 4). They are NOT the deployed
model: the demo and the frozen-v1 external validation use only the root-level
v1 files. Audit 2026-09-06 §9 found the v2 line reproducible only from its
prediction CSVs because these files existed nowhere but the author's disk.

Steps (one HF commit):
1. sha256 every v1 + v2 artifact under models/ -> models/CHECKSUMS.txt
   (sha256sum format; `cd models && shasum -a 256 -c CHECKSUMS.txt`).
2. Upload v2 files to <user>/breast-us-cad-weights under v2/ (the layout
   inference.hf_repo_path maps to), plus CHECKSUMS.txt and the model card.

Usage: python scripts/upload_v2_weights.py [--dry-run]
"""

import argparse
import hashlib
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from deploy_hf import WEIGHT_FILES as V1_FILES, WEIGHTS_CARD  # noqa: E402
from inference import hf_repo_path  # noqa: E402

V2_FILES = [
    *[f"v2_biomedclip_fold{k}.pt" for k in (1, 2, 3, 4, 5)],
    "v2_biomedclip_calibration.json",
    "v2_biomedclip_operating_point.json",
    *[f"v2_loco_{c}.pt" for c in ("breast", "busi", "gdph", "sysucc")],
    *[f"v2_loco_{c}_calibration.json" for c in ("breast", "busi", "gdph", "sysucc")],
    *[f"v2_loco_{c}_operating_point.json" for c in ("breast", "busi", "gdph", "sysucc")],
    "pretrained/biomedclip_vitb16_timm.pt",
]
CHECKSUMS = MODELS / "CHECKSUMS.txt"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums() -> str:
    lines = []
    for name in [*V1_FILES, *V2_FILES]:
        path = MODELS / name
        if not path.exists():
            raise FileNotFoundError(path)
        digest = sha256_of(path)
        lines.append(f"{digest}  {name}")
        print(f"{digest}  {name}")
    text = "\n".join(lines) + "\n"
    CHECKSUMS.write_text(text)
    print(f"written: {CHECKSUMS} ({len(lines)} files)")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="hash only, do not upload")
    args = parser.parse_args()

    write_checksums()
    if args.dry_run:
        return

    api = HfApi()
    user = api.whoami()["name"]
    weights_repo = f"{user}/breast-us-cad-weights"
    space_repo = f"{user}/breast-us-cad"
    card = MODELS / "_weights_card.md"
    card.write_text(WEIGHTS_CARD.format(repo=weights_repo, space=space_repo))

    ops = [CommitOperationAdd(path_in_repo=hf_repo_path(n), path_or_fileobj=str(MODELS / n)) for n in V2_FILES]
    ops.append(CommitOperationAdd(path_in_repo="CHECKSUMS.txt", path_or_fileobj=str(CHECKSUMS)))
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card)))
    for op in ops:
        print(f"queued: {op.path_in_repo}")
    info = api.create_commit(
        repo_id=weights_repo,
        repo_type="model",
        operations=ops,
        commit_message="v2 research artifacts under v2/ + CHECKSUMS.txt + card update (P3, 2026-09-07)",
    )
    card.unlink()
    print(f"HF commit: {info.commit_url}")
    print(f"weights: https://huggingface.co/{weights_repo}")


if __name__ == "__main__":
    main()

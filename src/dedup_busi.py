"""BUSI deduplication via perceptual hash — frozen keep-list for external validation.

Research prototype — not for diagnostic use.

Labels come from filename prefixes; no model inference happens here. The
normal class is excluded up front (out of scope per data/external_protocol.md).
64-bit pHash (imagehash, hash_size=8) on every benign+malignant image; pairs
with Hamming distance <= NEAR_DUP_MAX are duplicates (0 = exact-pHash match).
Duplicate groups (union-find) keep the lexicographically first filename;
groups mixing classes are dropped entirely (label conflict). Also reports
BUSI-vs-BrEaST cross-dataset matches (report only — BrEaST is untouched).

Outputs: data/splits/busi_clean.csv (frozen keep-list),
reports/busi_phash_distances.png (histogram to sanity-check the threshold),
reports/busi_dup_pairs.png (side-by-side view of every flagged pair).

Usage: python src/dedup_busi.py
"""

from itertools import combinations
from pathlib import Path

import imagehash
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

BUSI_IMAGES = Path("data/raw/busi/images")
BREAST_DIR = Path("data/raw/breast_poland")
OUT_CSV = Path("data/splits/busi_clean.csv")
HIST_PNG = Path("reports/busi_phash_distances.png")
NEAR_DUP_MAX = 8  # Hamming distance on 64-bit pHash; justified by the histogram
LABEL_MAP = {"benign": 0, "malignant": 1}


def phash_file(path: Path) -> imagehash.ImageHash:
    with Image.open(path) as im:
        return imagehash.phash(im.convert("L"), hash_size=8)


def find(parent: dict, a: str) -> str:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def union(parent: dict, a: str, b: str) -> None:
    parent[find(parent, a)] = find(parent, b)


def main() -> None:
    files = sorted(
        p for p in BUSI_IMAGES.glob("*.png") if not p.name.startswith("normal")
    )
    classes = {p.name: p.name.split("_")[0] for p in files}
    print(f"hashing {len(files)} BUSI benign+malignant images...")
    hashes = {p.name: phash_file(p) for p in files}

    names = list(hashes)
    pair_dists = []
    dup_pairs = []
    for a, b in combinations(names, 2):
        d = hashes[a] - hashes[b]
        pair_dists.append(d)
        if d <= NEAR_DUP_MAX:
            dup_pairs.append((a, b, d))
    pair_dists = np.array(pair_dists)

    # histogram of all pairwise distances (log y — dup tail is tiny vs ~74k pairs)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pair_dists, bins=np.arange(0, pair_dists.max() + 2) - 0.5, log=True)
    ax.axvline(NEAR_DUP_MAX + 0.5, color="red", ls="--",
               label=f"near-dup threshold <= {NEAR_DUP_MAX}")
    ax.set_xlabel("Pairwise Hamming distance (64-bit pHash)")
    ax.set_ylabel("Pair count (log)")
    ax.set_title("BUSI benign+malignant — pairwise pHash distances")
    ax.legend()
    fig.tight_layout()
    HIST_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HIST_PNG, dpi=150)
    plt.close(fig)

    exact = [p for p in dup_pairs if p[2] == 0]
    near = [p for p in dup_pairs if p[2] > 0]
    print(f"pairs total: {len(pair_dists)} | exact (d=0): {len(exact)} | "
          f"near (0 < d <= {NEAR_DUP_MAX}): {len(near)}")
    lo = pair_dists[pair_dists <= 16]
    print("distance counts (d <= 16):",
          dict(zip(*[a.tolist() for a in np.unique(lo, return_counts=True)])))
    for a, b, d in sorted(dup_pairs, key=lambda t: t[2]):
        print(f"  dup pair d={d:2d}: {a} <-> {b}")

    if dup_pairs:  # side-by-side grid of every flagged pair, for visual sanity check
        pairs = sorted(dup_pairs, key=lambda t: t[2])
        fig, axes = plt.subplots(len(pairs), 2, figsize=(6, 3 * len(pairs)),
                                 squeeze=False)
        for r, (a, b, d) in enumerate(pairs):
            for c, name in enumerate((a, b)):
                with Image.open(BUSI_IMAGES / name) as im:
                    axes[r, c].imshow(np.asarray(im.convert("L")), cmap="gray")
                axes[r, c].set_title(f"{name} (d={d})", fontsize=8)
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig("reports/busi_dup_pairs.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    parent = {n: n for n in names}
    for a, b, _ in dup_pairs:
        union(parent, a, b)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(parent, n), []).append(n)

    kept, dropped_dup, dropped_conflict = [], [], []
    for members in groups.values():
        members = sorted(members)
        if len({classes[m] for m in members}) > 1:
            dropped_conflict.extend(members)
            print(f"  LABEL-CONFLICT group dropped entirely: {members}")
        else:
            kept.append(members[0])
            dropped_dup.extend(members[1:])

    keep_df = pd.DataFrame(
        {
            "image_path": [str(BUSI_IMAGES / n) for n in sorted(kept)],
            "filename": sorted(kept),
        }
    )
    keep_df["class"] = keep_df["filename"].map(classes)
    keep_df["label"] = keep_df["class"].map(LABEL_MAP)
    keep_df["phash"] = keep_df["filename"].map(lambda n: str(hashes[n]))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    keep_df.to_csv(OUT_CSV, index=False)

    orig = pd.Series(classes).value_counts()
    after = keep_df["class"].value_counts()
    print(f"\ncounts (threshold d <= {NEAR_DUP_MAX}):")
    for c in ("benign", "malignant"):
        print(f"  {c:9s}: {orig.get(c, 0)} -> {after.get(c, 0)}")
    print(f"  total    : {len(names)} -> {len(keep_df)} "
          f"(dropped {len(dropped_dup)} dup, {len(dropped_conflict)} label-conflict)")

    # cross-dataset check: BUSI (benign+malignant) vs BrEaST main case images
    breast_files = sorted(
        p for p in BREAST_DIR.glob("case*.png")
        if "_tumor" not in p.name and "_other" not in p.name
    )
    print(f"\ncross-dataset check vs {len(breast_files)} BrEaST case images...")
    breast_hashes = {p.name: phash_file(p) for p in breast_files}
    cross = [
        (a, b, hashes[a] - hb)
        for a in names
        for b, hb in breast_hashes.items()
        if hashes[a] - hb <= NEAR_DUP_MAX
    ]
    min_cross = min(
        hashes[a] - hb for a in names for hb in breast_hashes.values()
    )
    if cross:
        for a, b, d in sorted(cross, key=lambda t: t[2]):
            print(f"  CROSS-DATASET dup d={d:2d}: busi/{a} <-> breast_poland/{b}")
    else:
        print(f"  no cross-dataset pairs at d <= {NEAR_DUP_MAX} "
              f"(minimum observed distance: {min_cross})")

    print(f"\nsaved: {OUT_CSV}, {HIST_PNG}")


if __name__ == "__main__":
    main()

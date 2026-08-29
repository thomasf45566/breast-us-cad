"""Full pairwise pHash sweep: BUS-BRA + all external cohorts (amendment E-2).

Research prototype — not for diagnostic use.

64-bit pHash on every image of every set; within-set and cross-set pairs at
Hamming d <= NEAR_DUP_MAX are flagged (same threshold as src/dedup_busi.py,
validated there by visual inspection). No model inference involved.

Rules (pre-registered in data/external_protocol.md amendment):
- Within GDPH / within SYSUCC: dedup like BUSI — union-find groups, keep the
  lexicographically first file, drop label-conflict groups entirely. Writes
  data/splits/gdph_clean.csv and data/splits/sysucc_clean.csv.
- Cross-set vs BUS-BRA (train contamination) and between external sets:
  REPORT ONLY here; any hits require a recorded exclusion before the run.

Outputs: reports/phash_sweep_hits.csv, reports/phash_sweep_hist.png,
data/splits/{gdph,sysucc}_clean.csv.

Usage: python src/phash_sweep.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import build_master_df
from dedup_busi import LABEL_MAP, NEAR_DUP_MAX, find, phash_file, union

HITS_CSV = Path("reports/phash_sweep_hits.csv")
HIST_PNG = Path("reports/phash_sweep_hist.png")
SPLITS_DIR = Path("data/splits")
BREAST_NORMAL_CASES = {45, 61, 209, 213}  # excluded per protocol


def collect_sets() -> dict[str, list[Path]]:
    breast = sorted(
        p for p in Path("data/raw/breast_poland").glob("case*.png")
        if "_tumor" not in p.name and "_other" not in p.name
        and int(p.stem.removeprefix("case")) not in BREAST_NORMAL_CASES
    )
    busi = [Path(p) for p in pd.read_csv(SPLITS_DIR / "busi_clean.csv")["image_path"]]
    return {
        "busbra": [Path(p) for p in build_master_df()["image_path"]],
        "breast": breast,
        "busi": busi,
        "gdph": sorted(Path("data/raw/gdph_sysucc/GDPH").glob("*.png")),
        "sysucc": sorted(Path("data/raw/gdph_sysucc/SYSUCC").glob("*.png")),
    }


def hash_matrix(paths: list[Path]) -> np.ndarray:
    return np.stack([phash_file(p).hash.flatten() for p in paths])


def pair_hits(A, B, names_a, names_b, same_set: bool, chunk: int = 256):
    """(name_a, name_b, d) for all pairs with d <= NEAR_DUP_MAX, plus a
    histogram of all pairwise distances (counts indexed by distance)."""
    hits, hist = [], np.zeros(65, dtype=np.int64)
    for i0 in range(0, len(A), chunk):
        d = (A[i0 : i0 + chunk, None, :] != B[None, :, :]).sum(-1)
        if same_set:  # keep strictly upper-triangular pairs only
            rows = np.arange(i0, i0 + d.shape[0])[:, None]
            d = np.where(rows < np.arange(len(B))[None, :], d, 65)
        valid = d <= 64
        hist += np.bincount(d[valid].ravel(), minlength=66)[:65]
        for ci, j in zip(*np.where(d <= NEAR_DUP_MAX)):
            hits.append((names_a[i0 + ci], names_b[j], int(d[ci, j])))
    return hits, hist


def dedup_keep_list(name: str, paths: list[Path], hits: list) -> pd.DataFrame:
    """BUSI-style within-set dedup: keep first of each group, drop conflicts."""
    names = [p.name for p in paths]
    classes = {p.name: p.name.split("(")[0] for p in paths}
    parent = {n: n for n in names}
    for a, b, _ in hits:
        union(parent, a, b)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(parent, n), []).append(n)

    kept, dropped, conflict = [], [], []
    for members in groups.values():
        members = sorted(members)
        if len({classes[m] for m in members}) > 1:
            conflict.extend(members)
            print(f"  [{name}] LABEL-CONFLICT group dropped entirely: {members}")
        else:
            kept.append(members[0])
            dropped.extend(members[1:])

    by_name = {p.name: p for p in paths}
    df = pd.DataFrame({"filename": sorted(kept)})
    df["image_path"] = df["filename"].map(lambda n: str(by_name[n]))
    df["class"] = df["filename"].map(classes)
    df["label"] = df["class"].map(LABEL_MAP)
    df = df[["image_path", "filename", "class", "label"]]

    orig = pd.Series(classes).value_counts()
    after = df["class"].value_counts()
    for c in ("benign", "malignant"):
        print(f"  [{name}] {c:9s}: {orig.get(c, 0)} -> {after.get(c, 0)}")
    print(f"  [{name}] total    : {len(names)} -> {len(df)} "
          f"(dropped {len(dropped)} dup, {len(conflict)} label-conflict)")
    return df


def main() -> None:
    sets = collect_sets()
    for name, paths in sets.items():
        print(f"{name}: {len(paths)} images")
    print("hashing...")
    mats = {name: hash_matrix(paths) for name, paths in sets.items()}
    names = {name: [p.name for p in paths] for name, paths in sets.items()}

    set_list = list(sets)
    all_hits, hists = [], {}
    for i, a in enumerate(set_list):
        for b in set_list[i:]:
            same = a == b
            hits, hist = pair_hits(mats[a], mats[b], names[a], names[b], same)
            hists[(a, b)] = hist
            n_pairs = int(hist.sum())
            mins = np.nonzero(hist)[0]
            print(f"{a} vs {b}: {n_pairs} pairs | min d = {mins[0] if len(mins) else '-'} "
                  f"| hits (d<={NEAR_DUP_MAX}): {len(hits)}")
            all_hits += [(a, b, ha, hb, d) for ha, hb, d in hits]

    hits_df = pd.DataFrame(
        all_hits, columns=["set_a", "set_b", "image_a", "image_b", "distance"]
    ).sort_values(["set_a", "set_b", "distance"])
    hits_df.to_csv(HITS_CSV, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cohort in zip(axes, ("gdph", "sysucc")):
        h = hists[(cohort, cohort)]
        ax.bar(np.arange(65), h, log=True, width=0.9)
        ax.axvline(NEAR_DUP_MAX + 0.5, color="red", ls="--",
                   label=f"near-dup <= {NEAR_DUP_MAX}")
        ax.set_title(f"within-{cohort.upper()} pairwise pHash distances")
        ax.set_xlabel("Hamming distance (64-bit pHash)")
        ax.set_ylabel("pair count (log)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(HIST_PNG, dpi=150)
    plt.close(fig)

    print("\nwithin-set dedup (GDPH, SYSUCC):")
    for cohort, out_csv in (("gdph", "gdph_clean.csv"), ("sysucc", "sysucc_clean.csv")):
        within = [
            (ha, hb, d) for sa, sb, ha, hb, d in all_hits if sa == sb == cohort
        ]
        keep = dedup_keep_list(cohort, sets[cohort], within)
        keep.to_csv(SPLITS_DIR / out_csv, index=False)
        print(f"  saved: {SPLITS_DIR / out_csv}")

    cross = hits_df[hits_df["set_a"] != hits_df["set_b"]]
    print(f"\ncross-set hits (d <= {NEAR_DUP_MAX}): {len(cross)}")
    if len(cross):
        print(cross.to_string(index=False))
        print("-> record exclusions in data/external_protocol.md before the run.")
    busbra_hits = cross[(cross["set_a"] == "busbra") | (cross["set_b"] == "busbra")]
    print(f"BUS-BRA contamination hits: {len(busbra_hits)}")
    print(f"saved: {HITS_CSV}, {HIST_PNG}")


if __name__ == "__main__":
    main()

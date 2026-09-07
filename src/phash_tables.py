"""Aggregate/tabular replacements for two pHash evidence figures that embedded
GDPH/SYSUCC images and are excluded from distribution (P4c, 2026-09-07).

Research prototype — not for diagnostic use.

From reports/phash_sweep_hits.csv (SAVED artifact of the pre-registered
cross-set sweep, protocol (g)) only:
- reports/phash_cross_pairs_table.csv — the two d <= 8 cross-set candidates
  with the visual adjudication recorded in data/external_protocol.md (g);
- reports/phash_cross_pairs_busbra_thumb.png — BUS-BRA-side thumbnail only
  (bus_0999-l, CC BY 4.0) for the BUS-BRA/SYSUCC pair; no SYSUCC pixels;
- reports/phash_within_d8_table.csv — the within-set pairs that were shown
  in the former phash_within_d8_sample.png, as a table.

Usage: python src/phash_tables.py
"""

from pathlib import Path

import cv2
import pandas as pd

HITS_CSV = Path("reports/phash_sweep_hits.csv")
CROSS_CSV = Path("reports/phash_cross_pairs_table.csv")
THUMB_PNG = Path("reports/phash_cross_pairs_busbra_thumb.png")
WITHIN_CSV = Path("reports/phash_within_d8_table.csv")
BUSBRA_IMAGES = Path("data/raw/busbra/Images")
THUMB_WIDTH = 384

# Adjudication text is the pre-registered record in data/external_protocol.md (g).
ADJUDICATION = {
    ("busbra", "sysucc"): "visually REFUTED — different scans; no training contamination",
    ("gdph", "sysucc"): "visually REFUTED — different scans",
}
# The pairs that were displayed in the former phash_within_d8_sample.png.
SAMPLED_WITHIN = [
    ("sysucc", "benign(105).png", "benign(107).png"),
    ("sysucc", "benign(188).png", "benign(190).png"),
    ("sysucc", "benign(216).png", "benign(219).png"),
    ("sysucc", "benign(274).png", "malignant(1023).png"),
]


def cross_set_table(hits: pd.DataFrame) -> pd.DataFrame:
    cross = hits[hits.set_a != hits.set_b].copy()
    assert len(cross) == 2, f"expected exactly 2 cross-set candidates, found {len(cross)}"
    cross["adjudication"] = [ADJUDICATION[(a, b)] for a, b in zip(cross.set_a, cross.set_b)]
    cross["thumbnail"] = [
        THUMB_PNG.name if a == "busbra" else "none (no licence for redistribution)" for a in cross.set_a
    ]
    return cross[["set_a", "image_a", "set_b", "image_b", "distance", "adjudication", "thumbnail"]]


def within_set_table(hits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, a, b in SAMPLED_WITHIN:
        hit = hits[(hits.set_a == s) & (hits.set_b == s) & (hits.image_a == a) & (hits.image_b == b)]
        assert len(hit) == 1, f"pair {a} / {b} not found once in {HITS_CSV}"
        d = int(hit.distance.iloc[0])
        label_conflict = a.split("(")[0] != b.split("(")[0]
        rows.append({"set": s, "image_a": a, "image_b": b, "distance": d,
                     "label_conflict": label_conflict,
                     "dedup_action": "group dropped entirely (label conflict)" if label_conflict
                     else "keep lexicographically first, drop the other"})
    return pd.DataFrame(rows)


def write_thumbnail() -> None:
    img = cv2.imread(str(BUSBRA_IMAGES / "bus_0999-l.png"), cv2.IMREAD_GRAYSCALE)
    assert img is not None, "bus_0999-l.png not found (BUS-BRA raw data required)"
    scale = THUMB_WIDTH / img.shape[1]
    thumb = cv2.resize(img, (THUMB_WIDTH, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(THUMB_PNG), thumb)


def main() -> None:
    hits = pd.read_csv(HITS_CSV)
    cross = cross_set_table(hits)
    cross.to_csv(CROSS_CSV, index=False)
    within = within_set_table(hits)
    within.to_csv(WITHIN_CSV, index=False)
    write_thumbnail()
    print(cross.to_string(index=False))
    print(within.to_string(index=False))
    print(f"written: {CROSS_CSV}, {WITHIN_CSV}, {THUMB_PNG}")


if __name__ == "__main__":
    main()

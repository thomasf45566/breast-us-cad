"""v2 line 4 (Amendment 4) split integrity — LOCO holdout, disjointness, fold 5."""

import pytest

from data import build_master_df
from v2_data import (
    BUSBRA_VAL_FOLD,
    EXTERNAL_COHORTS,
    build_multisource_df,
    verify_frozen_preprocessing,
)


@pytest.fixture(scope="module")
def loco_dfs():
    return {c: build_multisource_df(c) for c in EXTERNAL_COHORTS}


def test_held_out_cohort_absent(loco_dfs):
    from external_val import DATASETS

    for hold_out, df in loco_dfs.items():
        assert hold_out not in set(df["cohort"])
        held_paths = set(DATASETS[hold_out][0]()["image_path"])
        assert set(df["image_path"]).isdisjoint(held_paths)


def test_train_val_disjoint(loco_dfs):
    for hold_out, df in loco_dfs.items():
        train = set(df.loc[df["split"] == "train", "image_path"])
        val = set(df.loc[df["split"] == "val", "image_path"])
        assert train.isdisjoint(val), f"train/val overlap for hold_out={hold_out}"
        assert train | val == set(df["image_path"])


def test_busbra_val_is_exactly_fold5(loco_dfs):
    master = build_master_df("data/raw/busbra")
    fold5 = set(master.loc[master["fold"] == BUSBRA_VAL_FOLD, "image_path"])
    for hold_out, df in loco_dfs.items():
        bus_val = set(
            df.loc[(df["cohort"] == "busbra") & (df["split"] == "val"), "image_path"]
        )
        assert bus_val == fold5, f"BUS-BRA val != fold 5 for hold_out={hold_out}"


def test_external_val_slices_identical_across_runs(loco_dfs):
    """Amendment 4 (t): the 15% slices are drawn once and reused across runs."""
    for cohort in EXTERNAL_COHORTS:
        vals = [
            frozenset(
                df.loc[(df["cohort"] == cohort) & (df["split"] == "val"), "image_path"]
            )
            for hold_out, df in loco_dfs.items()
            if hold_out != cohort
        ]
        assert len(set(vals)) == 1, f"{cohort} val slice differs across LOCO runs"
        assert len(vals[0]) > 0


def test_frozen_preprocessing_parity(loco_dfs):
    """20 random external images: format/normalize stages bitwise-identical to
    src/inference.py; any tensor difference confined to cv2's size-dependent
    resize kernel dispatch, bounded at 1 gray level (asserted inside)."""
    n, _ = verify_frozen_preprocessing(loco_dfs["breast"])
    assert n == 20
    n, _ = verify_frozen_preprocessing(loco_dfs["gdph"])  # covers BrEaST RGBA
    assert n == 20

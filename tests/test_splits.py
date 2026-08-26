"""Split integrity tests — golden rule #1: patient-level splits only."""

import pytest

from data import build_master_df


@pytest.fixture(scope="module")
def master_df():
    return build_master_df("data/raw/busbra")


def test_patient_in_single_fold(master_df):
    folds_per_patient = master_df.groupby("patient_id")["fold"].nunique()
    leaky = folds_per_patient[folds_per_patient > 1]
    assert leaky.empty, (
        f"{len(leaky)} patient(s) span multiple folds: {leaky.index.tolist()[:10]}"
    )


def test_expected_folds(master_df):
    assert set(master_df["fold"].unique()) == {1, 2, 3, 4, 5}


def test_labels_complete(master_df):
    assert master_df["label"].notna().all()
    assert set(master_df["label"].unique()) == {0, 1}

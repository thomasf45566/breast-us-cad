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


def test_committed_fold_file_is_the_official_partition():
    """data/splits/busbra_official_5fold.csv must hash to the official file and,
    when the raw dataset is present, be byte-identical to its 5-fold-cv.csv."""
    import hashlib
    from pathlib import Path

    from data import OFFICIAL_FOLDS_CSV, OFFICIAL_FOLDS_SHA256, resolve_fold_file

    assert OFFICIAL_FOLDS_CSV.exists(), "committed fold file missing"
    assert hashlib.sha256(OFFICIAL_FOLDS_CSV.read_bytes()).hexdigest() == OFFICIAL_FOLDS_SHA256
    assert resolve_fold_file("data/raw/busbra") == OFFICIAL_FOLDS_CSV
    raw = Path("data/raw/busbra/5-fold-cv.csv")
    if raw.exists():
        assert raw.read_bytes() == OFFICIAL_FOLDS_CSV.read_bytes()


def test_fold_file_fallback_and_hash_check(tmp_path, monkeypatch):
    """Without the committed copy the raw location is used; a tampered file is refused."""
    import data
    from data import OFFICIAL_FOLDS_CSV, resolve_fold_file

    root = tmp_path / "busbra"
    root.mkdir()
    (root / "5-fold-cv.csv").write_bytes(OFFICIAL_FOLDS_CSV.read_bytes())
    monkeypatch.setattr(data, "OFFICIAL_FOLDS_CSV", tmp_path / "absent.csv")
    assert resolve_fold_file(root) == root / "5-fold-cv.csv"
    (root / "5-fold-cv.csv").write_bytes(b"ID,Pathology,kFold\nbus_0001-l,malignant,1\n")
    with pytest.raises(ValueError, match="not the official"):
        resolve_fold_file(root)

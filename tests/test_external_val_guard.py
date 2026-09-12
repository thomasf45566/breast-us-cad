"""external_val single-shot guard (post-hoc enforcement addition, 2026-09-12).

Pure-function tests: no data, no model, no inference. The guard must refuse
an existing prediction CSV unless --overwrite is given, and must do so
before anything is scored.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from external_val import preds_path_for, refuse_existing_output  # noqa: E402


def test_guard_allows_missing_output(tmp_path):
    assert refuse_existing_output("breast", overwrite=False, reports_dir=tmp_path) == tmp_path / "external_breast_preds.csv"


def test_guard_refuses_existing_output(tmp_path):
    (tmp_path / "external_breast_preds.csv").write_text("image_path,y_true\n")
    with pytest.raises(SystemExit, match="REFUSED"):
        refuse_existing_output("breast", overwrite=False, reports_dir=tmp_path)


def test_guard_overwrite_flag_permits(tmp_path):
    (tmp_path / "external_busi_preds.csv").write_text("image_path,y_true\n")
    assert refuse_existing_output("busi", overwrite=True, reports_dir=tmp_path) == preds_path_for("busi", tmp_path)


def test_v1_outputs_present_and_guarded():
    """The four external-v1 CSVs exist in the repository, so a repeated
    `--dataset all --confirm` without --overwrite is refused for every cohort."""
    for name in ("breast", "busi", "gdph", "sysucc"):
        assert preds_path_for(name).exists()
        with pytest.raises(SystemExit):
            refuse_existing_output(name, overwrite=False)

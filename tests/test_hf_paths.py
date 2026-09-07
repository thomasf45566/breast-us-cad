"""inference.hf_repo_path: v1 files at the HF repo root, v2 files under v2/."""

from inference import hf_repo_path


def test_v1_files_stay_at_root():
    for name in ("cv_vit_fold5.pt", "seg_unet_effb0.pt", "calibration.json", "operating_point.json"):
        assert hf_repo_path(name) == name


def test_v2_files_map_under_v2_prefix():
    assert hf_repo_path("v2_biomedclip_fold1.pt") == "v2/v2_biomedclip_fold1.pt"
    assert hf_repo_path("v2_loco_gdph_operating_point.json") == "v2/v2_loco_gdph_operating_point.json"
    assert hf_repo_path("pretrained/biomedclip_vitb16_timm.pt") == "v2/pretrained/biomedclip_vitb16_timm.pt"

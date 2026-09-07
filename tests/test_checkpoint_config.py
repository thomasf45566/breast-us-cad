"""Checkpoints must record the folds actually used (audit 2026-09-06 §1)."""

import copy

import torch

from train import checkpoint_payload

CFG = {
    "data": {"root": "data/raw/busbra", "img_size": 224, "train_folds": [1, 2, 3, 4], "val_folds": [5]},
    "model": {"name": "vit_base_patch16_224", "pretrained": True},
}


def test_payload_stores_resolved_folds_not_yaml_defaults():
    state = {"w": torch.zeros(1)}
    ckpt = checkpoint_payload(CFG, state, epoch=7, val_auc=0.9, train_folds=[2, 3, 4, 5], val_folds=[1])
    assert ckpt["config"]["data"]["train_folds"] == [2, 3, 4, 5]
    assert ckpt["config"]["data"]["val_folds"] == [1]
    assert ckpt["epoch"] == 7 and ckpt["val_auc"] == 0.9
    assert ckpt["model_state"] is state


def test_payload_keeps_other_config_and_does_not_mutate_input():
    before = copy.deepcopy(CFG)
    ckpt = checkpoint_payload(CFG, {}, epoch=1, val_auc=0.5, train_folds=[1, 2, 3, 5], val_folds=[4])
    assert CFG == before, "checkpoint_payload must not mutate the caller's config"
    assert ckpt["config"]["model"] == CFG["model"]
    assert ckpt["config"]["data"]["root"] == CFG["data"]["root"]


def test_payload_round_trips_through_torch_save(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint_payload(CFG, {}, epoch=3, val_auc=0.8, train_folds=[1, 2, 4, 5], val_folds=[3]), path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["config"]["data"]["val_folds"] == [3]

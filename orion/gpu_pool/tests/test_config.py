from __future__ import annotations

import copy

import pytest
import yaml

from orion.gpu_pool.config import DEFAULT_PATH, PoolConfig, check_vram, load_pool_config

RAW = yaml.safe_load(DEFAULT_PATH.read_text())


def bad(mutate):
    data = copy.deepcopy(RAW)
    mutate(data)
    with pytest.raises(ValueError):
        PoolConfig.model_validate(data)


def test_shipped_config_is_valid_and_names_no_models():
    cfg = load_pool_config()
    body = "\n".join(l.split("#")[0] for l in DEFAULT_PATH.read_text().splitlines()).lower()
    for model_word in ("qwen", "27b", "35b", "gguf", "deepseek"):
        assert model_word not in body, model_word
    assert cfg.digest


def test_unknown_card_rejected():
    bad(lambda d: d["roles"]["chat"].update(cards=["gpu9"]))


def test_unknown_role_in_class_rejected():
    bad(lambda d: d["classes"]["agent"]["roles"].append("nope"))


def test_duplicate_port_rejected():
    bad(lambda d: d["roles"]["fast"].update(port=8012))


def test_owner_must_list_the_role():
    bad(lambda d: d["roles"]["chat"].update(owner="world"))


def test_service_role_needs_slots_and_vram():
    bad(lambda d: d["roles"]["world"].pop("slots"))


def test_eviction_must_share_a_card():
    bad(lambda d: d["roles"]["agent-gpu2"]["swap"].update(evicts=["chat"]))


def test_vram_overflow_detected():
    cfg = load_pool_config()
    assert check_vram(cfg, {"world": 1, "diffusion": 24}) == []
    assert check_vram(cfg, {"world": 10, "diffusion": 24})
    assert check_vram(cfg, {"world": 1, "diffusion": 24, "agent-gpu2": 32})

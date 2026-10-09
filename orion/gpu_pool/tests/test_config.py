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
    """Discovery learns what each worker runs; the pool YAML never assumes it. The one exception
    (stage 5, Decision 2): a role's launch.profiles allow-list names config/llm_profiles.yaml
    profiles the actuator may load -- checked against that file by check_launch, so it cannot
    drift into a free-form model name. Those entries are removed before the check."""
    cfg = load_pool_config()
    body = "\n".join(l.split("#")[0] for l in DEFAULT_PATH.read_text().splitlines()).lower()
    allowed = {p.lower() for spec in cfg.roles.values() if spec.launch for p in spec.launch.profiles}
    for profile in sorted(allowed, key=len, reverse=True):
        body = body.replace(profile, "<profile>")
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


def test_urgent_is_the_highest_priority_with_its_defaults():
    cfg = load_pool_config()
    assert cfg.priorities[0] == "urgent"
    assert cfg.priority_rank("urgent") < cfg.priority_rank("interactive")
    assert cfg.defaults.urgent_preempt_grace_sec == 5
    assert cfg.defaults.urgent_max_concurrent == 3


def test_priorities_must_include_urgent():
    bad(lambda d: d.update(priorities=["interactive", "system", "background"]))


def test_urgent_route_priority_parses():
    data = copy.deepcopy(RAW)
    data.setdefault("routes", {})["probe"] = {"class": next(iter(data["classes"])), "priority": "urgent"}
    assert PoolConfig.model_validate(data).routes["probe"].priority == "urgent"


def test_resource_requirement_accepts_urgent_only_besides_background():
    from pydantic import ValidationError

    from orion.schemas.resource_admission import ResourceRequirementV1

    assert ResourceRequirementV1(priority="urgent").priority == "urgent"
    assert ResourceRequirementV1().priority == "background"
    with pytest.raises(ValidationError):
        ResourceRequirementV1(priority="interactive")

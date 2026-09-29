"""GPU pool stage 5.1: config + contracts
(docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md).

serialize_with (config + scheduler), launch.profile_var/profiles, the cuda_env interpolation gate,
the agent-burst compose migration, world on_unavailable=wait, experiment not_actuatable, and the
agent-gpu2 seat limit (Juniper 2026-09-29). Worked examples A (add gpu4) and B (a second model
option for gpu2's seat) must pass the validator with config-only edits.
"""
from __future__ import annotations

import copy
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orion.gpu_pool.config import (
    DEFAULT_PATH, PoolConfig, check_launch, launch_digest, load_pool_config,
)
from orion.gpu_pool.scheduler import (
    Backlog, CardLive, Grant, LeaseView, Recall, RoleLive, Serialized, SwapUnload, Unavailable, schedule,
)

ROOT = Path(__file__).resolve().parents[3]
RAW = yaml.safe_load(DEFAULT_PATH.read_text())
CFG = load_pool_config()
T0 = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
AGENT_27B = "qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex"
VISION_PROFILE = "muse-glimmer-30b-udq4kxl-v100-32gb-agent-vision"
TEMPLATES = ("services/orion-llamacpp-host/docker-compose.atlas-workers.yml",
             "services/orion-llamacpp-host/.env_example",
             "services/orion-diffusion-host/docker-compose.yml",
             "services/orion-diffusion-host/.env_example",
             "config/llm_profiles.yaml")


def _tree(tmp_path: Path) -> Path:
    for rel in TEMPLATES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text((ROOT / rel).read_text())
    return tmp_path


def _edit_compose(tmp_path: Path, rel: str, edit) -> None:
    path = tmp_path / rel
    compose = yaml.safe_load(path.read_text())
    edit(compose["services"])
    path.write_text(yaml.safe_dump(compose))


def _set_env(svc: dict, key: str, value: str) -> None:
    svc["environment"] = [e for e in svc["environment"] if not e.startswith(key + "=")] + [f"{key}={value}"]


BURST = "services/orion-llamacpp-host/docker-compose.atlas-workers.yml"
DIFF = "services/orion-diffusion-host/docker-compose.yml"


# --- the live config ------------------------------------------------------------------------------
def test_live_config_is_accepted_and_gate_is_clean():
    assert check_launch(CFG, ROOT) == []


def test_live_config_carries_the_5_1_values():
    assert CFG.roles["agent-gpu2"].max_hold_sec == 9000
    assert CFG.classes["world"].on_unavailable == "wait"
    assert CFG.roles["world"].serialize_with == ["diffusion"]
    assert CFG.serialized_with("world") == ["diffusion"]
    assert CFG.serialized_with("diffusion") == ["world"]          # symmetric
    assert CFG.serialized_with("agent-gpu2") == []
    launch = CFG.roles["agent-gpu2"].launch
    assert launch.cuda_env == "ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES"
    assert launch.profile_var == "ATLAS_AGENT_BURST_PROFILE_NAME"
    assert launch.profiles == []          # nothing produces a profile until 5.3: compose default
    assert CFG.roles["diffusion"].launch.cuda_env == "CUDA_VISIBLE_DEVICES"
    # agent-gpu2 keeps its bridge until 5.3; experiment lost its dead verbs and is not actuatable
    assert CFG.roles["agent-gpu2"].swap.bridged
    assert not CFG.roles["experiment"].swap.bridged and CFG.roles["experiment"].launch is None


def test_agent_burst_compose_resolves_to_todays_device_and_profile():
    """No behaviour change: with the committed templates the seat still lands on CUDA device 2 and
    LLM_PROFILE_NAME still falls back to ATLAS_AGENT_PROFILE_NAME."""
    from orion.gpu_pool.config import _env_pairs, _read_env_template, _resolve

    svc = yaml.safe_load((ROOT / BURST).read_text())["services"]["atlas-agent-burst"]
    env = _env_pairs(svc["environment"])
    template = _read_env_template(ROOT / "services/orion-llamacpp-host/.env_example")
    assert env["CUDA_VISIBLE_DEVICES_OVERRIDE"] == "${ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES:-2}"
    assert _resolve(env["CUDA_VISIBLE_DEVICES_OVERRIDE"], template) == "2"
    assert _resolve(env["CUDA_VISIBLE_DEVICES_OVERRIDE"], {}) == "2"          # key absent on a host
    assert template["ATLAS_AGENT_BURST_PROFILE_NAME"] == ""
    assert _resolve(env["LLM_PROFILE_NAME"], {**template, "ATLAS_AGENT_PROFILE_NAME": AGENT_27B}) == AGENT_27B
    assert _resolve(env["LLM_PROFILE_NAME"], {"ATLAS_AGENT_PROFILE_NAME": AGENT_27B}) == AGENT_27B
    assert _resolve(env["LLM_PROFILE_NAME"], {"ATLAS_AGENT_BURST_PROFILE_NAME": "x",
                                              "ATLAS_AGENT_PROFILE_NAME": AGENT_27B}) == "x"


# --- cuda_env gate --------------------------------------------------------------------------------
@pytest.mark.parametrize("value,match", [
    ("2", "must be ${ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES}"),          # acceptance 4: the old literal
    ("${SOMETHING_ELSE:-2}", "must be ${ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES}"),
    ("${ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES-2}", "must be"),          # only ${X} or ${X:-index}
    ("${ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES:-3}", "defaults to 3"),
    ("0,${ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES}", "must be"),
])
def test_gate_rejects_a_device_the_actuator_cannot_set(tmp_path, value, match):
    _tree(tmp_path)
    _edit_compose(tmp_path, BURST, lambda s: _set_env(s["atlas-agent-burst"], "CUDA_VISIBLE_DEVICES_OVERRIDE", value))
    problems = check_launch(CFG, tmp_path)
    assert any(match in p for p in problems), problems


def test_gate_rejects_a_second_device_key_left_literal(tmp_path):
    _tree(tmp_path)
    _edit_compose(tmp_path, BURST, lambda s: _set_env(s["atlas-agent-burst"], "CUDA_VISIBLE_DEVICES", "2"))
    problems = check_launch(CFG, tmp_path)
    assert any("CUDA_VISIBLE_DEVICES=2 must be" in p for p in problems), problems


def test_gate_rejects_template_value_off_the_card_index(tmp_path):
    _tree(tmp_path)
    example = tmp_path / "services/orion-llamacpp-host/.env_example"
    example.write_text(example.read_text().replace("ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES=2",
                                                   "ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES=1"))
    problems = check_launch(CFG, tmp_path)
    assert any("CUDA_VISIBLE_DEVICES_OVERRIDE=1 (from templates)" in p for p in problems), problems


def test_gate_accepts_a_device_ids_pin_through_cuda_env_and_rejects_another_var(tmp_path):
    _tree(tmp_path)
    ids = lambda v: (lambda s: s["atlas-agent-burst"]["deploy"]["resources"]["reservations"]["devices"][0]
                     .update(device_ids=[v]))
    _edit_compose(tmp_path, BURST, ids("${ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES:-2}"))
    assert check_launch(CFG, tmp_path) == []
    _edit_compose(tmp_path, BURST, ids("${SOME_OTHER_VAR}"))
    problems = check_launch(CFG, tmp_path)
    assert any("pins device_ids ['${SOME_OTHER_VAR}']" in p for p in problems), problems


def test_gate_nested_device_default_is_not_read_as_a_literal(tmp_path):
    _tree(tmp_path)
    _edit_compose(tmp_path, BURST, lambda s: _set_env(
        s["atlas-agent-burst"], "CUDA_VISIBLE_DEVICES_OVERRIDE",
        "${ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES:-${SOME_UNSET_VAR:-2}}"))
    assert check_launch(CFG, tmp_path) == []


def test_gate_checks_profiles_against_the_tree_it_checks(tmp_path):
    _tree(tmp_path)
    (tmp_path / "config/llm_profiles.yaml").unlink()
    problems = check_launch(_with_profiles([AGENT_27B]), tmp_path)
    assert any("llm_profiles.yaml not found" in p for p in problems), problems


def test_gate_rejects_a_literal_compose_device_pin(tmp_path):
    _tree(tmp_path)
    _edit_compose(tmp_path, BURST, lambda s: s["atlas-agent-burst"]["deploy"]["resources"]["reservations"]
                  ["devices"][0].update(device_ids=["2"]))
    problems = check_launch(CFG, tmp_path)
    assert any("pins device_ids ['2']" in p for p in problems), problems


def test_diffusions_bare_interpolation_already_satisfies_the_gate(tmp_path):
    _tree(tmp_path)
    _edit_compose(tmp_path, DIFF, lambda s: _set_env(s["diffusion-host"], "CUDA_VISIBLE_DEVICES", "2"))
    problems = check_launch(CFG, tmp_path)
    assert any("diffusion-host CUDA_VISIBLE_DEVICES=2 must be ${CUDA_VISIBLE_DEVICES}" in p for p in problems)


# --- profile_var / profiles -----------------------------------------------------------------------
def _with_profiles(profiles, **launch_overrides) -> PoolConfig:
    data = copy.deepcopy(RAW)
    data["roles"]["agent-gpu2"]["launch"].update(profiles=profiles, **launch_overrides)
    return PoolConfig.model_validate(data)


def test_worked_example_b_second_model_option_is_config_only(tmp_path):
    """Spec worked example B: a second model option for gpu2's seat = one YAML list (the profile
    already exists in llm_profiles.yaml); the compose file is untouched."""
    cfg = _with_profiles([AGENT_27B, VISION_PROFILE])
    assert check_launch(cfg, _tree(tmp_path)) == []
    assert cfg.roles["agent-gpu2"].launch.profiles[0] == AGENT_27B     # first = default
    assert launch_digest(cfg, "agent-gpu2") != launch_digest(CFG, "agent-gpu2")  # actuator must agree


def test_gate_rejects_an_unknown_profile(tmp_path):
    problems = check_launch(_with_profiles([AGENT_27B, "no-such-profile"]), _tree(tmp_path))
    assert any("no-such-profile is not a profile in config/llm_profiles.yaml" in p for p in problems), problems


def test_gate_rejects_llm_profile_name_that_ignores_profile_var(tmp_path):
    _tree(tmp_path)
    _edit_compose(tmp_path, BURST, lambda s: _set_env(s["atlas-agent-burst"], "LLM_PROFILE_NAME",
                                                      "${ATLAS_AGENT_PROFILE_NAME}"))
    problems = check_launch(CFG, tmp_path)
    assert any("LLM_PROFILE_NAME=${ATLAS_AGENT_PROFILE_NAME} must be ${ATLAS_AGENT_BURST_PROFILE_NAME}" in p
               for p in problems), problems


@pytest.mark.parametrize("launch,match", [
    ({"profile_var": None, "profiles": [AGENT_27B]}, "needs launch.profile_var"),
    ({"profiles": [AGENT_27B, AGENT_27B]}, "duplicates"),
    ({"profile_var": "ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES"}, "must be different"),
    ({"profile_var": "lower-case"}, "profile_var"),
])
def test_rejects_bad_profile_shape(launch, match):
    data = copy.deepcopy(RAW)
    data["roles"]["agent-gpu2"]["launch"].update(launch)
    with pytest.raises(ValidationError, match=match):
        PoolConfig.model_validate(data)


# --- serialize_with: validation -------------------------------------------------------------------
@pytest.mark.parametrize("names,match", [
    (["nope"], "serialize_with unknown role nope"),
    (["world"], "serialize_with names itself"),
    (["agent"], "serialize_with agent, which shares no card"),
    (["diffusion", "diffusion"], "serialize_with has duplicates"),
])
def test_rejects_bad_serialize_with(names, match):
    data = copy.deepcopy(RAW)
    data["roles"]["world"]["serialize_with"] = names
    with pytest.raises(ValidationError, match=match):
        PoolConfig.model_validate(data)


def test_non_operator_swap_seat_without_launch_or_bridge_is_still_refused():
    data = copy.deepcopy(RAW)
    data["roles"]["agent-gpu2"]["swap"].pop("load"), data["roles"]["agent-gpu2"]["swap"].pop("unload")
    data["roles"]["agent-gpu2"].pop("launch")
    with pytest.raises(ValidationError, match="need a launch"):
        PoolConfig.model_validate(data)


# --- serialize_with: scheduler (Z1) --------------------------------------------------------------
def _live():
    return {
        "chat": RoleLive("chat", True, 1, 65536, True), "agent": RoleLive("agent", True, 1, 131072),
        "agent-gpu2": RoleLive("agent-gpu2", True, 1, 131072), "metacog": RoleLive("metacog", True, 4, 4096),
        "fast": RoleLive("fast", True, 4, 4096), "world": RoleLive("world", True, 2),
        "diffusion": RoleLive("diffusion", True, 1), "experiment": RoleLive("experiment", True, 1, 8192),
    }


def _lease(lid, work_class, status="queued", role=None, age=0, **kw):
    return LeaseView(lease_id=lid, work_class=work_class, priority=kw.pop("priority", "system"), status=status,
                     role=role, created_at=T0 - timedelta(seconds=age), **kw)


def _run(leases, cfg=CFG):
    return schedule(cfg, _live(), {c: CardLive(c) for c in cfg.cards}, leases, T0)


def _of(kind, decisions):
    return [d for d in decisions if isinstance(d, kind)]


def test_world_waits_while_a_diffusion_hold_is_active_and_says_why():
    hold = _lease("h", "diffusion", "granted", "diffusion", kind="hold", priority="background")
    w = _lease("w", "world", deadline_at=T0 + timedelta(seconds=2))
    d = _run([hold, w])
    assert not _of(Grant, d)
    assert [(s.lease_id, s.role, s.reason) for s in _of(Serialized, d)] == [("w", "world", "serialized:diffusion")]
    assert not _of(Recall, d)                                # no preemption across the pair
    assert not _of(Unavailable, d) and not _of(Backlog, d)   # it is serviceable: it waits for its deadline


def test_world_past_its_deadline_is_unavailable_deadline():
    hold = _lease("h", "diffusion", "granted", "diffusion", kind="hold", priority="background")
    w = _lease("w", "world", deadline_at=T0)
    assert [(u.lease_id, u.reason) for u in _of(Unavailable, _run([hold, w]))] == [("w", "deadline")]


def test_diffusion_hold_waits_its_turn_while_world_computes():
    w = _lease("w", "world", "granted", "world")
    hold = _lease("h", "diffusion", kind="hold", priority="background", retryable=True)
    d = _run([w, hold])
    assert "h" not in {g.lease_id for g in _of(Grant, d)}
    assert [(s.lease_id, s.reason) for s in _of(Serialized, d)] == [("h", "serialized:world")]
    assert not _of(Recall, d)
    assert not _of(Backlog, d)


def test_a_child_on_the_partner_role_also_blocks():
    hold = _lease("h", "diffusion", "granted", "diffusion", kind="hold", priority="background")
    call = _lease("c", "diffusion", "granted", "diffusion", hold_lease_id="h")
    w = _lease("w", "world")
    d = _run([hold, call, w])
    assert not _of(Grant, d)
    assert [(s.lease_id, s.reason) for s in _of(Serialized, d)] == [("w", "serialized:diffusion")]


def test_an_older_waiter_keeps_its_place_against_a_stream_on_the_other_side():
    """world has 2 slots: without the reservation, overlapping world calls would starve an older
    diffusion hold forever (a younger w2 slipping in while w1 still runs)."""
    w1 = _lease("w1", "world", "granted", "world", priority="background")
    h = _lease("h", "diffusion", kind="hold", priority="background", age=100, retryable=True)
    w2 = _lease("w2", "world", priority="background", age=1)
    d = _run([w1, h, w2])
    assert not _of(Grant, d)                                  # w2 waits behind the older hold
    assert {(s.lease_id, s.reason) for s in _of(Serialized, d)} == {
        ("h", "serialized:world"), ("w2", "serialized:diffusion")}
    d = _run([h, w2])                                         # w1 finished: the hold goes first
    assert {g.lease_id: g.role for g in _of(Grant, d)} == {"h": "diffusion"}


def test_higher_priority_still_goes_first_across_the_pair():
    w1 = _lease("w1", "world", "granted", "world")
    h = _lease("h", "diffusion", kind="hold", priority="background", age=100, retryable=True)
    w2 = _lease("w2", "world", priority="system", age=1)
    assert {g.lease_id for g in _of(Grant, _run([w1, h, w2]))} == {"w2"}   # queue order is priority first


def test_urgent_capped_lease_is_not_reported_serialized():
    import dataclasses
    data = copy.deepcopy(RAW)
    data["defaults"]["urgent_max_concurrent"] = 1
    cfg = PoolConfig.model_validate(data)
    busy = _lease("u0", "agent", "granted", "agent", priority="urgent")
    hold = _lease("h", "diffusion", "granted", "diffusion", kind="hold", priority="background")
    w = _lease("w", "world", priority="urgent")
    assert not _of(Serialized, _run([busy, hold, w], cfg=cfg))


def test_both_queued_in_one_tick_grants_only_one_side():
    d = _run([_lease("w", "world", age=5), _lease("h", "diffusion", kind="hold", priority="background", age=1)])
    granted = {g.lease_id: g.role for g in _of(Grant, d)}
    assert granted == {"w": "world"}                       # queue order: system before background
    assert [(s.lease_id, s.reason) for s in _of(Serialized, d)] == [("h", "serialized:world")]


def test_world_still_uses_both_its_own_slots():
    d = _run([_lease("w1", "world", "granted", "world"), _lease("w2", "world")])
    assert {g.lease_id for g in _of(Grant, d)} == {"w2"}
    assert not _of(Serialized, d)


def test_removing_the_yaml_line_removes_the_mutex():
    data = copy.deepcopy(RAW)
    data["roles"]["world"].pop("serialize_with")
    cfg = PoolConfig.model_validate(data)
    hold = _lease("h", "diffusion", "granted", "diffusion", kind="hold", priority="background")
    d = _run([hold, _lease("w", "world")], cfg=cfg)
    assert {g.lease_id: g.role for g in _of(Grant, d)} == {"w": "world"}
    assert not _of(Serialized, d)


def test_no_serialized_report_when_the_role_is_full_anyway():
    # both world slots taken by world itself: waiting is ordinary, not the mutex
    held = [_lease(f"w{i}", "world", "granted", "world") for i in range(2)]
    d = _run(held + [_lease("w", "world")])
    assert not _of(Serialized, d)


# --- world is `wait`, agent-gpu2's seat limit ----------------------------------------------------
def test_unserviceable_world_request_waits_instead_of_backlogging():
    live = _live()
    live["world"] = RoleLive("world", False, 2)
    d = schedule(CFG, live, {c: CardLive(c) for c in CFG.cards}, [_lease("w", "world", retryable=True)], T0)
    assert not _of(Backlog, d) and not _of(Unavailable, d)


@pytest.mark.parametrize("loaded_sec,drains", [(8999, False), (9000, True)])
def test_agent_gpu2_seat_limit_is_9000s(loaded_sec, drains):
    crds = {c: CardLive(c) for c in CFG.cards}
    crds["gpu2"] = CardLive("gpu2", swapped_in={"agent-gpu2"}, loaded_at=T0 - timedelta(seconds=loaded_sec))
    d = schedule(CFG, _live(), crds, [], T0)
    assert bool([u for u in _of(SwapUnload, d) if u.reason == "max_hold"]) is drains


# --- worked example A: add gpu4 hosting an 8B or a vision model -----------------------------------
GPU4_COMPOSE = textwrap.dedent("""
    services:
      atlas-fast2:
        environment:
          - LLM_ROLE=fast2
          - LLM_ANNOUNCE_PORT=${ATLAS_FAST2_HOST_PORT:-8017}
          - LLM_PROFILE_NAME=${ATLAS_FAST2_PROFILE_NAME}
          - CUDA_VISIBLE_DEVICES_OVERRIDE=${ATLAS_FAST2_CUDA_VISIBLE_DEVICES:-4}
      atlas-vision4:
        profiles: ["vision4"]
        environment:
          - LLM_ROLE=vision4
          - LLM_ANNOUNCE_PORT=${ATLAS_VISION4_HOST_PORT:-8018}
          - LLM_PROFILE_NAME=${ATLAS_VISION4_PROFILE_NAME}
          - CUDA_VISIBLE_DEVICES_OVERRIDE=${ATLAS_VISION4_CUDA_VISIBLE_DEVICES:-4}
""")


def _example_a() -> dict:
    """The spec's YAML for example A, verbatim except: `metacog` also lists fast2 (its owner rule;
    the spec text shows it), and <vision profile> is a real profile."""
    data = copy.deepcopy(RAW)
    compose = "services/orion-llamacpp-host/docker-compose.atlas-workers.yml"
    env_file = "services/orion-llamacpp-host/.env"
    data["cards"]["gpu4"] = {"vram_gb": 32, "index": 4}
    data["roles"]["fast2"] = {
        "kind": "llm", "cards": ["gpu4"], "owner": ["metacog", "fast"], "port": 8017,
        "launch": {"actuator": "circe", "compose": compose, "env_file": env_file, "service": "atlas-fast2",
                   "cuda_env": "ATLAS_FAST2_CUDA_VISIBLE_DEVICES", "ready": "/health", "timeout_sec": 300}}
    data["roles"]["vision4"] = {
        "kind": "llm", "cards": ["gpu4"], "owner": "vision", "port": 8018,
        "launch": {"actuator": "circe", "compose": compose, "env_file": env_file, "service": "atlas-vision4",
                   "compose_profile": "vision4", "cuda_env": "ATLAS_VISION4_CUDA_VISIBLE_DEVICES",
                   "profile_var": "ATLAS_VISION4_PROFILE_NAME", "profiles": [VISION_PROFILE],
                   "ready": "/health", "timeout_sec": 600},
        "swap": {"evicts": ["fast2"], "guards": ["thermal"]}}
    data["classes"]["fast"] = {"roles": ["fast", "metacog", "fast2", "agent", "agent-gpu2", "chat"],
                               "on_unavailable": "wait"}
    data["classes"]["metacog"] = {"roles": ["metacog", "fast", "fast2", "agent", "agent-gpu2", "chat"],
                                  "on_unavailable": "backlog"}
    data["classes"]["vision"] = {"roles": ["vision4"], "on_unavailable": "backlog"}
    return data


def test_worked_example_a_add_gpu4_is_config_only(tmp_path):
    cfg = PoolConfig.model_validate(_example_a())
    _tree(tmp_path)
    _edit_compose(tmp_path, BURST, lambda s: s.update(yaml.safe_load(GPU4_COMPOSE)["services"]))
    assert check_launch(cfg, tmp_path) == []
    assert cfg.evicted_by("vision4") == ["fast2"] and not cfg.roles["vision4"].swap.bridged


def test_worked_example_a_with_a_literal_device_is_refused(tmp_path):
    cfg = PoolConfig.model_validate(_example_a())
    _tree(tmp_path)
    literal = GPU4_COMPOSE.replace("${ATLAS_VISION4_CUDA_VISIBLE_DEVICES:-4}", "4")
    _edit_compose(tmp_path, BURST, lambda s: s.update(yaml.safe_load(literal)["services"]))
    problems = check_launch(cfg, tmp_path)
    assert any("atlas-vision4 CUDA_VISIBLE_DEVICES_OVERRIDE=4 must be" in p for p in problems), problems

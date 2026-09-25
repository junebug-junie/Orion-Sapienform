"""Stage 4.1 contracts: hold/attach lease verbs, the lease reference, actuation messages, and the
YAML `launch` / `index` / `actuators` / per-seat swap keys with their static gate.

Spec: docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md
"""
from __future__ import annotations

import copy
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orion.gpu_pool.config import (
    DEFAULT_PATH, PoolConfig, check_launch, launch_digest, load_pool_config,
)
from orion.llm.resource_lease import (
    GPU_LEASE_HEADER, ResourceLeaseRejected, decode_gpu_lease_header, encode_gpu_lease_header,
)
from orion.schemas.gpu_pool import (
    GpuActuateResultV1, GpuActuateV1, GpuCardStateV1, GpuLeaseRefV1, GpuLeaseRequestV1, GpuPoolEventV1,
)
from orion.schemas.registry import SCHEMA_REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[3]
RAW = yaml.safe_load(DEFAULT_PATH.read_text())
NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)


# --- lease verbs -----------------------------------------------------------------------------
def test_attach_round_trips_and_names_the_hold_not_a_lease_id():
    req = GpuLeaseRequestV1(verb="attach", request_id="call-1", holder="gateway", work_class="agent",
                            hold_lease_id="hold-1", hold_generation=3, turn_correlation_id="turn-9")
    back = GpuLeaseRequestV1.model_validate(req.model_dump(mode="json"))
    assert back == req and back.hold_lease_id == "hold-1" and back.hold_generation == 3


@pytest.mark.parametrize("kw", [
    {"hold_lease_id": "hold-1"},                                    # no generation
    {"hold_generation": 2},                                         # no hold
    {"hold_lease_id": "hold-1", "hold_generation": 0},            # generations start at 1
    {"hold_lease_id": "hold-1", "hold_generation": 2, "lease_id": "x"},  # the child's id is the pool's
])
def test_attach_rejects_incomplete_parent(kw):
    with pytest.raises(ValidationError):
        GpuLeaseRequestV1(verb="attach", request_id="c", work_class="agent", **kw)


@pytest.mark.parametrize("missing", ["request_id", "work_class"])
def test_attach_needs_idempotency_key_and_class(missing):
    kw = dict(request_id="c", work_class="agent", hold_lease_id="hold-1", hold_generation=1)
    kw.pop(missing)
    with pytest.raises(ValidationError):
        GpuLeaseRequestV1(verb="attach", **kw)


def test_status_needs_lease_id():
    with pytest.raises(ValidationError):
        GpuLeaseRequestV1(verb="status")


@pytest.mark.parametrize("verb", ["acquire", "heartbeat", "release", "cancel", "status"])
def test_parent_fields_only_on_attach(verb):
    with pytest.raises(ValidationError):
        GpuLeaseRequestV1(verb=verb, lease_id="l", hold_lease_id="hold-1", hold_generation=1)


def test_status_verb_and_existing_verbs_still_parse():
    assert GpuLeaseRequestV1(verb="status", lease_id="hold-1").verb == "status"
    # What today's producers send (orion/gpu_pool/client.py) is unchanged.
    GpuLeaseRequestV1(verb="acquire", request_id="r", holder="h", work_class="agent", kind="hold", retryable=True)
    GpuLeaseRequestV1(verb="heartbeat", lease_id="l")


def test_lease_ref_round_trip_and_header():
    ref = GpuLeaseRefV1(lease_id="hold-1", generation=2, role="agent", holder="durable-runs:run-7")
    assert GpuLeaseRefV1.model_validate(ref.model_dump(mode="json")) == ref
    assert GPU_LEASE_HEADER == "X-Orion-Gpu-Lease"
    assert decode_gpu_lease_header(encode_gpu_lease_header(ref)) == ref
    assert decode_gpu_lease_header(encode_gpu_lease_header(ref).rstrip("=")) == ref   # proxy-stripped padding
    for junk in ("", "not base64!", encode_gpu_lease_header(ref)[:-4] + "AAAA"):
        with pytest.raises(ResourceLeaseRejected):
            decode_gpu_lease_header(junk)
    with pytest.raises(ValidationError):
        GpuLeaseRefV1(lease_id="hold-1", generation=0, role="agent", holder="h")


# --- pool events / card state ----------------------------------------------------------------
@pytest.mark.parametrize("event", ["swap_started", "swap_failed", "actuate_refused", "swap_requested", "swapped"])
def test_new_swap_events(event):
    assert GpuPoolEventV1(event=event, role="agent-gpu2", cards=["gpu2"]).event == event


def test_card_fault_state():
    assert GpuCardStateV1(card="gpu2", vram_gb=32, swap_state="fault").swap_state == "fault"


# --- actuation --------------------------------------------------------------------------------
def _actuate(**kw):
    base = dict(action_id="a1", generation=9, actuator="circe", role="agent-gpu2", action="load",
                cards=["gpu2"], launch_digest="d" * 64, deadline_at=NOW, reason="demand")
    return GpuActuateV1(**{**base, **kw})


def test_actuate_round_trip():
    msg = _actuate()
    assert msg.profile is None
    assert GpuActuateV1.model_validate(msg.model_dump(mode="json")) == msg


@pytest.mark.parametrize("kw", [{"generation": 0}, {"action": "restart"}, {"cards": []},
                                {"launch_digest": ""}, {"target": "gpu2/agent"}])
def test_actuate_rejects(kw):
    with pytest.raises(ValidationError):
        _actuate(**kw)


def test_actuate_result_round_trip_and_restored_only_on_failure():
    res = GpuActuateResultV1(action_id="a1", generation=9, role="agent-gpu2", action="load", status="failed",
                             phase="rolling_back", restored=True, elapsed_ms=120_000.0, reason="ready_timeout",
                             observed={"agent-gpu2": "exited", "diffusion": "running"})
    assert GpuActuateResultV1.model_validate(res.model_dump(mode="json")) == res
    with pytest.raises(ValidationError):
        GpuActuateResultV1(action_id="a1", generation=9, role="r", action="load", status="succeeded", restored=True)
    with pytest.raises(ValidationError):
        GpuActuateResultV1(action_id="a1", generation=9, role="r", action="unload", status="failed", restored=False)
    with pytest.raises(ValidationError):
        GpuActuateResultV1(action_id="a1", generation=9, role="r", action="load", status="accepted",
                           observed={"r": "sleeping"})


def test_actuate_schemas_resolve_through_the_bus_registry():
    # resolve() reads _REGISTRY -- the map the bus validates publishes against.
    assert resolve("GpuActuateV1") is GpuActuateV1
    assert resolve("GpuActuateResultV1") is GpuActuateResultV1
    assert SCHEMA_REGISTRY["GpuActuateV1"].kind == "gpu_pool.actuate.v1"
    assert SCHEMA_REGISTRY["GpuActuateResultV1"].kind == "gpu_pool.actuate.result.v1"
    channels = {c["name"]: c for c in yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text())["channels"]}
    assert channels["orion:gpu_pool:actuate:request"]["schema_id"] == "GpuActuateV1"
    assert channels["orion:gpu_pool:actuate:result"]["schema_id"] == "GpuActuateResultV1"


# --- config: the shipped file -------------------------------------------------------------------
def test_shipped_config_launch_blocks_match_compose():
    cfg = load_pool_config()
    assert cfg.roles["agent-gpu2"].launch.service == "atlas-agent-burst"
    assert cfg.roles["diffusion"].launch.drain.set_path == "/v1/lifecycle/drain"
    assert cfg.cards["gpu2"].index == 2
    assert cfg.defaults.hold_clawback_grace_sec == 600
    assert cfg.defaults.swap_min_residency_sec == 600
    assert cfg.defaults.actuate_ack_sec == 10
    assert check_launch(cfg, ROOT) == []


def test_launch_digest_is_stable_and_moves_with_launch():
    cfg = load_pool_config()
    d = launch_digest(cfg, "agent-gpu2")
    assert len(d) == 64 and d == launch_digest(load_pool_config(), "agent-gpu2")
    data = copy.deepcopy(RAW)
    data["roles"]["diffusion"]["launch"]["timeout_sec"] = 601     # an EVICTED role's launch moved
    assert launch_digest(PoolConfig.model_validate(data), "agent-gpu2") != d
    data = copy.deepcopy(RAW)
    data["roles"]["agent-gpu2"]["port"] = 8116                    # the seat's own port (ready check target)
    assert launch_digest(PoolConfig.model_validate(data), "agent-gpu2") != d
    data = copy.deepcopy(RAW)
    data["roles"]["chat"]["port"] = 8111                          # unrelated role: no effect
    assert launch_digest(PoolConfig.model_validate(data), "agent-gpu2") == d


def _bad(mutate, match):
    data = copy.deepcopy(RAW)
    mutate(data)
    with pytest.raises(ValueError, match=match):
        PoolConfig.model_validate(data)


def test_rejects_unknown_actuator():
    _bad(lambda d: d["roles"]["diffusion"]["launch"].update(actuator="atlas"), "not in actuators")


def test_rejects_launch_card_without_index():
    _bad(lambda d: d["cards"]["gpu2"].pop("index"), "has no index")


def test_rejects_duplicate_index():
    _bad(lambda d: d["cards"]["gpu3"].update(index=2), "share index 2")


def test_rejects_actuator_on_another_host():
    _bad(lambda d: d["actuators"].update(atlas={"host": "atlas"}), "is not the pool host")


def test_rejects_half_a_bridge():
    _bad(lambda d: d["roles"]["agent-gpu2"]["swap"].pop("unload"), "come as a pair")


def test_rejects_unbridged_seat_whose_evicted_role_has_no_launch():
    def mutate(d):
        d["roles"]["agent-gpu2"]["swap"].pop("load")
        d["roles"]["agent-gpu2"]["swap"].pop("unload")
        d["roles"]["diffusion"].pop("launch")
    _bad(mutate, r"missing on \['diffusion'\]")


def test_rejects_unknown_guard():
    _bad(lambda d: d["roles"]["agent-gpu2"]["swap"].update(guards=["moon_phase"]), "guards")


@pytest.mark.parametrize("field,value", [
    ("compose", "/etc/compose.yml"), ("compose", "../elsewhere/compose.yml"),
    ("ready", "health"), ("cuda_env", "cuda-devices"), ("timeout_sec", 0), ("container", "x"),
])
def test_rejects_bad_launch_block(field, value):
    _bad(lambda d: d["roles"]["agent-gpu2"]["launch"].update({field: value}), "")


def test_unbridged_seat_with_launches_is_valid_and_after_wait_overrides():
    data = copy.deepcopy(RAW)
    swap = data["roles"]["agent-gpu2"]["swap"]
    swap.pop("load"), swap.pop("unload")
    swap.update(after_wait_sec=1200, guards=["thermal", "visual_baseline"])
    cfg = PoolConfig.model_validate(data)
    assert cfg.swap_after_wait_sec("agent-gpu2") == 1200
    assert cfg.swap_after_wait_sec("experiment") == cfg.defaults.swap_after_wait_sec


# --- the static gate against compose ------------------------------------------------------------
def _compose_mutation(tmp_path: Path, edit) -> list[str]:
    """Copy the real compose + env templates into tmp, edit the agent-burst service, re-check."""
    for rel in ("services/orion-llamacpp-host/docker-compose.atlas-workers.yml",
                "services/orion-llamacpp-host/.env_example",
                "services/orion-diffusion-host/docker-compose.yml",
                "services/orion-diffusion-host/.env_example"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text((ROOT / rel).read_text())
    path = tmp_path / "services/orion-llamacpp-host/docker-compose.atlas-workers.yml"
    compose = yaml.safe_load(path.read_text())
    edit(compose["services"]["atlas-agent-burst"], compose)
    path.write_text(yaml.safe_dump(compose))
    return check_launch(load_pool_config(), tmp_path)


def _set_env(svc, key, value):
    svc["environment"] = [e for e in svc["environment"] if not e.startswith(key + "=")] + [f"{key}={value}"]


@pytest.mark.parametrize("edit,match", [
    (lambda s, c: _set_env(s, "LLM_ROLE", "agent"), "LLM_ROLE=agent"),
    (lambda s, c: _set_env(s, "LLM_ANNOUNCE_PORT", "${SOME_UNSET_PORT_VAR:-8017}"), "announces port 8017"),
    (lambda s, c: s.update(profiles=["burst"]), "compose_profile agent-burst"),
    (lambda s, c: _set_env(s, "CUDA_VISIBLE_DEVICES_OVERRIDE", "1"), "CUDA_VISIBLE_DEVICES_OVERRIDE=1"),
    (lambda s, c: s.update(environment=[e for e in s["environment"] if "CUDA" not in e]), "does not set"),
    (lambda s, c: c["services"].pop("atlas-agent-burst"), "is not a service"),
])
def test_gate_catches_compose_drift(tmp_path, edit, match):
    problems = _compose_mutation(tmp_path, edit)
    assert any(match in p for p in problems), problems


def test_gate_is_clean_on_unmodified_copy(tmp_path):
    assert _compose_mutation(tmp_path, lambda s, c: None) == []


def test_gate_env_template_value_beats_compose_default(tmp_path):
    # Compose uses ${VAR:-d}'s default only when VAR is unset/empty; a set template value wins.
    _compose_mutation(tmp_path, lambda s, c: _set_env(
        s, "LLM_ANNOUNCE_PORT", "${ATLAS_AGENT_BURST_HOST_PORT:-8016}"))
    rel = "services/orion-llamacpp-host/.env_example"
    text = (tmp_path / rel).read_text()
    (tmp_path / rel).write_text(text + "\nATLAS_AGENT_BURST_HOST_PORT=9999\n")
    problems = check_launch(load_pool_config(), tmp_path)
    assert any("announces port 9999" in p for p in problems), problems


def test_gate_template_default_equal_to_role_port_is_not_drift(tmp_path):
    # ${VAR:-8017} with VAR=8016 in the template resolves to 8016: no false alarm.
    _compose_mutation(tmp_path, lambda s, c: _set_env(
        s, "LLM_ANNOUNCE_PORT", "${ATLAS_AGENT_BURST_HOST_PORT:-8017}"))
    rel = "services/orion-llamacpp-host/.env_example"
    (tmp_path / rel).write_text((tmp_path / rel).read_text() + "\nexport ATLAS_AGENT_BURST_HOST_PORT=8016\n")
    assert check_launch(load_pool_config(), tmp_path) == []


@pytest.mark.parametrize("ports", [["127.0.0.1:${HOST_PORT}:6700"], [{"published": "${HOST_PORT}", "target": 6700}],
                                   ["${HOST_PORT}:6700/tcp"]])
def test_gate_port_forms(tmp_path, ports):
    _compose_mutation(tmp_path, lambda s, c: None)
    path = tmp_path / "services/orion-diffusion-host/docker-compose.yml"
    compose = yaml.safe_load(path.read_text())
    compose["services"]["diffusion-host"]["ports"] = ports
    path.write_text(yaml.safe_dump(compose))
    assert check_launch(load_pool_config(), tmp_path) == []


def test_gate_resolves_service_port_from_env_template(tmp_path):
    _compose_mutation(tmp_path, lambda s, c: None)
    example = tmp_path / "services/orion-diffusion-host/.env_example"
    example.write_text(example.read_text().replace("HOST_PORT=8014", "HOST_PORT=8114"))
    problems = check_launch(load_pool_config(), tmp_path)
    assert any("publishes host ports ['8114']" in p for p in problems), problems


# --- the spec's worked example: gpu4 hosts either an 8B or a vision model -------------------------
GPU4_COMPOSE = textwrap.dedent("""
    services:
      atlas-fast2:
        environment:
          - LLM_ROLE=fast2
          - LLM_ANNOUNCE_PORT=${ATLAS_FAST2_HOST_PORT:-8017}
          - CUDA_VISIBLE_DEVICES_OVERRIDE=4
      atlas-vision4:
        profiles: ["vision4"]
        environment:
          - LLM_ROLE=vision4
          - LLM_ANNOUNCE_PORT=${ATLAS_VISION4_HOST_PORT:-8018}
          - CUDA_VISIBLE_DEVICES_OVERRIDE=4
""")


def _gpu4(data: dict, *, list_fast2_in_metacog: bool) -> dict:
    launch = {"actuator": "circe", "compose": "services/orion-llamacpp-host/docker-compose.atlas-workers.yml",
              "env_file": "services/orion-llamacpp-host/.env", "cuda_env": "CUDA_VISIBLE_DEVICES_OVERRIDE",
              "ready": "/health"}
    data["cards"]["gpu4"] = {"vram_gb": 32, "index": 4}
    data["roles"]["fast2"] = {"kind": "llm", "cards": ["gpu4"], "owner": ["metacog", "fast"], "port": 8017,
                              "launch": {**launch, "service": "atlas-fast2", "timeout_sec": 300}}
    data["roles"]["vision4"] = {"kind": "llm", "cards": ["gpu4"], "owner": "vision", "port": 8018,
                                "launch": {**launch, "service": "atlas-vision4", "compose_profile": "vision4",
                                           "timeout_sec": 600},
                                "swap": {"evicts": ["fast2"], "guards": ["thermal"]}}
    data["classes"]["fast"] = {"roles": ["fast", "metacog", "fast2", "agent", "agent-gpu2", "chat"],
                               "on_unavailable": "wait"}
    data["classes"]["vision"] = {"roles": ["vision4"], "on_unavailable": "backlog"}
    if list_fast2_in_metacog:
        data["classes"]["metacog"]["roles"].insert(2, "fast2")
    return data


def test_spec_gpu4_example_as_written_is_rejected_because_metacog_does_not_list_fast2():
    # The spec's example gives fast2 owner [metacog, fast] but only adds it to class `fast`; the
    # existing owner rule refuses that. Recorded in the 4.1 PR report as a spec erratum.
    with pytest.raises(ValueError, match="owner class metacog does not list it"):
        PoolConfig.model_validate(_gpu4(copy.deepcopy(RAW), list_fast2_in_metacog=False))


def test_spec_gpu4_example_corrected_is_accepted_with_no_pool_code(tmp_path):
    cfg = PoolConfig.model_validate(_gpu4(copy.deepcopy(RAW), list_fast2_in_metacog=True))
    assert cfg.evicted_by("vision4") == ["fast2"]
    assert "fast2" in cfg.resident_roles() and "vision4" not in cfg.resident_roles()
    assert not cfg.roles["vision4"].swap.bridged
    # The gate against a compose file that has both services (and the real gpu2 ones):
    _compose_mutation(tmp_path, lambda s, c: c["services"].update(yaml.safe_load(GPU4_COMPOSE)["services"]))
    assert check_launch(cfg, tmp_path) == []
    assert launch_digest(cfg, "vision4") != launch_digest(cfg, "fast2")

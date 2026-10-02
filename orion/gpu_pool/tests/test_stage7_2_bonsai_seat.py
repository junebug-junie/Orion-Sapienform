"""GPU pool stage 7.2: the pool loads Ternary-Bonsai-2-27B on agent-gpu2 with 2 slots x 131072
(docs/superpowers/specs/2026-09-30-gpu-pool-stage7-concurrency.md, the 7.2 row).

Config-only for the pool: the seat's first launch profile changes; slots and per-slot context are
discovered from llama.cpp /props. H1 (one durable-run hold per role) is unchanged, so the second
slot serves one-off calls only. Covers acceptance check 3 (discovery) and the profile half of
check 10 (rollback drill); the max_holds half of check 10 belongs to stage 7.3.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from orion.gpu_pool.config import DEFAULT_PATH, PoolConfig, check_launch, launch_digest, load_pool_config
from orion.gpu_pool.discovery import Probe, load_profiles, profile_model_file, resolve_roles
from orion.gpu_pool.scheduler import CardLive, Grant, LeaseView, RoleLive, schedule
from orion.schemas.gpu_pool import LlmWorkerAnnounceV1

ROOT = Path(__file__).resolve().parents[3]
RAW = yaml.safe_load(DEFAULT_PATH.read_text())
CFG = load_pool_config()
PROFILES = load_profiles()
NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
BONSAI = "ternary-bonsai2-27b-pq2-v100-32gb-circe-agent"
Q4 = "qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex"
BONSAI_FILE = "Ternary-Bonsai-2-27B-PQ2_0.gguf"
Q4_FILE = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
GPU2_LOADED = {"gpu2": CardLive("gpu2", swapped_in={"agent-gpu2"})}


def _cards(**over):
    base = {c: CardLive(c) for c in CFG.cards}
    base.update(over)
    return base


def _props(file: str, slots: int, ctx: int) -> dict:
    return {"model_path": f"/models/gguf/{file}", "total_slots": slots,
            "default_generation_settings": {"n_ctx": ctx}, "modalities": {"vision": False}}


def _ann(profile: str) -> LlmWorkerAnnounceV1:
    return LlmWorkerAnnounceV1(host="circe", role="agent-gpu2", profile_name=profile, port=8016,
                               announced_at=NOW - timedelta(seconds=5))


# --- the seat's profile -----------------------------------------------------------------------------
def test_pool_loads_bonsai_first_and_keeps_q4_as_rollback():
    assert CFG.roles["agent-gpu2"].launch.profiles == [BONSAI, Q4]
    assert CFG.load_profile("agent-gpu2") == BONSAI


def test_bonsai_profile_is_two_slots_of_131k_on_the_fork_with_flash_attention():
    llamacpp = PROFILES[BONSAI]["llamacpp"]
    assert llamacpp["n_parallel"] == 2
    assert llamacpp["ctx_size"] // llamacpp["n_parallel"] == 131072
    assert llamacpp["flash_attn"] == "on"
    assert llamacpp["server_build"] == "prism"
    # The rollback profile runs the image's stock binary.
    assert PROFILES[Q4]["llamacpp"].get("server_build") in (None, "stock")
    # The #27148 knobs stay at the binary's defaults until the canary says otherwise (spec D2).
    assert "cache_ram_mib" not in llamacpp and "cache_idle_slots" not in llamacpp
    # Bonsai's template 500s on reasoning_effort "none"/"high".
    assert llamacpp["chat_template_kwargs"]["reasoning_effort"] not in ("none", "high")
    assert PROFILES[BONSAI]["gpu"]["device_ids"] == [CFG.cards["gpu2"].index]


# --- acceptance check 3: discovery -----------------------------------------------------------------
def test_discovery_confirms_two_slots_of_131k_on_agent_gpu2():
    assert profile_model_file(PROFILES[BONSAI]) == BONSAI_FILE
    discovered, live, _ = resolve_roles(
        CFG, PROFILES, {"agent-gpu2": _ann(BONSAI)},
        {"agent-gpu2": Probe(True, _props(BONSAI_FILE, slots=2, ctx=131072))}, _cards(**GPU2_LOADED), NOW)
    row = {d.role: d for d in discovered}["agent-gpu2"]
    assert (row.status, row.slots, row.ctx_per_slot, row.profile_name) == ("confirmed", 2, 131072, BONSAI)
    assert live["agent-gpu2"].healthy and live["agent-gpu2"].slots == 2 and live["agent-gpu2"].ctx_per_slot == 131072


def test_discovery_flags_a_bonsai_announce_serving_the_q4_file():
    discovered, live, _ = resolve_roles(
        CFG, PROFILES, {"agent-gpu2": _ann(BONSAI)},
        {"agent-gpu2": Probe(True, _props(Q4_FILE, slots=1, ctx=131072))}, _cards(**GPU2_LOADED), NOW)
    assert {d.role: d for d in discovered}["agent-gpu2"].status == "mismatch"
    assert not live["agent-gpu2"].healthy


# --- scheduling: H1 unchanged, the second slot serves one-off calls --------------------------------
def _live(gpu2_slots: int, gpu2_ctx: int = 131072) -> dict[str, RoleLive]:
    return {
        "chat": RoleLive("chat", True, 1, 65536, True),
        "agent": RoleLive("agent", True, 1, 131072, False),
        "agent-gpu2": RoleLive("agent-gpu2", True, gpu2_slots, gpu2_ctx, False),
        "metacog": RoleLive("metacog", True, 4, 4096, False),
        "fast": RoleLive("fast", True, 4, 4096, False),
        "world": RoleLive("world", True, 2),
        "diffusion": RoleLive("diffusion", True, 1),
        "experiment": RoleLive("experiment", True, 1, 8192, False),
    }


def _lease(lease_id, status="queued", role=None, priority="system", **kw):
    return LeaseView(lease_id=lease_id, work_class="agent", priority=priority, status=status, role=role,
                     created_at=NOW - timedelta(seconds=kw.pop("age", 0)), **kw)


def _hold(lease_id, status="queued", role=None):
    return _lease(lease_id, status, role, priority="background", kind="hold", retryable=True)


def _grants(slots: int, leases, ctx: int = 131072) -> dict[str, str]:
    decisions = schedule(CFG, _live(slots, ctx), _cards(**GPU2_LOADED), leases, NOW)
    return {d.lease_id: d.role for d in decisions if isinstance(d, Grant)}


def test_h1_unchanged_a_second_run_never_holds_the_two_slot_seat():
    held = [_hold("h1", "granted", "agent"), _hold("h2", "granted", "agent-gpu2")]
    assert _grants(2, [*held, _hold("h3")]) == {}


def test_second_slot_serves_a_one_off_call_while_a_run_is_mid_call():
    # Today (1 slot): the run's active call fills agent-gpu2, so a one-off agent call waits.
    run_busy = [_hold("h1", "granted", "agent"), _lease("c1", "granted", "agent", priority="background",
                                                        hold_lease_id="h1"),
                _hold("h2", "granted", "agent-gpu2"), _lease("c2", "granted", "agent-gpu2",
                                                             priority="background", hold_lease_id="h2")]
    assert "s" not in _grants(1, [*run_busy, _lease("s")])
    # 7.2 (2 slots): the same call gets the second slot at once.
    assert _grants(2, [*run_busy, _lease("s")]) == {"s": "agent-gpu2"}


def test_131k_per_slot_keeps_the_agent_class_largest_context():
    # min_ctx_exceeds_class only fires if per-slot ctx drops (4 x 65K would); 2 x 131K does not.
    big = _lease("big", min_ctx_tokens=100_000)
    busy = [_lease("a", "granted", "agent")]
    assert _grants(2, [*busy, big]) == {"big": "agent-gpu2"}
    # Control: the 4 x 65K layout the spec rejects would not fit it.
    assert "big" not in _grants(4, [*busy, big], ctx=65536)


# --- acceptance check 10, profile half: the rollback drill ------------------------------------------
def _reordered() -> PoolConfig:
    data = copy.deepcopy(RAW)
    launch = data["roles"]["agent-gpu2"]["launch"]
    launch["profiles"] = list(reversed(launch["profiles"]))
    return PoolConfig.model_validate(data)


def test_rollback_is_a_reorder_and_the_next_load_serves_q4():
    rolled = _reordered()
    assert rolled.load_profile("agent-gpu2") == Q4
    assert check_launch(rolled, ROOT) == []
    # Both profiles stay on the allow-list, so a seat still running Bonsai is confirmed, not refused.
    assert set(rolled.roles["agent-gpu2"].launch.profiles) == {BONSAI, Q4}


def test_profile_order_moves_the_launch_digest_so_pool_and_controller_deploy_together():
    # The controller refuses a load whose digest differs from its checkout (launch_digest_mismatch):
    # this change and its rollback both need athena's pool and circe's checkout on the same commit.
    assert launch_digest(_reordered(), "agent-gpu2") != launch_digest(CFG, "agent-gpu2")
    assert launch_digest(_reordered(), "agent") == launch_digest(CFG, "agent")


# --- the static gate: a prism profile needs the prism image ----------------------------------------
def _tree(tmp_path: Path) -> Path:
    for rel in ("config/llm_profiles.yaml",
                "services/orion-llamacpp-host/docker-compose.atlas-workers.yml",
                "services/orion-llamacpp-host/.env_example",
                "services/orion-diffusion-host/docker-compose.yml",
                "services/orion-diffusion-host/.env_example"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text((ROOT / rel).read_text())
    return tmp_path


def test_gate_refuses_a_prism_profile_on_the_stock_image(tmp_path):
    root = _tree(tmp_path)
    path = root / "services/orion-llamacpp-host/docker-compose.atlas-workers.yml"
    compose = yaml.safe_load(path.read_text())
    compose["services"]["atlas-agent-burst"]["build"]["dockerfile"] = "services/orion-llamacpp-host/Dockerfile"
    path.write_text(yaml.safe_dump(compose))
    problems = check_launch(CFG, root)
    assert any(BONSAI in p and "Dockerfile.prism" in p for p in problems), problems
    # Control: Q4 alone on the stock image is fine.
    data = copy.deepcopy(RAW)
    data["roles"]["agent-gpu2"]["launch"]["profiles"] = [Q4]
    assert check_launch(PoolConfig.model_validate(data), root) == []


def test_only_the_burst_seat_runs_the_prism_image():
    compose = yaml.safe_load((ROOT / "services/orion-llamacpp-host/docker-compose.atlas-workers.yml").read_text())
    prism = {name for name, svc in compose["services"].items()
             if str((svc.get("build") or {}).get("dockerfile", "")).endswith("Dockerfile.prism")}
    assert prism == {"atlas-agent-burst"}
    assert compose["services"]["atlas-agent-burst"]["image"] != compose["services"]["atlas-chat"]["image"]

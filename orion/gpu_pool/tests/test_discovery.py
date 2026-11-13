from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.gpu_pool.config import load_pool_config
from orion.gpu_pool.discovery import Probe, load_profiles, profile_model_file, resolve_roles
from orion.gpu_pool.scheduler import CardLive
from orion.schemas.gpu_pool import LlmWorkerAnnounceV1

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
CFG = load_pool_config()
PROFILES = load_profiles()
SHARED_PROFILE = "qwen35-9b-udq5kxl-v100-32gb-circe-shared-32k"


def cards(**over):
    base = {c: CardLive(c) for c in CFG.cards}
    base.update(over)
    return base


def props(file, slots=10, ctx=32768, vision=False):
    return {"model_path": f"/models/gguf/{file}", "total_slots": slots,
            "default_generation_settings": {"n_ctx": ctx}, "modalities": {"vision": vision}}


def ann(role, profile, port, age=0):
    return LlmWorkerAnnounceV1(host="circe", role=role, profile_name=profile, port=port,
                               announced_at=NOW - timedelta(seconds=age))


def by_role(discovered):
    return {d.role: d for d in discovered}


def test_live_profiles_name_their_model_files():
    assert profile_model_file(PROFILES[SHARED_PROFILE]) == "Qwen3.5-9B-UD-Q5_K_XL.gguf"


def test_confirmed_when_announcement_profile_and_props_agree():
    d, live, _ = resolve_roles(
        CFG, PROFILES, {"fast": ann("fast", SHARED_PROFILE, 8013)},
        {"fast": Probe(True, props("Qwen3.5-9B-UD-Q5_K_XL.gguf"))}, cards(), NOW)
    row = by_role(d)["fast"]
    assert row.status == "confirmed" and row.slots == 10 and row.ctx_per_slot == 32768
    assert row.model_path == "/models/gguf/Qwen3.5-9B-UD-Q5_K_XL.gguf"
    assert live["fast"].healthy


def test_mismatch_when_loaded_file_differs_gets_no_grants():
    d, live, _ = resolve_roles(
        CFG, PROFILES, {"fast": ann("fast", SHARED_PROFILE, 8013)},
        {"fast": Probe(True, props("Something-Else.gguf"))}, cards(), NOW)
    assert by_role(d)["fast"].status == "mismatch" and not live["fast"].healthy


def test_unknown_profile_is_mismatch():
    d, _, _ = resolve_roles(CFG, PROFILES, {"fast": ann("fast", "no-such-profile", 8013)},
                            {"fast": Probe(True, props("x.gguf"))}, cards(), NOW)
    assert "not in llm_profiles" in by_role(d)["fast"].detail


def test_silent_when_port_answers_without_fresh_announcement():
    d, live, _ = resolve_roles(
        CFG, PROFILES, {"fast": ann("fast", SHARED_PROFILE, 8013, age=600)},
        {"fast": Probe(True, props("Qwen3.5-9B-UD-Q5_K_XL.gguf"))}, cards(), NOW)
    assert by_role(d)["fast"].status == "silent" and not live["fast"].healthy


def test_down_when_probe_fails():
    d, live, _ = resolve_roles(CFG, PROFILES, {}, {"chat": Probe(False, error="refused")}, cards(), NOW)
    assert by_role(d)["chat"].status == "down" and not live["chat"].healthy


def test_swap_seat_unloaded_and_residents_evicted():
    d, _, _ = resolve_roles(CFG, PROFILES, {}, {}, cards(), NOW)
    assert by_role(d)["agent-gpu2"].status == "unloaded"
    d, live, _ = resolve_roles(CFG, PROFILES, {}, {}, cards(gpu2=CardLive("gpu2", swapped_in={"agent-gpu2"})), NOW)
    assert by_role(d)["diffusion"].status == "evicted" and not live["diffusion"].healthy


def test_service_roles_use_declared_slots():
    d, live, _ = resolve_roles(CFG, PROFILES, {}, {"world": Probe(True)}, cards(), NOW)
    assert by_role(d)["world"].status == "static" and live["world"].slots == 2


def test_unclaimed_announcement_is_listed():
    _, _, unclaimed = resolve_roles(CFG, PROFILES, {"mystery": ann("mystery", "p", 8777)}, {}, cards(), NOW)
    assert unclaimed == ["circe:8777 role=mystery profile=p"]

"""The chat role runs Strata: confirmed through llama.cpp's /props shape, but only when the probe says strata."""
from __future__ import annotations

from orion.gpu_pool.discovery import Probe, profile_model_file, resolve_roles
from orion.gpu_pool.tests.test_discovery import CFG, NOW, PROFILES, ann, by_role, cards

STRATA_PROFILE = "qwen38-flash-next-iq3s-strata-1xv100-circe-chat"
STRATA_FILE = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"


def strata_props(file=STRATA_FILE):
    # what Strata 0.2.7's GET /props answers (serve/server.py _props)
    return {"model_path": f"/data/Strata-data/models/IQ3_S/{file}", "total_slots": 1,
            "default_generation_settings": {"n_ctx": 131072}, "modalities": {"vision": True},
            "model_alias": "qwen3.8-flash-next-iq3_s", "build_info": "Strata 0.1.52"}


def test_chat_role_is_strata_and_profile_names_its_file():
    assert CFG.roles["chat"].backend == "strata"
    assert profile_model_file(PROFILES[STRATA_PROFILE]) == STRATA_FILE


def test_strata_chat_confirmed_from_props():
    d, live, _ = resolve_roles(CFG, PROFILES, {"chat": ann("chat", STRATA_PROFILE, 8011)},
                               {"chat": Probe(True, strata_props(), backend="strata")}, cards(), NOW)
    row = by_role(d)["chat"]
    assert row.status == "confirmed" and row.slots == 1 and row.ctx_per_slot == 131072 and row.vision
    assert live["chat"].healthy


def test_llamacpp_probe_on_strata_role_is_mismatch():
    d, live, _ = resolve_roles(CFG, PROFILES, {"chat": ann("chat", STRATA_PROFILE, 8011)},
                               {"chat": Probe(True, strata_props())}, cards(), NOW)
    assert by_role(d)["chat"].detail == "probe backend mismatch" and not live["chat"].healthy


def test_wrong_model_file_is_mismatch():
    d, _, _ = resolve_roles(CFG, PROFILES, {"chat": ann("chat", STRATA_PROFILE, 8011)},
                            {"chat": Probe(True, strata_props("Other.gguf"), backend="strata")}, cards(), NOW)
    assert by_role(d)["chat"].status == "mismatch"


def test_agent_role_is_strata_at_the_burst_seat_window():
    """gpu1's agent lane runs the same engine at 131072, agent-gpu2's window, so a recalled run can move."""
    agent = "qwen38-flash-next-iq3s-strata-1xv100-circe-agent"
    assert CFG.roles["agent"].backend == "strata"
    assert PROFILES[agent]["strata"]["ctx_size"] == 131072
    d, live, _ = resolve_roles(CFG, PROFILES, {"agent": ann("agent", agent, 8015)},
                               {"agent": Probe(True, {**strata_props(), "default_generation_settings": {"n_ctx": 131072}},
                                               backend="strata")}, cards(), NOW)
    assert by_role(d)["agent"].status == "confirmed" and live["agent"].healthy

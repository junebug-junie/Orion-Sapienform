"""Juniper's live turn (utterance_origin="juniper") rides `metacog_turn` (interactive: never shed
under heat); Orion's own, unmarked, and non-turn runs keep `metacog` (system: shed).
Spec 2026-10-06-thermal-controller-redesign D4; tiers attach to callers, not lanes (body-economy
design Organ 3b, 2026-10-23).
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from uuid import uuid4

import pytest

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "app" / "config"


def _mind_prep() -> None:
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()


@pytest.fixture(autouse=True)
def _prep() -> None:
    _mind_prep()


def _three_phase_responses() -> list[dict]:
    spec = importlib.util.spec_from_file_location(
        "_mind_pipeline_fixtures", Path(__file__).resolve().parent / "test_mind_llm_pipeline.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [mod._semantic_payload(claim_label="shared evening moment"), mod._frontier_payload(),
            {"stance_payload": dict(mod._VALID_STANCE)}]


def _routes_called(monkeypatch, *, trigger="user_turn", origin=None, turn_route="metacog_turn",
                   phase_routes=("metacog", "metacog", "metacog"), session_id=None) -> list[str]:
    return [c["route"] for c in _calls(monkeypatch, trigger=trigger, origin=origin, turn_route=turn_route,
                                       phase_routes=phase_routes, session_id=session_id)]


def _calls(monkeypatch, *, trigger="user_turn", origin=None, turn_route="metacog_turn",
           phase_routes=("metacog", "metacog", "metacog"), session_id=None, gpu_lease=None) -> list[dict]:
    from app.engine import run_mind
    from app.llm_client import FakeMindLLMClient, set_llm_client_override
    from app.settings import settings
    from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1

    monkeypatch.setattr(settings, "MIND_LLM_SYNTHESIS_ENABLED", True)
    monkeypatch.setattr(settings, "MIND_TURN_MODEL_ROUTE", turn_route)
    for key, route in zip(("MIND_SEMANTIC_MODEL_ROUTE", "MIND_APPRAISAL_MODEL_ROUTE", "MIND_STANCE_MODEL_ROUTE"),
                          phase_routes):
        monkeypatch.setattr(settings, key, route)
    fake = FakeMindLLMClient(_three_phase_responses())   # all three phases succeed, so each is asked
    set_llm_client_override(fake)
    try:
        run_mind(
            MindRunRequestV1(correlation_id=str(uuid4()), trigger=trigger, utterance_origin=origin,
                             session_id=session_id, gpu_lease=gpu_lease,
                             snapshot_inputs={"user_text": "hello"},
                             policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=60_000)),
            router_profiles_dir=_CONFIG_DIR, snapshot_max_bytes=512_000, mind_settings=settings,
        )
    finally:
        set_llm_client_override(None)
    assert len(fake.calls) == 3, [c["route"] for c in fake.calls]
    return fake.calls


def test_turn_route_setting_defaults_to_metacog_turn(monkeypatch) -> None:
    monkeypatch.delenv("MIND_TURN_MODEL_ROUTE", raising=False)
    settings_module = importlib.import_module("app.settings")
    importlib.reload(settings_module)
    assert settings_module.settings.MIND_TURN_MODEL_ROUTE == "metacog_turn"


def test_juniper_turn_lifts_all_three_phases(monkeypatch) -> None:
    assert _routes_called(monkeypatch, origin="juniper") == ["metacog_turn"] * 3


def test_orions_own_turn_stays_sheddable(monkeypatch) -> None:
    assert set(_routes_called(monkeypatch, origin="orion")) == {"metacog"}


def test_unmarked_turn_stays_sheddable(monkeypatch) -> None:
    """Unknown origin is not a live turn: "a human is waiting" must be positively marked."""
    assert set(_routes_called(monkeypatch, origin=None)) == {"metacog"}


def test_world_pulse_read_turn_is_not_lifted(monkeypatch) -> None:
    """Regression, live 2026-10-22..23: world-pulse read turns (reading_turn_listener, no origin
    set) rode Juniper's interactive tier -- 72 interactive orion-mind grants with no Juniper chat."""
    for session in ("orion_world_pulse_read", "orion_world_pulse_read_stage2"):
        assert set(_routes_called(monkeypatch, origin=None, session_id=session)) == {"metacog"}


def test_carries_live_turn_truth_table() -> None:
    from app.engine import carries_live_turn
    from orion.mind.v1 import MindRunRequestV1

    def live(trigger, origin):
        return carries_live_turn(MindRunRequestV1(correlation_id="c", trigger=trigger, utterance_origin=origin))

    assert live("user_turn", "juniper") is True
    for origin in (None, "orion"):
        assert live("user_turn", origin) is False
    for trigger in ("scheduled", "operator", "replay"):
        assert live(trigger, "juniper") is False


@pytest.mark.parametrize("trigger", ["scheduled", "operator", "replay"])
def test_non_turn_runs_stay_sheddable(monkeypatch, trigger) -> None:
    assert set(_routes_called(monkeypatch, trigger=trigger)) == {"metacog"}


def test_empty_setting_is_the_rollback(monkeypatch) -> None:
    assert set(_routes_called(monkeypatch, origin="juniper", turn_route="")) == {"metacog"}


def test_only_metacog_phases_are_lifted(monkeypatch) -> None:
    """An operator's `quick` or a deliberately yielding `metacog_background` phase is left alone."""
    routes = _routes_called(monkeypatch, origin="juniper", phase_routes=("quick", "metacog", "metacog_background"))
    assert routes == ["quick", "metacog_turn", "metacog_background"]


def test_turn_route_is_a_pool_route_at_interactive_priority() -> None:
    from orion.gpu_pool.config import load_pool_config

    spec = load_pool_config().routes["metacog_turn"]
    assert (spec.work_class, spec.priority) == ("metacog", "interactive")


_HOLD = {"lease_id": "hold-1", "generation": 2, "role": "agent-gpu2", "holder": "durable-runs:reading-1"}


@pytest.mark.parametrize("session", ["orion_world_pulse_read", "orion_world_pulse_read_stage2"])
def test_held_world_pulse_turn_runs_every_phase_on_its_seat(monkeypatch, session) -> None:
    """Juniper 2026-11-12: world-pulse work never goes to the fast or metacog lane. 11-12: 111 of a
    world-pulse day's Mind calls rode `metacog` (Qwen3-8B, 4k slot) and stage-2 runs timed out.
    A held turn names the hold's work-class route and carries the hold ref on every phase."""
    calls = _calls(monkeypatch, origin=None, session_id=session, gpu_lease=_HOLD)
    assert [c["route"] for c in calls] == ["agent"] * 3
    assert all(c["context"].gpu_lease == _HOLD for c in calls)


def test_held_turn_beats_the_juniper_lift(monkeypatch) -> None:
    """The hold wins over any configured or lifted small-lane route."""
    calls = _calls(monkeypatch, origin="juniper", gpu_lease=_HOLD,
                   phase_routes=("quick", "metacog", "metacog_background"))
    assert [c["route"] for c in calls] == ["agent"] * 3


def test_unheld_turn_carries_no_hold(monkeypatch) -> None:
    calls = _calls(monkeypatch, origin="juniper")
    assert [c["route"] for c in calls] == ["metacog_turn"] * 3
    assert all(c["context"].gpu_lease is None for c in calls)


def test_bus_client_sends_the_hold_as_gateway_option(monkeypatch) -> None:
    """The real client puts the context's hold on options.gpu_lease (llm-gateway attaches on it)."""
    from app import llm_client as client_mod
    from app.llm_context import MindLLMRequestContext
    from app.settings import settings

    monkeypatch.setattr(settings, "MIND_LLM_USE_BUS", True)
    monkeypatch.setattr(settings, "ORION_BUS_ENABLED", True)
    seen: dict = {}

    def _fake_bus_chat(self, **kwargs):
        seen.update(kwargs["options"])
        return '{"ok": true}', {}, "m", {}

    monkeypatch.setattr(client_mod.MindLLMClient, "_bus_chat", _fake_bus_chat)
    ctx = MindLLMRequestContext(correlation_id="c", mind_run_id="r", phase_name="semantic_synthesis",
                                gpu_lease=_HOLD)
    client_mod.MindLLMClient(rpc_health=False).request_json(
        system_prompt="s", user_prompt="u", route="agent", max_tokens=8, context=ctx)
    assert seen["gpu_lease"] == _HOLD
    seen.clear()
    client_mod.MindLLMClient(rpc_health=False).request_json(
        system_prompt="s", user_prompt="u", route="metacog", max_tokens=8,
        context=MindLLMRequestContext(correlation_id="c", mind_run_id="r", phase_name="x"))
    assert "gpu_lease" not in seen

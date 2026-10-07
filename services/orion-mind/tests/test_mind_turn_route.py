"""A live Hub turn's Mind calls ride `metacog_turn` (interactive: never shed under heat); Orion's own
turns and non-turn runs keep `metacog` (system: shed). Spec 2026-10-06-thermal-controller-redesign D4.
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
                   phase_routes=("metacog", "metacog", "metacog")) -> list[str]:
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
                             snapshot_inputs={"user_text": "hello"},
                             policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=60_000)),
            router_profiles_dir=_CONFIG_DIR, snapshot_max_bytes=512_000, mind_settings=settings,
        )
    finally:
        set_llm_client_override(None)
    assert len(fake.calls) == 3, [c["route"] for c in fake.calls]
    return [c["route"] for c in fake.calls]


def test_turn_route_setting_defaults_to_metacog_turn(monkeypatch) -> None:
    monkeypatch.delenv("MIND_TURN_MODEL_ROUTE", raising=False)
    settings_module = importlib.import_module("app.settings")
    importlib.reload(settings_module)
    assert settings_module.settings.MIND_TURN_MODEL_ROUTE == "metacog_turn"


@pytest.mark.parametrize("origin", [None, "juniper"])
def test_a_live_hub_turn_uses_the_turn_route(monkeypatch, origin) -> None:
    assert set(_routes_called(monkeypatch, origin=origin)) == {"metacog_turn"}


def test_orions_own_turn_stays_sheddable(monkeypatch) -> None:
    assert set(_routes_called(monkeypatch, origin="orion")) == {"metacog"}


@pytest.mark.parametrize("trigger", ["scheduled", "operator", "replay"])
def test_non_turn_runs_stay_sheddable(monkeypatch, trigger) -> None:
    assert set(_routes_called(monkeypatch, trigger=trigger)) == {"metacog"}


def test_empty_setting_is_the_rollback(monkeypatch) -> None:
    assert set(_routes_called(monkeypatch, turn_route="")) == {"metacog"}


def test_only_metacog_phases_are_lifted(monkeypatch) -> None:
    """An operator's `quick` or a deliberately yielding `metacog_background` phase is left alone."""
    routes = _routes_called(monkeypatch, phase_routes=("quick", "metacog", "metacog_background"))
    assert routes == ["quick", "metacog_turn", "metacog_background"]


def test_turn_route_is_a_pool_route_at_interactive_priority() -> None:
    from orion.gpu_pool.config import load_pool_config

    spec = load_pool_config().routes["metacog_turn"]
    assert (spec.work_class, spec.priority) == ("metacog", "interactive")

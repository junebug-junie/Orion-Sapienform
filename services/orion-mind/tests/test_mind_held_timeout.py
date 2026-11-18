"""A held run's Mind phases get MIND_LLM_HELD_TIMEOUT_SEC, not the small-lane MIND_LLM_TIMEOUT_SEC.

Live 11-10..12: on the agent-gpu2 seat (Bonsai-2-27B, thinks first) Mind phases took 10-56 s when
served and 18 of 59 calls hit the 60 s cap as `upstream_timeout`. The run wall still bounds a phase.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from uuid import uuid4

import pytest

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "app" / "config"
_HOLD = {"lease_id": "hold-1", "generation": 2, "role": "agent-gpu2", "holder": "durable-runs:reading-1"}


@pytest.fixture(autouse=True)
def _prep() -> None:
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()


def _three_phase_responses() -> list[dict]:
    spec = importlib.util.spec_from_file_location(
        "_mind_pipeline_fixtures", Path(__file__).resolve().parent / "test_mind_llm_pipeline.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [mod._semantic_payload(claim_label="shared evening moment"), mod._frontier_payload(),
            {"stance_payload": dict(mod._VALID_STANCE)}]


def _phase_timeouts(monkeypatch, *, gpu_lease, wall_ms: int, held_sec: float = 150.0) -> list[float]:
    from app.engine import run_mind
    from app.llm_client import FakeMindLLMClient, set_llm_client_override
    from app.settings import settings
    from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1

    monkeypatch.setattr(settings, "MIND_LLM_SYNTHESIS_ENABLED", True)
    monkeypatch.setattr(settings, "MIND_LLM_TIMEOUT_SEC", 60.0)
    monkeypatch.setattr(settings, "MIND_LLM_HELD_TIMEOUT_SEC", held_sec)
    fake = FakeMindLLMClient(_three_phase_responses())
    set_llm_client_override(fake)
    try:
        run_mind(
            MindRunRequestV1(correlation_id=str(uuid4()), trigger="user_turn", gpu_lease=gpu_lease,
                             session_id="orion_world_pulse_read", snapshot_inputs={"user_text": "hello"},
                             policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=wall_ms)),
            router_profiles_dir=_CONFIG_DIR, snapshot_max_bytes=512_000, mind_settings=settings,
        )
    finally:
        set_llm_client_override(None)
    assert len(fake.calls) == 3
    return [float(c["timeout_sec"]) for c in fake.calls]


def test_held_setting_defaults_to_150(monkeypatch) -> None:
    monkeypatch.delenv("MIND_LLM_HELD_TIMEOUT_SEC", raising=False)
    settings_module = importlib.import_module("app.settings")
    importlib.reload(settings_module)
    assert settings_module.settings.MIND_LLM_HELD_TIMEOUT_SEC == 150.0


def test_env_example_ships_150() -> None:
    text = (Path(__file__).resolve().parents[1] / ".env_example").read_text()
    assert "\nMIND_LLM_HELD_TIMEOUT_SEC=150\n" in text


def test_held_run_phases_use_the_held_budget(monkeypatch) -> None:
    timeouts = _phase_timeouts(monkeypatch, gpu_lease=_HOLD, wall_ms=180_000)
    # Fake calls return at once, so the wall leaves every phase its full held cap (less safety ms).
    assert all(149.0 < t <= 150.0 for t in timeouts), timeouts


def test_unheld_run_keeps_the_small_lane_budget(monkeypatch) -> None:
    timeouts = _phase_timeouts(monkeypatch, gpu_lease=None, wall_ms=180_000)
    assert all(59.0 < t <= 60.0 for t in timeouts), timeouts


def test_run_wall_still_caps_a_held_phase(monkeypatch) -> None:
    """The caller's wall (orion-thought sends 180 s, under its 210 s HTTP wait) still wins."""
    timeouts = _phase_timeouts(monkeypatch, gpu_lease=_HOLD, wall_ms=90_000)
    assert all(t <= 90.0 for t in timeouts), timeouts
    assert timeouts[0] > 89.0

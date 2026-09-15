"""Fail-closed Cursor contested-budget gate.

Mirrors ask_claude_trigger._budget_refusal: only a fresh, observed `clear`
authorises a hire. Missing / unobserved / limited / unknown / incoherent all
refuse. Default with no env/file meter is unobserved so production cannot
accidentally authorise spend.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.dev_economics.cursor_limit_events import (
    ENV_FILE,
    ENV_STATE,
    CursorLimitObservation,
    decide_cursor_budget,
    observe_cursor_limit,
)


def _obs(
    *,
    observed: bool = True,
    state: str = "clear",
    staleness_sec: float | None = 1.0,
) -> CursorLimitObservation:
    return CursorLimitObservation(
        observed=observed,
        state=state,  # type: ignore[arg-type]
        staleness_sec=staleness_sec,
    )


def test_missing_observation_refuses():
    assert decide_cursor_budget(None) == "budget_observation_missing"


def test_unobserved_refuses():
    assert decide_cursor_budget(_obs(observed=False, state="unknown")) == "budget_unobserved"


def test_limited_refuses():
    assert decide_cursor_budget(_obs(state="limited")) == "budget_limited"


def test_unknown_state_with_observation_refuses():
    assert decide_cursor_budget(_obs(state="unknown", observed=True)) == "budget_unknown"


def test_incoherent_staleness_on_observed_clear_refuses():
    assert decide_cursor_budget(_obs(staleness_sec=None)) == "budget_observation_incoherent"


def test_observed_clear_with_staleness_allows_hire():
    assert decide_cursor_budget(_obs(observed=True, state="clear", staleness_sec=1.0)) is None


def test_default_observe_is_unobserved_fail_closed(monkeypatch: pytest.MonkeyPatch):
    """No env/file meter — default must refuse, not invent clear."""
    monkeypatch.delenv(ENV_STATE, raising=False)
    monkeypatch.delenv(ENV_FILE, raising=False)
    obs = observe_cursor_limit()
    assert obs.observed is False
    assert obs.state == "unknown"
    assert decide_cursor_budget(obs) == "budget_unobserved"


def test_observe_fixture_seam_passes_through():
    fixture = _obs(observed=True, state="clear", staleness_sec=0.5)
    assert observe_cursor_limit(fixture=fixture) is fixture
    assert decide_cursor_budget(observe_cursor_limit(fixture=fixture)) is None


def test_env_state_clear_allows_hire(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_FILE, raising=False)
    monkeypatch.setenv(ENV_STATE, "clear")
    obs = observe_cursor_limit()
    assert obs.observed is True
    assert obs.state == "clear"
    assert obs.staleness_sec == 0.0
    assert decide_cursor_budget(obs) is None


def test_env_state_limited_refuses(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_FILE, raising=False)
    monkeypatch.setenv(ENV_STATE, "limited")
    obs = observe_cursor_limit()
    assert decide_cursor_budget(obs) == "budget_limited"


def test_env_state_garbage_stays_unobserved(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_FILE, raising=False)
    monkeypatch.setenv(ENV_STATE, "totally-fine-trust-me")
    obs = observe_cursor_limit()
    assert decide_cursor_budget(obs) == "budget_unobserved"


def test_file_json_clear_allows_hire(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv(ENV_STATE, raising=False)
    meter = tmp_path / "budget.json"
    meter.write_text(
        json.dumps(
            {
                "state": "clear",
                "observed_at": "2026-09-15T02:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_FILE, str(meter))
    now = datetime(2026, 9, 15, 2, 0, 30, tzinfo=timezone.utc)
    obs = observe_cursor_limit(now=now)
    assert obs.observed is True
    assert obs.state == "clear"
    assert obs.staleness_sec == pytest.approx(30.0)
    assert decide_cursor_budget(obs) is None


def test_file_plain_text_clear_allows_hire(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv(ENV_STATE, raising=False)
    meter = tmp_path / "budget.txt"
    meter.write_text("clear\n", encoding="utf-8")
    monkeypatch.setenv(ENV_FILE, str(meter))
    obs = observe_cursor_limit()
    assert obs.observed is True
    assert obs.state == "clear"
    assert obs.staleness_sec is not None
    assert decide_cursor_budget(obs) is None


def test_file_beats_env_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    meter = tmp_path / "budget.txt"
    meter.write_text("limited\n", encoding="utf-8")
    monkeypatch.setenv(ENV_FILE, str(meter))
    monkeypatch.setenv(ENV_STATE, "clear")
    obs = observe_cursor_limit()
    assert obs.state == "limited"
    assert decide_cursor_budget(obs) == "budget_limited"


def test_explicit_state_arg_bypasses_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_STATE, "limited")
    monkeypatch.delenv(ENV_FILE, raising=False)
    obs = observe_cursor_limit(state="clear")
    assert decide_cursor_budget(obs) is None

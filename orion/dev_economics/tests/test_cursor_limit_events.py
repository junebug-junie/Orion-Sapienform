"""Fail-closed Cursor contested-budget gate.

Mirrors ask_claude_trigger._budget_refusal: only a fresh, observed `clear`
authorises a hire. Missing / unobserved / limited / unknown / incoherent all
refuse. Until a live Cursor meter exists, observe_cursor_limit() defaults to
unobserved so production cannot accidentally authorise spend.
"""

from __future__ import annotations

from orion.dev_economics.cursor_limit_events import (
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


def test_default_observe_is_unobserved_fail_closed():
    """No real Cursor meter yet — default must refuse, not invent clear."""
    obs = observe_cursor_limit()
    assert obs.observed is False
    assert obs.state == "unknown"
    assert decide_cursor_budget(obs) == "budget_unobserved"


def test_observe_fixture_seam_passes_through():
    fixture = _obs(observed=True, state="clear", staleness_sec=0.5)
    assert observe_cursor_limit(fixture=fixture) is fixture
    assert decide_cursor_budget(observe_cursor_limit(fixture=fixture)) is None

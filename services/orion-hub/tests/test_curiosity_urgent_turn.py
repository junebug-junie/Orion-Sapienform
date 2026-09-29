"""Urgent curiosity turns: the typed seed on the turn request reaches the unified turn.

Plan: docs/superpowers/plans/2026-09-28-urgent-curiosity-plan-3-seeded-urgent-runs.md (Task 4).
`CuriosityTurnRequestV1.urgent` set -> `execute_unified_turn(urgent=True)`, which overrides a
stance defer/refuse and fails the turn (never defers) when stance is unavailable. Keyed on the
typed contract field only; nothing branches on the prompt's words.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from orion.schemas.curiosity_urgent import CuriosityUrgentSeedV1
from orion.schemas.durable_run import CuriosityTurnRequestV1
from scripts.curiosity_investigation import CuriosityInvestigation
from test_curiosity_investigation import _CortexBus, _loop


def _seed() -> CuriosityUrgentSeedV1:
    return CuriosityUrgentSeedV1(
        incident_id="a1b2c3d4e5f6",
        question="Why is gpu2 running hot?",
        trigger="manual",
        subject="circe/gpu2",
        requested_at=datetime(2026, 9, 28, tzinfo=timezone.utc),
    )


def _request(*, urgent: CuriosityUrgentSeedV1 | None) -> CuriosityTurnRequestV1:
    return CuriosityTurnRequestV1(
        run_id="abc123def456", correlation_id="abc123def456", prompt="Study the incident",
        timeout_sec=42, urgent=urgent,
    )


def _drive(request: CuriosityTurnRequestV1, frames: list[dict]):
    """Real `_turn_result_for` -> real `_generate` -> stubbed `execute_unified_turn`."""
    import orion.hub.turn_orchestrator as orchestrator

    captured: dict = {}
    original = orchestrator.execute_unified_turn

    async def _stub(**kwargs):
        captured.update(kwargs)
        return frames

    orchestrator.execute_unified_turn = _stub
    try:
        loop = _loop(_CortexBus(), text=None, kickoff_via_cortex=True)
        loop._generate = CuriosityInvestigation._generate.__get__(loop)
        result = asyncio.run(loop._turn_result_for(request, hold_lock=False))
    finally:
        orchestrator.execute_unified_turn = original
    return result, captured


_LOOKED = [{"type": "final", "llm_response": "found it", "harness_step_count": 14}]


def test_urgent_seed_reaches_the_unified_turn_as_urgent_true() -> None:
    result, captured = _drive(_request(urgent=_seed()), _LOOKED)
    assert captured["urgent"] is True
    assert result.ok and result.text == "found it"


def test_no_seed_reaches_the_unified_turn_as_urgent_false() -> None:
    result, captured = _drive(_request(urgent=None), _LOOKED)
    assert captured["urgent"] is False
    assert result.ok


def test_urgent_turn_still_requires_a_lookup() -> None:
    """MIN_HARNESS_STEPS still applies: an urgent turn that did not look fails."""
    result, captured = _drive(
        _request(urgent=_seed()),
        [{"type": "final", "llm_response": "a guess", "harness_step_count": 1}],
    )
    assert captured["urgent"] is True
    assert not result.ok and result.error == "no_lookup"


def test_urgent_stance_unavailable_fails_the_turn_with_its_reason() -> None:
    frames = [{
        "type": "turn_error", "phase": "stance", "correlation_id": "abc123def456",
        "finalize_ran": False, "error": "urgent_stance_unavailable",
        "stance_failure_reason": "stance_react_timeout",
    }]
    result, _ = _drive(_request(urgent=_seed()), frames)
    assert not result.ok and result.text == ""
    assert result.error == "urgent_stance_unavailable"
    assert result.debug["frame_type"] == "turn_error"


def test_non_urgent_no_final_frame_debug_is_unchanged() -> None:
    frames = [{"type": "turn_deferred", "correlation_id": "abc123def456", "reason": "stale"}]
    result, _ = _drive(_request(urgent=None), frames)
    assert not result.ok
    assert result.error == "no_final_frame"
    assert set(result.debug) == {"error", "frame_type", "elapsed_sec"}

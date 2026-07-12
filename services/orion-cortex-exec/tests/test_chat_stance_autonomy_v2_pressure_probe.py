"""Unit tests for the Phase 4 drive-pressure measurement probe (2026-07-12).

Measurement-only: logs `AutonomyStateV2`'s pressure vector right after every
`_run_autonomy_reducer` fold so it can be compared offline (by grepping logs
and correlating on subject + nearby timestamp) against `DriveEngine`'s
independently-computed pressures, logged from
`orion/spark/concept_induction/bus_worker.py`. No schema change, no behavior
change -- see
docs/superpowers/plans/2026-07-12-self-state-mesh-substrate-redesign.md Phase 4.
"""
from __future__ import annotations

import logging

import pytest

from orion.autonomy.models import AutonomyStateV1
from orion.autonomy.reducer import reduce_autonomy_state as real_reduce_autonomy_state

from app import chat_stance


def _v1_baseline() -> AutonomyStateV1:
    return AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        dominant_drive="coherence",
        drive_pressures={"coherence": 0.4},
        active_drives=["coherence"],
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
    )


def _wire_common_mocks(monkeypatch, *, store: dict) -> None:
    monkeypatch.setattr(chat_stance, "load_autonomy_state_v2", lambda subject: store.get(subject))
    monkeypatch.setattr(chat_stance, "save_autonomy_state_v2", lambda subject, state: store.__setitem__(subject, state))
    monkeypatch.setattr(chat_stance, "load_action_outcomes", lambda subject: [])


@pytest.mark.asyncio
async def test_run_autonomy_reducer_logs_pressure_probe(monkeypatch, caplog) -> None:
    store: dict = {}
    _wire_common_mocks(monkeypatch, store=store)

    v1_baseline = _v1_baseline()
    autonomy = {"state": v1_baseline}
    ctx: dict = {"user_message": "hello", "correlation_id": "c-1"}

    caplog.set_level(logging.INFO)
    result = await chat_stance._run_autonomy_reducer(ctx, autonomy, social={}, social_bridge={}, reasoning={})

    assert "autonomy_state_v2_pressure_probe" in caplog.text
    assert "subject=orion" in caplog.text
    for drive_id, value in dict(result.state.drive_pressures or {}).items():
        assert drive_id in caplog.text
        assert f"{round(value, 4)}" in caplog.text


@pytest.mark.asyncio
async def test_run_autonomy_reducer_pressure_probe_matches_folded_state(monkeypatch, caplog) -> None:
    """The logged pressures must be the SAME dict the reducer actually folded
    into result.state -- not e.g. the before-pressures or the V1 baseline."""
    store: dict = {}
    _wire_common_mocks(monkeypatch, store=store)

    def spy_reduce(inp):
        return real_reduce_autonomy_state(inp)

    monkeypatch.setattr(chat_stance, "reduce_autonomy_state", spy_reduce)

    v1_baseline = _v1_baseline()
    autonomy = {"state": v1_baseline}
    ctx: dict = {"user_message": "first turn, feeling curious", "correlation_id": "c-1"}

    caplog.set_level(logging.INFO)
    result = await chat_stance._run_autonomy_reducer(ctx, autonomy, social={}, social_bridge={}, reasoning={})

    probe_lines = [line for line in caplog.text.splitlines() if "autonomy_state_v2_pressure_probe" in line]
    assert len(probe_lines) == 1
    for drive_id, value in dict(result.state.drive_pressures or {}).items():
        assert f"'{drive_id}': {round(value, 4)}" in probe_lines[0]


@pytest.mark.asyncio
async def test_pressure_probe_failure_is_swallowed_and_reducer_still_completes(monkeypatch, caplog) -> None:
    """A probe logging failure must never break the hot chat-turn path. The
    call site in `_run_autonomy_reducer` has NO outer try/except around
    `_log_autonomy_pressure_probe` -- the helper's own internal try/except is
    the only thing protecting this path, so raise from inside the real
    logger.info call (not by replacing the whole helper, which would bypass
    that internal guard and prove nothing)."""
    store: dict = {}
    _wire_common_mocks(monkeypatch, store=store)

    original_info = chat_stance.logger.info

    def _raise_on_probe(msg, *args, **kwargs):
        if isinstance(msg, str) and msg.startswith("autonomy_state_v2_pressure_probe"):
            raise RuntimeError("logging exploded")
        return original_info(msg, *args, **kwargs)

    monkeypatch.setattr(chat_stance.logger, "info", _raise_on_probe)

    v1_baseline = _v1_baseline()
    autonomy = {"state": v1_baseline}
    ctx: dict = {"user_message": "hello", "correlation_id": "c-1"}

    # Must not raise, and must still return a real result / write-back.
    result = await chat_stance._run_autonomy_reducer(ctx, autonomy, social={}, social_bridge={}, reasoning={})
    assert result is not None
    assert "orion" in store


def test_log_autonomy_pressure_probe_never_raises_on_logging_failure(monkeypatch, caplog) -> None:
    """Direct unit test of the helper itself: a raising logger.info must be
    caught and turned into a warning, never propagate."""

    class _ExplodingLogger:
        def info(self, *_a, **_kw):
            raise RuntimeError("boom")

        def warning(self, *args, **kwargs):
            logging.getLogger("orion.cortex.exec.chat_stance").warning(*args, **kwargs)

    original_logger = chat_stance.logger
    chat_stance.logger = _ExplodingLogger()
    try:
        caplog.set_level(logging.WARNING)
        # Must not raise.
        chat_stance._log_autonomy_pressure_probe("orion", {"coherence": 0.5})
    finally:
        chat_stance.logger = original_logger

    assert "autonomy_state_v2_pressure_probe_failed" in caplog.text

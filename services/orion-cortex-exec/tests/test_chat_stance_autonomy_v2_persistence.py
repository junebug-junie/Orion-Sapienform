"""Non-amnesiac acceptance check for `_run_autonomy_reducer`.

Verifies the reducer's own output now closes its fold loop via
`orion.autonomy.state_store` instead of being discarded every turn: the
second call's effective `previous_state` must be the first call's own
`result.state`, not the V1/graph baseline passed in via `ctx`/`autonomy`.
"""
from __future__ import annotations

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


def test_run_autonomy_reducer_uses_persisted_state_not_v1_baseline_on_second_call(monkeypatch) -> None:
    store: dict[str, object] = {}

    def fake_load(subject):
        return store.get(subject)

    def fake_save(subject, state):
        store[subject] = state

    monkeypatch.setattr(chat_stance, "load_autonomy_state_v2", fake_load)
    monkeypatch.setattr(chat_stance, "save_autonomy_state_v2", fake_save)
    monkeypatch.setattr(chat_stance, "load_action_outcomes", lambda subject: [])

    captured_previous_states: list[object] = []

    def spy_reduce(inp):
        captured_previous_states.append(inp.previous_state)
        return real_reduce_autonomy_state(inp)

    monkeypatch.setattr(chat_stance, "reduce_autonomy_state", spy_reduce)

    v1_baseline = _v1_baseline()
    autonomy = {"state": v1_baseline}

    ctx1: dict = {"user_message": "first turn, feeling curious about the lab", "correlation_id": "c-1"}
    result1 = chat_stance._run_autonomy_reducer(
        ctx1, autonomy, social={}, social_bridge={}, reasoning={}
    )

    ctx2: dict = {"user_message": "second turn, entirely different evidence", "correlation_id": "c-2"}
    result2 = chat_stance._run_autonomy_reducer(
        ctx2, autonomy, social={}, social_bridge={}, reasoning={}
    )

    assert len(captured_previous_states) == 2
    # First-ever turn for this subject: nothing persisted yet, falls back to
    # the V1/graph baseline exactly as before.
    assert captured_previous_states[0] is v1_baseline

    # Second call: previous_state must be the FIRST call's own reducer output
    # (round-tripped through the store), not the V1 baseline again. This is
    # the core non-amnesiac acceptance check.
    assert captured_previous_states[1] is result1.state
    assert captured_previous_states[1] is not v1_baseline

    # The store now holds the second call's output -- the write-back happened.
    assert store["orion"] is result2.state


def test_run_autonomy_reducer_falls_back_to_v1_baseline_when_store_unreachable(monkeypatch) -> None:
    """Store returning None (unreachable / no DSN) must not change today's
    fallback behavior: previous_state stays the V1/graph baseline."""

    monkeypatch.setattr(chat_stance, "load_autonomy_state_v2", lambda subject: None)
    saved: dict[str, object] = {}
    monkeypatch.setattr(chat_stance, "save_autonomy_state_v2", lambda subject, state: saved.__setitem__(subject, state))
    monkeypatch.setattr(chat_stance, "load_action_outcomes", lambda subject: [])

    captured_previous_states: list[object] = []

    def spy_reduce(inp):
        captured_previous_states.append(inp.previous_state)
        return real_reduce_autonomy_state(inp)

    monkeypatch.setattr(chat_stance, "reduce_autonomy_state", spy_reduce)

    v1_baseline = _v1_baseline()
    autonomy = {"state": v1_baseline}
    ctx: dict = {"user_message": "hello", "correlation_id": "c-1"}

    chat_stance._run_autonomy_reducer(ctx, autonomy, social={}, social_bridge={}, reasoning={})

    assert captured_previous_states == [v1_baseline]
    assert "orion" in saved  # write-back still attempted even though load returned None


def test_run_autonomy_reducer_write_failure_is_swallowed(monkeypatch) -> None:
    """save_autonomy_state_v2 raising must not propagate out of the reducer --
    belt-and-suspenders fail-open, even though the store already fails open
    internally."""

    monkeypatch.setattr(chat_stance, "load_autonomy_state_v2", lambda subject: None)

    def boom(subject, state):
        raise RuntimeError("db hiccup")

    monkeypatch.setattr(chat_stance, "save_autonomy_state_v2", boom)
    monkeypatch.setattr(chat_stance, "load_action_outcomes", lambda subject: [])

    v1_baseline = _v1_baseline()
    autonomy = {"state": v1_baseline}
    ctx: dict = {"user_message": "hello", "correlation_id": "c-1"}

    # Must not raise.
    result = chat_stance._run_autonomy_reducer(ctx, autonomy, social={}, social_bridge={}, reasoning={})
    assert result is not None

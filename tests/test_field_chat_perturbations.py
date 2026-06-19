from __future__ import annotations

from orion.schemas.state_delta import StateDeltaV1

from app.ingest.state_deltas import delta_to_perturbations


def _make_chat_delta(
    *,
    operation: str = "update",
    node_id: str = "athena",
    pressure_hints: dict | None = None,
    include_hints: bool = True,
) -> StateDeltaV1:
    after: dict = {"node_id": node_id}
    if include_hints:
        after["pressure_hints"] = pressure_hints or {}
    return StateDeltaV1(
        delta_id="delta_chat_1",
        target_projection="chat_session",
        target_kind="chat_turn",
        target_id=f"chat:{node_id}:sess-1",
        operation=operation,  # type: ignore[arg-type]
        after=after,
        caused_by_event_ids=["gev_chat_1"],
        reducer_id="chat_reducer",
    )


def test_chat_turn_delta_produces_conversation_load() -> None:
    delta = _make_chat_delta(pressure_hints={"conversation_load": 0.6, "repair_pressure": 0.2})
    perturbations = delta_to_perturbations(delta)
    channels = {p.channel: p.intensity for p in perturbations}
    assert "conversation_load" in channels, f"expected conversation_load, got {list(channels)}"
    assert channels["conversation_load"] == 0.6
    assert perturbations[0].node_id == "node:athena"


def test_chat_turn_delta_produces_repair_pressure() -> None:
    delta = _make_chat_delta(pressure_hints={"conversation_load": 0.3, "repair_pressure": 0.8})
    perturbations = delta_to_perturbations(delta)
    channels = {p.channel: p.intensity for p in perturbations}
    assert "repair_pressure" in channels, f"expected repair_pressure, got {list(channels)}"
    assert channels["repair_pressure"] == 0.8


def test_chat_turn_noop_delta_skipped() -> None:
    delta = _make_chat_delta(
        operation="noop",
        pressure_hints={"conversation_load": 0.9, "repair_pressure": 0.9},
    )
    perturbations = delta_to_perturbations(delta)
    assert perturbations == [], f"expected no perturbations for noop, got {perturbations}"


def test_chat_turn_missing_hints_skipped() -> None:
    delta = _make_chat_delta(include_hints=False)
    perturbations = delta_to_perturbations(delta)
    assert perturbations == [], f"expected no perturbations when hints absent, got {perturbations}"


def test_chat_turn_does_not_emit_topic_coherence() -> None:
    delta = _make_chat_delta(
        pressure_hints={
            "conversation_load": 0.5,
            "repair_pressure": 0.4,
            "topic_coherence": 0.9,
        }
    )
    perturbations = delta_to_perturbations(delta)
    channels = [p.channel for p in perturbations]
    assert "topic_coherence" not in channels, (
        f"topic_coherence must not reach the lattice, but found in {channels}"
    )
    # the other two should still be present
    assert "conversation_load" in channels
    assert "repair_pressure" in channels

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from orion.substrate.attention_broadcast import (
    attention_broadcast_enabled,
    broadcast_projection_from_frame,
    build_substrate_attention_frame,
    substrate_pressure_signals,
)

_NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


def _node(node_id: str, label: str, **metadata) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        label=label,
        metadata=metadata,
        signals=SimpleNamespace(confidence=0.8),
    )


def test_flag_default_off(monkeypatch) -> None:
    monkeypatch.delenv("ORION_ATTENTION_BROADCAST_ENABLED", raising=False)
    assert attention_broadcast_enabled() is False
    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "true")
    assert attention_broadcast_enabled() is True


def test_high_pressure_node_wins_over_calm() -> None:
    nodes = [
        _node("node:calm", "calm background concept", dynamic_pressure=0.05),
        _node("node:hot", "unresolved execution contradiction", dynamic_pressure=0.9),
        _node("node:mild", "mildly active concept", dynamic_pressure=0.3),
    ]
    frame = build_substrate_attention_frame(nodes=nodes, now=_NOW)
    assert frame.selected_action is not None
    winner = next(
        loop for loop in frame.open_loops if loop.id == frame.selected_action.open_loop_id
    )
    assert winner.description == "unresolved execution contradiction"
    # calm node below min_salience never entered the competition
    assert all(loop.description != "calm background concept" for loop in frame.open_loops)


def test_prediction_error_beats_equal_plain_pressure() -> None:
    # dynamic_pressure carries magnitude for both nodes (what the dynamics
    # engine would have populated by the time either node is scored --
    # magnitude must never come from the raw, non-decaying prediction_error
    # field directly, see _node_salience()). prediction_error is still set
    # on node:surprise purely for its anomaly-vs-concept typing effect.
    nodes = [
        _node("node:pressure", "steady pressure region", dynamic_pressure=0.8),
        _node(
            "node:surprise",
            "surprising transport batch",
            dynamic_pressure=0.8,
            prediction_error=0.8,
        ),
    ]
    frame = build_substrate_attention_frame(nodes=nodes, now=_NOW)
    assert frame.selected_action is not None
    winner = next(
        loop for loop in frame.open_loops if loop.id == frame.selected_action.open_loop_id
    )
    assert winner.description == "surprising transport batch"
    assert winner.target_type == "anomaly"


def test_salience_uses_decayed_pressure_not_raw_prediction_error() -> None:
    """A node whose prediction_error seed has decayed (dynamic_pressure near
    zero, as SubstrateDynamicsEngine.tick() would compute after enough
    elapsed time) must report low salience -- not near-maximal salience just
    because the raw, non-decaying metadata['prediction_error'] field is
    still 1.0. Regression for the live bug where _node_salience() raced the
    two and always picked the raw value (dynamic_pressure = raw * weight(<1)
    * decay(<=1) can mathematically never exceed it), silently discarding
    the dynamics engine's decay on every tick, forever, for any node that
    ever had a prediction_error written. `kind` should still resolve to
    "prediction_error" (anomaly typing) even though magnitude is low."""
    nodes = [
        _node(
            "node:stale-surprise",
            "long-stale transport anomaly",
            dynamic_pressure=0.02,  # decayed by the dynamics engine over time
            prediction_error=1.0,  # raw seed -- never decays on its own
        ),
    ]
    signals = substrate_pressure_signals(nodes, min_salience=0.0)
    assert signals
    assert signals[0].salience == 0.02
    assert signals[0].target_type_hint == "anomaly"


def test_broadcast_never_generates_questions() -> None:
    nodes = [_node("node:hot", "very hot region", dynamic_pressure=0.95, prediction_error=0.9)]
    frame = build_substrate_attention_frame(nodes=nodes, now=_NOW)
    assert all(action.action_type != "ask" for action in frame.candidate_actions)
    assert all(action.question_text is None for action in frame.candidate_actions)
    assert frame.selected_action is not None
    assert frame.selected_action.action_type != "ask"


def test_calm_substrate_selects_none() -> None:
    nodes = [_node("node:calm", "calm concept", dynamic_pressure=0.01)]
    frame = build_substrate_attention_frame(nodes=nodes, now=_NOW)
    assert frame.open_loops == []
    assert frame.selected_action is not None
    assert frame.selected_action.action_type == "none"


def test_malformed_nodes_are_skipped_not_fatal() -> None:
    nodes = [
        object(),
        SimpleNamespace(node_id="node:x", label="", metadata={"dynamic_pressure": "not-a-float"}),
        _node("node:ok", "well formed node", dynamic_pressure=0.7),
    ]
    signals = substrate_pressure_signals(nodes)
    assert [s.target_text for s in signals] == ["well formed node"]


def test_projection_carries_selected_coalition() -> None:
    nodes = [_node("node:hot", "hot node", dynamic_pressure=0.9)]
    frame = build_substrate_attention_frame(nodes=nodes, now=_NOW)
    projection = broadcast_projection_from_frame(frame)
    assert projection.projection_id == "substrate.attention.broadcast.v1"
    assert projection.selected_open_loop_id == frame.selected_action.open_loop_id
    assert projection.selected_description == "hot node"
    assert projection.attended_node_ids == ["node:hot"]
    assert projection.generated_at == _NOW

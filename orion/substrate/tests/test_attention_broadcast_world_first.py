"""World-first substrate broadcast (spec 2026-10-07 self-calibration, section A).

The pre-world-first docstring said "always one winner" and "magnitude is never
read by scoring, so it cannot change who wins". Both change here ON PURPOSE:
with world_first=True magnitude is the gate and a calm tick has no winner.
The flag-off behaviour is pinned by tests/test_attention_world_first_parity.py
and the existing test_attention_broadcast*.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import orion.substrate.attention_broadcast as ab
from orion.attention.world_first import WORLD_CHAT_SOURCE_ID, chat_candidate
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setenv("ORION_ATTENTION_TOPDOWN_ENABLED", "false")
    ab._coalition_history.clear()
    ab._transition_history.clear()
    ab._current_active_coalition = None
    ab._dwell_ticks = 0
    with patch.object(ab, "load_terminal_verdict_loop_ids", return_value=set()):
        yield


def _node(node_id, label, *, pressure=0.2, pe=0.3, **extra_md):
    md = {"dynamic_pressure": pressure, "dynamic_pressure_reason": "prediction_error_seed"}
    if pe is not None:
        md["prediction_error"] = pe
    md.update(extra_md)
    return SimpleNamespace(
        node_id=node_id, label=label, node_kind="concept", metadata=md,
        signals=SimpleNamespace(confidence=0.8),
        temporal=SimpleNamespace(observed_at=NOW - timedelta(seconds=30)),
    )


def _mag(pct, value=0.3, age=30.0):
    band = "quiet" if pct < 0.5 else "usual" if pct < 0.9 else "high" if pct < 0.99 else "unusual"
    return PredictionErrorMagnitudeV1(
        value=value, age_sec=age, percentile_now=pct, n_readings_7d=4000, band=band, trend="flat"
    )


def _frame(nodes, mags, external=None):
    return ab.build_substrate_attention_frame(
        nodes=nodes, now=NOW, magnitude_by_node_id=mags,
        world_first=True, external_candidates=external or [],
    )


def test_calm_tick_has_no_winner() -> None:
    nodes = [_node("node:substrate.execution", "Execution PE"), _node("node:substrate.biometrics", "Bio PE")]
    frame = _frame(nodes, {"node:substrate.execution": _mag(0.7), "node:substrate.biometrics": _mag(0.3)})
    assert frame.open_loops == []
    assert frame.selected_action.action_type == "none"
    wf = frame.debug["world_first"]
    assert wf["no_winner"] is True and len(wf["candidates"]) == 2
    proj = ab.broadcast_projection_from_frame(frame)
    assert proj.attended_node_ids == [] and proj.selected_open_loop_id is None


def test_magnitude_now_decides_who_wins() -> None:
    """Pressure says biometrics; its own history says execution is the alarm."""
    nodes = [
        _node("node:substrate.execution", "Execution PE", pressure=0.06),
        _node("node:substrate.biometrics", "Bio PE", pressure=0.9),
    ]
    frame = _frame(nodes, {"node:substrate.execution": _mag(0.995), "node:substrate.biometrics": _mag(0.6)})
    proj = ab.broadcast_projection_from_frame(frame)
    assert proj.attended_node_ids == ["node:substrate.execution"]
    loop = frame.open_loops[0]
    assert loop.salience == 0.995 and loop.provenance["source_kind"] == "internal"
    assert "borda_salience" in loop.provenance


def test_internal_below_high_cannot_enter_even_with_huge_pressure() -> None:
    frame = _frame([_node("node:substrate.route", "Route PE", pressure=1.0)], {"node:substrate.route": _mag(0.89)})
    assert frame.open_loops == []


def test_chat_external_candidate_wins_on_a_calm_body() -> None:
    turns = [NOW - timedelta(days=d, minutes=30) for d in range(1, 7)] + [NOW - timedelta(minutes=2)]
    frame = _frame(
        [_node("node:substrate.biometrics", "Bio PE")],
        {"node:substrate.biometrics": _mag(0.4)},
        external=[chat_candidate(turns, now=NOW)],
    )
    proj = ab.broadcast_projection_from_frame(frame)
    assert proj.attended_node_ids == [WORLD_CHAT_SOURCE_ID]
    assert frame.open_loops[0].provenance["source_kind"] == "external"
    assert proj.selected_action_type == "watch"


def test_perception_is_external_and_absent_when_embeddings_stale() -> None:
    stale = _node("node:substrate.perception", "Perception PE", pe=0.0, embedding_staleness=1.0)
    frame = _frame([stale], {"node:substrate.perception": _mag(0.97)})
    assert frame.open_loops == []
    (c,) = frame.debug["world_first"]["candidates"]
    assert c["source_kind"] == "external" and c["absent"] is True
    fresh = _node("node:substrate.perception", "Perception PE", pe=0.4, embedding_staleness=0.0)
    frame = _frame([fresh], {"node:substrate.perception": _mag(0.7)})
    assert frame.open_loops and frame.open_loops[0].provenance["source_kind"] == "external"


def test_uncalibrated_graph_nodes_are_traced_not_attended() -> None:
    concept = _node("node:concept.x", "unresolved contradiction", pressure=0.9, pe=None)
    frame = _frame([concept], {})
    assert frame.open_loops == []
    assert frame.debug["world_first"]["uncalibrated"] == ["node:concept.x"]


def test_body_node_without_magnitude_is_absent_not_calm() -> None:
    frame = _frame([_node("node:substrate.execution", "Execution PE")], {})
    (c,) = frame.debug["world_first"]["candidates"]
    assert c["absent"] is True and c["eligible"] is False


def test_empty_coalition_never_activates_or_dwells() -> None:
    """Two no-winner ticks used to 'activate' a size-0 coalition."""
    calm = _frame([_node("node:substrate.execution", "Execution PE")], {"node:substrate.execution": _mag(0.2)})
    for _ in range(4):
        proj = ab.broadcast_projection_from_frame(calm)
    assert proj.dwell_ticks == 0
    assert not any(e["event"] == "activated" for e in proj.coalition_history)


def test_real_coalition_still_activates_and_decays_across_empty_ticks() -> None:
    hot = _frame([_node("node:substrate.execution", "Execution PE")], {"node:substrate.execution": _mag(0.995)})
    calm = _frame([_node("node:substrate.execution", "Execution PE")], {"node:substrate.execution": _mag(0.2)})
    ab.broadcast_projection_from_frame(hot)
    proj = ab.broadcast_projection_from_frame(hot)
    assert proj.dwell_ticks == 1 and proj.coalition_history[-1]["event"] == "activated"
    for _ in range(3):
        proj = ab.broadcast_projection_from_frame(calm)
    assert proj.coalition_history[-1]["event"] == "decayed" and proj.dwell_ticks == 0

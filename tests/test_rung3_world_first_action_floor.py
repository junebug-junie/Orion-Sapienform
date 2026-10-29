"""Rung 3 binding fix: an eligible world-first winner must not be dropped to "none".

Live 48 h to 2026-10-21: 48 broadcast ticks had an eligible world-first winner and an open loop but
selected_action_type "none", because select_actions' absolute 0.48/0.35 ladder ran on the gate's
FADING rank score (percentile x event decay). Nothing could bind to those winners.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import orion.substrate.attention_broadcast as ab
from orion.schemas.attention_frame import CuriosityCandidateActionV1, OpenLoopV1, PredictionErrorMagnitudeV1

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
ROUTE = "node:substrate.route"


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    monkeypatch.setattr(ab, "load_terminal_verdict_loop_ids", lambda ids, now=None: set())


def _route_node(age_sec: float):
    return SimpleNamespace(
        node_id=ROUTE, label="route", node_kind="concept",
        metadata={"prediction_error": 0.5, "dynamic_pressure": 0.5},
        signals=SimpleNamespace(confidence=0.8),
        temporal=SimpleNamespace(observed_at=NOW - timedelta(seconds=age_sec)),
    )


def _frame(age_sec: float):
    mag = PredictionErrorMagnitudeV1(value=0.5, age_sec=age_sec, percentile_now=0.93,
                                     n_readings_7d=1439, band="high", trend="flat")
    return ab.build_substrate_attention_frame(
        nodes=[_route_node(age_sec)], now=NOW, world_first=True, external_candidates=[],
        magnitude_by_node_id={ROUTE: mag})


def test_fading_eligible_winner_is_selected_not_dropped() -> None:
    """250 s into the 300 s orienting window the rank score is ~0.19 (< 0.35): still eligible, so
    it must be selected (defer), not 'none'. Fails on the pre-fix code."""
    frame = _frame(250.0)
    assert frame.debug["world_first"]["winner"] == ROUTE
    assert frame.open_loops and frame.open_loops[0].salience < 0.35
    assert frame.selected_action.action_type == "defer"
    assert frame.selected_action.open_loop_id == frame.open_loops[0].id
    assert frame.selected_action.provenance.get("world_first_floor") is True


def test_projection_binds_the_fading_winner() -> None:
    proj = ab.broadcast_projection_from_frame(_frame(250.0))
    assert proj.selected_action_type == "defer"
    assert proj.attended_node_ids[:1] == [ROUTE]


def test_fresh_winner_keeps_the_ladder_grade() -> None:
    assert _frame(10.0).selected_action.action_type == "watch"


def test_past_the_window_there_is_still_no_winner() -> None:
    """The floor never invents a winner: once the gate says ineligible, the tick stays calm."""
    frame = _frame(310.0)
    assert frame.debug["world_first"]["winner"] is None
    assert frame.selected_action.action_type == "none"


def test_floor_only_touches_eligible_loops() -> None:
    loop = OpenLoopV1(id="l1", description="d", source_refs=["node:substrate.other"], salience=0.1)
    none = CuriosityCandidateActionV1(action_type="none", open_loop_id="l1", score=0.1)
    selected = CuriosityCandidateActionV1(action_type="none", score=0.0)
    actions, out, deferred = ab.world_first_action_floor(
        [none], selected, [], open_loops=[loop], eligible_scores={ROUTE: 0.9})
    assert actions == [none] and out is selected and deferred == []


# --- world-first WORLD winners bind (Juniper's chat is world:chat) --------------------------------

def _projection(node: str):
    return {"selected_action_type": "watch", "selected_open_loop_id": "open-loop-chat", "generated_at": NOW,
            "dwell_ticks": 3, "attended_node_ids": [node]}


def test_a_world_winner_binds_when_listed() -> None:
    """Fails on pre-rung-3 code: only node: refs were scanned, so world:chat was node_not_bound."""
    from orion.autonomy.self_shed import bind_workspace_winner

    winner, why = bind_workspace_winner(_projection("world:chat"), broadcast_log_id="b1",
                                        binds_to_nodes=("world:chat",), now=NOW)
    assert why is None and winner.node_id == "world:chat" and winner.open_loop_id == "open-loop-chat"


def test_a_world_winner_still_needs_to_be_listed() -> None:
    from orion.autonomy.self_shed import bind_workspace_winner

    winner, why = bind_workspace_winner(_projection("world:chat"), broadcast_log_id="b1",
                                        binds_to_nodes=("node:substrate.cabinet",), now=NOW)
    assert winner is None and why == "winner_unbindable:node_not_bound"

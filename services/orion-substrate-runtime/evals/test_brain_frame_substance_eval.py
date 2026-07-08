from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.brain_frame_producer import assemble_brain_frame


def _node(node_id, kind, activation, pressure=0.0):
    # Real node shape: activation nested at signals.activation.activation.
    return SimpleNamespace(
        node_id=node_id, node_kind=kind, label=f"{kind}:{node_id}",
        signals=SimpleNamespace(
            salience=activation,
            activation=SimpleNamespace(activation=activation),
        ),
        metadata={"dynamic_pressure": pressure},
    )


def _settings():
    return SimpleNamespace(
        brain_frame_sample_nodes=40, brain_frame_sample_edges=60,
        brain_frame_firing_threshold=0.5, brain_frame_starving_threshold=0.1,
        brain_frame_self_state_cadence_sec=30.0, brain_frame_spotlight_cadence_sec=30.0,
    )


def test_active_and_dormant_graph_yields_firing_and_starving_and_samples():
    """Acceptance §2: >=1 firing region, >=1 starving region, non-empty samples."""
    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    nodes = [
        _node("e1", "event", 0.9, 0.85),
        _node("t1", "tension", 0.8, 0.7),
        _node("c1", "concept", 0.01),
        _node("o1", "ontology_branch", 0.0),
    ]
    frame = assemble_brain_frame(
        nodes=nodes, edges=[],
        lane_health={"cursor_lag_by_reducer": {}, "pending_backlog_by_reducer": {}, "quarantine_by_reducer": {}},
        self_state=None, attention=None, settings=_settings(), now=now, tick_seq=1,
    )
    states = [r.state for r in frame.regions if r.dimension == "node_kind"]
    assert "firing" in states, "expected at least one firing node-kind region"
    assert "starving" in states, "expected at least one starving node-kind region"
    assert len(frame.nodes) >= 1, "node samples must be non-empty for an active graph"
    assert all(r.node_count >= 0 for r in frame.regions)


def test_heavy_tool_turn_shape_execution_lit_concept_dim():
    """Acceptance §1 shape: execution lane fires, concept node-kind stays dim."""
    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    nodes = [_node("c1", "concept", 0.05), _node("e1", "event", 0.2)]
    lane_health = {
        "cursor_lag_by_reducer": {"execution_trajectory": 2.0},
        "pending_backlog_by_reducer": {"execution_trajectory": 15},
        "quarantine_by_reducer": {},
    }
    frame = assemble_brain_frame(
        nodes=nodes, edges=[], lane_health=lane_health,
        self_state=None, attention=None, settings=_settings(), now=now, tick_seq=1,
    )
    lanes = {r.region_id: r for r in frame.regions if r.dimension == "lane"}
    kinds = {r.region_id: r for r in frame.regions if r.dimension == "node_kind"}
    assert lanes["lane:execution_trajectory"].state in {"firing", "steady"}
    assert kinds["node_kind:concept"].state == "starving"

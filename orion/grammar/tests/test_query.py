from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from orion.grammar.constants import GRAMMAR_DIMENSIONS, GRAMMAR_LAYERS
from orion.grammar.graph_view import layer_y_position
from orion.grammar.query import get_trace_graph, list_traces


def _trace_row(
    *,
    trace_id: str = "trace:demo",
    session_id: str | None = "sess:1",
) -> MagicMock:
    now = datetime.now(timezone.utc)
    row = MagicMock()
    row.trace_id = trace_id
    row.trace_type = "vision_observation"
    row.session_id = session_id
    row.turn_id = "turn:1"
    row.root_event_id = "evt:root"
    row.started_at = now
    row.ended_at = None
    row.status = "open"
    row.summary = "demo trace"
    row.created_at = now
    return row


def _atom_row(
    *,
    atom_id: str,
    layer: str = "sensor_raw",
    dimensions: list[str] | None = None,
) -> MagicMock:
    now = datetime.now(timezone.utc)
    row = MagicMock()
    row.atom_id = atom_id
    row.trace_id = "trace:demo"
    row.atom_type = "observation"
    row.semantic_role = "motion"
    row.layer = layer
    row.dimensions = dimensions or ["visual"]
    row.summary = f"summary for {atom_id}"
    row.text_value = None
    row.confidence = 0.8
    row.salience = None
    row.uncertainty = None
    row.time_start = None
    row.time_end = None
    row.source_event_id = "evt:1"
    row.payload_ref = None
    row.renderer_hint = None
    row.atom_json = {}
    row.created_at = now
    return row


def _edge_row(*, edge_id: str, from_id: str, to_id: str) -> MagicMock:
    now = datetime.now(timezone.utc)
    row = MagicMock()
    row.edge_id = edge_id
    row.trace_id = "trace:demo"
    row.from_atom_id = from_id
    row.to_atom_id = to_id
    row.relation_type = "derived_from"
    row.confidence = 0.7
    row.salience = None
    row.layer_from = "sensor_raw"
    row.layer_to = "sensor_semantic"
    row.temporal_relation = None
    row.evidence_event_ids = []
    row.edge_json = {}
    row.created_at = now
    return row


def _chain_query(session: MagicMock, *, traces, atoms, edges) -> None:
    """Wire session.query(Model).filter(...).<terminal> for trace graph tests."""

    def query_side_effect(model: type) -> MagicMock:
        q = MagicMock()
        model_name = getattr(model, "__name__", str(model))

        if model_name == "GrammarTraceSQL":
            q.filter.return_value.first.return_value = traces[0] if traces else None
            q.order_by.return_value.limit.return_value.all.return_value = traces
        elif model_name == "GrammarAtomSQL":
            terminal = q.filter.return_value
            terminal.first.return_value = atoms[0] if len(atoms) == 1 else None
            terminal.all.return_value = atoms
            terminal.order_by.return_value.all.return_value = atoms
            terminal.count.return_value = len(atoms)
            terminal.distinct.return_value.all.return_value = [(a.layer,) for a in atoms]
            terminal.in_.return_value.all.return_value = atoms
        elif model_name == "GrammarEdgeSQL":
            terminal = q.filter.return_value
            terminal.all.return_value = edges
            terminal.count.return_value = len(edges)
        elif model_name == "GrammarTemporalHopSQL":
            q.filter.return_value.order_by.return_value.all.return_value = []
            q.filter.return_value.all.return_value = []
        elif model_name == "GrammarCompactionSQL":
            q.filter.return_value.order_by.return_value.all.return_value = []
            q.filter.return_value.all.return_value = []
        elif model_name == "GrammarProjectionSQL":
            q.filter.return_value.order_by.return_value.all.return_value = []
        elif model_name == "GrammarEventSQL":
            q.filter.return_value.first.return_value = None
        return q

    session.query.side_effect = query_side_effect


def test_constants_match_spec() -> None:
    assert "sensor_raw" in GRAMMAR_LAYERS
    assert "sensor_semantic" in GRAMMAR_LAYERS
    assert "visual" in GRAMMAR_DIMENSIONS
    assert "epistemic" in GRAMMAR_DIMENSIONS


def test_layer_y_position_orders_by_layer_index() -> None:
    assert layer_y_position("sensor_raw") < layer_y_position("sensor_semantic")


def test_list_traces_returns_items() -> None:
    session = MagicMock()
    trace = _trace_row()
    atom = _atom_row(atom_id="atom:a1")
    _chain_query(session, traces=[trace], atoms=[atom], edges=[])

    items = list_traces(session, session_id=None, limit=50)
    assert len(items) == 1
    assert items[0]["trace_id"] == "trace:demo"
    assert items[0]["atom_count"] == 1
    assert items[0]["edge_count"] == 0


def test_get_trace_graph_nodes_include_layer() -> None:
    session = MagicMock()
    trace = _trace_row()
    a1 = _atom_row(atom_id="atom:a1", layer="sensor_raw")
    a2 = _atom_row(atom_id="atom:a2", layer="sensor_semantic", dimensions=["visual", "epistemic"])
    edge = _edge_row(edge_id="edge:1", from_id="atom:a1", to_id="atom:a2")
    _chain_query(session, traces=[trace], atoms=[a1, a2], edges=[edge])

    graph = get_trace_graph(session, "trace:demo", layout="layered", depth=2)
    assert graph is not None
    assert graph["nodes"]
    assert all("layer" in n for n in graph["nodes"])
    assert graph["groups"]["layers"]
    assert graph["groups"]["dimensions"]


def test_get_trace_graph_unknown_trace_returns_none() -> None:
    session = MagicMock()
    q = MagicMock()
    q.filter.return_value.first.return_value = None
    session.query.return_value = q

    assert get_trace_graph(session, "trace:missing") is None

"""Tests for scripts/check_metric_lineage.py's `_liveness_for_nodes` wiring.

Mocks `orion.metrics.liveness.open_readonly_connection`/`liveness_for_node`
rather than touching real Postgres -- the classification logic itself is
covered by tests/test_metric_liveness.py. This only checks the CLI-level
dispatch: connect-once, skip entirely when nothing needs it, and report
DB-unreachable as UNKNOWN rather than silently as "not computed".
"""
from __future__ import annotations

import importlib
import sys
from unittest import mock

sys.path.insert(0, "scripts")

check_mod = importlib.import_module("check_metric_lineage")


class _FakeNode:
    def __init__(self, urn: str, *, name: str, schema_id=None, metric_field=None):
        self.urn = urn
        self.name = name
        self.schema_id = schema_id
        self.metric_field = metric_field


def test_skips_db_entirely_when_no_node_has_a_source():
    nodes = [_FakeNode("metric://a", name="cpu_pressure")]
    with mock.patch.object(check_mod, "open_readonly_connection") as m:
        result = check_mod._liveness_for_nodes(nodes)
    m.assert_not_called()
    assert result == {"metric://a": None}


def test_reports_db_unreachable_for_source_bearing_nodes_only():
    nodes = [
        _FakeNode("metric://live", name="confidence", schema_id="AttentionSelfModelV1", metric_field="confidence"),
        _FakeNode("metric://dead", name="cpu_pressure"),
    ]
    with mock.patch.object(check_mod, "open_readonly_connection", return_value=None):
        result = check_mod._liveness_for_nodes(nodes)
    assert result["metric://live"] == check_mod._DB_UNREACHABLE
    assert "metric://dead" not in result  # no source, no unreachable claim either


def test_computes_and_closes_connection_once():
    node = _FakeNode("metric://live", name="confidence", schema_id="AttentionSelfModelV1", metric_field="confidence")
    fake_conn = mock.Mock()
    fake_outcome = mock.Mock()
    with mock.patch.object(check_mod, "open_readonly_connection", return_value=fake_conn), \
         mock.patch.object(check_mod, "liveness_for_node", return_value=fake_outcome) as fake_lfn:
        result = check_mod._liveness_for_nodes([node])
    assert result == {"metric://live": fake_outcome}
    fake_lfn.assert_called_once_with(node, fake_conn)
    fake_conn.close.assert_called_once()


def test_query_failure_degrades_to_db_unreachable_not_a_crash():
    """liveness_for_node() deliberately does not catch query-time errors
    itself (see orion/metrics/liveness.py's docstring) -- this is the one
    place that must. A failure on one node must not prevent other nodes in
    the same --metric call from getting a real verdict."""
    broken = _FakeNode("metric://broken", name="confidence", schema_id="AttentionSelfModelV1", metric_field="confidence")
    healthy = _FakeNode("metric://healthy", name="l7_l11_ladder", schema_id=None, metric_field=None)
    fake_conn = mock.Mock()
    fake_outcome = mock.Mock()

    def _side_effect(node, conn):
        if node is broken:
            raise RuntimeError("connection reset by peer")
        return fake_outcome

    with mock.patch.object(check_mod, "open_readonly_connection", return_value=fake_conn), \
         mock.patch.object(check_mod, "liveness_for_node", side_effect=_side_effect):
        result = check_mod._liveness_for_nodes([broken, healthy])
    assert result["metric://broken"] == check_mod._DB_UNREACHABLE
    assert result["metric://healthy"] is fake_outcome
    fake_conn.close.assert_called_once()

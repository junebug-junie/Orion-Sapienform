"""Unit tests for the periodic bus-synaptic-graph tick (feeds
``bus_synaptic_prediction_error`` from the live ``orion_bus_synapse`` FalkorDB
graph, mirroring the shared writer the other five Active-Inference domains
use -- see ``docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-
redesign.md``'s 2026-07-25 revision).

Default-off, fail-open, no grammar-event batch -- closest existing sibling is
``test_worker_dynamics_tick.py``'s pattern, not the four grammar-driven
``_*_tick`` tests.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker


def _make_worker(monkeypatch, *, enabled: bool = True) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("SUBSTRATE_BUS_SYNAPTIC_TICK_ENABLED", "true" if enabled else "false")
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._bus_synaptic_client = None
    worker._substrate_graph_store = None
    worker._store = MagicMock()
    return worker


def _client_returning(publishes_rows, causal_rows):
    """A client whose graph_query alternates publishes/causal rows forever,
    so it supports more than one tick (a fixed 2-item side_effect list would
    raise StopIteration, silently swallowed by the tick's own fail-open
    except-Exception, on any second call -- passing the test for the wrong
    reason)."""
    client = MagicMock()
    responses = [publishes_rows, causal_rows]
    call_count = {"n": 0}

    def _respond(_cypher, _params):
        row = responses[call_count["n"] % 2]
        call_count["n"] += 1
        return row

    client.graph_query.side_effect = _respond
    return client


def test_bus_synaptic_tick_disabled_is_noop(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=False)
    with patch("orion.graph.falkor_client.RedisGraphQueryClient") as client_cls:
        worker._bus_synaptic_tick()
    client_cls.assert_not_called()
    worker._store.save_receipt.assert_not_called()


def test_bus_synaptic_tick_fails_open_on_client_init_error(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    with patch(
        "orion.graph.falkor_client.RedisGraphQueryClient",
        side_effect=RuntimeError("falkor down"),
    ):
        worker._bus_synaptic_tick()  # must not raise
    worker._store.save_receipt.assert_not_called()


def test_bus_synaptic_tick_fails_open_on_query_error(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    client = MagicMock()
    client.graph_query.side_effect = RuntimeError("cypher failed")
    worker._bus_synaptic_client = client
    worker._bus_synaptic_tick()  # must not raise
    worker._store.save_receipt.assert_not_called()


def test_bus_synaptic_tick_skips_nan_and_infinite_zscores(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    worker._bus_synaptic_client = _client_returning(
        [{"zscore": float("nan")}, {"zscore": float("inf")}, {"zscore": 1.5}],
        [],
    )
    with patch.object(worker, "_write_prediction_error_node") as write_node:
        worker._bus_synaptic_tick()
    # Only the real 1.5 should count -- not NaN-poisoned or maxed out by the
    # infinite value. Expected value updated 2026-07-26 for the calm-floor
    # fix (bus_synaptic_prediction_error now subtracts sqrt(2/pi) before
    # saturating): (1.5 - sqrt(2/pi)) / (3.0 - sqrt(2/pi)), not the old
    # mean(1.5)/3.0 = 0.5.
    write_node.assert_called_once()
    assert write_node.call_args.kwargs["error"] == pytest.approx(0.3188367996970756)


def test_bus_synaptic_edge_queries_filter_stale_edges() -> None:
    """Regression guard for the review finding (2026-07-25): a frozen edge
    from a decommissioned organ/channel must not contribute forever."""
    for cypher in BiometricsSubstrateWorker._BUS_SYNAPTIC_EDGE_QUERIES:
        assert "last_seen_epoch" in cypher
        assert "$stale_cutoff_epoch" in cypher


def test_bus_synaptic_tick_no_edges_still_writes_calm_node(monkeypatch):
    """Regression guard (2026-07-30): the node write must happen every tick,
    including a calm error=0.0 tick -- gating it the same way as the receipt
    below left `node:substrate.bus_synaptic` frozen at a stale nonzero value
    for hours (confirmed live) while orion-equilibrium-service polled that
    frozen value with no staleness check of its own, producing permanent
    false "Bus Anomaly Detected" alerts. The receipt stays gated on
    error > 0.0 -- it's an audit trail of notable events, not a polled
    current-state read, so no receipt on a calm tick is correct."""
    worker = _make_worker(monkeypatch, enabled=True)
    worker._bus_synaptic_client = _client_returning([], [])
    with patch.object(worker, "_write_prediction_error_node") as write_node:
        worker._bus_synaptic_tick()
    write_node.assert_called_once()
    _, kwargs = write_node.call_args
    assert kwargs["node_id"] == "node:substrate.bus_synaptic"
    assert kwargs["error"] == 0.0
    assert kwargs["reducer_key"] == "bus_synaptic"
    worker._store.save_receipt.assert_not_called()


def test_bus_synaptic_tick_aggregates_both_edge_kinds_and_writes(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    worker._bus_synaptic_client = _client_returning(
        [{"zscore": 1.5}, {"zscore": -3.0}],
        [{"zscore": 4.5}],
    )
    with patch.object(worker, "_write_prediction_error_node") as write_node:
        worker._bus_synaptic_tick()

    # mean(|1.5|, |-3.0|, |4.5|) / 3.0 saturation = mean(1.5,3.0,4.5)/3.0 = 1.0 (saturated)
    write_node.assert_called_once()
    _, kwargs = write_node.call_args
    assert kwargs["node_id"] == "node:substrate.bus_synaptic"
    assert kwargs["error"] == 1.0
    assert kwargs["reducer_key"] == "bus_synaptic"
    worker._store.save_receipt.assert_called_once()


def test_bus_synaptic_tick_passes_configured_min_count_and_stale_cutoff(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_BUS_SYNAPTIC_MIN_EDGE_COUNT", "9")
    monkeypatch.setenv("SUBSTRATE_BUS_SYNAPTIC_MAX_EDGE_AGE_SEC", "120")
    worker = _make_worker(monkeypatch, enabled=True)
    client = _client_returning([], [])
    worker._bus_synaptic_client = client
    before = time.time()
    worker._bus_synaptic_tick()
    after = time.time()
    assert len(client.graph_query.call_args_list) == 2
    for call in client.graph_query.call_args_list:
        params = call.args[1]
        assert params["min_count"] == 9
        # cutoff = now - max_age_sec, computed inside the tick -- bound it
        # against wall-clock time taken around the call instead of asserting
        # an exact float.
        assert (before - 120) <= params["stale_cutoff_epoch"] <= (after - 120)


def test_bus_synaptic_client_is_cached_across_ticks(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    with patch("orion.graph.falkor_client.RedisGraphQueryClient") as client_cls:
        client_cls.return_value = _client_returning([], [])
        worker._bus_synaptic_tick()
        worker._bus_synaptic_tick()
    client_cls.assert_called_once()
    # second tick's two graph_query calls must have actually run (not just
    # silently no-op'd on a cached-but-broken client).
    assert client_cls.return_value.graph_query.call_count == 4

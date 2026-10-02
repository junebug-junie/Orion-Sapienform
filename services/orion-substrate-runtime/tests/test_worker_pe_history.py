"""Prediction-error magnitude history writer in the attention-broadcast tick
(docs/superpowers/specs/2026-10-02-reverie-prediction-error-magnitude-proposal.md,
step 1): flag-gated, records only when observed_at moved, prunes, attaches
OpenLoopV1.magnitude, fails open.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker

OBSERVED = (datetime.now(timezone.utc) - timedelta(minutes=1)).replace(microsecond=0)


def _make_worker(monkeypatch, *, history_enabled: bool) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "true")
    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_MIN_SALIENCE", "0.05")
    monkeypatch.setenv("SUBSTRATE_PE_HISTORY_ENABLED", "true" if history_enabled else "false")
    monkeypatch.setenv("SUBSTRATE_PE_HISTORY_RETENTION_HOURS", "168")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._substrate_graph_store = None
    worker._store = MagicMock()
    worker._store.fetch_prediction_error_history.return_value = []
    return worker


def _pe_node(node_id: str, value: float, observed_at: datetime, pressure: float = 0.4):
    return SimpleNamespace(
        node_id=node_id,
        label="Chat prediction error",
        metadata={
            "dynamic_pressure": pressure,
            "prediction_error": value,
            "dynamic_pressure_reason": "prediction_error_seed",
        },
        signals=SimpleNamespace(confidence=0.8),
        temporal=SimpleNamespace(observed_at=observed_at),
    )


def _run_tick(worker, nodes):
    # The worker caches its graph store after the first build; reset it so
    # each tick snapshots the nodes passed here.
    worker._substrate_graph_store = None
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(nodes={n.node_id: n for n in nodes})
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()
    return worker._store.save_attention_broadcast.call_args.args[0]


def test_flag_off_touches_no_history_and_attaches_nothing(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=False)
    projection = _run_tick(worker, [_pe_node("node:substrate.chat", 0.14, OBSERVED)])
    worker._store.fetch_prediction_error_history.assert_not_called()
    worker._store.save_prediction_error_history_samples.assert_not_called()
    worker._store.prune_prediction_error_history.assert_not_called()
    assert projection.frame.open_loops
    assert all(loop.magnitude is None for loop in projection.frame.open_loops)
    dumped = projection.model_dump(mode="json")
    assert dumped["frame"]["open_loops"][0]["magnitude"] is None


def test_flag_on_records_sample_and_attaches_magnitude(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    projection = _run_tick(worker, [_pe_node("node:substrate.chat", 0.14, OBSERVED)])
    worker._store.save_prediction_error_history_samples.assert_called_once_with(
        [("node:substrate.chat", OBSERVED, 0.14)]
    )
    worker._store.prune_prediction_error_history.assert_called_once()
    loop = projection.frame.open_loops[0]
    assert loop.magnitude is not None
    assert loop.magnitude.value == 0.14
    assert loop.magnitude.n_readings_7d == 1
    assert loop.magnitude.band == "insufficient_history"


def test_stale_node_adds_no_new_rows(monkeypatch):
    """Spec acceptance check 1: unchanged observed_at -> no new history row."""
    worker = _make_worker(monkeypatch, history_enabled=True)
    node = _pe_node("node:substrate.chat", 0.14, OBSERVED)
    _run_tick(worker, [node])
    _run_tick(worker, [node])
    _run_tick(worker, [node])
    assert worker._store.save_prediction_error_history_samples.call_count == 1

    moved = _pe_node("node:substrate.chat", 0.2, OBSERVED + timedelta(seconds=30))
    projection = _run_tick(worker, [moved])
    assert worker._store.save_prediction_error_history_samples.call_count == 2
    assert worker._store.save_prediction_error_history_samples.call_args.args[0] == [
        ("node:substrate.chat", OBSERVED + timedelta(seconds=30), 0.2)
    ]
    assert projection.frame.open_loops[0].magnitude.n_readings_7d == 2


def test_node_already_in_seeded_history_is_not_rewritten(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    worker._store.fetch_prediction_error_history.return_value = [
        ("node:substrate.chat", OBSERVED - timedelta(minutes=1), 0.1),
        ("node:substrate.chat", OBSERVED, 0.14),
    ]
    projection = _run_tick(worker, [_pe_node("node:substrate.chat", 0.14, OBSERVED)])
    worker._store.save_prediction_error_history_samples.assert_not_called()
    assert projection.frame.open_loops[0].magnitude.n_readings_7d == 2


def test_week_stale_node_is_not_rewritten_every_tick(monkeypatch):
    """A node silent for >7 days falls out of the in-memory window; it must
    still read as already-recorded instead of re-writing every tick."""
    worker = _make_worker(monkeypatch, history_enabled=True)
    old = datetime.now(timezone.utc) - timedelta(days=9)
    node = _pe_node("node:substrate.harness_closure", 0.65, old)
    _run_tick(worker, [node])
    _run_tick(worker, [node])
    assert worker._store.save_prediction_error_history_samples.call_count == 1


def test_only_substrate_nodes_with_numeric_prediction_error_are_recorded(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    other = _pe_node("node:something.else", 0.5, OBSERVED)
    no_pe = SimpleNamespace(
        node_id="node:substrate.route",
        label="Route",
        metadata={"dynamic_pressure": 0.3},
        signals=SimpleNamespace(confidence=0.8),
        temporal=SimpleNamespace(observed_at=OBSERVED),
    )
    _run_tick(worker, [other, no_pe, _pe_node("node:substrate.chat", 0.0, OBSERVED)])
    samples = worker._store.save_prediction_error_history_samples.call_args.args[0]
    assert samples == [("node:substrate.chat", OBSERVED, 0.0)]


def test_write_failure_is_retried_next_tick_and_never_breaks_broadcast(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    worker._store.save_prediction_error_history_samples.side_effect = [RuntimeError("db"), 1]
    node = _pe_node("node:substrate.chat", 0.14, OBSERVED)
    _run_tick(worker, [node])
    _run_tick(worker, [node])
    assert worker._store.save_prediction_error_history_samples.call_count == 2
    assert worker._store.save_attention_broadcast.call_count == 2


def test_seed_failure_fails_open_without_magnitude(monkeypatch):
    """E.g. migration not applied: broadcast still persists, loops carry None."""
    worker = _make_worker(monkeypatch, history_enabled=True)
    worker._store.fetch_prediction_error_history.side_effect = RuntimeError(
        'relation "substrate_node_prediction_error_history" does not exist'
    )
    projection = _run_tick(worker, [_pe_node("node:substrate.chat", 0.14, OBSERVED)])
    worker._store.save_attention_broadcast.assert_called_once()
    assert all(loop.magnitude is None for loop in projection.frame.open_loops)


def test_prune_is_throttled_and_uses_retention(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    node = _pe_node("node:substrate.chat", 0.14, OBSERVED)
    before = datetime.now(timezone.utc)
    _run_tick(worker, [node])
    _run_tick(worker, [node])
    assert worker._store.prune_prediction_error_history.call_count == 1
    cutoff = worker._store.prune_prediction_error_history.call_args.kwargs["older_than"]
    assert before - timedelta(hours=168, seconds=5) <= cutoff <= datetime.now(timezone.utc) - timedelta(hours=168)

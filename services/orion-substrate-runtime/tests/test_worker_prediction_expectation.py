"""Rung 6 wiring in the substrate runtime: each execution tick records its raw
reading (the replay corpus the Hub tuner learns from) and scores surprise with
Orion's adopted expectation, read through a short cache. The kill switch
(SUBSTRATE_PREDICTION_EXPECTATION_LIVE_ENABLED=false) restores the constants."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

import app.worker as worker_module
from app.worker import BiometricsSubstrateWorker
from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.grammar import GrammarEventV1, GrammarProvenanceV1
from orion.substrate.prediction_expectation import Expectation

_NOW = datetime(2026, 10, 16, 0, 0, 0, tzinfo=timezone.utc)
_TUNED = Expectation(alpha=0.1, min_variance=1e-9)


def _worker(monkeypatch, *, live: bool) -> tuple[BiometricsSubstrateWorker, list, list]:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()  # other tick settings are irrelevant here
    worker._settings.prediction_expectation_live_enabled = live
    worker._settings.prediction_expectation_refresh_sec = 60.0
    worker._settings.prediction_raw_readings_enabled = True
    worker._settings.prediction_raw_reading_retention_hours = 192.0
    projection = ExecutionTrajectoryProjectionV1(
        projection_id="active_execution_trajectory",
        generated_at=_NOW,
        runs={
            "t": {
                "trace_id": "t",
                "correlation_id": "c",
                "node_id": "athena",
                "pressure_hints": {"reasoning_load": 0.3},
                "last_updated_at": _NOW,
            }
        },
    )
    worker._store = MagicMock()
    worker._store.load_execution_trajectory.return_value = projection
    worker._store.fetch_execution_grammar_events.return_value = [
        GrammarEventV1(
            event_id="gev-1",
            event_kind="atom_emitted",
            trace_id="t",
            emitted_at=_NOW,
            provenance=GrammarProvenanceV1(source_service="orion-cortex-exec"),
        )
    ]
    worker._get_substrate_graph_store = MagicMock(return_value=None)
    # Never a real store in tests: a stub that already reads as Postgres.
    worker._control_surface_store = SimpleNamespace(source_kind=lambda: "postgres", postgres_url="stub")
    surface_reads: list[str] = []
    passed: list = []
    monkeypatch.setattr(worker_module, "process_execution_grammar_events", lambda **kwargs: None)
    monkeypatch.setattr(
        worker_module,
        "get_prediction_expectation",
        lambda domain, store, strict=False: surface_reads.append(domain) or _TUNED,
    )

    def fake_error(prev, curr, *, expectation=None):
        passed.append(expectation)
        return 0.0

    monkeypatch.setattr(worker_module, "execution_prediction_error", fake_error)
    return worker, surface_reads, passed


def test_execution_tick_records_raw_reading_and_uses_the_adopted_expectation(monkeypatch) -> None:
    worker, surface_reads, passed = _worker(monkeypatch, live=True)

    worker._execution_tick()
    worker._execution_tick()

    assert passed == [_TUNED, _TUNED]
    assert surface_reads == ["execution"]  # second tick served from the cache
    call = worker._store.save_prediction_raw_reading.call_args_list[0]
    assert call.kwargs["domain"] == "execution"
    assert call.kwargs["raw_value"] == 0.0  # same projection both sides: a real, calm reading
    worker._store.prune_prediction_raw_readings.assert_called_once()  # hourly, first tick prunes


def test_kill_switch_scores_with_the_calibrated_constants(monkeypatch) -> None:
    worker, surface_reads, passed = _worker(monkeypatch, live=False)

    worker._execution_tick()

    assert passed == [None]
    assert surface_reads == []
    worker._store.save_prediction_raw_reading.assert_called_once()  # corpus still recorded


def test_a_failed_read_keeps_the_last_value_read_not_the_defaults(monkeypatch) -> None:
    worker, _, passed = _worker(monkeypatch, live=True)
    worker._execution_tick()
    worker._settings.prediction_expectation_refresh_sec = 0.0  # force a re-read

    def failing(domain, store, strict=False):
        raise RuntimeError("db blip")

    monkeypatch.setattr(worker_module, "get_prediction_expectation", failing)
    worker._execution_tick()

    assert passed == [_TUNED, _TUNED]

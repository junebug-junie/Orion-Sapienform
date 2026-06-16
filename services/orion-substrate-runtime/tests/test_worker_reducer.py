"""Unit tests for substrate worker reducer cursor advancement and poison isolation."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
SQL_WRITER_TESTS = REPO_ROOT / "services/orion-sql-writer/tests"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))
if str(SQL_WRITER_TESTS) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_TESTS))

from grammar_integration_helpers import bus_transport_trace_batch

from app.reducer_health import clear_health_for_tests
from app.worker import BiometricsSubstrateWorker, REDUCER_SPECS
from orion.schemas.grammar import GrammarEventV1


@pytest.fixture(autouse=True)
def _clear_health() -> None:
    clear_health_for_tests()


def test_poison_event_quarantine_advances_cursor_past_bad_event() -> None:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    worker._settings.enable_transport_bus_reducer = True
    worker._settings.reducer_poison_max_retries = 1
    worker._store = MagicMock()
    worker._store.save_receipt = MagicMock()

    trace = bus_transport_trace_batch(trace_suffix="poison01", event_count=3)
    bad = trace[1]
    spec = REDUCER_SPECS[2]

    def process_batch(batch: list[GrammarEventV1]) -> None:
        if any(e.event_id == bad.event_id for e in batch):
            raise ValueError("poison payload")

    with pytest.raises(ValueError):
        worker._process_events_with_poison_isolation(
            spec=spec,
            events=[bad],
            process_batch=process_batch,
        )

    last_id = worker._process_events_with_poison_isolation(
        spec=spec,
        events=[bad],
        process_batch=process_batch,
    )
    assert last_id == bad.event_id
    worker._store.save_receipt.assert_called()


def test_advance_cursor_records_commit_failure_when_created_at_missing() -> None:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    worker._settings.enable_execution_trajectory_reducer = True
    worker._store = MagicMock()
    worker._store.grammar_event_created_at.return_value = None

    spec = REDUCER_SPECS[1]
    worker._advance_cursor(spec, "gev_missing", worker._store.advance_execution_cursor)
    worker._store.advance_execution_cursor.assert_not_called()

    from app.reducer_health import health_snapshots

    snap = health_snapshots()["execution_trajectory"]
    assert snap.last_error_event_id == "gev_missing"

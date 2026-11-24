"""node:substrate.execution was retired 2026-11-24 (PE suite rebuild retire step 6, #3576).

_execution_tick still reduces execution grammar into the trajectory projection
(/projections/execution_trajectory and the Hub chat-turn trace read it), but it
writes no prediction-error receipt and no prediction-error node, and the worker
no longer caches an execution scalar for the world-model publish tick. Kill means
kill: this pins that the retired half cannot come back through the tick.

(Replaces the 2026-07-28 EWMA-baseline persistence tests: the baseline they
guarded belonged to the deleted execution_prediction_error.)"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker
from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.grammar import GrammarEventV1, GrammarProvenanceV1

_NOW = datetime(2026, 7, 28, 0, 0, 0, tzinfo=timezone.utc)


def _grammar_event(event_id: str) -> GrammarEventV1:
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id="trace-1",
        emitted_at=_NOW,
        provenance=GrammarProvenanceV1(source_service="orion-cortex-exec"),
    )


def test_execution_tick_reduces_but_writes_no_prediction_error(monkeypatch) -> None:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    projection = ExecutionTrajectoryProjectionV1(
        projection_id="active_execution_trajectory", generated_at=_NOW, runs={}
    )
    worker._store = MagicMock()
    worker._store.load_execution_trajectory.return_value = projection
    worker._store.fetch_execution_grammar_events.return_value = [_grammar_event("gev-1")]
    worker._write_prediction_error_node = MagicMock()
    reduced: list = []
    # Patch the globals the tick actually reads (another test can re-import `app`).
    g = BiometricsSubstrateWorker._execution_tick.__globals__
    monkeypatch.setitem(
        g, "process_execution_grammar_events", lambda **kwargs: reduced.append(kwargs)
    )

    worker._execution_tick()

    assert [e.event_id for e in reduced[0]["events"]] == ["gev-1"]
    # The reducer half stays wired: the projection the Hub trace and
    # /projections/execution_trajectory read is still loaded and saved (review 2026-11-24).
    assert reduced[0]["save_projection"] == worker._store.save_execution_trajectory
    assert reduced[0]["load_projection"]() is projection
    worker._write_prediction_error_node.assert_not_called()
    worker._store.save_receipt.assert_not_called()
    assert "execution_prediction_error" not in g
    assert not hasattr(worker, "_last_execution_prediction_error")


def test_execution_tick_with_no_events_does_nothing() -> None:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    worker._store = MagicMock()
    worker._store.fetch_execution_grammar_events.return_value = []
    assert worker._execution_tick() is None
    worker._store.save_receipt.assert_not_called()

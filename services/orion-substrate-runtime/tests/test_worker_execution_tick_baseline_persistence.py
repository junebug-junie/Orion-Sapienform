"""Regression coverage for the 2026-07-28 execution_prediction_error EWMA-baseline
fix: _execution_tick must persist the projection's mutated baseline fields on
every tick, not only when error > 0.0, or the baseline never survives past this
process's lifetime and every tick after a restart re-cold-starts at n=0."""

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

import app.worker as worker_module
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


def test_execution_tick_persists_baseline_even_when_error_is_zero(monkeypatch) -> None:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()

    projection = ExecutionTrajectoryProjectionV1(
        projection_id="active_execution_trajectory",
        generated_at=_NOW,
        runs={},
    )
    worker._store = MagicMock()
    worker._store.load_execution_trajectory.return_value = projection
    worker._store.fetch_execution_grammar_events.return_value = [_grammar_event("gev-1")]

    # Stub out the real grammar-reduction pipeline (irrelevant to this test --
    # only the tick's own save-orchestration logic is under test here) and force
    # execution_prediction_error's return to 0.0, the exact condition that used
    # to skip the receipt-writing block entirely.
    monkeypatch.setattr(worker_module, "process_execution_grammar_events", lambda **kwargs: None)
    monkeypatch.setattr(worker_module, "execution_prediction_error", lambda prev, curr: 0.0)

    worker._execution_tick()

    worker._store.save_execution_trajectory.assert_called_once_with(projection)
    worker._store.save_receipt.assert_not_called()


def test_execution_tick_still_writes_receipt_when_error_is_nonzero(monkeypatch) -> None:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()

    projection = ExecutionTrajectoryProjectionV1(
        projection_id="active_execution_trajectory",
        generated_at=_NOW,
        runs={},
    )
    worker._store = MagicMock()
    worker._store.load_execution_trajectory.return_value = projection
    worker._store.fetch_execution_grammar_events.return_value = [_grammar_event("gev-1")]
    worker._get_substrate_graph_store = MagicMock(return_value=None)

    monkeypatch.setattr(worker_module, "process_execution_grammar_events", lambda **kwargs: None)
    monkeypatch.setattr(worker_module, "execution_prediction_error", lambda prev, curr: 0.42)

    worker._execution_tick()

    worker._store.save_execution_trajectory.assert_called_once_with(projection)
    worker._store.save_receipt.assert_called_once()

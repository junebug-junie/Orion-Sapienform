"""Regression coverage for the 2026-08-19 chat_prediction_error EWMA-baseline
persistence fix: _chat_tick must persist the projection's mutated baseline
fields on every tick, not only when error > 0.0, or the baseline never
survives past this process's lifetime and every tick after a restart
re-cold-starts at n=0 -- reproducing this same patch's own bug (chat
permanently reading near-zero via predicted_shift's argmax) via a different
mechanism. Found in code review of the sibling EWMA fix; mirrors
test_worker_execution_tick_baseline_persistence.py's identical shape for
execution_prediction_error's own 2026-07-28 fix."""

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
from orion.schemas.chat_projection import ChatSessionProjectionV1
from orion.schemas.grammar import GrammarEventV1, GrammarProvenanceV1

_NOW = datetime(2026, 8, 19, 0, 0, 0, tzinfo=timezone.utc)


def _grammar_event(event_id: str) -> GrammarEventV1:
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id="trace-1",
        emitted_at=_NOW,
        provenance=GrammarProvenanceV1(source_service="orion-hub"),
    )


def _make_worker() -> tuple[BiometricsSubstrateWorker, ChatSessionProjectionV1]:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    projection = ChatSessionProjectionV1(
        projection_id="chat_session_projection",
        generated_at=_NOW,
        turns={},
    )
    worker._store = MagicMock()
    worker._store.load_chat_session_projection.return_value = projection
    worker._store.fetch_chat_grammar_events.return_value = [_grammar_event("gev-1")]
    worker._write_prediction_error_node = MagicMock()
    return worker, projection


def test_chat_tick_persists_baseline_even_when_error_is_zero(monkeypatch) -> None:
    worker, projection = _make_worker()

    monkeypatch.setattr(worker_module, "process_chat_grammar_events", lambda **kwargs: None)
    monkeypatch.setattr(worker_module, "chat_prediction_error", lambda prev, curr: 0.0)

    worker._chat_tick()

    worker._store.save_chat_session_projection.assert_called_once_with(projection)
    worker._store.save_receipt.assert_not_called()


def test_chat_tick_still_writes_receipt_when_error_is_nonzero(monkeypatch) -> None:
    worker, projection = _make_worker()

    monkeypatch.setattr(worker_module, "process_chat_grammar_events", lambda **kwargs: None)
    monkeypatch.setattr(worker_module, "chat_prediction_error", lambda prev, curr: 0.42)

    worker._chat_tick()

    worker._store.save_chat_session_projection.assert_called_once_with(projection)
    worker._store.save_receipt.assert_called_once()

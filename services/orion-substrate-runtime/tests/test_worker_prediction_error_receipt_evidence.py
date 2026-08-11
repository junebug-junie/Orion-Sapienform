"""Unit tests for `_prediction_error_receipt()`'s `caused_by_event_ids` evidence
threading (2026-07-31, PR #1547 proposal).

Before this patch, every prediction-error receipt hardcoded
`caused_by_event_ids=[]` -- the sole holdout among this repo's reducer family,
all of which already thread real event_ids into `StateDeltaV1.caused_by_event_ids`
(orion/substrate/chat_loop/reducer.py, execution_loop/reducer.py,
route_loop/reducer.py, transport_loop/reducer.py). That left the receipts
backing `prediction_error_confidence`/`prediction_error_by_domain`
(orion/substrate/attention_self_model.py) -- the composite "why do I feel this
way" signal -- with no evidence trail, even though
services/orion-hub/scripts/substrate_biometrics_routes.py's
`/biometrics-node/{node_id}/latest` route and
orion/substrate/receipts/retention.py::primary_event_id() both already read
this field on every other receipt.
"""

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
from app.worker import (
    REDUCER_SPECS,
    BiometricsSubstrateWorker,
    _PREDICTION_ERROR_EVIDENCE_CAP,
    _prediction_error_evidence_event_ids,
    _prediction_error_receipt,
)
from orion.schemas.grammar import GrammarEventV1

NOW = datetime(2026, 7, 31, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clear_health() -> None:
    clear_health_for_tests()


def test_default_stays_empty_matching_prior_behavior() -> None:
    """bus_synaptic/codebase call sites don't pass caused_by_event_ids -- must
    still produce [] exactly as every call site did before this patch."""
    receipt = _prediction_error_receipt(
        reducer_key="bus_synaptic",
        node_id="node:substrate.bus_synaptic",
        prediction_error=0.5,
        now=NOW,
    )
    assert receipt.state_deltas[0].caused_by_event_ids == []


def test_real_event_ids_thread_through() -> None:
    receipt = _prediction_error_receipt(
        reducer_key="node_biometrics",
        node_id="node:substrate.biometrics",
        prediction_error=0.3,
        now=NOW,
        caused_by_event_ids=["evt-1", "evt-2", "evt-3"],
    )
    assert receipt.state_deltas[0].caused_by_event_ids == ["evt-1", "evt-2", "evt-3"]


def test_evidence_capped_not_unbounded() -> None:
    """Same discipline as chat_loop/reducer.py's _EVIDENCE_CAP -- a raised
    batch-limit env var must not let this list grow unbounded."""
    many_ids = [f"evt-{i}" for i in range(_PREDICTION_ERROR_EVIDENCE_CAP + 25)]
    receipt = _prediction_error_receipt(
        reducer_key="execution_trajectory",
        node_id="node:substrate.execution",
        prediction_error=0.7,
        now=NOW,
        caused_by_event_ids=many_ids,
    )
    got = receipt.state_deltas[0].caused_by_event_ids
    assert len(got) == _PREDICTION_ERROR_EVIDENCE_CAP
    assert got == many_ids[:_PREDICTION_ERROR_EVIDENCE_CAP]


def test_accepts_any_sequence_not_just_list() -> None:
    """Call sites pass a list comprehension, but the param type is Sequence[str]
    -- confirm a tuple/generator-derived sequence works too, since worker.py's
    own call sites build the list inline (`[e.event_id for e in events if
    e.atom]`) and nothing should force that shape."""
    receipt = _prediction_error_receipt(
        reducer_key="chat_session",
        node_id="node:substrate.chat",
        prediction_error=0.1,
        now=NOW,
        caused_by_event_ids=("evt-a", "evt-b"),
    )
    assert receipt.state_deltas[0].caused_by_event_ids == ["evt-a", "evt-b"]


class TestPredictionErrorEvidenceEventIds:
    """`_prediction_error_evidence_event_ids()` (review finding, 2026-07-31):
    must only name atom-backed events that actually succeeded, never an
    event this same tick quarantined as poison -- a quarantined event is
    already recorded elsewhere with `rejected_event_ids=[...]`, so citing it
    as `caused_by_event_ids` evidence too would contradict its own
    quarantine receipt."""

    def _events(self) -> list[GrammarEventV1]:
        # event_count=4 -> [trace_started (no atom), atom event, atom event,
        # trace_ended (no atom)] -- same fixture test_worker_reducer.py uses.
        return bus_transport_trace_batch(trace_suffix="evidence01", event_count=4)

    def test_excludes_atomless_events(self) -> None:
        events = self._events()
        ids = _prediction_error_evidence_event_ids(events)
        assert ids == [e.event_id for e in events if e.atom]
        assert len(ids) == 2

    def test_excludes_quarantined_event_even_if_atom_backed(self) -> None:
        events = self._events()
        atom_backed = [e for e in events if e.atom]
        quarantined_id = atom_backed[0].event_id
        ids = _prediction_error_evidence_event_ids(
            events, quarantined_event_ids=[quarantined_id]
        )
        assert quarantined_id not in ids
        assert ids == [atom_backed[1].event_id]

    def test_no_quarantine_arg_behaves_as_before(self) -> None:
        events = self._events()
        assert _prediction_error_evidence_event_ids(events) == [
            e.event_id for e in events if e.atom
        ]


def test_poison_isolation_quarantined_out_excludes_bad_event_from_evidence() -> None:
    """Reproduces the review-flagged bug end-to-end: a >1-event batch where
    one event is poison must not have that event's id land in
    `caused_by_event_ids`, even though it was part of the originally-fetched
    `events` batch and did briefly hold `last_id`'s per-event retry cursor."""
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    worker._settings.enable_transport_bus_reducer = True
    worker._settings.reducer_poison_max_retries = 1
    worker._store = MagicMock()
    worker._store.save_receipt = MagicMock()
    worker._store.save_quarantine = MagicMock()

    events = bus_transport_trace_batch(trace_suffix="evidence02", event_count=4)
    atom_backed = [e for e in events if e.atom]
    bad = atom_backed[0]
    spec = REDUCER_SPECS[2]

    def process_batch(batch: list[GrammarEventV1]) -> None:
        if any(e.event_id == bad.event_id for e in batch):
            raise ValueError("poison payload")

    # First pass over the batch: the whole-batch call fails (bad is inside
    # it), falls into the per-event retry loop, and reaches `bad` with no
    # recorded failures yet -- _should_quarantine_poison(spec, bad.event_id)
    # is False on this first hit (reducer_poison_max_retries=1, 0 prior
    # failures recorded), so it records an error and re-raises instead of
    # quarantining. Same two-call shape test_worker_reducer.py's own
    # poison-quarantine test uses.
    quarantined: list[str] = []
    with pytest.raises(ValueError):
        worker._process_events_with_poison_isolation(
            spec=spec,
            events=events,
            process_batch=process_batch,
            quarantined_out=quarantined,
        )
    assert quarantined == []

    # Second pass: `bad`'s recorded failure count now meets
    # reducer_poison_max_retries, so this time it actually quarantines
    # instead of re-raising, and the other 3 events in the batch (including
    # the second atom-backed one) still process successfully around it.
    last_id = worker._process_events_with_poison_isolation(
        spec=spec,
        events=events,
        process_batch=process_batch,
        quarantined_out=quarantined,
    )

    assert bad.event_id in quarantined
    assert last_id is not None

    evidence = _prediction_error_evidence_event_ids(
        events, quarantined_event_ids=quarantined
    )
    assert bad.event_id not in evidence
    assert atom_backed[1].event_id in evidence

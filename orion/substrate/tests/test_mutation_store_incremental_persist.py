"""Tests for the incremental persist paths added to `record_pressure`/`add_proposal`.

Context: `SubstrateMutationStore._persist()` (the pre-existing full path) rewrites
every row in all ~19 store-backed tables on every call. `record_pressure` was
calling it once per signal in a live mutation cycle -- 12+ times/cycle -- to
change exactly one row in one table (the pressures table had 1 real row while
the signals table it dragged along had 17,805). That was the entire measured
64-125s mutation-cycle cost as of 2026-09-03.

None of the existing tests in this module reload from a *second* store
instance pointed at the same database -- they all check the same live
instance's in-memory state right after a write. That means a broken
incremental rewrite would not have been caught by anything already here, so
every test below either does a real two-instance reload, or asserts the exact
fallback-to-full-persist control flow directly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from orion.core.schemas.substrate_mutation import (
    MutationPatchV1,
    MutationPressureV1,
    MutationProposalV1,
    MutationSignalV1,
)
from orion.substrate import mutation_queue as mutation_queue_module
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_proposals import ProposalFactory


def _pressure(*, target_surface: str = "routing", pressure_score: float = 8.0) -> MutationPressureV1:
    return MutationPressureV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface=target_surface,
        pressure_kind="runtime_failure",
        pressure_score=pressure_score,
        evidence_refs=["telemetry:1"],
        source_signal_ids=["signal-1"],
    )


def _routing_surface(value: float = 0.50):
    return lambda: {"value": value, "raw": {"value": value}, "degraded": False}


def _proposal(*, proposal_id: str | None = None) -> MutationProposalV1:
    made = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(_pressure())
    assert made is not None
    if proposal_id is not None:
        made = made.model_copy(update={"proposal_id": proposal_id})
    return made


def _fake_signal(i: int) -> MutationSignalV1:
    return MutationSignalV1(
        event_kind="runtime_failure",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="routing",
        evidence_refs=[f"telemetry:{i}"],
    )


# ---------------------------------------------------------------------------
# Real reload round trips (two separate SubstrateMutationStore instances)
# ---------------------------------------------------------------------------


def test_record_pressure_round_trips_through_a_fresh_store_reload(tmp_path: Path) -> None:
    db = tmp_path / "mutation.sqlite3"
    store = SubstrateMutationStore(sql_db_path=str(db))
    pressure = _pressure()
    store.record_pressure(pressure)

    reloaded = SubstrateMutationStore(sql_db_path=str(db))
    key = store._pressure_key(pressure)
    assert key in reloaded._pressures
    assert reloaded._pressures[key].model_dump(mode="json") == pressure.model_dump(mode="json")


def test_add_proposal_new_item_round_trips_through_a_fresh_store_reload(tmp_path: Path) -> None:
    db = tmp_path / "mutation.sqlite3"
    store = SubstrateMutationStore(sql_db_path=str(db))
    proposal = _proposal()
    queue_item = store.add_proposal(proposal)

    reloaded = SubstrateMutationStore(sql_db_path=str(db))
    assert proposal.proposal_id in reloaded._proposals
    assert reloaded._proposals[proposal.proposal_id].model_dump(mode="json") == proposal.model_dump(mode="json")
    assert queue_item.queue_item_id in reloaded._queue
    assert reloaded._queue[queue_item.queue_item_id].proposal_id == proposal.proposal_id


def test_add_proposal_update_of_an_existing_item_round_trips(tmp_path: Path) -> None:
    """Hits the *other* branch of add_proposal -- an existing queue item for
    the same proposal_id, which only touches the proposals table, not queue."""
    db = tmp_path / "mutation.sqlite3"
    store = SubstrateMutationStore(sql_db_path=str(db))
    proposal = _proposal()
    queue_item = store.add_proposal(proposal)

    updated = proposal.model_copy(update={"rollout_state": "trialed", "notes": ["updated"]})
    same_queue_item = store.add_proposal(updated)
    assert same_queue_item.queue_item_id == queue_item.queue_item_id

    reloaded = SubstrateMutationStore(sql_db_path=str(db))
    assert reloaded._proposals[proposal.proposal_id].rollout_state == "trialed"
    assert reloaded._proposals[proposal.proposal_id].notes == ["updated"]


# ---------------------------------------------------------------------------
# Fallback contract: matches record_signal's existing shape exactly --
# incremental failure must still reach Postgres/sqlite via the full sweep,
# incremental success must NOT trigger a redundant full sweep.
# ---------------------------------------------------------------------------


def test_record_pressure_falls_back_to_full_persist_when_incremental_fails(monkeypatch) -> None:
    store = SubstrateMutationStore()
    calls = {"full_persist": 0}
    monkeypatch.setattr(store, "_persist_pressure", lambda pressure: False)
    monkeypatch.setattr(store, "_persist", lambda: calls.__setitem__("full_persist", calls["full_persist"] + 1))

    store.record_pressure(_pressure())

    assert calls["full_persist"] == 1


def test_record_pressure_clears_a_stale_degraded_flag_on_a_successful_postgres_write(monkeypatch) -> None:
    """Review finding: the fast path previously never cleared source_kind/
    last_error on success, so a store that had drifted to 'fallback' from a
    past outage would report degraded() forever once the database recovered,
    since nothing but a full _persist() sweep ever un-set it."""
    store = SubstrateMutationStore()
    store.postgres_url = "postgresql://example-not-real/db"
    store._source_kind = "fallback"
    store._last_error = "connection refused"
    monkeypatch.setattr(store, "_persist_pressure_postgres", lambda pressure: None)

    store.record_pressure(_pressure())

    assert store.source_kind() == "postgres"
    assert store.last_error() is None
    assert not store.degraded()


def test_record_pressure_skips_full_persist_when_incremental_succeeds(monkeypatch) -> None:
    store = SubstrateMutationStore()
    calls = {"full_persist": 0}
    monkeypatch.setattr(store, "_persist_pressure", lambda pressure: True)
    monkeypatch.setattr(store, "_persist", lambda: calls.__setitem__("full_persist", calls["full_persist"] + 1))

    store.record_pressure(_pressure())

    assert calls["full_persist"] == 0


def test_add_proposal_new_item_falls_back_to_full_persist_when_incremental_fails(monkeypatch) -> None:
    store = SubstrateMutationStore()
    calls = {"full_persist": 0}
    monkeypatch.setattr(store, "_persist_proposal_and_queue_item", lambda proposal, queue_item: False)
    monkeypatch.setattr(store, "_persist", lambda: calls.__setitem__("full_persist", calls["full_persist"] + 1))

    store.add_proposal(_proposal())

    assert calls["full_persist"] == 1


def test_add_proposal_existing_item_falls_back_to_full_persist_when_incremental_fails(monkeypatch) -> None:
    store = SubstrateMutationStore()
    proposal = _proposal()
    store.add_proposal(proposal)  # creates the queue item via the real path
    calls = {"full_persist": 0}
    monkeypatch.setattr(store, "_persist_proposal", lambda proposal: False)
    monkeypatch.setattr(store, "_persist", lambda: calls.__setitem__("full_persist", calls["full_persist"] + 1))

    store.add_proposal(proposal.model_copy(update={"rollout_state": "trialed"}))

    assert calls["full_persist"] == 1


# ---------------------------------------------------------------------------
# The actual scaling claim: record_pressure's SQL cost must NOT grow with how
# much history the store has already accumulated. This is the assertion that
# would have failed against the pre-fix code (which always called the full,
# every-table rewrite) and is written to prove that, not just assert the new
# behavior in isolation.
# ---------------------------------------------------------------------------


class _StatementCounter:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def __call__(self, sql: str) -> None:
        self.statements.append(sql)


def _counting_sqlite_connect(monkeypatch, counter: _StatementCounter) -> None:
    real_connect = sqlite3.connect

    def wrapped(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(counter)
        return conn

    monkeypatch.setattr(mutation_queue_module.sqlite3, "connect", wrapped)


def test_record_pressure_sql_cost_does_not_scale_with_store_size(tmp_path, monkeypatch) -> None:
    db = tmp_path / "mutation.sqlite3"
    store = SubstrateMutationStore(sql_db_path=str(db))

    # Simulate an already-busy store: 300 accumulated signals plus a few dozen
    # proposals/queue rows, all in memory only (mirrors what a live store
    # looks like after real use, without paying for 300 real writes here).
    store._signals = [_fake_signal(i) for i in range(300)]
    for i in range(20):
        proposal = _proposal(proposal_id=f"prop-{i}")
        store._proposals[proposal.proposal_id] = proposal

    counter = _StatementCounter()
    _counting_sqlite_connect(monkeypatch, counter)

    store.record_pressure(_pressure())

    # One row written (INSERT ... ON CONFLICT), no per-signal or per-proposal
    # fan-out. A generous bound (5) rather than an exact count so this isn't
    # brittle against an incidental extra PRAGMA/BEGIN, while still being far
    # below "one statement per accumulated row" (320+).
    assert len(counter.statements) <= 5, counter.statements

    # Prove the comparison is real: the pre-fix path (still reachable as
    # `_persist()`, used for the multi-table mutators) really does scale with
    # everything accumulated above -- this is what record_pressure used to
    # call on every invocation.
    counter.statements.clear()
    store._persist()
    assert len(counter.statements) >= 300, (
        "expected the full persist sweep to touch every accumulated signal; "
        f"got {len(counter.statements)} statements -- if this shrank, the "
        "scaling claim this test exists to prove may no longer hold"
    )

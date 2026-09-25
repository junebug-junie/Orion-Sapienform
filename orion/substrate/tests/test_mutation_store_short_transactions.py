"""Regression tests for the 2026-09-25 long-transaction incident.

Live: the substrate mutation worker held Postgres transactions open for 7+
minutes. (1) `_acquire_leader_lock` left its advisory-lock connection "idle in
transaction" for the whole cycle (SQLAlchemy 2 autobegin). (2)
`_persist_to_postgres` re-UPSERTed every in-memory row (550k+ signals) inside
ONE transaction, one statement at a time, on every `_persist()` call. Any
`CREATE INDEX CONCURRENTLY` in the database waited on them -- orion-gpu-pool
hung ~10 minutes at boot (LangGraph saver.setup()), a full LLM outage.

Every test here fails against the pre-fix code:
  * persist-after-one-change wrote every row (not 1),
  * a large backlog went out in one transaction (not <= 500-row batches),
  * a fresh engine was created per persist call (not cached),
  * a reloaded store's first persist rewrote everything it had just read,
  * the leader-lock connection had a transaction open after acquire.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy

from orion.core.schemas.substrate_mutation import (
    MutationPatchV1,
    MutationProposalV1,
    MutationQueueItemV1,
    MutationSignalV1,
)
from orion.substrate import mutation_queue as mutation_queue_module
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_worker import build_default_worker

ACTIVE_SURFACE = "substrate_mutation_active_surface"


def _signal(i: int) -> MutationSignalV1:
    return MutationSignalV1(
        event_kind="runtime_failure",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="recall",
        evidence_refs=[f"telemetry:{i}"],
    )


def _proposal(i: int) -> MutationProposalV1:
    return MutationProposalV1(
        proposal_id=f"prop-{i}",
        mutation_class="recall_weighting_patch",
        risk_tier="low",
        target_surface="recall",
        anchor_scope="orion",
        subject_ref="entity:orion",
        evidence_refs=["telemetry:1"],
        source_signal_ids=["signal-1"],
        source_pressure_id="pressure-1",
        patch=MutationPatchV1(
            mutation_class="recall_weighting_patch",
            target_surface="recall",
            target_ref="recall_weights",
            patch={"w": i},
            rollback_payload={"w": 0},
        ),
    )


def _populate(store: SubstrateMutationStore, *, signals: int = 300, proposals: int = 20) -> None:
    store._signals.extend(_signal(i) for i in range(signals))
    for i in range(proposals):
        proposal = _proposal(i)
        store._proposals[proposal.proposal_id] = proposal
        item = MutationQueueItemV1(
            queue_item_id=f"queue-{i}",
            proposal_id=proposal.proposal_id,
            mutation_class=proposal.mutation_class,
            target_surface=proposal.target_surface,
        )
        store._queue[item.queue_item_id] = item


# ---------------------------------------------------------------------------
# A fake Postgres: records every transaction, and holds upserted rows so a
# second store can load them back (real round trip through the pg code path).
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None


class _FakePostgres:
    def __init__(self) -> None:
        self.tables: dict[str, dict[str, tuple[Any, Any]]] = {}
        #: One entry per committed transaction: [(sql, row_count), ...]
        self.transactions: list[list[tuple[str, int]]] = []
        self.engines_created = 0
        self.jsonb_reorder = False

    # sqlalchemy.create_engine replacement
    def create_engine(self, *_args: Any, **_kwargs: Any) -> "_FakePostgres":
        self.engines_created += 1
        return self

    def dispose(self) -> None:
        pass

    @contextmanager
    def begin(self):
        txn: list[tuple[str, int]] = []
        yield _FakeConn(self, txn)
        self.transactions.append(txn)

    def data_rows_written(self, since: int = 0) -> dict[str, int]:
        out: dict[str, int] = {}
        for txn in self.transactions[since:]:
            for sql, n in txn:
                m = re.match(r"\s*INSERT INTO (\w+)", sql)
                if m and m.group(1) != ACTIVE_SURFACE:
                    out[m.group(1)] = out.get(m.group(1), 0) + n
        return out


class _FakeConn:
    def __init__(self, db: _FakePostgres, txn: list[tuple[str, int]]) -> None:
        self.db = db
        self.txn = txn

    def execute(self, stmt: Any, params: Any = None) -> _FakeResult:
        sql = str(stmt)
        rows = params if isinstance(params, list) else ([params] if params else [])
        self.txn.append((sql, len(rows)))
        head = sql.strip().upper()
        if head.startswith("CREATE"):
            return _FakeResult([])
        if head.startswith("DELETE"):
            table = re.search(r"FROM (\w+)", sql).group(1)
            self.db.tables[table] = {}
            return _FakeResult([])
        if head.startswith("INSERT"):
            table = re.match(r"\s*INSERT INTO (\w+)", sql).group(1)
            dest = self.db.tables.setdefault(table, {})
            for r in rows:
                if table == ACTIVE_SURFACE:
                    dest[r["surface"]] = (r["updated_at"], r["adoption_id"])
                else:
                    # Accept the pre-fix param names too (:created_at /
                    # :updated_at / ...), so these tests fail on old code for
                    # the reason they claim, not on a fake-shape mismatch.
                    ts = next(r[k] for k in ("ts", "created_at", "updated_at", "detected_at", "completed_at") if k in r)
                    dest[r["id"]] = (ts, r["payload"])
            return _FakeResult([])
        if head.startswith("SELECT"):
            table = re.search(r"FROM (\w+)", sql).group(1)
            stored = self.db.tables.get(table, {})
            ordered = sorted(stored.items(), key=lambda kv: str(kv[1][0]))
            if self.db.jsonb_reorder:
                # Real JSONB `::text` output: its own key order and spacing,
                # not the sorted compact string that was written.
                ordered = [
                    (k, (ts, json.dumps(dict(reversed(list(json.loads(v).items()))), indent=None, separators=(", ", ": "))))
                    if isinstance(v, str) and v.startswith("{") else (k, (ts, v))
                    for k, (ts, v) in ordered
                ]
            if "payload_json::text FROM" in sql and not re.search(r"SELECT \w+, payload_json", sql):
                return _FakeResult([(value,) for _row_id, (_ts, value) in ordered])  # pre-fix query shape
            return _FakeResult([(row_id, value) for row_id, (_ts, value) in ordered])
        raise AssertionError(f"unexpected SQL in fake: {sql}")


@pytest.fixture()
def fake_pg(monkeypatch) -> _FakePostgres:
    db = _FakePostgres()
    monkeypatch.setattr(sqlalchemy, "create_engine", db.create_engine)
    return db


def _pg_store() -> SubstrateMutationStore:
    store = SubstrateMutationStore(postgres_url="postgresql://fake/db")
    assert store.source_kind() == "postgres", store.last_error()
    return store


def _persist_ok(store: SubstrateMutationStore) -> None:
    """_persist() swallows a failed Postgres write into a "fallback" state;
    make sure every write in these tests really went through the pg path."""
    store._persist()
    assert store.source_kind() == "postgres" and store.last_error() is None, store.last_error()


# ---------------------------------------------------------------------------
# (a) persist writes only what changed, in short batches, on a cached engine
# ---------------------------------------------------------------------------


def test_postgres_persist_after_one_change_writes_only_that_row(fake_pg) -> None:
    store = _pg_store()
    assert store.source_kind() == "postgres"
    _populate(store)
    _persist_ok(store)
    assert fake_pg.data_rows_written()["substrate_mutation_signal"] == 300

    before = len(fake_pg.transactions)
    proposal = store._proposals["prop-3"]
    store._proposals["prop-3"] = proposal.model_copy(update={"rollout_state": "trialed"})
    _persist_ok(store)

    assert fake_pg.data_rows_written(since=before) == {"substrate_mutation_proposal": 1}
    # One short transaction: the changed row plus the (tiny, always-rewritten)
    # active-surface lock table.
    assert len(fake_pg.transactions) - before == 1


def test_postgres_persist_detects_in_place_mutation(fake_pg) -> None:
    store = _pg_store()
    _populate(store)
    _persist_ok(store)
    before = len(fake_pg.transactions)
    store._queue["queue-7"].status = "trialed"  # mutated in place, not model_copy
    _persist_ok(store)
    assert fake_pg.data_rows_written(since=before) == {"substrate_mutation_queue": 1}


def test_postgres_noop_persist_writes_no_data_rows(fake_pg) -> None:
    store = _pg_store()
    _populate(store)
    _persist_ok(store)
    before = len(fake_pg.transactions)
    _persist_ok(store)
    _persist_ok(store)
    assert fake_pg.data_rows_written(since=before) == {}


def test_postgres_backlog_is_split_into_bounded_transactions(fake_pg) -> None:
    store = _pg_store()
    _populate(store, signals=1200, proposals=0)
    _persist_ok(store)
    sizes = [sum(n for _sql, n in txn) for txn in fake_pg.transactions if any("INSERT" in sql for sql, _ in txn)]
    assert sum(sizes) >= 1200
    assert max(sizes) <= 501  # 500 rows + the active-surface DELETE
    assert mutation_queue_module._PERSIST_BATCH_ROWS == 500
    assert len(sizes) >= 3


def test_postgres_engine_is_created_once_per_store(fake_pg) -> None:
    store = _pg_store()
    _populate(store, signals=5, proposals=2)
    for _ in range(5):
        _persist_ok(store)
    store.record_signal(_signal(999))
    assert fake_pg.engines_created == 1


def test_incremental_signal_write_advances_the_mark(fake_pg) -> None:
    store = _pg_store()
    _populate(store, signals=10, proposals=0)
    _persist_ok(store)
    store.record_signal(_signal(1000))  # written immediately via the single-row path
    before = len(fake_pg.transactions)
    _persist_ok(store)
    assert fake_pg.data_rows_written(since=before) == {}


# ---------------------------------------------------------------------------
# Round trips: tables end up identical; a reload is not followed by a rewrite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("jsonb_reorder", [False, True])
def test_postgres_round_trip_and_reload_does_not_rewrite(fake_pg, jsonb_reorder) -> None:
    fake_pg.jsonb_reorder = jsonb_reorder
    store = _pg_store()
    _populate(store, signals=50, proposals=5)
    _persist_ok(store)
    store._proposals["prop-1"] = store._proposals["prop-1"].model_copy(update={"rollout_state": "rejected"})
    _persist_ok(store)

    reloaded = _pg_store()
    assert [s.signal_id for s in reloaded._signals] == [s.signal_id for s in store._signals]
    assert reloaded._proposals == store._proposals
    assert reloaded._queue == store._queue

    before = len(fake_pg.transactions)
    _persist_ok(reloaded)
    assert fake_pg.data_rows_written(since=before) == {}

    reloaded.record_signal(_signal(5000))
    reloaded._queue["queue-2"] = reloaded._queue["queue-2"].model_copy(update={"status": "applied"})
    before = len(fake_pg.transactions)
    _persist_ok(reloaded)
    assert fake_pg.data_rows_written(since=before) == {"substrate_mutation_queue": 1}


def _sqlite_row_counter(monkeypatch) -> list[str]:
    statements: list[str] = []
    real_connect = sqlite3.connect

    def wrapped(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(mutation_queue_module.sqlite3, "connect", wrapped)
    return statements


def _data_inserts(statements: list[str]) -> list[str]:
    return [s for s in statements if s.lstrip().upper().startswith("INSERT") and ACTIVE_SURFACE not in s]


def test_sqlite_round_trip_and_dirty_only(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "mutation.sqlite3"
    store = SubstrateMutationStore(sql_db_path=str(db))
    _populate(store, signals=40, proposals=6)
    store._persist()

    statements = _sqlite_row_counter(monkeypatch)
    store._queue["queue-4"].status = "rejected"
    store._persist()
    inserts = _data_inserts(statements)
    assert len(inserts) == 1 and "substrate_mutation_queue" in inserts[0], inserts

    statements.clear()
    reloaded = SubstrateMutationStore(sql_db_path=str(db))
    assert reloaded._queue["queue-4"].status == "rejected"
    assert reloaded._proposals == store._proposals
    assert [s.signal_id for s in reloaded._signals] == [s.signal_id for s in store._signals]
    statements.clear()
    reloaded._persist()
    assert _data_inserts(statements) == []


# ---------------------------------------------------------------------------
# (b) the leader-lock connection holds no open transaction
# ---------------------------------------------------------------------------


def test_leader_lock_connection_is_autocommit(monkeypatch) -> None:
    real_create_engine = sqlalchemy.create_engine
    held: dict[str, int] = {}

    def fake_create_engine(_url: str, **kwargs: Any):
        engine = real_create_engine("sqlite://", **kwargs)

        @sqlalchemy.event.listens_for(engine, "connect")
        def _register(dbapi_conn, _record):  # emulate Postgres' session advisory lock
            dbapi_conn.create_function("pg_try_advisory_lock", 1, lambda k: 0 if held.get("k") else held.setdefault("k", k) and 1)
            dbapi_conn.create_function("pg_advisory_unlock", 1, lambda k: 1 if held.pop("k", None) else 0)

        return engine

    store = SubstrateMutationStore()
    store.postgres_url = "postgresql://fake/db"
    worker = build_default_worker(store=store)
    monkeypatch.setattr(sqlalchemy, "create_engine", fake_create_engine)

    ctx = worker._acquire_leader_lock()
    try:
        assert ctx["acquired"] is True
        conn = ctx["conn"]
        # The incident: autobegin left this connection "idle in transaction"
        # for the whole mutation cycle. SQLAlchemy's own in_transaction() is
        # True either way (it tracks a Python-side marker even under
        # AUTOCOMMIT), so check what the driver was told: under AUTOCOMMIT
        # the DBAPI connection never issues BEGIN (psycopg2 `autocommit=True`;
        # pysqlite `isolation_level=None`).
        assert conn.get_execution_options().get("isolation_level") == "AUTOCOMMIT"
        assert conn.connection.dbapi_connection.isolation_level is None
        assert conn.connection.dbapi_connection.in_transaction is False
    finally:
        worker._release_leader_lock(ctx)
    assert "k" not in held  # unlocked on release


def test_concurrent_saves_are_serialized(fake_pg) -> None:
    """Review finding: two overlapping saves could commit an older row last yet
    record the newer digest, leaving the row stale forever. Saves now hold a
    store-level io lock from snapshot to record."""
    import threading

    store = _pg_store()
    _populate(store, signals=5, proposals=3)
    _persist_ok(store)
    entered = threading.Event()
    release = threading.Event()
    real = store._persist_locked

    def slow_persist() -> None:
        entered.set()
        release.wait(5)
        real()

    store._persist_locked = slow_persist  # type: ignore[method-assign]
    t = threading.Thread(target=store._persist)
    t.start()
    assert entered.wait(5)
    second_done = threading.Event()
    t2 = threading.Thread(target=lambda: (store._persist_proposal(store._proposals["prop-1"]), second_done.set()))
    t2.start()
    assert not second_done.wait(0.3), "single-row save ran while a full save was mid-flight"
    release.set()
    t.join(5)
    t2.join(5)
    assert second_done.is_set()


def test_postgres_url_change_resets_known_state(fake_pg) -> None:
    store = _pg_store()
    _populate(store, signals=10, proposals=2)
    _persist_ok(store)
    store.postgres_url = "postgresql://fake/other"
    before = len(fake_pg.transactions)
    _persist_ok(store)
    assert fake_pg.data_rows_written(since=before)["substrate_mutation_signal"] == 10

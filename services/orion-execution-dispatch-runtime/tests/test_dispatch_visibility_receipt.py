from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from app.store import ExecutionDispatchRuntimeStore
from app.worker import ExecutionDispatchRuntimeWorker
from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1


STAMP = datetime(2026, 11, 3, 1, 0, 1, tzinfo=timezone.utc)


class _Result:
    def __init__(self, scalar=None):
        self.scalar = scalar
        self.rowcount = 1

    def scalar_one(self):
        return self.scalar


class _Connection:
    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, params=None):
        sql = " ".join(str(statement).split())
        self.engine.calls.append((sql, params))
        return _Result(STAMP if "RETURNING generated_at" in sql else None)


class _Engine:
    def __init__(self):
        self.calls = []
        self.begin_count = 0

    def begin(self):
        self.begin_count += 1
        return _Connection(self)


def _store():
    store = ExecutionDispatchRuntimeStore.__new__(ExecutionDispatchRuntimeStore)
    store._engine = _Engine()
    return store


def _frame(generated_at=STAMP):
    return ExecutionDispatchFrameV1(
        frame_id="dispatch-frame-1",
        generated_at=generated_at,
        source_policy_frame_id="policy-1",
        source_proposal_frame_id="proposal-1",
    )


def test_materialization_receipt_assigns_database_event_time():
    store = _store()

    generated_at = store.start_dispatch_frame_materialization("dispatch-frame-1")

    assert generated_at == STAMP
    sql, params = store._engine.calls[0]
    assert "INSERT INTO substrate_execution_dispatch_inflight" in sql
    assert "ON CONFLICT (frame_id) DO UPDATE" in sql
    assert "RETURNING generated_at" in sql
    assert params == {"frame_id": "dispatch-frame-1"}


def test_real_frame_replaces_inflight_receipt_in_the_same_transaction():
    store = _store()

    store.save_dispatch_frame(_frame())

    sql = [statement for statement, _params in store._engine.calls]
    assert store._engine.begin_count == 1
    assert "INSERT INTO substrate_execution_dispatch_frames" in sql[0]
    assert "SET dispatch_pending = false" in sql[1]
    assert "DELETE FROM substrate_execution_dispatch_inflight" in sql[2]


def test_worker_replaces_pre_receipt_timestamp_before_any_send_path():
    worker = ExecutionDispatchRuntimeWorker.__new__(ExecutionDispatchRuntimeWorker)
    worker._store = SimpleNamespace(
        start_dispatch_frame_materialization=lambda _frame_id: STAMP
    )
    preliminary = _frame(datetime(2026, 11, 3, 0, 59, 55, tzinfo=timezone.utc))

    visible = worker._start_frame_materialization(preliminary)

    assert visible.generated_at == STAMP

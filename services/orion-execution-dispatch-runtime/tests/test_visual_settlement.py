"""Durable render settlement in execution-dispatch.

A render_scene verb that only submitted its `reverie.visual` run is stored as
pending (no latency, no action outcome). The reconcile step later replays the
run's terminal DurableRunStateV1 row and settles the SAME result row in place.
A run that ends without an image is never scored as Orion failing: visual
outcome unknown, success False, latency NULL (no motor seconds, no cost sample).

Design: docs/superpowers/specs/2026-09-28-visual-reverie-durable-graph-design.md.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.worker as worker_mod
from app.store import ExecutionDispatchRuntimeStore
from app.worker import DISPATCH_STATUS_PENDING, ExecutionDispatchRuntimeWorker
from orion.execution_dispatch.visual_settlement import SETTLEMENT_ORPHAN_MARGIN_SEC, settle_visual_result
from orion.schemas.durable_run import DurableRunStateV1
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.reverie_visual_run import reverie_visual_run_id

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
CREATED = NOW - timedelta(minutes=20)
DISPATCH_ID = "dispatch:render:1"
RUN_ID = reverie_visual_run_id(DISPATCH_ID)


def _pending_json(**over) -> dict:
    settlement = {
        "state": "pending",
        "durable_run_id": RUN_ID,
        "workflow": "reverie.visual",
        "submitted_at": CREATED.isoformat(),
        "deadline_at": (CREATED + timedelta(seconds=5400)).isoformat(),
    }
    settlement.update(over)
    return {
        "visual_outcome": "unknown",
        "result_kind": "structured",
        "structured_result": {"outcome": "unknown", "settlement": dict(settlement)},
        "settlement": settlement,
        "submit_latency_ms": 812.0,
        "evidence_refs": [f"result:{DISPATCH_ID}"],
        "latency_ms": None,
    }


def _terminal(status: str, detail: dict) -> DurableRunStateV1:
    """A real terminal row, validated by the frozen schema, then read back the
    way sql-writer stores it (substrate_durable_run_state.detail is JSON)."""
    return DurableRunStateV1(
        run_id=RUN_ID,
        workflow="reverie.visual",
        thread_id=RUN_ID,
        node="finish" if status == "completed" else "failed",
        status=status,
        correlation_id="corr-1",
        detail=detail,
    )


def _row(run: DurableRunStateV1 | None, **over) -> dict:
    row = {
        "result_id": f"result:{DISPATCH_ID}",
        "dispatch_id": DISPATCH_ID,
        "frame_id": "execution.dispatch.frame:original",
        "status": DISPATCH_STATUS_PENDING,
        "result_json": _pending_json(),
        "dispatch_kind": "express",
        "target_id": "host:circe_gpu",
        "created_at": CREATED,
        "run_status": run.status if run else None,
        "run_detail": json.loads(json.dumps(run.detail)) if run else None,
    }
    row.update(over)
    return row


def _settle(row: dict, now: datetime = NOW):
    return settle_visual_result(
        result_json=row["result_json"],
        run_status=row["run_status"],
        run_detail=row["run_detail"],
        created_at=row["created_at"],
        now=now,
        dispatch_kind=row["dispatch_kind"],
        target_id=row["target_id"],
    )


# --- the pure settlement decision -------------------------------------------


def test_completed_produced_run_charges_the_runs_own_gpu_seconds():
    receipt = {"outcome": "produced", "gate_reason": "ok", "artifact_persisted": True}
    run = _terminal("completed", {"outcome": "produced", "visual_elapsed_sec": 61.25, "chain_id": "chain-1", "execution_receipt": receipt})
    settled = _settle(_row(run))
    assert settled.visual_outcome == "produced"
    assert settled.success is True
    assert settled.status == "success"
    assert settled.latency_ms == pytest.approx(61250.0)
    assert settled.result_json["latency_ms"] == pytest.approx(61250.0)
    assert settled.result_json["execution_receipt"] == receipt
    assert settled.result_json["chain_id"] == "chain-1"
    block = settled.result_json["settlement"]
    assert block["state"] == "settled" and block["durable_status"] == "completed"
    assert block["settled_at"] == NOW.isoformat()
    assert block["durable_run_id"] == RUN_ID
    assert "reason" not in block


def test_failed_run_is_not_orion_failing_and_costs_nothing():
    run = _terminal("failed", {"error": "retry_window_expired", "visual_elapsed_sec": 12.0})
    settled = _settle(_row(run))
    assert settled.visual_outcome == "unknown"
    assert settled.success is False
    assert settled.latency_ms is None
    assert settled.result_json["latency_ms"] is None
    assert settled.status != "failed"
    assert settled.result_json["settlement"]["reason"] == "retry_window_expired"
    assert settled.result_json["settlement"]["durable_status"] == "failed"
    assert "not counted as a failure" in settled.summary


@pytest.mark.parametrize("status", ["cancelled", "abandoned"])
def test_cancelled_or_abandoned_run_settles_unknown(status):
    settled = _settle(_row(_terminal(status, {"last_error": "operator_cancel"})))
    assert settled.visual_outcome == "unknown"
    assert settled.latency_ms is None
    assert settled.result_json["settlement"]["reason"] == "operator_cancel"


def test_completed_run_with_an_unrecognised_outcome_settles_unknown():
    settled = _settle(_row(_terminal("completed", {"outcome": "accepted", "visual_elapsed_sec": 50.0})))
    assert settled.visual_outcome == "unknown"
    assert settled.latency_ms is None


def test_completed_non_produced_outcome_keeps_its_value_without_cost():
    settled = _settle(_row(_terminal("completed", {"outcome": "already_satisfied", "visual_elapsed_sec": 3.0})))
    assert settled.visual_outcome == "already_satisfied"
    assert settled.success is False
    assert settled.latency_ms is None


def test_in_flight_run_stays_pending_until_the_orphan_deadline():
    row = _row(None)
    deadline = CREATED + timedelta(seconds=5400)
    assert _settle(row, now=deadline + timedelta(seconds=SETTLEMENT_ORPHAN_MARGIN_SEC - 1)) is None
    orphan = _settle(row, now=deadline + timedelta(seconds=SETTLEMENT_ORPHAN_MARGIN_SEC + 1))
    assert orphan.visual_outcome == "unknown"
    assert orphan.latency_ms is None
    assert orphan.result_json["settlement"]["reason"] == "settlement_timeout"
    assert orphan.result_json["settlement"]["durable_status"] is None


def test_already_settled_row_is_left_alone():
    row = _row(_terminal("completed", {"outcome": "produced", "visual_elapsed_sec": 60.0}))
    row["result_json"]["settlement"]["state"] = "settled"
    assert _settle(row) is None


def test_zero_gpu_seconds_is_missing_not_free():
    """durable-runs writes 0.0 when it recorded nothing; charging 0 motor
    seconds for a produced image would read as a free render."""
    settled = _settle(_row(_terminal("completed", {"outcome": "produced", "visual_elapsed_sec": 0.0})))
    assert settled.visual_outcome == "produced"
    assert settled.latency_ms is None
    assert settled.result_json["settlement"]["latency_missing"] is True


def test_settled_block_carries_when_the_run_really_ended():
    finished = CREATED + timedelta(minutes=9)
    run = _terminal("completed", {"outcome": "produced", "visual_elapsed_sec": 60.0,
                                  "finished_at": finished.isoformat()})
    block = _settle(_row(run)).result_json["settlement"]
    assert block["finished_at"] == finished.isoformat()
    # No run row, no end time: an orphan never claims one.
    deadline = CREATED + timedelta(seconds=5400 + SETTLEMENT_ORPHAN_MARGIN_SEC + 1)
    assert "finished_at" not in _settle(_row(None), now=deadline).result_json["settlement"]


def test_nested_verb_result_no_longer_claims_pending():
    run = _terminal("completed", {"outcome": "produced", "visual_elapsed_sec": 60.0})
    nested = _settle(_row(run)).result_json["structured_result"]
    assert nested["settlement"]["state"] == "settled"
    assert nested["outcome"] == "produced"


def _not_submitted_row(run):
    result_json = _pending_json(state="not_submitted", reason="rpc_error:timeout")
    return _row(run, status="empty", result_json=result_json)


def test_unconfirmed_kickoff_settles_once_its_run_ends():
    """The receipt RPC can time out after orch admitted the run; if that run
    produces an image, the row must not stay failed and uncharged forever."""
    run = _terminal("completed", {"outcome": "produced", "visual_elapsed_sec": 42.0})
    settled = _settle(_not_submitted_row(run))
    assert settled.visual_outcome == "produced" and settled.success is True
    assert settled.latency_ms == pytest.approx(42000.0)
    block = settled.result_json["settlement"]
    assert block["state"] == "settled" and block["settled_from"] == "not_submitted"
    assert "reason" not in block


def test_unconfirmed_kickoff_without_a_run_is_never_orphaned():
    far_future = NOW + timedelta(days=30)
    assert _settle(_not_submitted_row(None), now=far_future) is None
    no_run_id = _row(None, result_json=_pending_json(state="not_submitted", durable_run_id=None))
    no_run_id["run_status"] = "completed"
    assert _settle(no_run_id) is None


def test_unconfirmed_kickoff_whose_run_made_no_image_is_not_a_failure():
    settled = _settle(_not_submitted_row(_terminal("failed", {"error": "retry_window_expired", "visual_elapsed_sec": 9.0})))
    assert settled.visual_outcome == "unknown" and settled.success is False
    assert settled.latency_ms is None
    assert settled.status == "empty"
    assert settled.result_json["settlement"]["settled_from"] == "not_submitted"


# --- the worker ---------------------------------------------------------------


def _worker(monkeypatch) -> ExecutionDispatchRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    worker._store = MagicMock()
    worker._store.latest_bus_synaptic_prediction_error.return_value = 0.2
    worker._store.load_dispatch_result_by_dispatch_id.return_value = None
    return worker


def _fake_bus(monkeypatch, *, publish_error: Exception | None = None) -> MagicMock:
    bus = MagicMock()
    bus.connect = AsyncMock()
    bus.close = AsyncMock()
    bus.publish = AsyncMock(side_effect=publish_error)
    monkeypatch.setattr(worker_mod, "OrionBusAsync", MagicMock(return_value=bus))
    return bus


def test_reconcile_settles_the_same_row_and_reemits_the_outcome(monkeypatch):
    worker = _worker(monkeypatch)
    bus = _fake_bus(monkeypatch)
    run = _terminal("completed", {"outcome": "produced", "visual_elapsed_sec": 58.5, "execution_receipt": {"outcome": "produced", "gate_reason": "ok"}})
    worker._store.load_pending_visual_settlements.return_value = [_row(run)]
    worker._store.settle_dispatch_result.return_value = True

    settled = asyncio.run(worker._reconcile_visual_settlements(now=NOW))

    assert settled == 1
    kwargs = worker._store.settle_dispatch_result.call_args.kwargs
    assert kwargs["result_id"] == f"result:{DISPATCH_ID}"
    assert kwargs["latency_ms"] == pytest.approx(58500.0)
    assert kwargs["status"] == "success"
    assert kwargs["result_json"]["visual_outcome"] == "produced"
    assert kwargs["result_json"]["settlement"]["state"] == "settled"
    # The frame and day bucket belong to the original row; settle never rewrites them.
    assert "frame_id" not in kwargs and "created_at" not in kwargs
    _, env = bus.publish.await_args.args
    assert env.payload["action_id"] == DISPATCH_ID
    assert env.payload["kind"] == "express"
    assert env.payload["success"] is True
    assert env.payload["visual_outcome"] == "produced"


def test_reconcile_failed_run_emits_unsuccessful_unknown_with_no_cost(monkeypatch):
    worker = _worker(monkeypatch)
    bus = _fake_bus(monkeypatch)
    worker._store.load_pending_visual_settlements.return_value = [_row(_terminal("failed", {"error": "retry_window_expired"}))]
    worker._store.settle_dispatch_result.return_value = True

    asyncio.run(worker._reconcile_visual_settlements(now=NOW))

    kwargs = worker._store.settle_dispatch_result.call_args.kwargs
    assert kwargs["latency_ms"] is None
    assert kwargs["result_json"]["visual_outcome"] == "unknown"
    _, env = bus.publish.await_args.args
    assert env.payload["success"] is False
    assert env.payload["visual_outcome"] == "unknown"


def test_reconcile_orphan_times_out(monkeypatch):
    worker = _worker(monkeypatch)
    bus = _fake_bus(monkeypatch)
    worker._store.load_pending_visual_settlements.return_value = [_row(None)]
    worker._store.settle_dispatch_result.return_value = True
    late = CREATED + timedelta(seconds=5400 + SETTLEMENT_ORPHAN_MARGIN_SEC + 60)

    asyncio.run(worker._reconcile_visual_settlements(now=late))

    kwargs = worker._store.settle_dispatch_result.call_args.kwargs
    assert kwargs["result_json"]["settlement"]["reason"] == "settlement_timeout"
    assert kwargs["latency_ms"] is None
    assert bus.publish.await_count == 1


def test_reconcile_leaves_in_flight_rows_and_opens_no_bus(monkeypatch):
    worker = _worker(monkeypatch)
    ctor = MagicMock()
    monkeypatch.setattr(worker_mod, "OrionBusAsync", ctor)
    worker._store.load_pending_visual_settlements.return_value = [_row(None)]

    assert asyncio.run(worker._reconcile_visual_settlements(now=NOW)) == 0
    worker._store.settle_dispatch_result.assert_not_called()
    ctor.assert_not_called()


def test_publish_failure_keeps_the_row_pending_for_the_next_pass(monkeypatch):
    worker = _worker(monkeypatch)
    _fake_bus(monkeypatch, publish_error=RuntimeError("bus down"))
    worker._store.load_pending_visual_settlements.return_value = [_row(_terminal("failed", {"error": "x"}))]

    assert asyncio.run(worker._reconcile_visual_settlements(now=NOW)) == 0
    worker._store.settle_dispatch_result.assert_not_called()


def test_reconcile_is_rate_limited(monkeypatch):
    worker = _worker(monkeypatch)
    _fake_bus(monkeypatch)
    worker._store.load_pending_visual_settlements.return_value = []
    asyncio.run(worker._reconcile_visual_settlements(now=NOW))
    asyncio.run(worker._reconcile_visual_settlements(now=NOW))
    assert worker._store.load_pending_visual_settlements.call_count == 1


def _render_candidate() -> ExecutionDispatchCandidateV1:
    return ExecutionDispatchCandidateV1(
        dispatch_id=DISPATCH_ID,
        source_decision_id="pd:render",
        source_proposal_id="proposal:render",
        dispatch_status="prepared_for_dispatch",
        dispatch_mode="dispatch_read_only",
        dispatch_kind="express",
        target_id="host:circe_gpu",
        target_kind="host",
        cortex_verb="skills.imagination.render_scene.v1",
        cortex_mode="brain",
        request_envelope={"context": {}},
        risk_score=0.1,
        confidence_score=0.9,
    )


def _frame() -> ExecutionDispatchFrameV1:
    return ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:render",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:render",
        source_proposal_frame_id="proposal.frame:render",
        source_field_tick_id="field.tick:render",
        dispatch_mode="dispatch_read_only",
        candidates=[_render_candidate()],
    )


class _Client:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def dispatch(self, **_kwargs):
        return self.payload


def _verb_payload(result: dict, *, status: str = "success") -> dict:
    return {"result": {"status": status, "final_text": json.dumps(result, sort_keys=True)}}


def test_pending_submit_is_stored_uncosted_and_emits_nothing(monkeypatch):
    worker = _worker(monkeypatch)
    bus = MagicMock()
    bus.publish = AsyncMock()
    settlement = _pending_json()["settlement"]
    client = _Client(_verb_payload({"outcome": "unknown", "ran": False, "refused": False, "settlement": settlement}))

    out = asyncio.run(worker._send_one_inner(client, bus, _frame(), _render_candidate()))

    kwargs = worker._store.save_dispatch_result.call_args.kwargs
    assert kwargs["status"] == DISPATCH_STATUS_PENDING
    assert kwargs["latency_ms"] is None
    assert kwargs["result_json"]["latency_ms"] is None
    assert kwargs["result_json"]["settlement"] == settlement
    assert kwargs["result_json"]["submit_latency_ms"] >= 0
    assert kwargs["result_json"]["visual_outcome"] == "unknown"
    bus.publish.assert_not_called()
    assert list(worker._recent_dispatch_statuses) == [DISPATCH_STATUS_PENDING]
    assert out.result_ref == f"result:{DISPATCH_ID}"
    assert out.visual_outcome == "unknown"


def test_replay_of_a_pending_result_emits_nothing(monkeypatch):
    worker = _worker(monkeypatch)
    bus = MagicMock()
    bus.publish = AsyncMock()
    worker._store.load_dispatch_result_by_dispatch_id.return_value = {
        "result_id": f"result:{DISPATCH_ID}", "status": DISPATCH_STATUS_PENDING,
        "result_json": _pending_json(), "raw_len": 0,
    }
    client = MagicMock()

    asyncio.run(worker._send_one_inner(client, bus, _frame(), _render_candidate()))

    bus.publish.assert_not_called()
    worker._store.save_dispatch_result.assert_not_called()


def test_deferred_resource_is_an_allowed_visual_outcome(monkeypatch):
    worker = _worker(monkeypatch)
    bus = MagicMock()
    bus.publish = AsyncMock()
    client = _Client(_verb_payload({"outcome": "deferred_resource", "ran": False, "refused": True}))

    out = asyncio.run(worker._send_one_inner(client, bus, _frame(), _render_candidate()))

    assert out.visual_outcome == "deferred_resource"
    assert worker._store.save_dispatch_result.call_args.kwargs["result_json"]["visual_outcome"] == "deferred_resource"
    _, env = bus.publish.await_args.args
    assert env.payload["visual_outcome"] == "deferred_resource"
    assert env.payload["success"] is False


def _not_submitted_verb_result() -> dict:
    return {"outcome": "unknown", "ran": False, "refused": False, "durable_run_id": RUN_ID,
            "reason": "TimeoutError: x",
            "settlement": {"state": "not_submitted", "durable_run_id": RUN_ID, "reason": "TimeoutError: x"}}


@pytest.mark.parametrize("plan_status", ["fail", "unavailable"])
def test_unsubmitted_durable_render_is_not_a_failure_and_costs_nothing(monkeypatch, plan_status):
    """cortex-exec could not confirm the kickoff (ok=False, status unavailable):
    no image was made, so no failed row, no failure outcome, no motor seconds."""
    worker = _worker(monkeypatch)
    bus = MagicMock()
    bus.publish = AsyncMock()
    client = _Client(_verb_payload(_not_submitted_verb_result(), status=plan_status))

    out = asyncio.run(worker._send_one_inner(client, bus, _frame(), _render_candidate()))

    kwargs = worker._store.save_dispatch_result.call_args.kwargs
    assert kwargs["status"] == "empty"
    assert kwargs["latency_ms"] is None
    assert kwargs["result_json"]["latency_ms"] is None
    assert kwargs["result_json"]["visual_outcome"] == "unknown"
    assert kwargs["result_json"]["settlement"]["state"] == "not_submitted"
    assert kwargs["result_json"]["settlement"]["durable_run_id"] == RUN_ID
    assert kwargs["result_json"]["submit_latency_ms"] >= 0
    bus.publish.assert_not_called()
    assert list(worker._recent_dispatch_statuses) == ["empty"]
    assert out.visual_outcome == "unknown"
    assert out.result_ref == f"result:{DISPATCH_ID}" and out.dispatch_error is None


def test_unsubmitted_row_settles_through_reconcile_once_its_run_ends(monkeypatch):
    """The row the send path stores is the row reconcile picks up: an admitted
    run that produced an image settles it charged and emits its one outcome."""
    worker = _worker(monkeypatch)
    bus = MagicMock()
    bus.publish = AsyncMock()
    client = _Client(_verb_payload(_not_submitted_verb_result(), status="fail"))
    asyncio.run(worker._send_one_inner(client, bus, _frame(), _render_candidate()))
    stored = worker._store.save_dispatch_result.call_args.kwargs

    reconcile_bus = _fake_bus(monkeypatch)
    run = _terminal("completed", {"outcome": "produced", "visual_elapsed_sec": 30.0})
    worker._store.load_pending_visual_settlements.return_value = [
        _row(run, status=stored["status"], result_json=json.loads(json.dumps(stored["result_json"])))
    ]
    worker._store.settle_dispatch_result.return_value = True

    assert asyncio.run(worker._reconcile_visual_settlements(now=NOW)) == 1
    kwargs = worker._store.settle_dispatch_result.call_args.kwargs
    assert kwargs["status"] == "success"
    assert kwargs["latency_ms"] == pytest.approx(30000.0)
    assert kwargs["result_json"]["settlement"]["settled_from"] == "not_submitted"
    _, env = reconcile_bus.publish.await_args.args
    assert env.payload["success"] is True and env.payload["visual_outcome"] == "produced"


def test_unsubmitted_row_without_a_terminal_run_is_not_settled(monkeypatch):
    worker = _worker(monkeypatch)
    ctor = MagicMock()
    monkeypatch.setattr(worker_mod, "OrionBusAsync", ctor)
    worker._store.load_pending_visual_settlements.return_value = [_not_submitted_row(None)]

    assert asyncio.run(worker._reconcile_visual_settlements(now=NOW + timedelta(days=30))) == 0
    worker._store.settle_dispatch_result.assert_not_called()
    ctor.assert_not_called()


class _RaisingClient:
    async def dispatch(self, **_kwargs):
        raise TimeoutError("rpc timed out")


def test_render_send_exception_is_uncharged_and_not_scored(monkeypatch):
    """The RPC to cortex-exec threw: the whole timeout is not charged as motor
    time for a render that made no image, and nothing scores it as failing.
    cortex-exec may still have submitted, so the row stays settleable."""
    worker = _worker(monkeypatch)
    bus = MagicMock()
    bus.publish = AsyncMock()

    out = asyncio.run(worker._send_one_inner(_RaisingClient(), bus, _frame(), _render_candidate()))

    kwargs = worker._store.save_dispatch_result.call_args.kwargs
    assert kwargs["status"] == "empty"
    assert kwargs["latency_ms"] is None
    assert kwargs["result_json"]["latency_ms"] is None
    assert kwargs["result_json"]["visual_outcome"] == "unknown"
    settlement = kwargs["result_json"]["settlement"]
    assert settlement["state"] == "not_submitted"
    assert settlement["durable_run_id"] == RUN_ID
    assert settlement["reason"].startswith("send_error:TimeoutError")
    bus.publish.assert_not_called()
    assert out.visual_outcome == "unknown"
    assert out.result_ref == f"result:{DISPATCH_ID}"


def test_non_render_send_exception_is_still_charged_and_failed(monkeypatch):
    worker = _worker(monkeypatch)
    bus = MagicMock()
    bus.publish = AsyncMock()
    candidate = _render_candidate().model_copy(update={"cortex_verb": "skills.gpu.nvidia_smi_snapshot.v1", "dispatch_kind": "inspect"})

    asyncio.run(worker._send_one_inner(_RaisingClient(), bus, _frame(), candidate))

    kwargs = worker._store.save_dispatch_result.call_args.kwargs
    assert kwargs["status"] == "failed"
    assert kwargs["latency_ms"] is not None
    _, env = bus.publish.await_args.args
    assert env.payload["success"] is False


@pytest.mark.parametrize("status", ["empty", "failed"])
def test_replay_of_an_unsubmitted_result_emits_nothing(monkeypatch, status):
    """`failed` covers rows stored before unconfirmed kickoffs were non-failures."""
    worker = _worker(monkeypatch)
    bus = MagicMock()
    bus.publish = AsyncMock()
    worker._store.load_dispatch_result_by_dispatch_id.return_value = {
        "result_id": f"result:{DISPATCH_ID}", "status": status,
        "result_json": _pending_json(state="not_submitted", reason="TimeoutError: x"), "raw_len": 0,
    }

    asyncio.run(worker._send_one_inner(MagicMock(), bus, _frame(), _render_candidate()))

    bus.publish.assert_not_called()
    worker._store.save_dispatch_result.assert_not_called()


def test_direct_path_render_is_unchanged(monkeypatch):
    """No settlement block (setting off / manual): today's accounting."""
    worker = _worker(monkeypatch)
    bus = MagicMock()
    bus.publish = AsyncMock()
    client = _Client(_verb_payload({"outcome": "produced", "ran": True, "refused": False, "chain_id": "c"}))

    asyncio.run(worker._send_one_inner(client, bus, _frame(), _render_candidate()))

    kwargs = worker._store.save_dispatch_result.call_args.kwargs
    assert kwargs["status"] == "success"
    assert kwargs["latency_ms"] is not None
    _, env = bus.publish.await_args.args
    assert env.payload["success"] is True


# --- the store ----------------------------------------------------------------


def _store_with_capture(rowcount: int = 1):
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    engine.connect.return_value.__enter__.return_value = conn
    captured: list[tuple[str, dict]] = []

    def _execute(stmt, params=None):
        captured.append((str(stmt), params or {}))
        result = MagicMock()
        result.rowcount = rowcount
        result.first.return_value = (12.5, 0)
        result.mappings.return_value.all.return_value = []
        return result

    conn.execute.side_effect = _execute
    store._engine = engine
    return store, captured


def test_settle_update_keeps_frame_and_day_bucket_and_is_pending_guarded():
    store, captured = _store_with_capture(rowcount=1)
    assert store.settle_dispatch_result(result_id="result:x", status="success", result_json={"a": 1}, latency_ms=5.0) is True
    sql, params = captured[-1]
    set_clause = sql.split("SET", 1)[1].split("WHERE", 1)[0]
    assert "created_at" not in set_clause and "frame_id" not in set_clause
    assert "result_json->'settlement'->>'state' IN ('pending', 'not_submitted')" in sql
    assert params["result_id"] == "result:x"

    store, _ = _store_with_capture(rowcount=0)
    assert store.settle_dispatch_result(result_id="result:x", status="success", result_json={}, latency_ms=None) is False


def test_intentionally_uncosted_settlement_rows_are_not_warned_about():
    store, captured = _store_with_capture()
    store.sum_motor_seconds_for_day(NOW, NOW + timedelta(days=1))
    sql, _ = captured[-1]
    assert "NOT (result_json ? 'settlement')" in sql


def test_pending_loader_joins_only_terminal_reverie_visual_states():
    store, captured = _store_with_capture()
    store.load_pending_visual_settlements(limit=5, workflow="reverie.visual")
    sql, params = captured[-1]
    assert "substrate_durable_run_state" in sql
    assert "'completed', 'failed', 'cancelled', 'abandoned'" in sql
    assert params == {"workflow": "reverie.visual", "lookback_sec": 7 * 86400.0, "limit": 5}
    flat = " ".join(sql.split())
    assert "state' = 'not_submitted' AND s.status IS NOT NULL" in flat, (
        "an unconfirmed kickoff is only picked up once its run has ended"
    )
    assert "ORDER BY (s.status IS NULL), r.created_at ASC" in flat, (
        "in-flight rows must not crowd finished runs out of the batch"
    )

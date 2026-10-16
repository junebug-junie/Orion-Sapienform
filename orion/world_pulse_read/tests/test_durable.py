import asyncio
from uuid import uuid4
from unittest.mock import patch

import httpx
import pytest

from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.reading_turn import ReadingRunBriefV1, ReadingTurnResultV1
from orion.schemas.resource_admission import ResourceRequirementV1
from orion.world_pulse_read.durable import ReadingCancelled, ReadingPending, poll_turn


def request():
    return DurableRunRequestV1(
        run_id="reading-test",
        workflow="reading.turn",
        correlation_id=str(uuid4()),
        admission=ResourceRequirementV1(),
        brief=ReadingRunBriefV1(
            seed_id="reading:test",
            stage=1,
            prompt="Read source",
            session_id="reading",
            timeout_sec=900,
        ),
    )


@pytest.mark.parametrize(
    "status", ["waiting_resource", "admitted", "running", "paused"]
)
def test_waiting_never_returns_empty_success(status):
    req = request()

    def respond(http):
        return httpx.Response(
            200,
            json={
                "run_id": req.run_id,
                "workflow_kind": req.workflow,
                "status": status,
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://durable"
    )
    with patch("orion.world_pulse_read.durable.httpx.AsyncClient", return_value=client):
        with pytest.raises(ReadingPending):
            asyncio.run(poll_turn(req, "http://durable"))


def test_lost_acceptance_ack_is_pending_and_keeps_exact_request():
    req = request()
    seen = []

    def respond(http):
        seen.append(http.content)
        raise httpx.ReadTimeout("ack lost", request=http)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://durable"
    )
    with patch("orion.world_pulse_read.durable.httpx.AsyncClient", return_value=client):
        with pytest.raises(ReadingPending):
            asyncio.run(poll_turn(req, "http://durable"))
    assert DurableRunRequestV1.model_validate_json(seen[0]) == req


def test_completed_result_preserves_actual_turn_trace_and_evidence():
    req = request()
    result = ReadingTurnResultV1(
        run_id=req.run_id,
        correlation_id=str(uuid4()),
        ok=True,
        text="Source-grounded answer",
        source_fetches=[],
    )

    def respond(http):
        return httpx.Response(
            200,
            json={
                "run_id": req.run_id,
                "workflow_kind": req.workflow,
                "status": "completed",
                "reading_result": result.model_dump(),
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://durable"
    )
    with patch("orion.world_pulse_read.durable.httpx.AsyncClient", return_value=client):
        assert asyncio.run(poll_turn(req, "http://durable")) == result


def test_reading_requires_admission_and_cannot_use_curiosity_brief():
    req = request().model_dump()
    req["admission"] = None
    with pytest.raises(ValueError):
        DurableRunRequestV1.model_validate(req)


def test_reading_rpc_schemas_are_resolvable_by_the_live_bus():
    from orion.schemas.registry import resolve, SCHEMA_REGISTRY
    from orion.schemas.reading_turn import ReadingTurnRequestV1

    assert resolve("ReadingTurnRequestV1") is ReadingTurnRequestV1
    assert resolve("ReadingTurnResultV1") is ReadingTurnResultV1
    assert SCHEMA_REGISTRY["ReadingTurnRequestV1"].kind == "reading.turn.request.v1"
    assert SCHEMA_REGISTRY["ReadingTurnResultV1"].kind == "reading.turn.result.v1"


def test_cancelled_run_has_separate_noncharging_outcome():
    req = request()
    def respond(http):
        return httpx.Response(200, json={"run_id": req.run_id, "workflow_kind": req.workflow,
                                        "status": "cancelled", "work_started": False})
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://durable")
    with patch("orion.world_pulse_read.durable.httpx.AsyncClient", return_value=client):
        with pytest.raises(ReadingCancelled):
            asyncio.run(poll_turn(req, "http://durable"))


class _BindingConn:
    """Just enough of asyncpg for bind_turn's "an unconsumed binding exists" path."""

    def __init__(self, stored):
        self.stored = stored

    def transaction(self):
        class _Tx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False
        return _Tx()

    async def fetchval(self, sql, *args):
        return "reading:test" if "world_pulse_read_seed" in sql else self.stored


@pytest.mark.parametrize("as_text", [True, False])
def test_a_binding_stored_before_4_6_still_binds(as_text):
    """Regression (stage 4.6 review): bindings committed before 4.6 carry the deleted broker
    fields; ResourceRequirementV1 is extra="forbid", so reading them back used to raise and
    wedge the seed in reading_binding_unavailable forever. Live 2026-09-29: 39 unconsumed rows."""
    import json
    from orion.world_pulse_read.durable import bind_turn

    original = request()
    stored = original.model_dump(mode="json")
    stored["admission"].update(allow_elastic_activation=False, alternatives=[], pinned_lane=None,
                               operator_override=None)
    with pytest.raises(Exception):
        DurableRunRequestV1.model_validate(stored)   # the hazard: the bare model refuses the row
    conn = _BindingConn(json.dumps(stored) if as_text else stored)
    bound = asyncio.run(bind_turn(conn, original.brief, original.correlation_id))
    assert bound == original   # same run id, same prompt: the stored binding stays authoritative


# -- work start (the reading `look` act opens at it) ---------------------------

def test_work_started_at_reads_first_run_started_and_ignores_queued_runs():
    from datetime import datetime, timezone

    from orion.world_pulse_read.durable import work_started_at

    history = [
        {"event": "run.accepted", "generated_at": "2026-10-16T12:03:31Z"},
        {"event": "run.started", "generated_at": "2026-10-16T15:17:37.356183Z"},
        {"event": "run.started", "generated_at": "2026-10-16T15:40:00Z"},
    ]
    assert work_started_at({"work_started": True, "history": history}) == datetime(
        2026, 10, 16, 15, 17, 37, 356183, tzinfo=timezone.utc)
    # Queued, or refused before any work: no start.
    assert work_started_at({"work_started": False, "history": history[:1]}) is None
    assert work_started_at({"history": history}) is None
    # No usable history: creation plus the queue wait.
    assert work_started_at({"work_started": True, "history": [],
                            "created_at": "2026-10-16T12:00:00+00:00",
                            "queue_wait_seconds": 60.0}) == datetime(2026, 10, 16, 12, 1, tzinfo=timezone.utc)


def test_poll_reports_each_verified_state_and_a_broken_observer_never_fails_it():
    req = request()
    state = {"run_id": req.run_id, "workflow_kind": req.workflow, "status": "running",
             "work_started": True}
    seen = []

    def respond(http):
        return httpx.Response(200, json=state)

    def observer(s):
        seen.append(s["status"])
        raise RuntimeError("observer bug")

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://durable")
    with patch("orion.world_pulse_read.durable.httpx.AsyncClient", return_value=client):
        with pytest.raises(ReadingPending):
            asyncio.run(poll_turn(req, "http://durable", on_state=observer))
    assert seen == ["running"]


@pytest.mark.parametrize("status,raises", [("completed", None), ("failed", ValueError)])
def test_poll_reports_terminal_states_before_returning_or_raising(status, raises):
    """The poll that ends a read is often the first to see its work start: the
    observer must see it before poll_turn returns the result or raises."""
    req = request()
    result = ReadingTurnResultV1(run_id=req.run_id, correlation_id=str(uuid4()), ok=True,
                                 text="answer", source_fetches=[])
    state = {"run_id": req.run_id, "workflow_kind": req.workflow, "status": status,
             "work_started": True, "error": "boom", "reading_result": result.model_dump()}
    seen = []

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda http: httpx.Response(200, json=state)),
        base_url="http://durable")
    with patch("orion.world_pulse_read.durable.httpx.AsyncClient", return_value=client):
        if raises is None:
            assert asyncio.run(poll_turn(req, "http://durable", on_state=seen.append)) == result
        else:
            with pytest.raises(raises):
                asyncio.run(poll_turn(req, "http://durable", on_state=seen.append))
    assert [s["status"] for s in seen] == [status]

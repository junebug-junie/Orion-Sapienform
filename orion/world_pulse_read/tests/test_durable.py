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

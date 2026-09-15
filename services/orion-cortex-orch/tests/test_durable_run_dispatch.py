"""Cortex is the kickoff for durable runs (app/durable_runs.py).

A request carrying `context.metadata.durable_run` is dispatched to
`orion:durable:run:request` and answered `accepted` immediately; a request
without it is untouched; a malformed one fails loudly instead of routing
into chat.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from orion.core.bus.bus_schemas import LLMMessage, ServiceRef
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest
from orion.schemas.durable_run import DURABLE_RUN_REQUEST_CHANNEL, DURABLE_RUN_REQUEST_KIND

from app.durable_runs import dispatch_durable_run, durable_run_request_from, has_durable_run_request


def _req(metadata: dict | None) -> CortexClientRequest:
    return CortexClientRequest(
        mode="brain",
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="investigate")],
            user_message="investigate",
            session_id="orion_curiosity",
            metadata=metadata or {},
        ),
    )


def _durable_payload() -> dict:
    return {
        "run_id": "abc123def456",
        "workflow": "curiosity.investigate",
        "correlation_id": "7dcc3944-29bb-5d8f-915f-90f4e6968d47",
        "brief": {
            "prompt": "Pick something.",
            "session_id": "orion_curiosity",
            "timeout_sec": 3500.0,
            "graph_configured": True,
            "material": {"approved_total": 40, "approved_by_kind": {"semantic": 40}, "crystallization_count": 12, "relation_total": 300, "relation_count": 6},
        },
    }


def test_plain_requests_are_not_durable_runs() -> None:
    assert has_durable_run_request(_req(None)) is False
    assert has_durable_run_request(_req({"workflow_request": {"workflow_id": "x"}})) is False
    assert durable_run_request_from(_req(None)) is None


def test_a_malformed_kickoff_fails_loudly() -> None:
    with pytest.raises(Exception):
        durable_run_request_from(_req({"durable_run": {"run_id": "x"}}))


def _bus_with_subscriber(count: int = 1) -> AsyncMock:
    """A bus mock whose `.redis.pubsub_numsub(channel)` answers like a real
    Redis client would -- `[(channel, count)]`. A bare `AsyncMock()` used to
    be enough here; adding the subscriber check (below) means the fixture
    has to actually look like the thing being checked, or the mock silently
    validates nothing."""
    bus = AsyncMock()
    bus.redis.pubsub_numsub = AsyncMock(return_value=[(DURABLE_RUN_REQUEST_CHANNEL, count)])
    return bus


def test_dispatch_publishes_the_request_and_replies_accepted() -> None:
    bus = _bus_with_subscriber(1)
    req = _req({"durable_run": _durable_payload()})
    assert has_durable_run_request(req)
    result = asyncio.run(dispatch_durable_run(bus=bus, source=ServiceRef(name="orion-cortex-orch"), req=req, correlation_id="corr-1"))
    bus.publish.assert_awaited_once()
    channel, envelope = bus.publish.await_args.args
    assert channel == DURABLE_RUN_REQUEST_CHANNEL
    assert envelope.kind == DURABLE_RUN_REQUEST_KIND
    assert envelope.payload["run_id"] == "abc123def456"
    assert str(envelope.correlation_id) == "7dcc3944-29bb-5d8f-915f-90f4e6968d47"
    assert result.ok is True and result.status == "accepted"
    assert result.verb == "durable:curiosity.investigate"
    assert result.metadata["durable_run"]["status"] == "dispatched"


def test_dispatch_failure_is_a_failed_result_not_an_exception() -> None:
    bus = _bus_with_subscriber(1)
    bus.publish.side_effect = RuntimeError("bus down")
    result = asyncio.run(dispatch_durable_run(bus=bus, source=ServiceRef(name="orion-cortex-orch"), req=_req({"durable_run": _durable_payload()}), correlation_id="corr-1"))
    assert result.ok is False and result.status == "fail"
    assert result.metadata["durable_run"]["status"] == "dispatch_failed"


def test_a_publish_with_no_subscriber_is_a_failed_result_not_accepted() -> None:
    """A successful publish is not proof the run started: Redis drops a
    message with no subscriber, silently. Confirmed as a real gap 2026-09-07
    -- this used to report status="accepted" for a run that had just
    vanished."""
    bus = _bus_with_subscriber(0)
    result = asyncio.run(dispatch_durable_run(bus=bus, source=ServiceRef(name="orion-cortex-orch"), req=_req({"durable_run": _durable_payload()}), correlation_id="corr-1"))
    bus.publish.assert_awaited_once()
    assert result.ok is False and result.status == "fail"
    assert result.metadata["durable_run"]["status"] == "no_subscriber"


def test_a_subscriber_check_that_errors_fails_open_to_accepted() -> None:
    """The check itself is diagnostic, not load-bearing -- if it cannot run
    (a Redis client shape this script doesn't recognize, a transient error),
    a real dispatch must not be turned into a false "fail"."""
    bus = _bus_with_subscriber(1)
    bus.redis.pubsub_numsub.side_effect = RuntimeError("NUMSUB not supported")
    result = asyncio.run(dispatch_durable_run(bus=bus, source=ServiceRef(name="orion-cortex-orch"), req=_req({"durable_run": _durable_payload()}), correlation_id="corr-1"))
    assert result.ok is True and result.status == "accepted"


def _admission_bus(payload=None, kind="durable.run.receipt.v1"):
    from orion.core.bus.codec import OrionCodec
    from orion.core.bus.bus_schemas import BaseEnvelope
    bus = _bus_with_subscriber()
    bus.codec = OrionCodec()
    receipt = payload or {
        "run_id": "abc123def456", "status": "waiting_resource",
        "workflow_kind": "curiosity.investigate", "requested_resource": "llm.route.agent",
    }
    bus.rpc_request.return_value = {"data": bus.codec.encode(BaseEnvelope(
        kind=kind, source=ServiceRef(name="orion-durable-runs"), payload=receipt,
    ))}
    return bus


def test_admission_accepts_only_a_persisted_matching_receipt():
    payload = {**_durable_payload(), "admission": {}}
    bus = _admission_bus()
    result = asyncio.run(dispatch_durable_run(
        bus=bus, source=ServiceRef(name="orion-cortex-orch"), req=_req({"durable_run": payload}),
        correlation_id="corr-1", admission_enabled=True, receipt_timeout_sec=3,
    ))
    assert result.ok and result.status == "accepted"
    assert result.metadata["durable_run"]["status"] == "waiting_resource"
    assert result.metadata["durable_run"]["requested_resource"] == "llm.route.agent"
    assert bus.rpc_request.await_args.kwargs["timeout_sec"] == 3
    bus.publish.assert_not_awaited()


@pytest.mark.parametrize("case", ["timeout", "wrong_kind", "wrong_run", "disabled"])
def test_uncertain_admission_never_falls_back_to_pubsub(case):
    bus = _admission_bus(kind="unrelated" if case == "wrong_kind" else "durable.run.receipt.v1")
    if case == "timeout":
        bus.rpc_request.side_effect = TimeoutError("receipt missing")
    if case == "wrong_run":
        bus = _admission_bus({"run_id": "another-run", "status": "queued",
                              "workflow_kind": "curiosity.investigate", "requested_resource": "llm.route.agent"})
    result = asyncio.run(dispatch_durable_run(
        bus=bus, source=ServiceRef(name="orion-cortex-orch"),
        req=_req({"durable_run": {**_durable_payload(), "admission": {}}}),
        correlation_id="corr-1", admission_enabled=case != "disabled",
    ))
    assert not result.ok and result.error["type"] == "AdmissionUnconfirmed"
    bus.publish.assert_not_awaited()


def test_durable_kickoff_in_main_uses_get_settings_not_bare_settings() -> None:
    """Regression: live 2026-09-15 orch NameError'd every curiosity durable
    kickoff because main.py used bare ``settings.durable_admission_enabled``.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "app" / "main.py"
    text = src.read_text(encoding="utf-8")
    assert "settings.durable_admission_enabled" not in text
    assert "settings.durable_receipt_timeout_sec" not in text
    assert "get_settings()" in text
    assert "durable_admission_enabled" in text

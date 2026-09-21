"""A shed/refused gateway reply must fail the LLM step by name, not succeed empty.

Regression for 2026-09-19: ~70% of autonomous reading (world-pulse Stage 1/2)
and curiosity turns died as ``turn_deferred:stance_react_failed: stance_react
exec result missing thought payload``. The gateway had actually answered
``resource_lease_rejected ... reason=capacity_wait_budget_exhausted`` after a
235s wait for the single-slot agent lane -- but it says so via a normal
``llm.chat.result`` with empty content and ``raw.error`` set, which this
service carried forward as a *successful* step (the pre-existing gap
``_overloaded_result``'s own docstring in orion-llm-gateway records).

The reply shapes below are copied literally from
services/orion-llm-gateway/app/main.py (``_overloaded_result`` and
``_dispatch_chat``'s CapacityRejected catch). They are NOT imported: a
cross-service import would be reaching into another service's internals, and
this test exists precisely to pin the wire shape this consumer depends on.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.executor import call_step_services, gateway_error_step_failure
from orion.core.bus.bus_schemas import ChatResponsePayload, ServiceRef
from orion.schemas.cortex.schemas import ExecutionStep

# services/orion-llm-gateway/app/main.py::_dispatch_chat, `except (ResourceLeaseRejected,
# CapacityRejected)` -- the shape behind the live 2026-09-19 corr 5258cae8 failure.
CAPACITY_REJECTED_REPLY = {
    "text": "",
    "content": "",
    "route": "agent",
    "raw": {
        "error": "gateway_capacity_rejected",
        "details": {"reason": "capacity_wait_budget_exhausted"},
    },
}

# services/orion-llm-gateway/app/main.py::_overloaded_result
OVERLOADED_REPLY = {
    "text": "",
    "content": "",
    "spark_meta": {},
    "raw": {
        "error": "gateway_overloaded",
        "details": {
            "stage": "upstream_queue",
            "route": "agent",
            "upstream": "http://circe-worker-2:8000",
            "served_by": "qwen-agent",
            "waited_s": 235.001,
            "budget_s": 235.0,
            "lane": {"inflight": 1, "waiting": 0, "max_inflight": 1},
        },
    },
    "route": "agent",
    "served_by": "qwen-agent",
}


def test_capacity_rejected_reply_is_named_with_its_reason() -> None:
    assert (
        gateway_error_step_failure(CAPACITY_REJECTED_REPLY)
        == "gateway_capacity_rejected:capacity_wait_budget_exhausted"
    )


def test_overloaded_reply_is_named_with_its_stage() -> None:
    assert gateway_error_step_failure(OVERLOADED_REPLY) == "gateway_overloaded:upstream_queue"


def test_error_without_details_still_names_the_error() -> None:
    assert gateway_error_step_failure({"content": "", "raw": {"error": "llm_route_unavailable"}}) == (
        "llm_route_unavailable"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"content": "a real answer", "raw": {}},
        {"content": "", "raw": {}},  # empty but no error flag: not this check's business
        {"content": "answered anyway", "raw": {"error": "gateway_overloaded"}},  # text wins
        {"content": None, "raw": {"error": ""}},
        {"content": "", "raw": {"error": None}},
        {"content": "", "raw": "not-a-dict"},
        {"content": "", "text": "text-only shape", "raw": {"error": "x"}},
        None,
    ],
)
def test_replies_with_text_or_no_error_flag_are_not_failures(payload) -> None:
    assert gateway_error_step_failure(payload) is None


def _stance_step() -> ExecutionStep:
    return ExecutionStep(
        step_name="llm_stance_react",
        verb_name="stance_react",
        services=["LLMGatewayService"],
        order=0,
        prompt_template="{{ raw_user_text }}",
        timeout_ms=60000,
    )


def _ctx() -> dict:
    return {
        "mode": "brain",
        "llm_route": "agent",
        "session_id": "sess-1",
        "raw_user_text": "should I engage with this?",
        "messages": [{"role": "user", "content": "should I engage with this?"}],
    }


def _run_step(reply: ChatResponsePayload):
    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(return_value=reply)):
        return asyncio.run(
            call_step_services(
                bus=MagicMock(),
                source=ServiceRef(name="test"),
                step=_stance_step(),
                ctx=_ctx(),
                correlation_id=str(uuid4()),
            )
        )


def test_stance_step_fails_by_name_on_capacity_rejected_reply() -> None:
    """The whole point: the step result's ``error`` carries the gateway's own
    reason so orion-thought can surface ``stance_react_failed:
    gateway_capacity_rejected:capacity_wait_budget_exhausted`` instead of the
    generic "missing thought payload"."""
    result = _run_step(ChatResponsePayload.model_validate(CAPACITY_REJECTED_REPLY))

    assert result.status == "fail"
    assert result.error == "gateway_capacity_rejected:capacity_wait_budget_exhausted"
    assert result.result["error"]["message"] == result.error
    assert any("gateway_capacity_rejected" in line for line in result.logs)


def test_stance_step_fails_by_name_on_overloaded_reply() -> None:
    result = _run_step(ChatResponsePayload.model_validate(OVERLOADED_REPLY))

    assert result.status == "fail"
    assert result.error == "gateway_overloaded:upstream_queue"


def test_stance_step_with_real_content_still_succeeds() -> None:
    result = _run_step(ChatResponsePayload(content='{"disposition":"proceed"}'))

    assert result.status == "success"
    assert result.error is None

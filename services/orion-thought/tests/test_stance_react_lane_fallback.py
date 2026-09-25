from __future__ import annotations

import pytest

from app.bus_listener import (
    exec_failure_reason,
    execute_stance_react,
    extract_stance_react_payload,
    MISSING_THOUGHT_PAYLOAD,
)
from app.settings import settings
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


def _request(**overrides: object) -> StanceReactRequestV1:
    kwargs = dict(
        correlation_id="corr-1",
        session_id="sess-1",
        user_message="where is our work heading?",
        association=HubAssociationBundleV1(
            correlation_id="corr-1",
            broadcast=None,
            broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "where is our work heading?"},
    )
    kwargs.update(overrides)
    return StanceReactRequestV1(**kwargs)


# --- exec_failure_reason -----------------------------------------------------

def test_exec_failure_reason_reads_plan_level_error() -> None:
    assert exec_failure_reason({"error": "gateway_overloaded:upstream_queue", "steps": []}) == (
        "gateway_overloaded:upstream_queue"
    )


def test_exec_failure_reason_falls_back_to_failed_step() -> None:
    result = {
        "steps": [
            {"status": "success", "error": None},
            {"status": "fail", "error": "gateway_capacity_rejected:capacity_wait_budget_exhausted"},
        ]
    }
    assert exec_failure_reason(result) == "gateway_capacity_rejected:capacity_wait_budget_exhausted"


def test_exec_failure_reason_none_when_nothing_named() -> None:
    assert exec_failure_reason({"steps": [{"status": "success"}]}) is None
    assert exec_failure_reason({}) is None


def test_extract_stance_react_payload_raises_named_reason_over_generic() -> None:
    """The whole point of slice 1 + slice 2's plumbing: a named gateway failure
    must reach the deferred-turn label instead of the generic 'missing thought
    payload' message."""
    result = {
        "status": "fail",
        "error": "gateway_capacity_rejected:capacity_wait_budget_exhausted",
        "steps": [],
    }
    with pytest.raises(ValueError, match="gateway_capacity_rejected:capacity_wait_budget_exhausted"):
        extract_stance_react_payload(result)


def test_extract_stance_react_payload_still_raises_generic_when_nothing_named() -> None:
    with pytest.raises(ValueError, match=MISSING_THOUGHT_PAYLOAD):
        extract_stance_react_payload({"steps": []})


# --- execute_stance_react ---------------------------------------------------

class _FakeClient:
    def __init__(self, results: list) -> None:
        self._results = list(results)
        self.calls: list[dict] = []

    async def execute_plan(self, *, source, req, correlation_id, timeout_sec):
        self.calls.append(
            {
                "correlation_id": correlation_id,
                "timeout_sec": timeout_sec,
                "llm_route": req.context.get("llm_route"),
                "step_timeouts_ms": [s.timeout_ms for s in req.plan.steps],
            }
        )
        outcome = self._results[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["agent", "chat", None])
async def test_one_attempt_on_the_turns_own_route_and_correlation_id(monkeypatch, route) -> None:
    """Placement is orion-gpu-pool's job now: no agent-then-chat retry, no fresh correlation id,
    no route override, the whole budget on the single attempt."""
    monkeypatch.setattr(settings, "stance_react_timeout_sec", 360.0)
    client = _FakeClient([{"final_text": '{"disposition":"proceed"}', "steps": []}])
    result, payload = await execute_stance_react(_request(llm_route=route), client=client)
    assert payload == '{"disposition":"proceed"}'
    assert len(client.calls) == 1
    assert client.calls[0]["correlation_id"] == "corr-1" and client.calls[0]["timeout_sec"] == 360.0
    assert client.calls[0]["llm_route"] == route


@pytest.mark.asyncio
async def test_a_named_failure_is_raised_not_retried_elsewhere(monkeypatch) -> None:
    monkeypatch.setattr(settings, "stance_react_timeout_sec", 360.0)
    client = _FakeClient([{"status": "fail", "error": "gpu_pool_unavailable:deadline", "steps": []}])
    with pytest.raises(ValueError, match="gpu_pool_unavailable:deadline"):
        await execute_stance_react(_request(llm_route="agent"), client=client)
    assert len(client.calls) == 1

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.bus_listener import (
    exec_failure_reason,
    execute_stance_react_with_lane_fallback,
    extract_stance_react_payload,
    lane_fallback_applies,
    MISSING_THOUGHT_PAYLOAD,
)
from app.settings import settings
from orion.schemas.resource_admission import ResourceLeaseV1
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


def _lease(lane: str = "agent") -> ResourceLeaseV1:
    now = datetime.now(timezone.utc)
    return ResourceLeaseV1(
        run_id="run-1", demand_id="run-1:turn", lease_id="lease-1",
        resource_key=f"llm.route.{lane}", lane=lane, backend_key="http://worker:8000",
        generation=7, granted_at=now, heartbeat_at=now,
        expires_at=now + timedelta(seconds=60),
    )


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


# --- lane_fallback_applies ----------------------------------------------------

def test_fallback_applies_for_agent_preference_without_lease() -> None:
    assert lane_fallback_applies(_request(llm_route="agent"), agent_lane_budget_sec=60.0) is True


def test_fallback_does_not_apply_without_agent_preference() -> None:
    assert lane_fallback_applies(_request(llm_route=None), agent_lane_budget_sec=60.0) is False
    assert lane_fallback_applies(_request(llm_route="chat"), agent_lane_budget_sec=60.0) is False


def test_fallback_does_not_apply_with_a_durable_lease() -> None:
    req = _request(llm_route="agent", resource_lease=_lease().model_dump(mode="json"))
    assert lane_fallback_applies(req, agent_lane_budget_sec=60.0) is False


def test_fallback_does_not_apply_when_caller_handles_it() -> None:
    req = _request(llm_route="agent", caller_handles_lane_fallback=True)
    assert lane_fallback_applies(req, agent_lane_budget_sec=60.0) is False


def test_fallback_disabled_by_non_positive_budget() -> None:
    assert lane_fallback_applies(_request(llm_route="agent"), agent_lane_budget_sec=0.0) is False


# --- execute_stance_react_with_lane_fallback ----------------------------------

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
async def test_agent_lane_success_skips_chat_fallback(monkeypatch) -> None:
    monkeypatch.setattr(settings, "stance_react_agent_lane_budget_sec", 60.0)
    monkeypatch.setattr(settings, "stance_react_timeout_sec", 360.0)
    client = _FakeClient([{"final_text": '{"disposition":"proceed"}', "steps": []}])

    result, payload = await execute_stance_react_with_lane_fallback(
        _request(llm_route="agent"), client=client
    )

    assert payload == '{"disposition":"proceed"}'
    assert len(client.calls) == 1
    assert client.calls[0]["correlation_id"] == "corr-1"
    assert client.calls[0]["llm_route"] == "agent"
    assert all(ms <= 60_000 for ms in client.calls[0]["step_timeouts_ms"])


@pytest.mark.asyncio
async def test_agent_lane_named_failure_falls_back_to_chat_on_fresh_correlation_id(monkeypatch) -> None:
    monkeypatch.setattr(settings, "stance_react_agent_lane_budget_sec", 60.0)
    monkeypatch.setattr(settings, "stance_react_timeout_sec", 360.0)
    client = _FakeClient(
        [
            {  # agent attempt: named capacity failure, empty payload
                "status": "fail",
                "error": "gateway_capacity_rejected:capacity_wait_budget_exhausted",
                "steps": [],
            },
            {"final_text": '{"disposition":"proceed"}', "steps": []},  # chat attempt: succeeds
        ]
    )

    result, payload = await execute_stance_react_with_lane_fallback(
        _request(llm_route="agent"), client=client
    )

    assert payload == '{"disposition":"proceed"}'
    assert len(client.calls) == 2
    agent_call, chat_call = client.calls
    assert agent_call["correlation_id"] == "corr-1"
    assert agent_call["llm_route"] == "agent"
    assert chat_call["correlation_id"] != "corr-1"
    assert chat_call["llm_route"] == "chat"


@pytest.mark.asyncio
async def test_both_lanes_failing_names_both_reasons(monkeypatch) -> None:
    monkeypatch.setattr(settings, "stance_react_agent_lane_budget_sec", 60.0)
    monkeypatch.setattr(settings, "stance_react_timeout_sec", 360.0)
    client = _FakeClient(
        [
            {"status": "fail", "error": "gateway_capacity_rejected:capacity_wait_budget_exhausted", "steps": []},
            {"status": "fail", "error": "gateway_overloaded:upstream_queue", "steps": []},
        ]
    )

    with pytest.raises(ValueError) as excinfo:
        await execute_stance_react_with_lane_fallback(_request(llm_route="agent"), client=client)

    message = str(excinfo.value)
    assert "gateway_capacity_rejected:capacity_wait_budget_exhausted" in message
    assert "gateway_overloaded:upstream_queue" in message
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_agent_lane_budget_below_gateway_floor_is_clamped_up(monkeypatch) -> None:
    """Review finding (2026-09-19): a configured budget under cortex-exec's own
    45s gateway-read-timeout floor would make this service give up on its RPC
    before the gateway's shed/serve decision could land. Must clamp, not pass
    a sub-floor cap straight through to the plan request."""
    monkeypatch.setattr(settings, "stance_react_agent_lane_budget_sec", 10.0)
    monkeypatch.setattr(settings, "stance_react_timeout_sec", 360.0)
    client = _FakeClient([{"final_text": '{"disposition":"proceed"}', "steps": []}])

    await execute_stance_react_with_lane_fallback(_request(llm_route="agent"), client=client)

    assert client.calls[0]["step_timeouts_ms"] == [45_000]


@pytest.mark.asyncio
async def test_insufficient_remaining_budget_skips_chat_attempt_entirely(monkeypatch) -> None:
    """Boundary: after the agent attempt fails, if what is left of the total
    budget is already at/under cortex-exec's 45s gateway floor, a chat attempt
    would only generate for a caller that has already given up -- skip it and
    name that explicitly instead of making a doomed second RPC."""
    monkeypatch.setattr(settings, "stance_react_agent_lane_budget_sec", 60.0)
    monkeypatch.setattr(settings, "stance_react_timeout_sec", 40.0)  # < 45s floor once spent
    client = _FakeClient(
        [{"status": "fail", "error": "gateway_capacity_rejected:capacity_wait_budget_exhausted", "steps": []}]
    )

    with pytest.raises(ValueError) as excinfo:
        await execute_stance_react_with_lane_fallback(_request(llm_route="agent"), client=client)

    message = str(excinfo.value)
    assert "gateway_capacity_rejected:capacity_wait_budget_exhausted" in message
    assert "chat=skipped:budget_remaining=" in message
    assert len(client.calls) == 1  # no doomed second RPC


@pytest.mark.asyncio
async def test_chat_attempt_step_cap_floors_at_45s_when_remaining_is_tight(monkeypatch) -> None:
    """Boundary: remaining budget between the 45s gateway floor and the 30s RPC
    margin (i.e. `remaining - margin < floor`) must still floor the chat
    attempt's step cap at 45s rather than passing a smaller number through."""
    monkeypatch.setattr(settings, "stance_react_agent_lane_budget_sec", 20.0)
    monkeypatch.setattr(settings, "stance_react_timeout_sec", 70.0)  # remaining ~70s after agent fails
    client = _FakeClient(
        [
            {"status": "fail", "error": "gateway_capacity_rejected:capacity_wait_budget_exhausted", "steps": []},
            {"final_text": '{"disposition":"proceed"}', "steps": []},
        ]
    )

    result, payload = await execute_stance_react_with_lane_fallback(_request(llm_route="agent"), client=client)

    assert payload == '{"disposition":"proceed"}'
    assert len(client.calls) == 2
    # remaining (~70s) - RPC margin (30s) = 40s, under the 45s floor -> floored.
    assert client.calls[1]["step_timeouts_ms"] == [45_000]


@pytest.mark.asyncio
async def test_leased_request_makes_exactly_one_call_with_no_route_override(monkeypatch) -> None:
    """Admission already owns the lane -- no fallback, no route mutation."""
    monkeypatch.setattr(settings, "stance_react_agent_lane_budget_sec", 60.0)
    monkeypatch.setattr(settings, "stance_react_timeout_sec", 360.0)
    client = _FakeClient([{"final_text": '{"disposition":"proceed"}', "steps": []}])
    req = _request(llm_route="agent", resource_lease=_lease().model_dump(mode="json"))

    result, payload = await execute_stance_react_with_lane_fallback(req, client=client)

    assert payload == '{"disposition":"proceed"}'
    assert len(client.calls) == 1
    assert client.calls[0]["correlation_id"] == "corr-1"
    assert client.calls[0]["timeout_sec"] == 360.0


@pytest.mark.asyncio
async def test_caller_handled_request_makes_exactly_one_call(monkeypatch) -> None:
    """endogenous_outreach's own fallback owns this -- orion-thought must not
    stack a second one on top."""
    monkeypatch.setattr(settings, "stance_react_agent_lane_budget_sec", 60.0)
    monkeypatch.setattr(settings, "stance_react_timeout_sec", 360.0)
    client = _FakeClient([Exception("boom")])
    req = _request(llm_route="agent", caller_handles_lane_fallback=True)

    with pytest.raises(Exception, match="boom"):
        await execute_stance_react_with_lane_fallback(req, client=client)

    assert len(client.calls) == 1

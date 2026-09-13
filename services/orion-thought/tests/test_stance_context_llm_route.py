from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.bus_listener import build_stance_react_context, build_stance_react_plan_request
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


def test_context_without_llm_route_has_no_key() -> None:
    """The common case (every call site before 2026-09-08) must be
    byte-identical to before this field existed -- no new key appears in ctx
    unless a caller actually asked for a route override."""
    ctx = build_stance_react_context(_request())
    assert "llm_route" not in ctx
    assert "llm_lane" not in ctx
    assert "resource_lease" not in ctx


def test_context_with_agent_llm_route_threads_it_through() -> None:
    """Regression for the outreach-agent-lane-precedence gap: cortex-exec's
    _resolve_llm_route_override reads ctx["llm_route"] (or
    ctx["options"]["llm_route"]) -- this asserts build_stance_react_context
    actually surfaces the request's llm_route at the top-level key that
    resolver reads, not just carries it somewhere inert."""
    ctx = build_stance_react_context(_request(llm_route="agent"))
    assert ctx["llm_route"] == "agent"


def test_context_with_none_llm_route_omits_key() -> None:
    """Explicit None (the schema default) must behave the same as omitting
    the field entirely -- an empty-string/None override must never leak in
    as a falsy-but-present ctx["llm_route"] that could confuse a caller
    doing `"llm_route" in ctx`."""
    ctx = build_stance_react_context(_request(llm_route=None))
    assert "llm_route" not in ctx


@pytest.mark.parametrize("lane", ["agent", "chat", "metacog"])
def test_admitted_stance_plan_uses_owning_lease_and_assigned_lane(lane: str) -> None:
    now = datetime.now(timezone.utc)
    lease = ResourceLeaseV1(
        run_id="run-1", demand_id="run-1:turn", lease_id="lease-1",
        resource_key=f"llm.route.{lane}", lane=lane, backend_key="http://worker:8000",
        generation=7, granted_at=now, heartbeat_at=now,
        expires_at=now + timedelta(seconds=60),
    )
    request = _request(llm_route="agent", resource_lease=lease.model_dump(mode="json"))

    plan = build_stance_react_plan_request(request)

    assert isinstance(request.resource_lease, ResourceLeaseV1)
    assert plan.context["resource_lease"] == lease.model_dump(mode="json")
    assert plan.context["llm_route"] == lane
    assert plan.context["llm_lane"] == lane
    assert plan.context["metadata"]["correlation_id"] == request.correlation_id


def test_stance_rejects_invalid_lease_instead_of_dispatching_without_ownership() -> None:
    with pytest.raises(ValidationError):
        _request(resource_lease={"lease_id": "lease-1", "generation": 0})

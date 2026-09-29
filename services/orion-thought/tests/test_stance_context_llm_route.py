from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.bus_listener import build_stance_react_context, build_stance_react_plan_request
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


_REF = {"lease_id": "hold-1", "generation": 3, "role": "agent-gpu2", "holder": "durable-runs:run-1"}


def test_held_stance_carries_the_gpu_lease_ref_on_the_agent_route() -> None:
    """Stage 4.4: stance under a durable run's GPU pool hold forwards the ref (the gateway attaches
    the call to the hold) and names the hold's work-class route, never the role."""
    ctx = build_stance_react_plan_request(_request(llm_route="chat", gpu_lease=_REF)).context
    assert ctx["gpu_lease"] == _REF
    assert ctx["llm_route"] == ctx["llm_lane"] == "agent"
    assert "resource_lease" not in ctx


def test_a_legacy_resource_lease_is_no_longer_part_of_the_contract() -> None:
    """Stage 4.6: the durable token is gone. StanceReactRequestV1 ignores unknown fields, so a stray
    one from an old producer is dropped -- it never reaches cortex-exec's context or picks the lane."""
    assert "resource_lease" not in StanceReactRequestV1.model_fields
    ctx = build_stance_react_context(
        _request(llm_route="chat", resource_lease={"lease_id": "legacy", "lane": "metacog"}, gpu_lease=_REF))
    assert "resource_lease" not in ctx and ctx["llm_route"] == "agent" and ctx["gpu_lease"] == _REF


def test_malformed_ref_is_refused_instead_of_dispatching_unheld() -> None:
    with pytest.raises(ValidationError):
        _request(gpu_lease={"lease_id": "hold-1", "generation": 0, "role": "agent", "holder": "x"})

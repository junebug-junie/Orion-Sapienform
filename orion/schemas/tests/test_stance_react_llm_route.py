from __future__ import annotations

from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


def _request(**overrides: object) -> StanceReactRequestV1:
    kwargs = dict(
        correlation_id="c-1",
        session_id="sess-1",
        user_message="hello",
        association=HubAssociationBundleV1(
            correlation_id="c-1",
            broadcast=None,
            broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "hello"},
    )
    kwargs.update(overrides)
    return StanceReactRequestV1(**kwargs)


def test_llm_route_defaults_to_none() -> None:
    """Back-compat: an existing constructor call that omits llm_route (every
    call site before orion.hub.turn_orchestrator's 2026-09-08 change) still
    validates and carries no route override."""
    req = _request()
    assert req.llm_route is None


def test_llm_route_accepts_agent() -> None:
    req = _request(llm_route="agent")
    assert req.llm_route == "agent"


def test_llm_route_round_trips_through_json() -> None:
    req = _request(llm_route="agent")
    dumped = req.model_dump(mode="json")
    assert dumped["llm_route"] == "agent"
    restored = StanceReactRequestV1.model_validate(dumped)
    assert restored.llm_route == "agent"


def test_llm_route_absent_key_still_parses() -> None:
    """A payload from an OLDER producer that never had this field (e.g. a
    stale worker mid-rollout) must still validate -- the model has no
    extra="forbid", so a missing optional field is not a breaking change in
    either direction."""
    req = _request()
    dumped = req.model_dump(mode="json")
    dumped.pop("llm_route", None)
    restored = StanceReactRequestV1.model_validate(dumped)
    assert restored.llm_route is None

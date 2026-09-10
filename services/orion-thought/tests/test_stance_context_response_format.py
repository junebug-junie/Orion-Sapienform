from __future__ import annotations

from app.bus_listener import build_stance_react_context
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


def test_context_always_forces_json_object_mode() -> None:
    """Regression for the recurring 'stance_react exec result missing thought
    payload' deferred turn (root-caused live 2026-09-10, corr=9c7e9272):
    router.py's _structured_output_expected() treats stance_react as
    JSON-required and discards any reply it can't parse as JSON, but nothing
    was telling the gateway to actually constrain the model to JSON -- so a
    plain-prose (but perfectly good) reply got thrown away whole. This must
    hold for every request, agent-lane override or not."""
    ctx = build_stance_react_context(_request())
    assert ctx["response_format"] == {"type": "json_object"}


def test_context_with_agent_llm_route_still_forces_json_object_mode() -> None:
    ctx = build_stance_react_context(_request(llm_route="agent"))
    assert ctx["response_format"] == {"type": "json_object"}

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


def test_context_turns_thinking_off_on_every_route() -> None:
    """Regression for reading-turn stance timeouts (live 2026-10-14..31): the agent
    lanes think by default, so stance_react there wrote ~3-6k completion tokens
    (chat lane: ~340) and 108 reading turns failed at cortex-exec's 235 s cutoff as
    `turn_deferred:stance_react_failed: timeout:caller_budget_exhausted`. Thinking
    must be off whatever route or hold the request carries."""
    from orion.schemas.gpu_pool import GpuLeaseRefV1

    lease = GpuLeaseRefV1(lease_id="lease-1", generation=1, role="agent-gpu2", holder="reading")
    for request in (
        _request(),
        _request(llm_route="agent"),
        _request(llm_route="chat"),
        _request(gpu_lease=lease),
    ):
        ctx = build_stance_react_context(request)
        assert ctx["chat_template_kwargs"] == {"enable_thinking": False}

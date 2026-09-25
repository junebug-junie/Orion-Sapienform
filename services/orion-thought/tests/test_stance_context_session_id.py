"""The stance step must hand cortex-exec a TOP-LEVEL session_id.

cortex-exec merges a plan request as ``{**context, **args.extra, ...}`` and
reads ``ctx["session_id"]`` -- router.py's ``mark_orion_turn`` (conversation
phase), metacog traces and grammar events all key off it. Thought used to
nest the session only under ``context["metadata"]``, so every unified turn's
stance step recorded Orion's turn under the shared "global" phase key and
emitted its traces with no session.
"""

from __future__ import annotations

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


def test_context_carries_session_id_at_top_level() -> None:
    ctx = build_stance_react_context(_request())
    assert ctx["session_id"] == "sess-1"
    # metadata keeps its copy for anything already reading it there
    assert ctx["metadata"]["session_id"] == "sess-1"


def test_session_id_survives_cortex_exec_context_merge() -> None:
    """Mirrors the merge order in services/orion-cortex-exec/app/main.py:
    context first, then args.extra -- extra must not shadow the session."""
    plan_request = build_stance_react_plan_request(_request())
    ctx = {**plan_request.context, **(plan_request.args.extra or {})}
    assert ctx.get("session_id") == "sess-1"


def test_sessionless_request_stays_sessionless() -> None:
    ctx = build_stance_react_context(_request(session_id=None))
    assert ctx["session_id"] is None

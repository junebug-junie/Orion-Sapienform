"""orion-thought hands the caller's recall search text to cortex-exec as ctx["retrieval_query"].

Phase 3 of the recall retrieval design (2026-09-29). cortex-exec's run_recall_step
reads ctx["retrieval_query"] for both stance recalls (PCR 0+1 and phase 3).
"""

from __future__ import annotations

from app.bus_listener import build_stance_react_context, build_stance_react_plan_request
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


def _request(**overrides: object) -> StanceReactRequestV1:
    kwargs = dict(
        correlation_id="corr-1",
        session_id="sess-1",
        user_message="Investigation claim: not yet chosen.",
        association=HubAssociationBundleV1(
            correlation_id="corr-1",
            broadcast=None,
            broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "Investigation claim: not yet chosen."},
    )
    kwargs.update(overrides)
    return StanceReactRequestV1(**kwargs)


def test_request_retrieval_query_reaches_exec_ctx() -> None:
    ctx = build_stance_react_context(_request(retrieval_query="What am I, when nobody asks?"))
    assert ctx["retrieval_query"] == "What am I, when nobody asks?"
    assert ctx["user_message"] == "Investigation claim: not yet chosen."


def test_survives_cortex_exec_context_merge() -> None:
    plan_request = build_stance_react_plan_request(_request(retrieval_query="q"))
    ctx = {**plan_request.context, **(plan_request.args.extra or {})}
    assert ctx["retrieval_query"] == "q"


def test_stance_inputs_copy_is_used_when_request_field_absent() -> None:
    ctx = build_stance_react_context(
        _request(stance_inputs={"user_message": "m", "retrieval_query": "from inputs"})
    )
    assert ctx["retrieval_query"] == "from inputs"


def test_no_retrieval_query_means_no_ctx_key() -> None:
    ctx = build_stance_react_context(_request())
    assert "retrieval_query" not in ctx


def test_old_producer_payload_still_validates() -> None:
    payload = _request().model_dump(mode="json")
    payload.pop("retrieval_query")
    assert StanceReactRequestV1.model_validate(payload).retrieval_query is None

from __future__ import annotations

from orion.schemas.cortex.schemas import PlanExecutionResult, StepExecutionResult


def _build_trace_metadata(res: PlanExecutionResult, *, req_extra: dict | None = None) -> dict:
    from app.main import build_cognition_trace_metadata

    return build_cognition_trace_metadata(res, req_extra=req_extra or {})


def test_metadata_includes_routing_and_presence_flags() -> None:
    res = PlanExecutionResult(
        verb_name="chat_general",
        status="success",
        mode="brain",
        steps=[
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="collect_metacog_context",
                order=0,
                result={"MetacogContextService": {}},
                latency_ms=10,
            ),
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="synthesize_chat_stance_brief",
                order=1,
                result={"LLMGatewayService": {"stance_brief": "secret"}},
                latency_ms=20,
            ),
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="llm_chat_general",
                order=2,
                result={"LLMGatewayService": {"content": "hello"}},
                latency_ms=30,
            ),
        ],
        final_text="hello",
        memory_used=True,
        metadata={"recall_profile": "chat.general.v1"},
    )
    meta = _build_trace_metadata(
        res,
        req_extra={
            "route_intent": "none",
            "session_id": "sess-1",
            "message_id": "msg-1",
        },
    )
    assert meta["verb"] == "chat_general"
    assert meta["mode"] == "brain"
    assert meta["recall_profile"] == "chat.general.v1"
    assert meta["chat_stance_debug_present"] is True
    assert meta["final_text_present"] is True
    assert meta["canonical_final_step_name"] == "llm_chat_general"
    assert meta["session_id"] == "sess-1"
    assert meta["message_id"] == "msg-1"
    assert "secret" not in str(meta)

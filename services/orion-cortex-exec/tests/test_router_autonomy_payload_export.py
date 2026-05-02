from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app.router import PlanRunner, _autonomy_payload_from_ctx
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionArgs, PlanExecutionRequest, StepExecutionResult


def _request(*, verb_name: str = "chat_general", step_name: str = "llm_chat_general") -> PlanExecutionRequest:
    plan = ExecutionPlan(
        verb_name=verb_name,
        label=verb_name,
        description="",
        category="x",
        priority="normal",
        interruptible=True,
        can_interrupt_others=False,
        timeout_ms=1000,
        max_recursion_depth=1,
        metadata={"mode": "brain"},
        steps=[
            ExecutionStep(
                verb_name=verb_name,
                step_name=step_name,
                description="",
                order=0,
                services=["LLMGatewayService"],
                requires_memory=False,
            )
        ],
    )
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(request_id="r1", extra={"mode": "brain"}),
        context={"mode": "brain", "raw_user_text": "hello"},
    )


def test_router_exports_autonomy_metadata_from_chat_stance_context(monkeypatch) -> None:
    runner = PlanRunner()
    fake_step = StepExecutionResult(
        status="success",
        verb_name="chat_general",
        step_name="llm_chat_general",
        order=0,
        result={"LLMGatewayService": {"content": "ok"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    req = _request()
    ctx = {
        "mode": "brain",
        "raw_user_text": "hello",
        "chat_autonomy_summary": {
            "stance_hint": "favor synthesis and reduction",
            "top_drives": ["coherence"],
            "active_tensions": ["scope_sprawl"],
            "proposal_headlines": ["stabilize triage sequence"],
        },
        "chat_autonomy_debug": {"orion": {"availability": "available", "present": True}},
        "chat_autonomy_state": {
            "subject": "orion",
            "source": "graph",
            "dominant_drive": "coherence",
            "active_drives": ["coherence"],
            "tension_kinds": ["scope_sprawl"],
        },
        "chat_stance_debug": {
            "overview": {"fallback_invoked": False},
            "final_prompt_contract": {"chat_stance_brief": {"task_mode": "direct_response"}},
        },
        "chat_mutation_cognition_context": {
            "mutation_scope": "routing_threshold_patch_only",
            "live_ramp_active": True,
        },
    }
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-a1",
            ctx=ctx,
        )
    )
    assert result.metadata["autonomy_summary"]["stance_hint"] == "favor synthesis and reduction"
    assert result.metadata["autonomy_debug"]["orion"]["availability"] == "available"
    assert result.metadata["autonomy_state_preview"]["dominant_drive"] == "coherence"
    assert result.metadata["autonomy_backend"] == "graph"
    assert result.metadata["autonomy_selected_subject"] == "orion"
    assert result.metadata["chat_stance_debug"]["overview"]["fallback_invoked"] is False
    assert result.metadata["mutation_cognition_context"]["mutation_scope"] == "routing_threshold_patch_only"


def test_router_omits_autonomy_metadata_when_absent(monkeypatch) -> None:
    runner = PlanRunner()
    fake_step = StepExecutionResult(
        status="success",
        verb_name="chat_general",
        step_name="llm_chat_general",
        order=0,
        result={"LLMGatewayService": {"content": "ok"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    req = _request()
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-a2",
            ctx={"mode": "brain", "raw_user_text": "hello"},
        )
    )
    assert "autonomy_summary" not in result.metadata
    assert "autonomy_debug" not in result.metadata
    assert "autonomy_state_preview" not in result.metadata
    assert "chat_stance_debug" not in result.metadata


def test_router_prepares_brain_autonomy_context_for_non_chat_general_verbs(monkeypatch) -> None:
    runner = PlanRunner()
    fake_step = StepExecutionResult(
        status="success",
        verb_name="analyze_text",
        step_name="llm_analyze_text",
        order=0,
        result={"LLMGatewayService": {"content": "ok"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))

    def _fake_prepare(ctx):
        ctx["chat_autonomy_summary"] = {"stance_hint": "stabilize and clarify"}
        ctx["chat_autonomy_debug"] = {"orion": {"availability": "available", "present": True}}
        ctx["chat_autonomy_state"] = {"subject": "orion", "source": "graph"}
        return {"autonomy": {"summary": ctx["chat_autonomy_summary"]}}

    # synchronous helper; keep assertion surface simple
    monkeypatch.setattr("app.router.prepare_brain_reply_context", _fake_prepare)

    req = _request(verb_name="analyze_text", step_name="llm_analyze_text")
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-a3",
            ctx={"mode": "brain", "raw_user_text": "hello"},
        )
    )
    assert result.verb_name == "analyze_text"
    assert result.metadata["autonomy_summary"]["stance_hint"] == "stabilize and clarify"
    assert result.metadata["autonomy_debug"]["orion"]["availability"] == "available"


def test_router_exports_backend_and_repository_status_when_autonomy_unavailable(monkeypatch) -> None:
    runner = PlanRunner()
    fake_step = StepExecutionResult(
        status="success",
        verb_name="chat_general",
        step_name="llm_chat_general",
        order=0,
        result={"LLMGatewayService": {"content": "ok"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    req = _request()
    ctx = {
        "mode": "brain",
        "raw_user_text": "hello",
        "chat_autonomy_summary": {"stance_hint": "stay stable"},
        "chat_autonomy_debug": {
            "orion": {"availability": "unavailable", "present": False, "unavailable_reason": "query_error"},
            "_runtime": {"backend": "graph", "selected_subject": None, "repository_status": {"source_available": True, "source_path": "http://graphdb/repositories/collapse"}},
        },
        "chat_autonomy_backend": "graph",
        "chat_autonomy_selected_subject": None,
        "chat_autonomy_repository_status": {"backend": "graph", "source_path": "http://graphdb/repositories/collapse", "source_available": True},
    }
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-a4",
            ctx=ctx,
        )
    )
    assert result.metadata["autonomy_backend"] == "graph"
    assert result.metadata["autonomy_selected_subject"] is None
    assert result.metadata["autonomy_repository_status"]["source_available"] is True
    assert result.metadata["autonomy_debug"]["orion"]["unavailable_reason"] == "query_error"


def test_router_exports_goal_lineage_and_proposal_only_execution_mode(monkeypatch) -> None:
    runner = PlanRunner()
    fake_step = StepExecutionResult(
        status="success",
        verb_name="chat_general",
        step_name="llm_chat_general",
        order=0,
        result={"LLMGatewayService": {"content": "ok"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    req = _request()
    ctx = {
        "mode": "brain",
        "raw_user_text": "hello",
        "chat_autonomy_summary": {"stance_hint": "x", "proposal_headlines": ["consider boundaries"]},
        "chat_autonomy_state": {
            "subject": "orion",
            "source": "graph",
            "goal_headlines": [
                {"artifact_id": "goal-abc", "proposal_signature": "deadbeef01", "goal_statement": "y"},
            ],
        },
    }
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-lineage",
            ctx=ctx,
        )
    )
    assert result.metadata["autonomy_execution_mode"] == "proposal_only"
    assert result.metadata["autonomy_goal_lineage"]["goal_artifact_id"] == "goal-abc"
    assert result.metadata["autonomy_goal_lineage"]["proposal_signature"] == "deadbeef01"
    assert result.metadata["autonomy_state_preview"]["goal_lineage"]["proposal_signature"] == "deadbeef01"


def test_router_autonomy_preview_dominant_drive_falls_back_to_top_drive(monkeypatch) -> None:
    runner = PlanRunner()
    fake_step = StepExecutionResult(
        status="success",
        verb_name="chat_general",
        step_name="llm_chat_general",
        order=0,
        result={"LLMGatewayService": {"content": "ok"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    req = _request()
    ctx = {
        "mode": "brain",
        "raw_user_text": "hello",
        "chat_autonomy_summary": {"top_drives": ["relational", "predictive", "continuity"]},
        "chat_autonomy_state": {"dominant_drive": None, "active_drives": ["coherence"]},
    }
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-a5",
            ctx=ctx,
        )
    )
    assert result.metadata["autonomy_state_preview"]["dominant_drive"] == "relational"


def test_router_autonomy_preview_dominant_drive_falls_back_to_active_drive(monkeypatch) -> None:
    runner = PlanRunner()
    fake_step = StepExecutionResult(
        status="success",
        verb_name="chat_general",
        step_name="llm_chat_general",
        order=0,
        result={"LLMGatewayService": {"content": "ok"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    req = _request()
    ctx = {
        "mode": "brain",
        "raw_user_text": "hello",
        "chat_autonomy_summary": {},
        "chat_autonomy_state": {"dominant_drive": None, "active_drives": ["continuity"]},
    }
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-a6",
            ctx=ctx,
        )
    )
    assert result.metadata["autonomy_state_preview"]["dominant_drive"] == "continuity"


def test_autonomy_payload_includes_v2_preview_and_delta() -> None:
    ctx = {
        "chat_autonomy_state_v2": {
            "schema_version": "autonomy.state.v2",
            "dominant_drive": "coherence",
            "active_drives": ["coherence", "continuity", "relational"],
            "confidence": 0.5,
            "unknowns": ["a", "b", "c", "d", "e", "f"],
            "attention_items": [{"summary": "x"}, {"summary": "y"}],
            "inhibited_impulses": [{"inhibition_reason": "proxy_signal_not_canonical_state"}],
        },
        "chat_autonomy_state_delta": {"subject": "orion", "confidence_delta": 0.1},
    }
    md = _autonomy_payload_from_ctx(ctx)
    assert md["autonomy_state_v2_preview"]["dominant_drive"] == "coherence"
    assert len(md["autonomy_state_v2_preview"]["active_drives"]) <= 3
    assert md["autonomy_state_delta"]["confidence_delta"] == 0.1


def test_autonomy_payload_includes_turn_effect_when_present() -> None:
    md = _autonomy_payload_from_ctx(
        {
            "turn_effect": {"turn": {"coherence": -0.15, "novelty": 0.05}},
            "turn_effect_evidence": {"phi_before": {"coherence": 0.5}},
        }
    )
    assert md["turn_effect"]["turn"]["coherence"] == -0.15
    assert md["turn_effect_evidence"]["phi_before"]["coherence"] == 0.5
    assert md["turn_effect_status"] == "present"


def test_autonomy_payload_omits_empty_turn_effect() -> None:
    md = _autonomy_payload_from_ctx({"turn_effect": {}, "turn_effect_evidence": {}, "biometrics": {"reason": "no_state_reply"}})
    assert "turn_effect" not in md
    assert "turn_effect_evidence" not in md
    assert md["turn_effect_status"] == "missing"
    assert md["turn_effect_missing_reason"] == "no_state_reply"


def test_router_exports_turn_effect_in_plan_metadata(monkeypatch) -> None:
    runner = PlanRunner()
    fake_step = StepExecutionResult(
        status="success",
        verb_name="chat_general",
        step_name="llm_chat_general",
        order=0,
        result={"LLMGatewayService": {"content": "ok"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    req = _request()
    ctx = {
        "mode": "brain",
        "raw_user_text": "hello",
        "turn_effect": {"turn": {"novelty": 0.2}},
        "turn_effect_evidence": {"note": "x"},
    }
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-turn-effect",
            ctx=ctx,
        )
    )
    assert result.metadata["turn_effect"]["turn"]["novelty"] == 0.2
    assert result.metadata["turn_effect_evidence"]["note"] == "x"
    assert result.metadata["turn_effect_status"] == "present"


def test_router_exports_turn_effect_status_for_chat_quick_when_missing(monkeypatch) -> None:
    runner = PlanRunner()
    fake_step = StepExecutionResult(
        status="success",
        verb_name="chat_quick",
        step_name="llm_chat_quick",
        order=1,
        result={"LLMGatewayService": {"content": "ok"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    req = _request(verb_name="chat_quick", step_name="llm_chat_quick")
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-turn-effect-missing",
            ctx={"mode": "brain", "raw_user_text": "hello", "biometrics": {"reason": "no_state_reply"}},
        )
    )
    assert result.metadata["turn_effect_status"] == "missing"
    assert result.metadata["turn_effect_missing_reason"] == "no_state_reply"

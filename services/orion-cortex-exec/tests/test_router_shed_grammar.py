"""PlanRunner keeps a deliberate GPU-pool shed visible on the grammar atoms (2026-10-20).

Live 10-20: cabinet_hot sheds reached the execution reducer as plain gpu_pool_unavailable
step failures (short_error_kind cut the reason), so every shed read as failure_pressure=1.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.executor import gateway_error_step_failure
from app.router import PlanRunner
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanExecutionArgs,
    PlanExecutionRequest,
    StepExecutionResult,
)

# The shape orion-llm-gateway's _pool_unavailable_result() returns for a shed request
# (services/orion-llm-gateway/app/main.py): empty text, raw.error + details.reason only.
POOL_SHED_REPLY = {
    "text": "",
    "content": "",
    "spark_meta": {},
    "raw": {
        "error": "gpu_pool_unavailable",
        "details": {"reason": "shed:cabinet_hot", "route": "background", "work_class": "background"},
    },
    "route": "background",
    "served_by": None,
}


def test_gateway_shed_reply_names_the_shed_in_the_step_error():
    """Pins the producer side: if the gateway ever adds details.stage (which wins over
    reason), the shed reason would vanish silently -- this test would catch it."""
    assert gateway_error_step_failure(POOL_SHED_REPLY) == "gpu_pool_unavailable:shed:cabinet_hot"


def _req() -> PlanExecutionRequest:
    plan = ExecutionPlan(
        verb_name="chat_general", label="x", description="", category="x", priority="normal",
        interruptible=True, can_interrupt_others=False, timeout_ms=1000, max_recursion_depth=1,
        steps=[ExecutionStep(verb_name="chat_general", step_name="draft_journal_entry",
                             order=0, services=["LLMGatewayService"])],
        metadata={"execution_depth": "0", "mode": "brain"},
    )
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(request_id="r", extra={"mode": "brain", "recall": {"enabled": False}}),
        context={"mode": "brain"},
    )


@pytest.mark.parametrize(
    "error, shed",
    [("gpu_pool_unavailable:shed:cabinet_hot", "cabinet_hot"), ("gpu_pool_unavailable:deadline", None)],
)
def test_run_plan_marks_shed_step_and_shed_result(monkeypatch, error, shed):
    step = StepExecutionResult(
        status="fail", verb_name="chat_general", step_name="draft_journal_entry", order=0,
        result={}, latency_ms=1, node="n", logs=[], error=error,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=step))
    assembled: list[dict] = []
    import app.router as router

    real = router.record_assembled_grammar
    monkeypatch.setattr(
        router, "record_assembled_grammar",
        lambda ctx, **kw: (assembled.append(kw), real(ctx, **kw))[1],
    )
    ctx = {"mode": "brain"}
    result = asyncio.run(PlanRunner().run_plan(
        MagicMock(), source=ServiceRef(name="x", version="0", node="n"), req=_req(),
        correlation_id="c-shed", ctx=ctx,
    ))
    assert result.status != "success"
    assert result.error == error
    assert assembled and assembled[-1]["shed_reason"] == shed
    collector = ctx.get("_cortex_exec_grammar_collector")
    assert collector is not None
    from app.grammar_emit import build_cortex_exec_grammar_events

    summaries = {e.atom.semantic_role: e.atom.summary
                 for e in build_cortex_exec_grammar_events(collector) if e.atom}
    assert ("shed_reason=cabinet_hot" in summaries["exec_step_failed"]) is (shed is not None)
    assert ("shed_reason=cabinet_hot" in summaries["exec_result_assembled"]) is (shed is not None)

import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.router import PlanRunner
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionArgs, PlanExecutionRequest, StepExecutionResult


def _req(depth: int, mode: str, verb: str):
    plan = ExecutionPlan(
        verb_name=verb,
        label=verb,
        description="",
        category="x",
        priority="normal",
        interruptible=True,
        can_interrupt_others=False,
        timeout_ms=1000,
        max_recursion_depth=1,
        steps=[ExecutionStep(verb_name=verb, step_name="s1", order=0, services=["LLMGatewayService"])],
        metadata={"execution_depth": str(depth), "mode": mode},
    )
    return PlanExecutionRequest(plan=plan, args=PlanExecutionArgs(request_id="r", extra={"mode": mode, "recall": {"enabled": False}}), context={"mode": mode})


def test_depth0_and_depth1_skip_supervisor(monkeypatch):
    runner = PlanRunner()
    fake_step = StepExecutionResult(status="success", verb_name="chat_general", step_name="s1", order=0, result={}, latency_ms=1, node="n", logs=[], error=None)
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    sup = AsyncMock()
    monkeypatch.setattr("app.router.Supervisor", lambda *_a, **_k: MagicMock(execute=sup))

    for depth in (0, 1):
        req = _req(depth, "brain", "chat_general" if depth == 0 else "analyze_text")
        asyncio.run(runner.run_plan(MagicMock(), source=ServiceRef(name="x", version="0", node="n"), req=req, correlation_id=f"c{depth}", ctx={"mode": "brain"}))

    assert sup.await_count == 0


def test_depth2_uses_supervisor(monkeypatch):
    runner = PlanRunner()
    sup = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr("app.router.Supervisor", lambda *_a, **_k: MagicMock(execute=sup))
    req = _req(2, "agent", "agent_runtime")
    asyncio.run(runner.run_plan(MagicMock(), source=ServiceRef(name="x", version="0", node="n"), req=req, correlation_id="c2", ctx={"mode": "agent"}))
    assert sup.await_count == 1

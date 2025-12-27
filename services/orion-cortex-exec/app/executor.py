"""Cortex Exec executor.

This service is the *runtime* for plan steps. It does not decide what to do
(next-step logic is Cortex Orch). Instead, it:

- fans out an "exec_step" to one or more downstream services
- waits for "exec_step_result" replies
- returns aggregated results to the caller

The caller can be:
- Cortex Orch via bus RPC (preferred)
- the HTTP /execute endpoint (debug / legacy)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Tuple

from orion.core.bus.rpc_async import collect_replies
from orion.core.bus.service_async import OrionBusAsync

from .models import ExecutionPlan, PlanExecutionRequest, PlanExecutionResult, StepExecutionResult
from .settings import settings

logger = logging.getLogger("orion-cortex-exec")


def _ms() -> float:
    return time.monotonic() * 1000.0


class StepExecutor:
    """Executes steps by fanning out to downstream exec-target services."""

    def __init__(self, bus: OrionBusAsync):
        self.bus = bus

    async def execute_step_calls(
        self,
        *,
        trace_id: str,
        verb_name: str,
        step_name: str,
        order: int,
        origin_node: str,
        calls: List[Dict[str, Any]],
        timeout_ms: int | None = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Execute one step across N services.

        `calls` is a list of:
          {"service": "LLMGatewayService", "payload": {...}}

        Returns: (raw_exec_step_result_messages, elapsed_ms)
        """

        if not self.bus.enabled:
            raise RuntimeError("ORION_BUS_ENABLED is false")

        if not trace_id:
            raise ValueError("trace_id required")

        timeout_ms = int(timeout_ms or settings.step_timeout_ms)
        timeout_sec = max(0.2, timeout_ms / 1000.0)

        result_channel = f"{settings.exec_result_prefix}:{trace_id}"

        pubsub = self.bus.pubsub()
        await pubsub.subscribe(result_channel)

        started = _ms()
        try:
            # Fan-out
            for call in calls:
                svc = str(call.get("service") or "").strip()
                if not svc:
                    continue

                payload = call.get("payload") or {}
                request_channel = f"{settings.exec_request_prefix}:{svc}"

                envelope = {
                    "event": "exec_step",
                    "service": svc,
                    "correlation_id": trace_id,
                    "trace_id": trace_id,
                    "reply_channel": result_channel,
                    "payload": payload,
                }

                await self.bus.publish(request_channel, envelope)

            expected = max(0, len([c for c in calls if c.get("service")]))
            if expected == 0:
                return [], int(_ms() - started)

            def match(msg: Dict[str, Any]) -> bool:
                # tolerate legacy variants
                return (
                    str(msg.get("correlation_id") or msg.get("trace_id") or "") == trace_id
                    and str(msg.get("event") or "") in {"exec_step_result", "exec_result"}
                )

            replies = await collect_replies(
                pubsub,
                expected_count=expected,
                timeout_sec=timeout_sec,
                match=match,
            )
            elapsed = int(_ms() - started)

            # Telemetry (best-effort)
            try:
                await self.bus.publish(
                    settings.cortex_log_channel,
                    {
                        "event": "cortex.exec.step",
                        "trace_id": trace_id,
                        "verb": verb_name,
                        "step": step_name,
                        "order": order,
                        "origin_node": origin_node,
                        "elapsed_ms": elapsed,
                        "services": [c.get("service") for c in calls],
                        "ok": all(bool(r.get("payload", {}).get("ok", True)) for r in replies) if replies else True,
                    },
                )
            except Exception:
                pass

            return replies, elapsed
        finally:
            try:
                await pubsub.close()
            except Exception:
                pass

    async def execute_plan(self, request: PlanExecutionRequest) -> PlanExecutionResult:
        """Legacy/debug: execute a full ExecutionPlan sequentially."""

        plan: ExecutionPlan = request.plan

        if plan.blocked:
            return PlanExecutionResult(
                verb_name=plan.verb_name,
                request_id=request.args.request_id,
                status="fail",
                blocked=True,
                blocked_reason=plan.blocked_reason,
                steps=[],
            )

        prior_step_results: Dict[str, Any] = {}
        step_results: List[StepExecutionResult] = []

        for step in plan.steps:
            trace_id = request.args.request_id or f"exec-{int(time.time())}-{step.step_name}"
            calls: List[Dict[str, Any]] = []

            for svc in step.services:
                calls.append(
                    {
                        "service": svc,
                        "payload": {
                            "verb": step.verb_name,
                            "step": step.step_name,
                            "order": step.order,
                            "service": svc,
                            "origin_node": settings.node_name,
                            "prompt": None,
                            "prompt_template": step.prompt_template,
                            "context": request.context,
                            "args": request.args.model_dump(mode="json"),
                            "prior_step_results": prior_step_results,
                        },
                    }
                )

            replies, elapsed_ms = await self.execute_step_calls(
                trace_id=trace_id,
                verb_name=step.verb_name,
                step_name=step.step_name,
                order=step.order,
                origin_node=settings.node_name,
                calls=calls,
                timeout_ms=plan.timeout_ms,
            )

            ok = any(bool(r.get("payload", {}).get("ok")) for r in replies) if replies else False
            step_result = StepExecutionResult(
                status="success" if ok else "fail",
                verb_name=step.verb_name,
                step_name=step.step_name,
                order=step.order,
                result={"replies": replies},
                latency_ms=elapsed_ms,
                node=settings.node_name,
                error=None if ok else "no successful replies",
            )

            step_results.append(step_result)
            prior_step_results[step.step_name] = step_result.model_dump(mode="json")

        overall = "success" if all(s.status == "success" for s in step_results) else "partial"
        return PlanExecutionResult(
            verb_name=plan.verb_name,
            request_id=request.args.request_id,
            status=overall,
            steps=step_results,
        )

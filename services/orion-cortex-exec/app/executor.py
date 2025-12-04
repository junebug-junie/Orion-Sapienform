# orion-cortex-exec/app/executor.py

import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional

from orion.core.bus.service import OrionBus
from .settings import settings
from .models import ExecutionStep, StepExecutionResult
from .service_registry import resolve_service

logger = logging.getLogger("orion-cortex.executor")

bus = OrionBus()

class StepExecutor:
    def __init__(self, node_name: Optional[str] = None):
        self.node_name = node_name or settings.NODE_NAME

    async def execute_step(
        self,
        step: ExecutionStep,
        args: Dict[str, Any],
        context: Dict[str, Any],
    ) -> StepExecutionResult:

        start = time.monotonic()
        logs: List[str] = []

        correlation_id = str(uuid.uuid4())
        # RESULT PREFIX: orion-exec:result:<cid>
        reply_channel = f"{settings.EXEC_RESULT_PREFIX}:{correlation_id}"

        # Resolve semantic service aliases (e.g. "llm.brain") -> concrete bus services (e.g. "LLMGatewayService")
        target_services: List[str] = []
        for svc_alias in step.services:
            try:
                svc_name = resolve_service(svc_alias)  # may be identity if already concrete
                target_services.append(svc_name)
            except Exception as e:
                logs.append(f"Service alias '{svc_alias}' could not be resolved: {e}")

        if not target_services:
            latency_ms = int((time.monotonic() - start) * 1000)
            return StepExecutionResult(
                status="fail",
                verb_name=step.verb_name,
                step_name=step.step_name,
                order=step.order,
                result={},
                artifacts={},
                latency_ms=latency_ms,
                node=self.node_name,
                logs=logs + ["No resolvable services for this step"],
                error="No resolvable services for step",
            )

        # Base payload shared across services
        base_payload = {
            "event": "exec_step",  # 🔥 brain expects this
            "verb": step.verb_name,
            "step": step.step_name,
            "order": step.order,
            "requires_gpu": step.requires_gpu,
            "requires_memory": step.requires_memory,
            "prompt_template": step.prompt_template,
            "args": args,
            "context": context,
            "correlation_id": correlation_id,
            "reply_channel": reply_channel,
            "origin_node": self.node_name,
        }

        # Publish to each resolved service
        for service in target_services:
            payload = dict(base_payload)
            payload["service"] = service
            # REQUEST PREFIX: orion-exec:request:<ServiceName>
            channel = f"{settings.EXEC_REQUEST_PREFIX}:{service}"
            bus.publish(channel, payload)
            logs.append(f"Published exec_step to {channel}")

        # Listen for results on reply_channel, aggregating per service
        received: Dict[str, Dict[str, Any]] = {}
        timeout_s = settings.STEP_TIMEOUT_MS / 1000.0
        deadline = start + timeout_s

        pubsub = bus.client.pubsub()
        pubsub.subscribe(reply_channel)
        logs.append(
            f"Subscribed {reply_channel}; waiting for {len(target_services)} result(s). "
            f"Timeout={settings.STEP_TIMEOUT_MS}ms"
        )

        try:
            while time.monotonic() < deadline and len(received) < len(target_services):
                msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if not msg:
                    continue
                if msg.get("type") != "message":
                    continue

                try:
                    data = json.loads(msg["data"])
                except Exception as e:
                    logs.append(f"JSON parse error on reply: {e}")
                    continue

                if data.get("correlation_id") != correlation_id:
                    continue

                # Only accept proper exec_step_result envelopes
                if data.get("event") != "exec_step_result":
                    continue

                svc = data.get("service") or "unknown"
                received[svc] = data
        finally:
            pubsub.close()

        latency_ms = int((time.monotonic() - start) * 1000)

        # Compose final result
        if not received:
            return StepExecutionResult(
                status="fail",
                verb_name=step.verb_name,
                step_name=step.step_name,
                order=step.order,
                result={},
                artifacts={},
                latency_ms=latency_ms,
                node=self.node_name,
                logs=logs + [f"Timeout; no responses on {reply_channel}"],
                error=f"Timeout after {settings.STEP_TIMEOUT_MS} ms with 0/{len(target_services)} responses",
            )

        # Decide overall status: success only if all responded success
        statuses = [r.get("status", "success") for r in received.values()]
        overall = "success" if all(s == "success" for s in statuses) else "partial"

        # Aggregate result & artifacts per service
        aggregated_result = {svc: r.get("result", {}) for svc, r in received.items()}

        aggregated_artifacts: Dict[str, Any] = {}
        for svc, r in received.items():
            artifacts = r.get("artifacts") or {}
            for key, value in artifacts.items():
                # naive merge; if collisions, namespace by service
                if key in aggregated_artifacts and aggregated_artifacts[key] != value:
                    aggregated_artifacts[f"{svc}.{key}"] = value
                else:
                    aggregated_artifacts[key] = value

        logs.append(f"Collected {len(received)}/{len(target_services)} responses in {latency_ms} ms.")

        # emit step-level cognition summary on the bus
        try:
            # Trim outputs a bit to avoid spam huge texts
            result_preview = {}
            for svc, payload in received.items():
                svc_result = payload.get("result") or {}
                llm_out = svc_result.get("llm_output", "")
                if isinstance(llm_out, str):
                    llm_out = llm_out[:512]  # keep first 512 chars
                result_preview[svc] = {
                    "status": payload.get("status", "success"),
                    "llm_output": llm_out,
                }

            summary = {
                "event": "cortex_step_summary",
                "layer": "step",
                "correlation_id": correlation_id,
                "verb": step.verb_name,
                "step": step.step_name,
                "services": list(received.keys()),
                "status": overall,
                "latency_ms": latency_ms,
                "node": self.node_name,
                "args": args,
                "context": context,
                "result_preview": result_preview,
                "timestamp": time.time(),
            }

            bus.publish(settings.CORTEX_LOG_CHANNEL, summary)
        except Exception as e:
            # Don't fail the step if logging fails; just record the error in logs.
            logs.append(f"Failed to publish cortex_step_summary: {e}")


        return StepExecutionResult(
            status=overall,
            verb_name=step.verb_name,
            step_name=step.step_name,
            order=step.order,
            result=aggregated_result,
            artifacts=aggregated_artifacts,
            latency_ms=latency_ms,
            node=self.node_name,
            logs=logs,
            error=None if overall == "success" else "one or more services failed or timed out",
        )

    def _fail(
        self,
        step: ExecutionStep,
        logs: List[str],
        error_msg: str,
        start: float,
    ) -> StepExecutionResult:
        latency = int((time.monotonic() - start) * 1000)
        return StepExecutionResult(
            status="fail",
            verb_name=step.verb_name,
            step_name=step.step_name,
            order=step.order,
            result={},
            artifacts={},
            latency_ms=latency,
            node=self.node_name,
            logs=logs,
            error=error_msg,
        )

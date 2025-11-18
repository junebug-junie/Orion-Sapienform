# orion-cortex-exec/app/executor.py

import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional

from .bus import bus
from .settings import settings
from .models import ExecutionStep, StepExecutionResult

logger = logging.getLogger("orion-cortex.executor")


class StepExecutor:
    def __init__(self, node_name: Optional[str] = None):
        self.node_name = node_name or settings.ORION_NODE_NAME

    async def execute_step(
        self,
        step: ExecutionStep,
        args: Dict[str, Any],
        context: Dict[str, Any],
    ) -> StepExecutionResult:

        start = time.monotonic()
        logs: List[str] = []

        correlation_id = str(uuid.uuid4())
        reply_channel = f"{settings.EXEC_RESULT_PREFIX}.{correlation_id}"

        base_payload = {
            "event": "exec_step_request",
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

        # Publish to each required service
        for service in step.services:
            payload = dict(base_payload)
            payload["service"] = service
            channel = f"{settings.EXEC_REQUEST_PREFIX}.{service}"
            bus.publish(channel, payload)
            logs.append(f"Published to {channel}")

        # Listen for results on reply_channel, aggregating per service
        received: Dict[str, Dict[str, Any]] = {}
        timeout_s = settings.STEP_TIMEOUT_MS / 1000.0
        deadline = start + timeout_s

        pubsub = bus.client.pubsub()
        pubsub.subscribe(reply_channel)
        logs.append(f"Subscribed {reply_channel}; waiting for {len(step.services)} result(s).")

        try:
            while time.monotonic() < deadline and len(received) < len(step.services):
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
                error=f"Timeout after {settings.STEP_TIMEOUT_MS} ms with 0/{len(step.services)} responses",
            )

        # Decide overall status: success only if all responded success
        statuses = [r.get("status", "success") for r in received.values()]
        overall = "success" if all(s == "success" for s in statuses) else ("partial" if received else "fail")

        aggregated_result = {svc: r.get("result", {}) for svc, r in received.items()}
        aggregated_artifacts = {}
        for r in received.values():
            aggregated_artifacts.update(r.get("artifacts", {}))

        logs.append(f"Collected {len(received)}/{len(step.services)} responses in {latency_ms} ms.")

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
        latency = int((time.perf_counter() - start) * 1000)
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

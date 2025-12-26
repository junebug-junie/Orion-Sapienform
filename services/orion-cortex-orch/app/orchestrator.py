# ... existing code ...
import uuid
import time
import logging
import json
from typing import List, Dict, Any

from orion.core.bus.schemas import BaseEnvelope
# We assume settings, etc. are imported or available in the module scope
# This replacement focuses on run_cortex_verb

def run_cortex_verb(bus, req: OrchestrateVerbRequest) -> OrchestrateVerbResponse:
    """
    Execute a high-level "verb" as a sequence of Cortex steps.

    For each step:
      - Build a prompt (Jinja header + JSON footer)
      - Fan-out to {EXEC_REQUEST_PREFIX}:{service}
      - Fan-in from {EXEC_RESULT_PREFIX}:{exec_id}
      - Accumulate StepExecutionResult objects
    """
    # Determine step config list: caller-provided or loaded from cognition/verbs/<verb>.yaml
    if req.steps and len(req.steps) > 0:
        steps: List[CortexStepConfig] = list(req.steps)
    else:
        steps = _load_verb_steps_for_name(req.verb_name)

    # Ensure each step has a verb_name
    for s in steps:
        if not s.verb_name:
            s.verb_name = req.verb_name

    ordered_steps = sorted(steps, key=lambda s: s.order)
    timeout_ms = req.timeout_ms or settings.cortex_step_timeout_ms

    prior_step_results: List[StepExecutionResult] = []
    final_step_results: List[StepExecutionResult] = []

    for step in ordered_steps:
        # One execution id per step, shared across all services for that step
        exec_id = str(uuid.uuid4())
        result_channel = f"{settings.exec_result_prefix}:{exec_id}"

        if not step.services or len(step.services) == 0:
            raise ValueError(
                f"Step has no services: verb={req.verb_name} step={step.step_name} order={step.order}"
            )

        expected = len(step.services)

        # ─────────────────────────────────────────────
        # Fan-out: publish exec_step to each service
        # ─────────────────────────────────────────────
        for service in step.services:
            prompt = _build_prompt(
                step=step,
                service=service,
                origin_node=req.origin_node,
                context=req.context,
                prior_results=prior_step_results,
            )

            exec_channel = f"{settings.exec_request_prefix}:{service}"

            # REFACTORED: Use BaseEnvelope via dict construction or direct instantiation
            # Since 'bus' here is the legacy sync bus (OrionBus), we manually jsonify the envelope model.
            
            payload = {
                "verb": step.verb_name or req.verb_name,
                "step": step.step_name,
                "order": step.order,
                "service": service,
                "origin_node": req.origin_node,
                "prompt_template": step.prompt_template,
                "context": req.context,
                "args": {},  # reserved for richer argument passing later
                "prior_step_results": [
                    r.model_dump(mode="json") for r in prior_step_results
                ],
                "requires_gpu": step.requires_gpu,
                "requires_memory": step.requires_memory,
                "prompt": prompt,
            }

            # Create Envelope
            envelope = BaseEnvelope(
                event="exec_step",
                source="orion-cortex-orch", # Explicit source
                correlation_id=exec_id,
                reply_channel=result_channel,
                payload=payload
            )
            # Add causality (trace_id is same as correlation_id here)
            envelope.add_causality("orion-cortex-orch", "exec_step_publish")

            logger.info(
                "Published exec_step to %s (exec_id=%s, verb=%s, step=%s, service=%s)",
                exec_channel,
                exec_id,
                step.verb_name,
                step.step_name,
                service,
            )
            
            # Legacy bus.publish takes a dict and does json.dumps. 
            # BaseEnvelope.model_dump(mode='json') gives a dict.
            bus.publish(exec_channel, envelope.model_dump(mode='json'))

        # ─────────────────────────────────────────────
        # Fan-in: collect results for this step
        # ─────────────────────────────────────────────
        raw_results = _wait_for_exec_results(
            bus,
            result_channel,
            exec_id,      # positional: avoids keyword signature mismatches
            expected,
            timeout_ms,
        )

        step_results: List[ServiceStepResult] = []
        for raw in raw_results:
            service_name = raw.get("service") or raw.get("service_name") or "unknown"
            elapsed_ms = int(raw.get("elapsed_ms") or raw.get("elapsed") or 0)
            ok = bool(raw.get("ok", True))

            payload = {
                k: v
                for k, v in raw.items()
                if k
                not in {
                    "trace_id",
                    "correlation_id",
                    "service",
                    "service_name",
                    "elapsed_ms",
                    "elapsed",
                    "ok",
                }
            }

            step_results.append(
                ServiceStepResult(
                    service=service_name,
                    trace_id=exec_id,
                    ok=ok,
                    elapsed_ms=elapsed_ms,
                    payload=payload,
                )
            )

        prompt_preview = (
            step.prompt_template.strip()[:200].replace("\n", " ")
            if step.prompt_template
            else ""
        )

        step_execution = StepExecutionResult(
            verb_name=step.verb_name,
            step_name=step.step_name,
            order=step.order,
            services=step_results,
            prompt_preview=prompt_preview,
        )

        final_step_results.append(step_execution)
        prior_step_results.append(step_execution)

    return OrchestrateVerbResponse(
        verb_name=req.verb_name,
        origin_node=req.origin_node,
        steps_executed=len(final_step_results),
        step_results=final_step_results,
        context_echo=req.context,
    )

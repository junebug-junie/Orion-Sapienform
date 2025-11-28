import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .settings import get_settings

logger = logging.getLogger("orion-cortex-orchestrator")
settings = get_settings()


class CortexStepConfig(BaseModel):
    """
    Configuration for a single Cortex "step" in a verb.

    Matches NCIO/semantic schema:
    - verb_name: required
    - step_name: required
    - description: required
    - order: required
    - services: required
    - prompt_template: required
    - requires_gpu: optional (default False)
    - requires_memory: optional (default False)
    """

    verb_name: str = Field(..., description="High-level verb this step belongs to.")
    step_name: str = Field(..., description="Human-readable name for this step.")
    description: str = Field(..., description="What this step is trying to accomplish.")
    order: int = Field(..., description="Execution order within the verb.")
    services: list[str] = Field(
        ..., description="One or more cognitive services to receive this step."
    )
    prompt_template: str = Field(
        ..., description="Base instruction text; orchestrator appends context."
    )

    requires_gpu: bool = Field(
        False,
        description="If True, this step should be routed to a GPU-capable service.",
    )
    requires_memory: bool = Field(
        False,
        description="If True, this step needs access to long-term memory services.",
    )

class OrchestrateVerbRequest(BaseModel):
    verb_name: str
    origin_node: str = Field("unknown-node")
    context: dict = Field(default_factory=dict)
    steps: list[CortexStepConfig] = Field(
        ..., description="Ordered list of Cortex steps implementing this verb."
    )
    timeout_ms: int | None = Field(
        None,
        description="Optional per-step timeout override (ms).",
    )

class ServiceStepResult(BaseModel):
    service: str
    trace_id: str
    ok: bool
    elapsed_ms: int
    payload: Dict[str, Any]


class StepExecutionResult(BaseModel):
    verb_name: str
    step_name: str
    order: int
    services: List[ServiceStepResult]
    prompt_preview: str


class OrchestrateVerbResponse(BaseModel):
    verb_name: str
    origin_node: str
    steps_executed: int
    step_results: List[StepExecutionResult]
    context_echo: Dict[str, Any]


def _build_prompt(
    step: CortexStepConfig,
    service: str,
    origin_node: str,
    context: Dict[str, Any],
    prior_results: List[StepExecutionResult],
) -> str:
    prior_results_json = json.dumps(
        [r.model_dump(mode="json") for r in prior_results],
        indent=2,
        ensure_ascii=False,
    )
    context_json = json.dumps(context, indent=2, ensure_ascii=False)

    header = step.prompt_template.strip()
    suffix = (
        "\n\n"
        "# Orion Cortex Orchestrator Context\n"
        f"- Verb: {step.verb_name}\n"
        f"- Step: {step.step_name} (order={step.order})\n"
        f"- Target Service: {service}\n"
        f"- Origin Node: {origin_node}\n"
        "\n"
        "## Input Context (JSON)\n"
        f"{context_json}\n"
        "\n"
        "## Prior Step Results (JSON)\n"
        f"{prior_results_json}\n"
    )

    return f"{header}\n{suffix}"


def _wait_for_exec_results(
    bus,
    result_channel: str,
    trace_id: str,
    expected: int,
    timeout_ms: int,
) -> List[Dict[str, Any]]:
    timeout_s = timeout_ms / 1000.0
    started = time.time()
    collected: List[Dict[str, Any]] = []

    logger.info(
        "Subscribed %s; waiting for %d result(s). Timeout=%d ms",
        result_channel,
        expected,
        timeout_ms,
    )

    try:
        for msg in bus.raw_subscribe(result_channel):
            now = time.time()
            if now - started > timeout_s:
                logger.warning(
                    "Timeout waiting for results on %s (have %d/%d)",
                    result_channel,
                    len(collected),
                    expected,
                )
                break

            payload = msg.get("data") or {}
            if payload.get("trace_id") != trace_id:
                logger.debug(
                    "Ignoring message with mismatched trace_id on %s: %s",
                    result_channel,
                    payload,
                )
                continue

            collected.append(payload)

            if len(collected) >= expected:
                break
    except Exception as e:
        logger.error("Error while waiting for exec results on %s: %s", result_channel, e)

    return collected


def run_cortex_verb(bus, req: OrchestrateVerbRequest) -> OrchestrateVerbResponse:
    for s in req.steps:
        if not s.verb_name:
            s.verb_name = req.verb_name

    ordered_steps = sorted(req.steps, key=lambda s: s.order)
    timeout_ms = req.timeout_ms or settings.cortex_step_timeout_ms

    prior_step_results: List[StepExecutionResult] = []
    final_step_results: List[StepExecutionResult] = []

    for step in ordered_steps:
        trace_id = str(uuid.uuid4())
        result_channel = f"{settings.exec_result_prefix}:{trace_id}"
        expected = len(step.services)

        # Fan-out
        for service in step.services:
            prompt = _build_prompt(
                step=step,
                service=service,
                origin_node=req.origin_node,
                context=req.context,
                prior_results=prior_step_results,
            )
            exec_channel = f"{settings.exec_request_prefix}:{service}"

            message = {
                # Cortex → Brain: tell Brain this is an exec_step
                "event": "exec_step",
                "kind": "cortex_step",

                # Correlation & routing
                "trace_id": trace_id,
                "correlation_id": trace_id,
                # Where Brain should send exec_step_result
                "reply_channel": result_channel,

                # Names/keys Brain expects in build_cortex_prompt
                "verb": step.verb_name or req.verb_name,
                "step": step.step_name,
                "order": step.order,
                "service": service,
                "origin_node": req.origin_node,

                # Template + context for Brain to assemble its own prompt
                "prompt_template": step.prompt_template,
                "context": req.context,
                "args": {},  # reserved for richer argument passing later

                # For debugging / advanced use
                "prior_step_results": [
                    r.model_dump(mode="json") for r in prior_step_results
                ],

                "requires_gpu": step.requires_gpu,
                "requires_memory": step.requires_memory,

                # Optional: also include the fully-built prompt (nice for
                # human inspection / future services, but Brain won't rely on it)
                "prompt": prompt,
            }

            logger.info(
                "Published exec_step to %s (trace_id=%s, verb=%s, step=%s, service=%s)",
                exec_channel,
                trace_id,
                step.verb_name,
                step.step_name,
                service,
            )
            bus.publish(exec_channel, message)

        # Fan-in
        raw_results = _wait_for_exec_results(
            bus=bus,
            result_channel=result_channel,
            trace_id=trace_id,
            expected=expected,
            timeout_ms=timeout_ms,
        )

        step_results: List[ServiceStepResult] = []
        for raw in raw_results:
            service_name = raw.get("service") or raw.get("service_name") or "unknown"
            elapsed_ms = int(raw.get("elapsed_ms") or raw.get("elapsed") or 0)
            ok = bool(raw.get("ok", True))

            payload = {
                k: v
                for k, v in raw.items()
                if k not in {"trace_id", "service", "service_name", "elapsed_ms", "elapsed", "ok"}
            }

            step_results.append(
                ServiceStepResult(
                    service=service_name,
                    trace_id=trace_id,
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

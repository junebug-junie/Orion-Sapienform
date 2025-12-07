# services/orion-cortex-orch/app/orchestrator.py

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from .settings import get_settings
import orion  # Uses the installed orion package to find cognition/

logger = logging.getLogger("orion-cortex-orchestrator")
settings = get_settings()

# ─────────────────────────────────────
# Cognition / prompts / verbs locations
# ─────────────────────────────────────

COGNITION_ROOT = Path(orion.__file__).resolve().parent / "cognition"
VERBS_DIR = COGNITION_ROOT / "verbs"
PROMPTS_DIR = COGNITION_ROOT / "prompts"

_jinja_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=False,
)


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
    # 👇 NEW: steps are optional; if omitted, load from cognition/verbs/<verb>.yaml
    steps: Optional[List[CortexStepConfig]] = Field(
        default=None,
        description="Ordered list of Cortex steps. If omitted, load from verb YAML.",
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


# ─────────────────────────────────────
# Verb loader (cognition/verbs/*.yaml)
# ─────────────────────────────────────

def _load_verb_steps_for_name(verb_name: str) -> List[CortexStepConfig]:
    """
    Load a verb definition from cognition/verbs/<verb_name>.yaml and
    convert its plan[] into CortexStepConfig objects.

    Example (chat_general.yaml):

      name: chat_general
      services:
        - LLMGatewayService
      prompt_template: chat_general.j2
      plan:
        - name: llm_chat_general
          prompt_template: chat_general.j2
          services:
            - LLMGatewayService
          ...

    """
    path = VERBS_DIR / f"{verb_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No verb YAML found for '{verb_name}' at {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    name = data.get("name") or verb_name
    default_desc = data.get("description") or f"Verb {name}"
    default_services = data.get("services") or []
    default_prompt_template = data.get("prompt_template") or ""
    default_requires_gpu = bool(data.get("requires_gpu", False))
    default_requires_memory = bool(data.get("requires_memory", False))

    plan = data.get("plan") or []
    steps: List[CortexStepConfig] = []

    for raw_step in plan:
        step_name = raw_step.get("name") or raw_step.get("step_name") or "step"
        step_desc = raw_step.get("description") or default_desc
        order = int(raw_step.get("order", 0))
        services = raw_step.get("services") or default_services
        prompt_template = raw_step.get("prompt_template") or default_prompt_template
        requires_gpu = bool(raw_step.get("requires_gpu", default_requires_gpu))
        requires_memory = bool(raw_step.get("requires_memory", default_requires_memory))

        steps.append(
            CortexStepConfig(
                verb_name=name,
                step_name=step_name,
                description=step_desc,
                order=order,
                services=services,
                prompt_template=prompt_template,
                requires_gpu=requires_gpu,
                requires_memory=requires_memory,
            )
        )

    if not steps:
        raise ValueError(f"Verb YAML {path} has no plan[] steps")

    logger.info(
        "Loaded %d step(s) for verb '%s' from %s",
        len(steps),
        verb_name,
        path,
    )
    return steps


def _render_prompt_template(
    template_ref: str,
    context: Dict[str, Any],
    prior_results: List[StepExecutionResult],
) -> str:
    """
    Resolve a prompt_template reference.

    - If template_ref looks like a Jinja file name (e.g. 'chat_general.j2'),
      render it via Jinja2 from cognition/prompts.
    - Otherwise treat it as literal prompt text.
    """
    if not template_ref:
        return ""

    ref = template_ref.strip()

    # Heuristic: treat as template name if no whitespace and endswith '.j2'
    if (" " not in ref) and ("\n" not in ref) and ref.endswith(".j2"):
        try:
            tmpl = _jinja_env.get_template(ref)
            return tmpl.render(
                context=context,
                prior_step_results=[
                    r.model_dump(mode="json") for r in prior_results
                ],
            ).strip()
        except Exception as e:
            logger.error(
                "Failed to render Jinja prompt template '%s': %s", ref, e, exc_info=True
            )
            # Fall back to the raw ref so we don't blow up exec_step
            return ref

    # Literal inline text
    return ref


def _build_prompt(
    step: CortexStepConfig,
    service: str,
    origin_node: str,
    context: Dict[str, Any],
    prior_results: List[StepExecutionResult],
) -> str:
    """
    Build a rich, debuggable prompt for a Cortex exec_step.

    - Header from prompt_template (possibly via Jinja)
    - JSON-encoded context and prior results appended as a footer
    """
    prior_results_json = json.dumps(
        [r.model_dump(mode="json") for r in prior_results],
        indent=2,
        ensure_ascii=False,
    )
    context_json = json.dumps(context, indent=2, ensure_ascii=False)

    header = _render_prompt_template(
        step.prompt_template,
        context=context,
        prior_results=prior_results,
    )

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
    """
    Fan-in helper: wait for up to `expected` results on `result_channel`
    matching the given trace_id, or until timeout.
    """
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
    """
    Execute a high-level "verb" as a sequence of Cortex steps.

    For each step:
      - Build a rich prompt (header + JSON context + prior results).
      - Fan-out to all configured services on the Orion bus.
      - Fan-in results from each service on a per-step result channel.
      - Accumulate results into StepExecutionResult objects.

    The services are bus suffixes like "LLMGatewayService", and requests are
    published on channels:

        {EXEC_REQUEST_PREFIX}:{service}

    with result messages returned on:

        {EXEC_RESULT_PREFIX}:{trace_id}

    using a standard shape compatible with emit_cortex_step_result / LLM Gateway
    exec_step replies.
    """
    # Determine the step config list: either caller-provided or from verb YAML.
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
        trace_id = str(uuid.uuid4())
        result_channel = f"{settings.exec_result_prefix}:{trace_id}"
        expected = len(step.services)

        # ─────────────────────────────────────────────
        # Fan-out: publish ExecutionEnvelope-style messages
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

            envelope = {
                "event": "exec_step",
                "service": service,
                "correlation_id": trace_id,
                "reply_channel": result_channel,
                "payload": {
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
                    # Also include the fully-built prompt (what the LLM will see)
                    "prompt": prompt,
                },
            }

            logger.info(
                "Published exec_step to %s (trace_id=%s, verb=%s, step=%s, service=%s)",
                exec_channel,
                trace_id,
                step.verb_name,
                step.step_name,
                service,
            )
            bus.publish(exec_channel, envelope)

        # ─────────────────────────────────────────────
        # Fan-in: collect results for this step
        # ─────────────────────────────────────────────
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
                if k
                not in {
                    "trace_id",
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

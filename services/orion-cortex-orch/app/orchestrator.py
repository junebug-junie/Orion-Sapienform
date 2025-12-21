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
    verb_name: str = Field(..., description="High-level verb this step belongs to.")
    step_name: str = Field(..., description="Human-readable name for this step.")
    description: str = Field(..., description="What this step is trying to accomplish.")
    order: int = Field(..., description="Execution order within the verb.")
    services: list[str] = Field(..., description="One or more cognitive services to receive this step.")
    prompt_template: str = Field(..., description="Base instruction text; orchestrator appends context.")

    requires_gpu: bool = Field(False)
    requires_memory: bool = Field(False)


class OrchestrateVerbRequest(BaseModel):
    verb_name: str
    origin_node: str = Field("unknown-node")
    context: dict = Field(default_factory=dict)
    steps: Optional[List[CortexStepConfig]] = Field(default=None)
    timeout_ms: int | None = Field(None)


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

    logger.info("Loaded %d step(s) for verb '%s' from %s", len(steps), verb_name, path)
    return steps


def _render_prompt_template(
    template_ref: str,
    context: Dict[str, Any],
    prior_results: List[StepExecutionResult],
) -> str:
    if not template_ref:
        return ""

    ref = template_ref.strip()

    if (" " not in ref) and ("\n" not in ref) and ref.endswith(".j2"):
        try:
            tmpl = _jinja_env.get_template(ref)
            return tmpl.render(
                context=context,
                prior_step_results=[r.model_dump(mode="json") for r in prior_results],
            ).strip()
        except Exception as e:
            logger.error("Failed to render Jinja prompt template '%s': %s", ref, e, exc_info=True)
            return ref

    return ref


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

    header = _render_prompt_template(step.prompt_template, context=context, prior_results=prior_results)

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


# ✅ NEW: robust pubsub payload decoding
def _coerce_pubsub_data(data: Any) -> Dict[str, Any]:
    """
    bus.raw_subscribe() sometimes yields:
      - {"data": {...}} (already dict)
      - {"data": "<json string>"} (string)
      - {"data": b"<json bytes>"} (bytes)
    Normalize to dict.
    """
    if isinstance(data, dict):
        return data

    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8", errors="ignore")
        except Exception:
            return {}

    if isinstance(data, str):
        s = data.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {}


def _wait_for_exec_results(
    bus,
    reply_channel: str,
    correlation_id: str,
    expected_count: int,
    timeout_ms: int,
) -> List[Dict[str, Any]]:
    """
    Synchronously waits for 'expected_count' results on 'reply_channel'.
    """
    results: List[Dict[str, Any]] = []
    start_time = time.time()
    deadline = start_time + (timeout_ms / 1000.0)

    # Access the underlying Redis client for precise blocking control
    # Assuming bus.client is the redis.Redis instance
    pubsub = bus.client.pubsub()
    pubsub.subscribe(reply_channel)

    try:
        while len(results) < expected_count:
            now = time.time()
            if now >= deadline:
                break
            
            remaining = deadline - now
            # Wait for next message with dynamic timeout
            msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=remaining)
            
            if msg:
                if msg["type"] == "message":
                    try:
                        data = _coerce_pubsub_data(msg["data"])
                        # Verify it belongs to this execution batch
                        # (checking trace_id or correlation_id)
                        cid = data.get("correlation_id") or data.get("trace_id")
                        if cid == correlation_id:
                            results.append(data)
                    except Exception as e:
                        logger.warning(f"Error parsing exec result: {e}")
            else:
                # No message yet, short sleep to prevent busy loop 
                # (though get_message timeout handles most of this)
                time.sleep(0.005)
                
    except Exception as e:
        logger.error(f"Error waiting for results: {e}")
    finally:
        pubsub.unsubscribe()
        pubsub.close()

    return results

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

            # IMPORTANT:
            # - Orch remains agnostic: it only targets {prefix}:{ServiceName}
            # - We include both correlation_id + trace_id so mixed workers are fine.
            envelope = {
                "event": "exec_step",
                "service": service,
                "correlation_id": exec_id,
                "trace_id": exec_id,
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
                    "prompt": prompt,
                },
            }

            logger.info(
                "Published exec_step to %s (exec_id=%s, verb=%s, step=%s, service=%s)",
                exec_channel,
                exec_id,
                step.verb_name,
                step.step_name,
                service,
            )
            bus.publish(exec_channel, envelope)

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

# services/orion-cortex-orch/app/orchestrator.py

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
    """
    Build a rich, debuggable prompt for a Cortex exec_step.

    This is friendly both for humans (inspection) and for LLM backends.
    """
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
    """
    Fan-in helper: wait for up to `expected` results on `result_channel`
    matching the given trace_id, or until timeout.
    """
    timeout

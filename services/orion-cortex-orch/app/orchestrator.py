"""Cortex Orchestrator.

Full-fidelity pipeline:

Hub/UI  →  Cortex Orch  →  Cortex Exec  →  (LLM Gateway / other services)

Orch is responsible for:
- loading verb YAML (planning)
- building prompts / contexts
- deciding step order
- aggregating results

Exec is responsible for:
- fan-out to downstream service channels
- fan-in (collect exec_step_result)
- timeouts

This file intentionally *does not* implement any PubSub subscribe loops for
exec results; it delegates those mechanics to Cortex Exec via RPC.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orion.core.bus.rpc_async import request_and_wait
from orion.core.bus.service_async import OrionBusAsync

from .settings import settings
from .verbs import VerbLoader

logger = logging.getLogger("orion-cortex-orchestrator")


# ---- Request/Response models ----


class OrchestrateVerbRequest(BaseModel):
    verb_name: str = Field(..., description="Name of verb YAML to load")
    origin_node: str = Field("hub", description="Who asked for this")
    session_id: Optional[str] = None

    # Chat use-case
    text: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None

    # Global args/context for templates
    args: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

    # Backward-compat (older callers)
    steps: Optional[List[Dict[str, Any]]] = None
    timeout_ms: Optional[int] = None


class ServiceStepResult(BaseModel):
    service: str
    ok: bool = True
    elapsed_ms: Optional[int] = None
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class StepExecutionResult(BaseModel):
    verb: str
    step: str
    order: int
    ok: bool
    elapsed_ms: int
    service_results: List[ServiceStepResult]


class OrchestrateVerbResponse(BaseModel):
    verb_name: str
    exec_id: str
    ok: bool
    elapsed_ms: int

    # Stable for Hub/UI
    step_results: List[StepExecutionResult]

    # Convenience for chat
    text: Optional[str] = None


@dataclass
class CortexStepConfig:
    verb_name: str
    step_name: str
    order: int
    services: List[str]
    prompt_template: str
    requires: Dict[str, Any]


# ---- Orch Core ----


class CortexOrchestrator:
    def __init__(self, verbs_dir: str = "/app/orion/cognition/verbs") -> None:
        self.loader = VerbLoader(verbs_dir)

    def _build_prompt(
        self,
        *,
        verb_name: str,
        step_name: str,
        service: str,
        text: Optional[str],
        history: Optional[List[Dict[str, Any]]],
        args: Dict[str, Any],
        context: Dict[str, Any],
        prior_step_results: List[Dict[str, Any]],
    ) -> str:
        """Build a high-fidelity prompt wrapper.

        Downstream services can also use `prompt_template` + `context` directly,
        but this wrapper preserves observability and improves LLM steering.
        """

        history_len = len(history or [])
        user_text = (text or "").strip()

        lines = []
        lines.append("# Orion Cortex Orchestrator")
        lines.append(f"verb: {verb_name}")
        lines.append(f"step: {step_name}")
        lines.append(f"target_service: {service}")
        lines.append(f"history_len: {history_len}")
        if user_text:
            lines.append("\n## User input")
            lines.append(user_text)

        if context:
            lines.append("\n## Context")
            # keep it lightweight; downstream can consume full context dict
            for k, v in context.items():
                lines.append(f"- {k}: {v}")

        if prior_step_results:
            lines.append("\n## Prior step results")
            for r in prior_step_results[-3:]:
                step = r.get("step")
                ok = r.get("ok")
                lines.append(f"- step={step} ok={ok}")

        # Template-aware: include a marker so the LLM can anchor.
        lines.append("\n## Instruction")
        lines.append(
            "Follow the step template and produce the most helpful response. "
            "If tool calls are needed, emit them using the target service's contract." 
        )

        return "\n".join(lines)

    def _extract_text_from_results(self, step_results: List[StepExecutionResult]) -> Optional[str]:
        # Common path: LLM gateway returns choices[0].message.content
        for step in reversed(step_results):
            for sr in step.service_results:
                if not sr.payload:
                    continue
                payload = sr.payload
                # Some services return nested
                if isinstance(payload, dict):
                    # LLM Gateway: {ok, service, payload:{...}} or raw openai shape
                    if "choices" in payload:
                        try:
                            return payload["choices"][0]["message"]["content"]
                        except Exception:
                            pass
                    if "payload" in payload and isinstance(payload["payload"], dict):
                        inner = payload["payload"]
                        if "choices" in inner:
                            try:
                                return inner["choices"][0]["message"]["content"]
                            except Exception:
                                pass
        return None

    async def orchestrate_verb(self, *, bus, req: OrchestrateVerbRequest) -> OrchestrateVerbResponse:
        exec_id = str(uuid.uuid4())
        t0 = time.monotonic()

        verb_cfg = self.loader.load_verb(req.verb_name)
        steps_cfg = verb_cfg.get("steps") or []
        if not steps_cfg:
            return OrchestrateVerbResponse(
                verb_name=req.verb_name,
                exec_id=exec_id,
                ok=False,
                elapsed_ms=0,
                step_results=[],
                text=None,
            )

        # Normalize step configs
        steps: List[CortexStepConfig] = []
        for s in steps_cfg:
            steps.append(
                CortexStepConfig(
                    verb_name=req.verb_name,
                    step_name=s.get("name") or s.get("step") or "step",
                    order=int(s.get("order") or 0),
                    services=list(s.get("services") or []),
                    prompt_template=str(s.get("prompt_template") or ""),
                    requires=dict(s.get("requires") or {}),
                )
            )

        steps.sort(key=lambda x: x.order)
        logger.info(
            "Loaded %s step(s) for verb '%s' from %s",
            len(steps),
            req.verb_name,
            verb_cfg.get("_path") or "(memory)",
        )

        prior_step_results: List[Dict[str, Any]] = []
        step_results: List[StepExecutionResult] = []

        for step in steps:
            step_t0 = time.monotonic()
            step_exec_id = str(uuid.uuid4())

            if not step.services:
                step_results.append(
                    StepExecutionResult(
                        verb=req.verb_name,
                        step=step.step_name,
                        order=step.order,
                        ok=False,
                        elapsed_ms=0,
                        service_results=[
                            ServiceStepResult(
                                service="(none)",
                                ok=False,
                                error="No services configured for step",
                            )
                        ],
                    )
                )
                continue

            calls: List[Dict[str, Any]] = []
            for svc in step.services:
                prompt = self._build_prompt(
                    verb_name=req.verb_name,
                    step_name=step.step_name,
                    service=svc,
                    text=req.text,
                    history=req.history,
                    args=req.args,
                    context=req.context,
                    prior_step_results=prior_step_results,
                )

                payload_for_service = {
                    "verb": req.verb_name,
                    "step": step.step_name,
                    "order": step.order,
                    "service": svc,
                    "origin_node": req.origin_node,
                    "prompt": prompt,
                    "prompt_template": step.prompt_template,
                    "context": req.context,
                    "args": req.args,
                    "prior_step_results": prior_step_results,
                    "requires": step.requires,
                }

                calls.append({"service": svc, "payload": payload_for_service})

            # ---- Delegate runtime mechanics to Cortex Exec ----

            reply_channel = f"{settings.cortex_exec_result_prefix}:{step_exec_id}"
            exec_req = {
                "trace_id": step_exec_id,
                "result_channel": reply_channel,
                "verb_name": req.verb_name,
                "step_name": step.step_name,
                "order": step.order,
                "origin_node": req.origin_node,
                "timeout_ms": int(req.timeout_ms or settings.cortex_step_timeout_ms),
                "calls": calls,
            }

            try:
                exec_reply = await request_and_wait(
                    bus,
                    intake_channel=settings.cortex_exec_request_channel,
                    reply_channel=reply_channel,
                    payload=exec_req,
                    timeout_sec=(settings.cortex_step_timeout_ms / 1000.0) + 5.0,
                )
            except Exception as e:
                logger.exception("Cortex Exec RPC failed for step %s", step.step_name)
                step_results.append(
                    StepExecutionResult(
                        verb=req.verb_name,
                        step=step.step_name,
                        order=step.order,
                        ok=False,
                        elapsed_ms=int((time.monotonic() - step_t0) * 1000),
                        service_results=[
                            ServiceStepResult(
                                service="CortexExecService",
                                ok=False,
                                error=str(e),
                            )
                        ],
                    )
                )
                prior_step_results.append({"step": step.step_name, "ok": False, "error": str(e)})
                continue

            # Exec reply shape: {ok, results:[...], elapsed_ms, error?}
            raw_results: List[Dict[str, Any]] = []
            ok = True
            if isinstance(exec_reply, dict):
                ok = bool(exec_reply.get("ok", True))
                raw_results = list(exec_reply.get("results") or [])
                if not ok and exec_reply.get("error"):
                    raw_results.append(
                        {
                            "event": "exec_error",
                            "service": "CortexExecService",
                            "ok": False,
                            "error": exec_reply.get("error"),
                        }
                    )
            else:
                ok = False

            service_results: List[ServiceStepResult] = []
            for r in raw_results:
                # These are raw downstream replies; normalize lightly.
                svc = r.get("service") or r.get("payload", {}).get("service") or "unknown"
                svc_ok = bool(r.get("ok", True))
                service_results.append(
                    ServiceStepResult(
                        service=svc,
                        ok=svc_ok,
                        elapsed_ms=r.get("elapsed_ms"),
                        payload=r,
                        error=r.get("error"),
                    )
                )

            step_elapsed_ms = int((time.monotonic() - step_t0) * 1000)
            step_ok = ok and all(sr.ok for sr in service_results) and len(service_results) == len(calls)

            step_results.append(
                StepExecutionResult(
                    verb=req.verb_name,
                    step=step.step_name,
                    order=step.order,
                    ok=step_ok,
                    elapsed_ms=step_elapsed_ms,
                    service_results=service_results,
                )
            )

            prior_step_results.append(
                {
                    "step": step.step_name,
                    "ok": step_ok,
                    "results": raw_results,
                }
            )

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        overall_ok = all(s.ok for s in step_results) if step_results else False
        text = self._extract_text_from_results(step_results)

        return OrchestrateVerbResponse(
            verb_name=req.verb_name,
            exec_id=exec_id,
            ok=overall_ok,
            elapsed_ms=elapsed_ms,
            step_results=step_results,
            text=text,
        )



async def run_cortex_verb(bus: OrionBusAsync, req: OrchestrateVerbRequest) -> OrchestrateVerbResponse:
    """Compatibility wrapper used by app.main."""
    orch = CortexOrchestrator()
    return await orch.orchestrate_verb(bus=bus, req=req)

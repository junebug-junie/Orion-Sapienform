from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable

from orion.schemas.context_exec import (
    ContextExecRequestV1,
    ContextExecRunV1,
    ContextExecVerbStepV1,
)

from .artifact_builder import (
    artifact_type_for_mode,
    build_final_text,
    synthesize_findings_bundle,
    validate_artifact,
)
from .callable_namespace import ContextNamespace
from .rlm_engine import RLMEngine, build_engine
from .security import PolicyBlockedError, enforce_no_write_settings
from .settings import settings

logger = logging.getLogger("orion-context-exec.runner")


class FakeOrgans:
    """Test hooks for deterministic fake evidence."""

    memory_hits: list[dict[str, Any]] | None = None
    trace_hits: list[dict[str, Any]] | None = None


FAKE_ORGANS = FakeOrgans()


def _default_memory_search(query: str, limit: int = 20) -> list[dict[str, Any]]:
    if FAKE_ORGANS.memory_hits is not None:
        return FAKE_ORGANS.memory_hits[:limit]
    if "denver" in query.lower():
        return [
            {
                "claim": "User mentioned Denver in session",
                "source_ref": "memory:session:123",
                "verified": True,
                "confidence": 0.85,
            }
        ]
    return []


def _default_trace_search(**_kwargs: Any) -> list[dict[str, Any]]:
    if FAKE_ORGANS.trace_hits is not None:
        return FAKE_ORGANS.trace_hits
    return [{"handle": "trace:denver:1", "snippet": "Denver claim in trace", "corr_id": "abc"}]


class ContextExecRunner:
    def __init__(self, engine: RLMEngine | None = None) -> None:
        self.engine = engine or build_engine(settings.rlm_engine)

    async def run(self, request: ContextExecRequestV1) -> ContextExecRunV1:
        run_id = f"ctxrun_{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()
        enforce_no_write_settings(settings.context_exec_write_enabled, request.permissions)
        verb_trace: list[ContextExecVerbStepV1] = []
        failure_modes: list[str] = []
        status: str = "ok"

        namespace = ContextNamespace(
            permissions=request.permissions,
            memory_fn={"search_claims": _default_memory_search, "read": lambda h: {"handle": h}},
            recall_fn=lambda q, **kw: {"hits": []},
            traces_fn={"search": _default_trace_search, "read": lambda h: {"handle": h}},
        )

        try:
            budget_sec = min(request.budget.max_seconds, settings.context_exec_max_seconds)
            raw_final = await asyncio.wait_for(
                self.engine.run(request, namespace),
                timeout=budget_sec,
            )
            verb_trace.append(
                ContextExecVerbStepV1(
                    step_index=0,
                    verb="synthesize",
                    callable="rlm_engine.run",
                    status="ok",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    output_summary="rlm episode complete",
                )
            )
        except asyncio.TimeoutError:
            status = "timeout"
            failure_modes.append("timeout")
            raw_final = None
        except PolicyBlockedError as exc:
            status = "policy_blocked"
            failure_modes.append(str(exc))
            raw_final = None
        except Exception as exc:
            status = "error"
            failure_modes.append(str(exc))
            logger.exception("context-exec rlm error run_id=%s", run_id)
            raw_final = None

        artifact: dict[str, Any] = {}
        artifact_type: str | None = request.expected_artifact_type or artifact_type_for_mode(request.mode)
        schema_valid = False
        if isinstance(raw_final, dict):
            artifact, artifact_type, schema_valid = validate_artifact(request.mode, raw_final)
            if not schema_valid:
                status = "schema_invalid" if status == "ok" else status
                failure_modes.append("schema_invalid")

        fb = synthesize_findings_bundle(request, artifact, schema_valid=schema_valid)
        final_text = build_final_text(request.mode, artifact, status=status)
        ac_dump = request.answer_contract.model_dump(mode="json") if request.answer_contract else None

        return ContextExecRunV1(
            run_id=run_id,
            status=status,  # type: ignore[arg-type]
            mode=request.mode,
            text=request.text,
            answer_contract=ac_dump,
            findings_bundle=fb,
            artifact_type=artifact_type,
            artifact=artifact,
            final_text=final_text,
            verb_trace=verb_trace,
            runtime_debug={
                "engine": "context_exec",
                "rlm_depth": settings.context_exec_max_depth,
                "subcalls": 0,
                "schema_valid": schema_valid,
                "sandbox_mode": settings.context_exec_sandbox_mode,
            },
            failure_modes=failure_modes,
        )

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from orion.core.bus.async_service import OrionBusAsync
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
from .events import ContextExecEventEmitter
from .organ_runtime import OrganRuntime
from .rlm_engine import RLMEngine, build_engine
from .security import PolicyBlockedError, enforce_no_write_settings
from .settings import settings

logger = logging.getLogger("orion-context-exec.runner")


class FakeOrgans:
    """Test hooks for deterministic fake evidence (pytest only)."""

    memory_hits: list[dict[str, Any]] | None = None
    trace_hits: list[dict[str, Any]] | None = None


FAKE_ORGANS = FakeOrgans()


def _default_memory_search(query: str, limit: int = 20) -> list[dict[str, Any]]:
    if FAKE_ORGANS.memory_hits is not None:
        return FAKE_ORGANS.memory_hits[:limit]
    if not settings.context_exec_fake_organs_enabled:
        return []
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
    if not settings.context_exec_fake_organs_enabled:
        return []
    return [{"handle": "trace:denver:1", "snippet": "Denver claim in trace", "corr_id": "abc"}]


class ContextExecRunner:
    def __init__(
        self,
        engine: RLMEngine | None = None,
        *,
        bus: OrionBusAsync | None = None,
        events: ContextExecEventEmitter | None = None,
    ) -> None:
        self.engine = engine or build_engine(settings.rlm_engine)
        self.bus = bus
        self._default_events = events

    def _build_events(
        self,
        request: ContextExecRequestV1,
        *,
        causality_chain: list[str] | None = None,
    ) -> ContextExecEventEmitter:
        if self._default_events is not None:
            emitter = self._default_events
            emitter.bind_request(
                correlation_id=request.correlation_id,
                causality_chain=causality_chain,
            )
            return emitter
        return ContextExecEventEmitter(
            self.bus,
            correlation_id=request.correlation_id,
            causality_chain=causality_chain,
        )

    async def run(
        self,
        request: ContextExecRequestV1,
        *,
        causality_chain: list[str] | None = None,
    ) -> ContextExecRunV1:
        run_id = f"ctxrun_{uuid.uuid4().hex[:12]}"
        if not request.correlation_id:
            request = request.model_copy(update={"correlation_id": f"ctxcorr_{uuid.uuid4().hex[:12]}"})
        started = time.perf_counter()
        enforce_no_write_settings(settings.context_exec_write_enabled, request.permissions)
        verb_trace: list[ContextExecVerbStepV1] = []
        failure_modes: list[str] = []
        status: str = "ok"

        events = self._build_events(request, causality_chain=causality_chain)
        await events.started(run_id=run_id, mode=request.mode, text=request.text)

        organ_runtime = OrganRuntime(bus=self.bus, request=request, run_id=run_id)
        namespace = self._build_namespace(organ_runtime)
        await namespace._prefetch_organs()  # type: ignore[attr-defined]

        try:
            budget_sec = min(request.budget.max_seconds, settings.context_exec_max_seconds)
            raw_final = await asyncio.wait_for(
                self.engine.run(request, namespace, organ_runtime=organ_runtime),
                timeout=budget_sec,
            )
            step = ContextExecVerbStepV1(
                step_index=0,
                verb="synthesize",
                callable="rlm_engine.run",
                status="ok",
                duration_ms=int((time.perf_counter() - started) * 1000),
                output_summary="rlm episode complete",
            )
            verb_trace.append(step)
            await events.verb_step(run_id=run_id, mode=request.mode, step=step)
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
                await events.schema_invalid(run_id=run_id, mode=request.mode, artifact_type=artifact_type)

        fb = synthesize_findings_bundle(request, artifact, schema_valid=schema_valid)
        final_text = build_final_text(request.mode, artifact, status=status)
        ac_dump = request.answer_contract.model_dump(mode="json") if request.answer_contract else None

        await events.finished(
            run_id=run_id,
            mode=request.mode,
            status=status,
            artifact_type=artifact_type,
            schema_valid=schema_valid,
            failure_modes=failure_modes,
        )

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
                "fake_organs_enabled": settings.context_exec_fake_organs_enabled,
                "real_trace_enabled": settings.context_exec_real_trace_enabled,
                "real_recall_enabled": settings.context_exec_real_recall_enabled,
                "real_repo_enabled": settings.context_exec_real_repo_enabled,
                "correlation_id": request.correlation_id,
            },
            failure_modes=failure_modes,
        )

    def _build_namespace(self, organ_runtime: OrganRuntime) -> ContextNamespace:
        organ_cache: dict[str, Any] = {"traces": None, "recall": None, "trace_reads": {}}

        def traces_search(**kwargs: Any) -> list[dict[str, Any]]:
            if FAKE_ORGANS.trace_hits is not None:
                return FAKE_ORGANS.trace_hits
            if settings.context_exec_fake_organs_enabled:
                return _default_trace_search(**kwargs)
            if organ_cache["traces"] is not None:
                return organ_cache["traces"]
            return []

        def traces_read(handle: str) -> dict[str, Any]:
            return organ_cache["trace_reads"].get(handle, {"handle": handle})

        def recall_fn(q: str, **kw: Any) -> dict[str, Any]:
            if FAKE_ORGANS.memory_hits is not None:
                return {
                    "hits": [
                        {
                            "snippet": h.get("claim"),
                            "source_ref": h.get("source_ref"),
                            "score": h.get("confidence"),
                        }
                        for h in FAKE_ORGANS.memory_hits[: kw.get("limit") or 12]
                    ]
                }
            if settings.context_exec_fake_organs_enabled:
                return {"hits": _default_memory_search(q, limit=kw.get("limit") or 12)}
            if organ_cache["recall"] is not None:
                return organ_cache["recall"]
            return {"hits": []}

        async def _prefetch_organs() -> None:
            if settings.context_exec_fake_organs_enabled or FAKE_ORGANS.trace_hits is not None:
                return
            if settings.context_exec_real_recall_enabled and organ_runtime.request.permissions.read_recall:
                organ_cache["recall"] = await organ_runtime.recall_query(
                    organ_runtime.request.text,
                    limit=settings.context_exec_recall_limit,
                )
            if settings.context_exec_real_trace_enabled and organ_runtime.request.permissions.read_redis_traces:
                hits = await organ_runtime.traces_search(limit=settings.context_exec_trace_limit)
                organ_cache["traces"] = hits
                for hit in hits:
                    handle = hit.get("handle")
                    if handle:
                        organ_cache["trace_reads"][handle] = await organ_runtime.traces_read(handle)

        namespace = ContextNamespace(
            permissions=organ_runtime.request.permissions,
            memory_fn={"search_claims": _default_memory_search, "read": lambda h: {"handle": h}},
            recall_fn=recall_fn,
            traces_fn={"search": traces_search, "read": traces_read},
        )
        namespace._organ_cache = organ_cache  # type: ignore[attr-defined]
        namespace._organ_runtime = organ_runtime  # type: ignore[attr-defined]
        namespace._prefetch_organs = _prefetch_organs  # type: ignore[attr-defined]
        return namespace

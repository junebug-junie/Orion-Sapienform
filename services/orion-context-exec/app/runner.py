from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.context_exec import (
    ContextExecOperatorSummaryV1,
    ContextExecRequestV1,
    ContextExecRunV1,
    ContextExecSafetySummaryV1,
    ContextExecVerbStepV1,
    InvestigationReportV2,
    ProposalEnvelopeV1,
)

from .agent_synthesis import (
    LLM_GATEWAY_SYNTHESIS_UNAVAILABLE,
    run_agent_synthesis,
    synthesis_unavailable_for_llm_gateway_readiness,
)
from .bus_dependency_preflight import check_llm_gateway_bus_ready, effective_llm_gateway_ready, llm_gateway_http_alive
from .grounding_eval import evaluate_investigation_outcome, is_placeholder_investigation_summary
from .investigation_v2 import (
    INVESTIGATION_V2_ARTIFACT_TYPE,
    run_investigation_v2,
)
from .investigation_v2_reducers import apply_synthesis_to_report
from .artifact_builder import (
    artifact_type_for_mode,
    build_final_text,
    synthesize_findings_bundle,
    validate_artifact,
)
from .callable_namespace import ContextNamespace
from .events import ContextExecEventEmitter
from .organ_runtime import OrganRuntime
from .rlm_engine import FakeRLMEngine, RLMEngine, build_engine
from .alexzhang_rlm_engine import (
    AlexZhangInitError,
    AlexZhangRLMEngine,
    UnsupportedModeError,
)
from .llm_profile_resolver import (
    LLMProfileSelection,
    LLMProfileUnavailableError,
    resolve_llm_profile,
    selection_runtime_debug,
)
from .proposal_ledger_intake import (
    intake_final_text_line,
    intake_runtime_debug,
    maybe_persist_proposal_envelope,
)
from .organ_status import initial_organ_status, record_recall, record_repo, record_trace
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
    ) -> None:
        self.engine_selected = (settings.rlm_engine or "fake").strip().lower()
        self.engine = engine or build_engine(settings.rlm_engine)
        self.bus = bus

    def _engine_runtime_debug(
        self,
        *,
        engine_used: str,
        fallback_engine: str | None = None,
        fallback_reason: str | None = None,
        subcalls: int = 0,
        schema_valid: bool = False,
        mode: str,
        extra_steps: list[str] | None = None,
    ) -> dict[str, Any]:
        fallback_used = fallback_engine is not None
        debug: dict[str, Any] = {
            "engine_requested": self.engine_selected,
            "engine_selected": engine_used,
            "engine": engine_used,
            "fallback_used": fallback_used,
            "fallback_engine": fallback_engine,
            "rlm_depth": settings.context_exec_max_depth,
            "subcalls": subcalls,
            "schema_valid": schema_valid,
            "sandbox_mode": settings.context_exec_sandbox_mode,
            "write_enabled": settings.context_exec_write_enabled,
            "network_enabled": settings.context_exec_network_enabled,
            "shell_enabled": False,
            "mutation_allowed": False,
            "mode": mode,
            "fake_organs_enabled": settings.context_exec_fake_organs_enabled,
            "real_trace_enabled": settings.context_exec_real_trace_enabled,
            "real_recall_enabled": settings.context_exec_real_recall_enabled,
            "real_repo_enabled": settings.context_exec_real_repo_enabled,
        }
        if fallback_reason:
            debug["fallback_reason"] = fallback_reason
        if extra_steps:
            debug["rlm_steps"] = extra_steps
        return debug

    async def _execute_rlm(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        organ_runtime: OrganRuntime,
        *,
        started: float,
        events: ContextExecEventEmitter,
        run_id: str,
        verb_trace: list[ContextExecVerbStepV1],
    ) -> tuple[Any, str, int, list[str], list[str]]:
        """Run RLM engine with optional fallback to fake. Returns raw_final, engine_used, subcalls, steps, failure_modes."""
        failure_modes: list[str] = []
        engine_used = getattr(self.engine, "engine_name", "fake")
        subcalls = 0
        extra_steps: list[str] = []
        raw_final: Any = None

        async def _run_one(engine: RLMEngine) -> Any:
            nonlocal subcalls, extra_steps
            if isinstance(engine, AlexZhangRLMEngine) and not engine.is_ready:
                raise AlexZhangInitError(engine.init_error or "alexzhang_init_failed")
            result = await engine.run(request, namespace, organ_runtime=organ_runtime)
            subcalls = getattr(engine, "subcall_count", 0)
            extra_steps = getattr(engine, "debug_steps", [])
            return result

        try:
            budget_sec = min(request.budget.max_seconds, settings.context_exec_max_seconds)
            raw_final = await asyncio.wait_for(
                _run_one(self.engine),
                timeout=budget_sec,
            )
            engine_used = getattr(self.engine, "engine_name", engine_used)
        except UnsupportedModeError as exc:
            failure_modes.append(f"unsupported_mode:{exc.mode}")
            return None, engine_used, subcalls, extra_steps, failure_modes
        except PolicyBlockedError as exc:
            failure_modes.append(str(exc))
            return None, engine_used, subcalls, extra_steps, failure_modes + ["policy_blocked"]
        except asyncio.TimeoutError:
            failure_modes.append("timeout")
            if (
                self.engine_selected == "alexzhang"
                and settings.context_exec_rlm_fallback_enabled
                and not isinstance(self.engine, FakeRLMEngine)
            ):
                failure_modes.append("alexzhang_execution_failed")
                fake = FakeRLMEngine()
                raw_final = await fake.run(request, namespace, organ_runtime=organ_runtime)
                return raw_final, "fake", 0, ["fallback"], failure_modes + ["fallback_engine:fake"]
            return None, engine_used, subcalls, extra_steps, failure_modes
        except (AlexZhangInitError, Exception) as exc:
            failure_modes.append(str(exc))
            if (
                self.engine_selected == "alexzhang"
                and settings.context_exec_rlm_fallback_enabled
                and not isinstance(self.engine, FakeRLMEngine)
            ):
                reason = (
                    "alexzhang_init_failed"
                    if isinstance(exc, AlexZhangInitError)
                    else "alexzhang_execution_failed"
                )
                logger.warning("alexzhang engine failed (%s); falling back to fake", reason)
                fake = FakeRLMEngine()
                raw_final = await fake.run(request, namespace, organ_runtime=organ_runtime)
                return raw_final, "fake", 0, ["fallback"], failure_modes + [f"fallback_engine:fake:{reason}"]
            logger.exception("context-exec rlm error run_id=%s", run_id)
            return None, engine_used, subcalls, extra_steps, failure_modes

        step = ContextExecVerbStepV1(
            step_index=len(verb_trace),
            verb="synthesize",
            callable="rlm_engine.run",
            status="ok",
            duration_ms=int((time.perf_counter() - started) * 1000),
            output_summary="rlm episode complete",
        )
        verb_trace.append(step)
        await events.verb_step(run_id=run_id, mode=request.mode, step=step)
        return raw_final, engine_used, subcalls, extra_steps, failure_modes

    def _build_events(
        self,
        request: ContextExecRequestV1,
        *,
        causality_chain: list[str] | None = None,
    ) -> ContextExecEventEmitter:
        return ContextExecEventEmitter(
            self.bus,
            correlation_id=request.correlation_id,
            causality_chain=causality_chain,
        )

    def _persist_run_ledger(
        self,
        run: ContextExecRunV1,
        request: ContextExecRequestV1,
    ) -> None:
        """Persist the run ledger bundle. Fail-open: never fails the run."""
        if not settings.context_exec_run_ledger_enabled:
            return
        try:
            from .storage import persist_context_exec_run

            persist_context_exec_run(run, request=request)
        except Exception as exc:  # fail-open: never fail the run on ledger errors
            logger.warning(
                "failed to persist context-exec run ledger run_id=%s error=%s",
                getattr(run, "run_id", None),
                exc,
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

        if request.mode == "investigation_v2":
            return await self._run_investigation_v2(
                request=request,
                run_id=run_id,
                started=started,
                events=events,
                verb_trace=verb_trace,
                failure_modes=failure_modes,
            )

        try:
            profile_selection = await resolve_llm_profile(request.llm_profile)
        except LLMProfileUnavailableError as exc:
            failure_modes.append(str(exc))
            runtime_debug = {
                **selection_runtime_debug(
                    LLMProfileSelection(
                        requested=request.llm_profile,
                        selected=str(request.llm_profile or settings.context_exec_default_llm_profile),
                        route_used=str(request.llm_profile or settings.context_exec_default_llm_profile),
                    )
                ),
                **self._engine_runtime_debug(
                    engine_used=self.engine_selected,
                    mode=request.mode,
                ),
                "correlation_id": request.correlation_id,
            }
            await events.finished(
                run_id=run_id,
                mode=request.mode,
                status="error",
                artifact_type=None,
                schema_valid=False,
                failure_modes=failure_modes,
            )
            run = ContextExecRunV1(
                run_id=run_id,
                status="error",
                mode=request.mode,
                text=request.text,
                answer_contract=request.answer_contract.model_dump(mode="json")
                if request.answer_contract
                else None,
                findings_bundle=None,
                artifact_type=None,
                artifact={},
                final_text=build_final_text(request.mode, {}, status="error"),
                verb_trace=verb_trace,
                runtime_debug=runtime_debug,
                failure_modes=failure_modes,
            )
            self._persist_run_ledger(run, request)
            return run

        request = request.model_copy(update={"llm_profile": profile_selection.selected})

        organ_runtime = OrganRuntime(
            bus=self.bus,
            request=request,
            run_id=run_id,
            llm_route=profile_selection.route_used,
        )
        namespace = self._build_namespace(organ_runtime)
        await namespace._prefetch_organs()  # type: ignore[attr-defined]
        organ_cache = getattr(namespace, "_organ_cache", {}) or {}
        organ_status = getattr(namespace, "_organ_status", {}) or {}

        raw_final, engine_used, subcalls, rlm_steps, rlm_failures = await self._execute_rlm(
            request,
            namespace,
            organ_runtime,
            started=started,
            events=events,
            run_id=run_id,
            verb_trace=verb_trace,
        )
        await organ_runtime.flush_llm_subcalls()
        failure_modes.extend(rlm_failures)
        fallback_engine: str | None = None
        fallback_reason: str | None = None
        if engine_used == "fake" and self.engine_selected == "alexzhang":
            fallback_engine = "fake"
            for fm in failure_modes:
                if fm.startswith("fallback_engine:fake:"):
                    fallback_reason = fm.split(":", 2)[-1]
                elif fm == "fallback_engine:fake":
                    fallback_reason = "alexzhang_execution_failed"

        if raw_final is None and "policy_blocked" in failure_modes:
            status = "policy_blocked"
        elif raw_final is None and "unsupported_mode:" in " ".join(failure_modes):
            status = "error"
        elif raw_final is None and "timeout" in failure_modes:
            status = "timeout"
        elif raw_final is None and failure_modes:
            status = "error"
        elif raw_final is None:
            status = "error"

        artifact: dict[str, Any] = {}
        artifact_type: str | None = request.expected_artifact_type or artifact_type_for_mode(request.mode)
        schema_valid = False
        if isinstance(raw_final, dict):
            artifact, artifact_type, schema_valid = validate_artifact(request.mode, raw_final)
            if not schema_valid:
                if (
                    self.engine_selected == "alexzhang"
                    and settings.context_exec_rlm_fallback_enabled
                    and engine_used != "fake"
                ):
                    fake = FakeRLMEngine()
                    raw_final = await fake.run(request, namespace, organ_runtime=organ_runtime)
                    engine_used = "fake"
                    fallback_engine = "fake"
                    fallback_reason = "schema_invalid"
                    artifact, artifact_type, schema_valid = validate_artifact(request.mode, raw_final)
                if not schema_valid:
                    status = "schema_invalid" if status == "ok" else status
                    failure_modes.append("schema_invalid")
                    await events.schema_invalid(run_id=run_id, mode=request.mode, artifact_type=artifact_type)

        fb = synthesize_findings_bundle(request, artifact, schema_valid=schema_valid)
        final_text = build_final_text(request.mode, artifact, status=status)
        ac_dump = request.answer_contract.model_dump(mode="json") if request.answer_contract else None

        runtime_debug = self._engine_runtime_debug(
            engine_used=engine_used,
            fallback_engine=fallback_engine,
            fallback_reason=fallback_reason,
            subcalls=subcalls,
            schema_valid=schema_valid,
            mode=request.mode,
            extra_steps=rlm_steps or None,
        )
        runtime_debug.update(selection_runtime_debug(profile_selection))
        if profile_selection.fallback_used:
            runtime_debug["fallback_used"] = True
            if profile_selection.fallback_reason:
                runtime_debug["fallback_reason"] = profile_selection.fallback_reason
        if request.mode in {"patch_proposal", "memory_correction_proposal"} and schema_valid and artifact_type == "ProposalEnvelopeV1":
            runtime_debug["proposal_enveloped"] = True
            runtime_debug["proposal_type"] = artifact.get("proposal_type")
            runtime_debug["requires_human_approval"] = artifact.get("requires_human_approval", True)
            runtime_debug["mutation_allowed"] = artifact.get("mutation_allowed", False)

            envelope = ProposalEnvelopeV1.model_validate(artifact)
            intake_result = maybe_persist_proposal_envelope(
                envelope,
                settings,
                source_run_id=run_id,
            )
            runtime_debug.update(
                intake_runtime_debug(
                    intake_result,
                    enabled=settings.context_exec_proposal_ledger_enabled,
                )
            )
            ledger_line = intake_final_text_line(intake_result)
            if ledger_line:
                final_text = f"{final_text} {ledger_line}"

        synthesis_result = await run_agent_synthesis(
            request=request,
            artifact=artifact,
            profile_selection=profile_selection,
            runtime_debug=runtime_debug,
            bus=self.bus,
        )
        runtime_debug["model_synthesis_used"] = synthesis_result.model_synthesis_used
        if synthesis_result.fallback_reason:
            runtime_debug["synthesis_fallback_reason"] = synthesis_result.fallback_reason
        if synthesis_result.fallback_used:
            runtime_debug["synthesis_fallback_used"] = True
        runtime_debug["grounding_attempts"] = {
            "recall": organ_cache.get("recall") is not None,
            "trace": organ_cache.get("traces") is not None,
            "repo": bool(organ_cache.get("repo_hits")),
        }
        runtime_debug["organ_status"] = organ_status
        operator_summary = synthesis_result.operator_summary
        if operator_summary is not None and status == "ok":
            if synthesis_result.model_synthesis_used:
                if synthesis_result.synthesis_summary:
                    final_text = synthesis_result.synthesis_summary
                elif operator_summary.summary:
                    final_text = operator_summary.summary
            elif operator_summary.summary:
                final_text = operator_summary.summary

        answer_eval = evaluate_investigation_outcome(
            runtime_status=status,
            text=request.text,
            mode=request.mode,
            artifact=artifact,
            runtime_debug=runtime_debug,
            verb_trace=verb_trace,
            findings_bundle=fb,
            organ_cache=organ_cache,
            answer_contract=ac_dump,
            scopes=request.scopes,
            model_synthesis_used=synthesis_result.model_synthesis_used,
            current_summary=final_text,
        )
        runtime_debug["answer_evaluation"] = answer_eval
        if answer_eval.get("summary_text"):
            final_text = str(answer_eval["summary_text"])
        if operator_summary is not None and (
            answer_eval.get("answer_status")
            not in {"answered_grounded", "partial_or_weak_evidence"}
            or is_placeholder_investigation_summary(operator_summary.summary)
        ):
            operator_summary = operator_summary.model_copy(
                update={
                    "title": "Runtime completed (not grounded)",
                    "summary": final_text,
                }
            )

        await events.finished(
            run_id=run_id,
            mode=request.mode,
            status=status,
            artifact_type=artifact_type,
            schema_valid=schema_valid,
            failure_modes=failure_modes,
        )

        run = ContextExecRunV1(
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
            operator_summary=operator_summary,
            runtime_debug={
                **runtime_debug,
                "correlation_id": request.correlation_id,
            },
            failure_modes=failure_modes,
        )
        self._persist_run_ledger(run, request)
        return run

    async def _run_investigation_v2(
        self,
        *,
        request: ContextExecRequestV1,
        run_id: str,
        started: float,
        events: ContextExecEventEmitter,
        verb_trace: list[ContextExecVerbStepV1],
        failure_modes: list[str],
    ) -> ContextExecRunV1:
        organ_runtime = OrganRuntime(
            bus=self.bus,
            request=request,
            run_id=run_id,
            llm_route=str(request.llm_profile or settings.context_exec_default_llm_profile),
        )
        namespace = self._build_namespace(organ_runtime)
        organ_cache = getattr(namespace, "_organ_cache", {}) or {}
        organ_status = getattr(namespace, "_organ_status", {}) or {}

        artifact = await run_investigation_v2(request, namespace, organ_runtime, organ_cache)
        artifact_type = INVESTIGATION_V2_ARTIFACT_TYPE
        validated, artifact_type, schema_valid = validate_artifact(request.mode, artifact)
        artifact = validated
        status = "ok"
        if not schema_valid:
            status = "schema_invalid"
            failure_modes.append("schema_invalid")

        profile_selection = await resolve_llm_profile(request.llm_profile)
        runtime_debug_base = {
            **self._engine_runtime_debug(
                engine_used="investigation_v2",
                mode=request.mode,
                schema_valid=schema_valid,
            ),
            "correlation_id": request.correlation_id,
            "investigation_v2": True,
            "answer_status": artifact.get("answer_status"),
            "permissions_received": request.permissions.model_dump(mode="json"),
            "read_repo": request.permissions.read_repo,
            "grounding_attempts": {
                "recall": organ_cache.get("recall") is not None,
                "trace": organ_cache.get("traces") is not None,
                "repo": bool(organ_cache.get("repo_hits")) or organ_cache.get("repo_probe_attempted"),
            },
            "organ_status": organ_status,
            "evidence_sources": artifact.get("sources") or {},
        }
        runtime_debug_base.update(selection_runtime_debug(profile_selection))

        synthesis_result = None
        if (
            settings.context_exec_agent_synthesis_enabled
            and settings.orion_bus_enabled
            and self.bus is not None
        ):
            llm_ready, llm_http_ok, llm_effective_ok = await effective_llm_gateway_ready(
                self.bus,
                timeout_sec=float(settings.context_exec_bus_readiness_timeout_sec),
            )
            if not llm_effective_ok:
                synthesis_result = synthesis_unavailable_for_llm_gateway_readiness(
                    request=request,
                    artifact=artifact,
                    profile_selection=profile_selection,
                    runtime_debug=runtime_debug_base,
                    readiness=llm_ready,
                )
        if synthesis_result is None:
            synthesis_result = await run_agent_synthesis(
                request=request,
                artifact=artifact,
                profile_selection=profile_selection,
                runtime_debug=runtime_debug_base,
                bus=self.bus,
            )
        runtime_debug = dict(runtime_debug_base)
        runtime_debug["model_synthesis_used"] = synthesis_result.model_synthesis_used
        if synthesis_result.fallback_reason:
            runtime_debug["synthesis_fallback_reason"] = synthesis_result.fallback_reason
        if synthesis_result.fallback_used:
            runtime_debug["synthesis_fallback_used"] = True

        synthesis_failed = synthesis_result.fallback_used and not synthesis_result.model_synthesis_used
        if synthesis_result.model_synthesis_used:
            runtime_debug["synthesis_status"] = "completed"
        elif synthesis_failed:
            runtime_debug["synthesis_status"] = "synthesis_unavailable"
        else:
            runtime_debug["synthesis_status"] = "skipped"
        if synthesis_result.model_synthesis_used or synthesis_failed:
            report = InvestigationReportV2.model_validate(artifact)
            failure_message = None
            if synthesis_failed and synthesis_result.fallback_reason == LLM_GATEWAY_SYNTHESIS_UNAVAILABLE:
                failure_message = LLM_GATEWAY_SYNTHESIS_UNAVAILABLE
            updated = apply_synthesis_to_report(
                report,
                synthesis_summary=synthesis_result.synthesis_summary,
                synthesis_failed=synthesis_failed,
                synthesis_failure_message=failure_message,
            )
            artifact = updated.model_dump(mode="json")

        final_text = build_final_text(request.mode, artifact, status=status)
        fb = synthesize_findings_bundle(request, artifact, schema_valid=schema_valid)
        ac_dump = request.answer_contract.model_dump(mode="json") if request.answer_contract else None

        operator_summary = synthesis_result.operator_summary
        if operator_summary is None:
            operator_summary = ContextExecOperatorSummaryV1(
                title="Investigation v2 report",
                summary=final_text,
                agent_mode="investigation_v2",
                route_used=profile_selection.route_used,
                model_synthesis_used=False,
                safety=ContextExecSafetySummaryV1(),
            )
        elif status == "ok":
            if synthesis_result.model_synthesis_used and synthesis_result.synthesis_summary:
                final_text = synthesis_result.synthesis_summary
            elif operator_summary.summary:
                final_text = operator_summary.summary
        if synthesis_result.model_synthesis_used:
            operator_summary = operator_summary.model_copy(
                update={"model_synthesis_used": True, "summary": final_text}
            )

        grounded = artifact.get("grounded_sources") or []
        limitations = artifact.get("limitations") or []
        grounding_attempts = runtime_debug_base.get("grounding_attempts") or {}
        answer_eval = {
            "runtime_status": "ok" if status == "ok" else "failed",
            "answer_status": str(artifact.get("answer_status") or "no_reliable_evidence"),
            "grounding_status": "attempted"
            if isinstance(grounding_attempts, dict) and any(grounding_attempts.values())
            else "skipped",
            "synthesis_status": runtime_debug.get("synthesis_status"),
            "evidence_count": len(grounded) if isinstance(grounded, list) else 0,
            "grounding_required": True,
            "summary_text": final_text,
            "limitations": limitations if isinstance(limitations, list) else [],
        }
        runtime_debug["answer_evaluation"] = answer_eval

        await events.finished(
            run_id=run_id,
            mode=request.mode,
            status=status,
            artifact_type=artifact_type,
            schema_valid=schema_valid,
            failure_modes=failure_modes,
        )
        run = ContextExecRunV1(
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
            operator_summary=operator_summary,
            runtime_debug=runtime_debug,
            failure_modes=failure_modes,
        )
        self._persist_run_ledger(run, request)
        return run

    def _build_namespace(self, organ_runtime: OrganRuntime) -> ContextNamespace:
        organ_cache: dict[str, Any] = {"traces": None, "recall": None, "trace_reads": {}}
        organ_status = initial_organ_status(organ_runtime.request.permissions, settings)
        organ_runtime.organ_status = organ_status

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
                recall_result = await organ_runtime.recall_query(
                    organ_runtime.request.text,
                    limit=settings.context_exec_recall_limit,
                )
                organ_cache["recall"] = recall_result
            if settings.context_exec_real_trace_enabled and organ_runtime.request.permissions.read_redis_traces:
                try:
                    hits = await organ_runtime.traces_search(limit=settings.context_exec_trace_limit)
                except Exception as exc:
                    record_trace(organ_status, [], error=str(exc))
                    hits = []
                else:
                    organ_cache["traces"] = hits
                    for hit in hits:
                        handle = hit.get("handle")
                        if handle:
                            organ_cache["trace_reads"][handle] = await organ_runtime.traces_read(handle)

        route_used = organ_runtime.llm_route

        def llm_subcall(prompt: str, context: Any = None, schema: str | None = None) -> dict[str, Any]:
            organ_runtime.record_llm_subcall(
                route=route_used,
                prompt=prompt,
                context=context,
                schema=schema,
            )
            pending = organ_runtime.pending_llm_subcalls[-1]
            if pending.get("result") is not None:
                return pending["result"]
            return {
                "ok": True,
                "route": route_used,
                "summary": "llm subcall recorded (async flush via organ_runtime)",
            }

        namespace = ContextNamespace(
            permissions=organ_runtime.request.permissions,
            memory_fn={"search_claims": _default_memory_search, "read": lambda h: {"handle": h}},
            recall_fn=recall_fn,
            traces_fn={"search": traces_search, "read": traces_read},
            subcall_fn=llm_subcall,
        )
        namespace._organ_cache = organ_cache  # type: ignore[attr-defined]
        namespace._organ_status = organ_status  # type: ignore[attr-defined]
        namespace._organ_runtime = organ_runtime  # type: ignore[attr-defined]
        namespace._prefetch_organs = _prefetch_organs  # type: ignore[attr-defined]
        return namespace

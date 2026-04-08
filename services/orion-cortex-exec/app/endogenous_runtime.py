from __future__ import annotations

import hashlib
import json
import logging
import time
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from orion.core.schemas.endogenous import EndogenousTriggerDecisionV1, EndogenousTriggerRequestV1
from orion.core.schemas.endogenous_runtime import (
    EndogenousRuntimeAuditV1,
    EndogenousRuntimeConsumptionItemV1,
    EndogenousRuntimeExecutionRecordV1,
    EndogenousRuntimeQueryV1,
    EndogenousRuntimeResultV1,
    EndogenousRuntimeSignalDigestV1,
)
from orion.core.schemas.endogenous_eval import EndogenousEvaluationRequestV1
from orion.core.schemas.calibration_adoption import (
    CalibrationAdoptionRequestV1,
    CalibrationProfileAuditV1,
    CalibrationRollbackRequestV1,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.schemas.reasoning import ContradictionV1
from orion.core.schemas.reasoning_policy import EntityLifecycleEvaluationRequestV1
from orion.reasoning import (
    CalibrationRuntimeContext,
    EndogenousCalibrationEngine,
    EndogenousTriggerEvaluator,
    EndogenousWorkflowOrchestrator,
    EndogenousWorkflowPlanner,
    EndogenousOfflineEvaluator,
    CalibrationProfileStore,
    InMemoryCalibrationProfileStore,
    SqlCalibrationProfileStore,
    InMemoryReasoningRepository,
    InMemoryTriggerHistoryStore,
    TriggerPolicy,
    evaluate_entity_lifecycle,
    render_evaluation_report,
)

from .endogenous_runtime_store import (
    EndogenousRuntimeRecordStore,
    InMemoryEndogenousRuntimeRecordStore,
    JsonlEndogenousRuntimeRecordStore,
)
from .endogenous_runtime_sql_reader import EndogenousRuntimeSqlReader
from .settings import settings

logger = logging.getLogger("orion.cortex.exec.endogenous_runtime")

ENDOGENOUS_RUNTIME_RECORD_CHANNEL = "orion:endogenous:runtime:record"
ENDOGENOUS_RUNTIME_AUDIT_CHANNEL = "orion:endogenous:runtime:audit"
CALIBRATION_PROFILE_AUDIT_CHANNEL = "orion:calibration:profile:audit"

ENDOGENOUS_RUNTIME_RECORD_KIND = "endogenous.runtime.record.v1"
ENDOGENOUS_RUNTIME_AUDIT_KIND = "endogenous.runtime.audit.v1"
CALIBRATION_PROFILE_AUDIT_KIND = "calibration.profile.audit.v1"


@dataclass(frozen=True)
class EndogenousRuntimeConfig:
    enabled: bool = False
    surface_chat_reflective_enabled: bool = False
    surface_operator_enabled: bool = False
    allowed_workflow_types: frozenset[str] = frozenset({"contradiction_review", "concept_refinement", "reflective_journal"})
    allow_mentor_branch: bool = False
    sample_rate: float = 1.0
    max_actions: int = 5
    store_backend: str = "memory"
    store_path: str = "/tmp/orion_endogenous_runtime_records.jsonl"
    store_max_records: int = 2000

    @classmethod
    def from_settings(cls) -> "EndogenousRuntimeConfig":
        allowed = {
            item.strip()
            for item in (settings.endogenous_runtime_allowed_workflow_types or "").split(",")
            if item.strip()
        }
        return cls(
            enabled=settings.endogenous_runtime_enabled,
            surface_chat_reflective_enabled=settings.endogenous_runtime_surface_chat_reflective_enabled,
            surface_operator_enabled=settings.endogenous_runtime_surface_operator_enabled,
            allowed_workflow_types=frozenset(allowed or {"contradiction_review", "concept_refinement", "reflective_journal"}),
            allow_mentor_branch=settings.endogenous_runtime_allow_mentor_branch,
            sample_rate=min(1.0, max(0.0, float(settings.endogenous_runtime_sample_rate))),
            max_actions=max(1, min(10, int(settings.endogenous_runtime_max_actions))),
            store_backend=str(settings.endogenous_runtime_store_backend or "memory").lower().strip(),
            store_path=str(settings.endogenous_runtime_store_path or "/tmp/orion_endogenous_runtime_records.jsonl").strip(),
            store_max_records=max(100, int(settings.endogenous_runtime_store_max_records)),
        )


@dataclass(frozen=True)
class RuntimeSignalBuildResult:
    request: EndogenousTriggerRequestV1
    digest: EndogenousRuntimeSignalDigestV1


class EndogenousRuntimeSignalBuilder:
    """Richer bounded signal assembly using reasoning-backed inputs and repository/lifecycle helpers."""

    def build(
        self,
        *,
        ctx: dict[str, Any],
        reasoning_summary: dict[str, Any],
        reflective: dict[str, Any],
        autonomy: dict[str, Any],
        concept: dict[str, Any],
        history: InMemoryTriggerHistoryStore,
    ) -> RuntimeSignalBuildResult:
        repository = ctx.get("reasoning_repository") if isinstance(ctx.get("reasoning_repository"), InMemoryReasoningRepository) else None
        summary_refs = [str(v) for v in (reasoning_summary.get("active_subject_refs") or []) if str(v).strip()]
        subject_ref = str(ctx.get("reasoning_subject_ref") or (summary_refs[0] if summary_refs else "project:orion_sapienform"))

        contradiction_refs = [str(v) for v in (ctx.get("reasoning_contradiction_refs") or []) if str(v).strip()]
        unresolved_contradiction_artifact_ids: list[str] = []
        summary_contradiction_count = len(list(reasoning_summary.get("tensions") or []))
        contradiction_count = summary_contradiction_count
        if repository is not None:
            contradictions = [
                artifact
                for artifact in repository.list_by_type("contradiction", limit=40)
                if isinstance(artifact, ContradictionV1)
                and artifact.resolution_status != "resolved"
                and (not subject_ref or artifact.subject_ref == subject_ref)
            ]
            contradiction_count = max(summary_contradiction_count, len(contradictions))
            unresolved_contradiction_artifact_ids = [item.artifact_id for item in contradictions[:8]]
            if not contradiction_refs:
                contradiction_refs = [item.artifact_id for item in contradictions[:8]]

        lifecycle_state = str(ctx.get("reasoning_lifecycle_state") or "unknown")
        if repository is not None and lifecycle_state == "unknown":
            history_artifacts = repository.list_by_subject_ref(subject_ref, limit=80)
            lifecycle = evaluate_entity_lifecycle(
                EntityLifecycleEvaluationRequestV1(anchor_scope="orion", subject_ref=subject_ref),
                artifacts=history_artifacts,
            )
            action_to_state = {"retire": "retired", "dormant": "dormant", "decay": "dormant", "none": "active", "revive": "active"}
            lifecycle_state = action_to_state.get(lifecycle.lifecycle_action, "active")

        selected_artifact_ids = [str(v) for v in (ctx.get("reasoning_selected_artifact_ids") or []) if str(v).strip()][:12]
        if not selected_artifact_ids and repository is not None:
            selected_artifact_ids = [artifact.artifact_id for artifact in repository.list_by_subject_ref(subject_ref, limit=12)]

        hazards = list(reasoning_summary.get("hazards") or [])
        summary_debug = reasoning_summary.get("debug") if isinstance(reasoning_summary.get("debug"), dict) else {}
        autonomy_summary = autonomy.get("summary") if isinstance(autonomy.get("summary"), dict) else {}

        spark_pressure = min(1.0, len(list(reflective.get("tensions") or [])) / 4.0)
        spark_pressure = round(max(spark_pressure, 0.5 if "unresolved_contradictions_present" in hazards else 0.0), 3)
        autonomy_pressure = round(
            min(
                1.0,
                len(list(autonomy_summary.get("active_tensions") or [])) / 3.0 + len(list(autonomy_summary.get("top_drives") or [])) / 8.0,
            ),
            3,
        )
        concept_fragmentation_score = round(min(1.0, len(list(concept.get("tension") or [])) / 4.0), 3)
        mentor_gap_count = 1 if bool(reasoning_summary.get("fallback_recommended")) else 0
        low_confidence_count = max(0, int(summary_debug.get("suppressed_by_drift") or 0))

        recent_history = history.recent(subject_ref=subject_ref, limit=8)
        request = EndogenousTriggerRequestV1(
            anchor_scope="orion",
            subject_ref=subject_ref,
            selected_artifact_ids=selected_artifact_ids,
            contradiction_refs=contradiction_refs,
            unresolved_contradiction_count=contradiction_count,
            contradiction_severity_score=min(1.0, contradiction_count / 3.0),
            spark_pressure=spark_pressure,
            spark_instability=0.6 if "unresolved_contradictions_present" in hazards else 0.2,
            autonomy_pressure=autonomy_pressure,
            concept_fragmentation_score=concept_fragmentation_score,
            low_confidence_artifact_count=low_confidence_count,
            mentor_gap_count=mentor_gap_count,
            lifecycle_state=lifecycle_state if lifecycle_state in {"active", "dormant", "retired", "unknown"} else "unknown",
            recent_history=recent_history,
            trigger_version="phase10.runtime.v1",
            policy_version="phase10.runtime.policy.v1",
        )

        digest = EndogenousRuntimeSignalDigestV1(
            source_summary_request_id=str(reasoning_summary.get("request_id") or "") or None,
            contradiction_count=contradiction_count,
            contradiction_refs=contradiction_refs[:8],
            unresolved_contradiction_artifact_ids=unresolved_contradiction_artifact_ids[:8],
            lifecycle_state=request.lifecycle_state,
            spark_pressure=request.spark_pressure,
            autonomy_pressure=request.autonomy_pressure,
            concept_fragmentation_score=request.concept_fragmentation_score,
            low_confidence_artifact_count=request.low_confidence_artifact_count,
            mentor_gap_count=request.mentor_gap_count,
            selected_artifact_ids=request.selected_artifact_ids[:8],
            signal_sources={
                "reasoning_summary": "chat_reasoning_summary",
                "repository": "reasoning_repository" if repository is not None else "none",
                "lifecycle": "evaluate_entity_lifecycle",
                "history": f"recent:{len(recent_history)}",
                "autonomy": "chat_autonomy_summary",
                "reflective": "chat_reflective_summary",
            },
        )
        return RuntimeSignalBuildResult(request=request, digest=digest)


class EndogenousRuntimeAdoptionService:
    """Phase 10 bounded runtime service with durable store/replay seam."""

    def __init__(
        self,
        *,
        config: EndogenousRuntimeConfig,
        history: InMemoryTriggerHistoryStore | None = None,
        store: EndogenousRuntimeRecordStore | None = None,
        signal_builder: EndogenousRuntimeSignalBuilder | None = None,
        calibration_profiles: CalibrationProfileStore | None = None,
        event_bus: Any | None = None,
        sql_reader: EndogenousRuntimeSqlReader | None = None,
    ) -> None:
        self._config = config
        self._history = history or InMemoryTriggerHistoryStore()
        self._store = store or self._default_store(config)
        self._signal_builder = signal_builder or EndogenousRuntimeSignalBuilder()
        if calibration_profiles is not None:
            self._calibration_profiles = calibration_profiles
        else:
            try:
                self._calibration_profiles = SqlCalibrationProfileStore(
                    database_url=settings.endogenous_runtime_sql_database_url
                )
            except Exception:
                self._calibration_profiles = InMemoryCalibrationProfileStore()
        self._event_bus = event_bus
        self._sql_reader = sql_reader or EndogenousRuntimeSqlReader(
            enabled=settings.endogenous_runtime_sql_read_enabled,
            database_url=settings.endogenous_runtime_sql_database_url,
        )

    @staticmethod
    def _default_store(config: EndogenousRuntimeConfig) -> EndogenousRuntimeRecordStore:
        if config.store_backend == "jsonl":
            return JsonlEndogenousRuntimeRecordStore(path=config.store_path, max_records=config.store_max_records)
        return InMemoryEndogenousRuntimeRecordStore()

    def maybe_invoke(
        self,
        *,
        ctx: dict[str, Any],
        reasoning_summary: dict[str, Any],
        reflective: dict[str, Any],
        autonomy: dict[str, Any],
        concept: dict[str, Any],
    ) -> dict[str, Any]:
        result = self._maybe_invoke_result(
            ctx=ctx,
            reasoning_summary=reasoning_summary,
            reflective=reflective,
            autonomy=autonomy,
            concept=concept,
        )
        return result.model_dump(mode="json")

    def inspect_recent_records(
        self,
        *,
        limit: int = 25,
        invocation_surface: str | None = None,
        workflow_type: str | None = None,
        outcome: str | None = None,
        audit_status: str | None = None,
        subject_ref: str | None = None,
        anchor_scope: str | None = None,
        mentor_invoked: bool | None = None,
        execution_success: bool | None = None,
        calibration_profile_id: str | None = None,
        correlation_id: str | None = None,
        request_id: str | None = None,
        created_after: datetime | None = None,
    ) -> list[dict[str, Any]]:
        result = self.inspect_recent_records_with_source(
            limit=limit,
            invocation_surface=invocation_surface,
            workflow_type=workflow_type,
            outcome=outcome,
            audit_status=audit_status,
            subject_ref=subject_ref,
            anchor_scope=anchor_scope,
            mentor_invoked=mentor_invoked,
            execution_success=execution_success,
            calibration_profile_id=calibration_profile_id,
            correlation_id=correlation_id,
            request_id=request_id,
            created_after=created_after,
        )
        return list(result["records"])

    def inspect_recent_records_with_source(
        self,
        *,
        limit: int = 25,
        invocation_surface: str | None = None,
        workflow_type: str | None = None,
        outcome: str | None = None,
        audit_status: str | None = None,
        subject_ref: str | None = None,
        anchor_scope: str | None = None,
        mentor_invoked: bool | None = None,
        execution_success: bool | None = None,
        calibration_profile_id: str | None = None,
        correlation_id: str | None = None,
        request_id: str | None = None,
        created_after: datetime | None = None,
    ) -> dict[str, Any]:
        sql = self._sql_reader.runtime_records(
            limit=limit,
            invocation_surface=invocation_surface,
            workflow_type=workflow_type,
            outcome=outcome,
            audit_status=audit_status,
            subject_ref=subject_ref,
            anchor_scope=anchor_scope,
            mentor_invoked=mentor_invoked,
            execution_success=execution_success,
            calibration_profile_id=calibration_profile_id,
            correlation_id=correlation_id,
            request_id=request_id,
            created_after=created_after,
        )
        if sql.source == "sql":
            return {
                "source": "sql",
                "filters": sql.filters,
                "count": len(sql.rows),
                "records": sql.rows,
                "fallback_used": False,
            }

        rows = self._store.list_recent(
            limit=limit,
            invocation_surface=invocation_surface,
            workflow_type=workflow_type,
            outcome=outcome,
            subject_ref=subject_ref,
            mentor_invoked=mentor_invoked,
            created_after=created_after,
        )
        return {
            "source": f"fallback_local:{sql.source}",
            "filters": sql.filters,
            "count": len(rows),
            "records": [row.model_dump(mode="json") for row in rows],
            "fallback_used": True,
            "sql_error": sql.error,
        }

    def consume_for_reflective_review(self, *, query: EndogenousRuntimeQueryV1) -> list[EndogenousRuntimeConsumptionItemV1]:
        fetched = self.inspect_recent_records_with_source(
            limit=query.limit,
            invocation_surface=query.invocation_surface,
            workflow_type=query.workflow_type,
            outcome=query.outcome,
            subject_ref=query.subject_ref,
            mentor_invoked=query.mentor_invoked,
            created_after=query.created_after,
        )
        records = [EndogenousRuntimeExecutionRecordV1.model_validate(item) for item in fetched["records"]]
        return [
            EndogenousRuntimeConsumptionItemV1(
                runtime_record_id=record.runtime_record_id,
                subject_ref=record.subject_ref,
                invocation_surface=record.invocation_surface,
                workflow_type=record.decision.workflow_type,
                outcome=record.decision.outcome,
                mentor_invoked=record.mentor_invoked,
                reasons=list(record.decision.reasons),
                materialized_artifact_ids=list(record.materialized_artifact_ids),
                created_at=record.created_at,
            )
            for record in records
        ]

    def _persist_record(self, record: EndogenousRuntimeExecutionRecordV1) -> bool:
        try:
            self._store.write(record)
            return True
        except Exception as persist_exc:  # pragma: no cover
            logger.warning("endogenous_runtime_record_persist_failed error=%s", persist_exc)
            return False

    def _publish_calibration_audit_event(self, *, ctx: dict[str, Any], event: CalibrationProfileAuditV1) -> bool:
        return self._publish_event(
            channel=CALIBRATION_PROFILE_AUDIT_CHANNEL,
            kind=CALIBRATION_PROFILE_AUDIT_KIND,
            payload=event.model_dump(mode="json"),
            ctx=ctx,
        )

    def _publish_event(self, *, channel: str, kind: str, payload: dict[str, Any], ctx: dict[str, Any]) -> bool:
        bus = ctx.get("orion_bus") or self._event_bus
        if bus is None:
            return False
        try:
            envelope = BaseEnvelope(
                kind=kind,
                correlation_id=self._coerce_correlation_uuid(
                    str(ctx.get("correlation_id") or ctx.get("trace_id") or uuid4())
                ),
                source=ServiceRef(
                    name=settings.service_name,
                    version=settings.service_version,
                    node=settings.node_name,
                ),
                payload=payload,
            )
            self._dispatch_publish(bus=bus, channel=channel, envelope=envelope)
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("endogenous_publish_failed channel=%s kind=%s error=%s", channel, kind, exc)
            return False

    @staticmethod
    def _dispatch_publish(*, bus: Any, channel: str, envelope: BaseEnvelope) -> None:
        publish = getattr(bus, "publish", None)
        if publish is None:
            raise RuntimeError("bus_publish_missing")
        result = publish(channel, envelope)
        if asyncio.iscoroutine(result):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(result)
            except RuntimeError:
                asyncio.run(result)

    @staticmethod
    def _coerce_correlation_uuid(raw: str) -> UUID:
        try:
            return UUID(raw)
        except Exception:
            return uuid5(NAMESPACE_URL, raw)

    def apply_calibration_adoption(self, *, request: CalibrationAdoptionRequestV1) -> dict[str, Any]:
        result = self._calibration_profiles.apply(request)
        latest = self._calibration_profiles.list_audit(limit=1)
        if latest:
            self._publish_calibration_audit_event(
                ctx={"correlation_id": request.profile_id or request.operator_id},
                event=latest[0],
            )
        return result.model_dump(mode="json")

    def rollback_calibration_profile(self, *, request: CalibrationRollbackRequestV1) -> dict[str, Any]:
        result = self._calibration_profiles.rollback(request)
        latest = self._calibration_profiles.list_audit(limit=1)
        if latest:
            self._publish_calibration_audit_event(
                ctx={"correlation_id": request.operator_id},
                event=latest[0],
            )
        return result.model_dump(mode="json")

    def inspect_calibration_profiles(self, *, include_rolled_back: bool = True) -> list[dict[str, Any]]:
        return [profile.model_dump(mode="json") for profile in self._calibration_profiles.list_profiles(include_rolled_back=include_rolled_back)]

    def inspect_calibration_profile_state_with_source(self, *, profile_id: str | None = None) -> dict[str, Any]:
        source = "sql" if isinstance(self._calibration_profiles, SqlCalibrationProfileStore) else "fallback_local:memory"
        fallback_used = source != "sql"
        active = self._calibration_profiles.active_profile()
        profiles = self._calibration_profiles.list_profiles(include_rolled_back=True)
        staged = [item.model_dump(mode="json") for item in profiles if item.state == "staged"]
        selected = next((item for item in profiles if item.profile_id == profile_id), None) if profile_id else None
        active_previous_id = active.previous_profile_id if active else None
        return {
            "source": source,
            "fallback_used": fallback_used,
            "active_profile": active.model_dump(mode="json") if active else None,
            "active_previous_profile_id": active_previous_id,
            "staged_profiles": staged,
            "profile": selected.model_dump(mode="json") if selected else None,
            "profile_found": True if (selected is not None or profile_id is None) else False,
        }

    def inspect_calibration_audit(
        self,
        *,
        limit: int = 50,
        profile_id: str | None = None,
        event_type: str | None = None,
        previous_profile_id: str | None = None,
        operator_id: str | None = None,
        created_after: datetime | None = None,
    ) -> list[dict[str, Any]]:
        result = self.inspect_calibration_audit_with_source(
            limit=limit,
            profile_id=profile_id,
            event_type=event_type,
            previous_profile_id=previous_profile_id,
            operator_id=operator_id,
            created_after=created_after,
        )
        return list(result["records"])

    def inspect_calibration_audit_with_source(
        self,
        *,
        limit: int = 50,
        profile_id: str | None = None,
        event_type: str | None = None,
        previous_profile_id: str | None = None,
        operator_id: str | None = None,
        created_after: datetime | None = None,
    ) -> dict[str, Any]:
        sql = self._sql_reader.calibration_audit(
            limit=limit,
            profile_id=profile_id,
            event_type=event_type,
            previous_profile_id=previous_profile_id,
            operator_id=operator_id,
            created_after=created_after,
        )
        if sql.source == "sql":
            return {
                "source": "sql",
                "filters": sql.filters,
                "count": len(sql.rows),
                "records": sql.rows,
                "fallback_used": False,
            }
        rows = [item.model_dump(mode="json") for item in self._calibration_profiles.list_audit(limit=limit)]
        return {
            "source": f"fallback_local:{sql.source}",
            "filters": sql.filters,
            "count": len(rows),
            "records": rows,
            "fallback_used": True,
            "sql_error": sql.error,
        }

    def resolve_calibration_profile(self, *, invocation_surface: str | None, ctx: dict[str, Any]) -> dict[str, Any]:
        resolved = self._calibration_profiles.resolve(
            context=CalibrationRuntimeContext(
                invocation_surface=invocation_surface,
                mentor_enabled=self._config.allow_mentor_branch,
                operator_surface=invocation_surface == "operator_review",
                seed=str(ctx.get("correlation_id") or ctx.get("trace_id") or ctx.get("session_id") or ""),
            )
        )
        return resolved.model_dump(mode="json")

    def compare_runtime_profile_outcomes(
        self,
        *,
        profile_id: str,
        baseline_profile_id: str | None = None,
        limit: int = 300,
        created_after: datetime | None = None,
    ) -> dict[str, Any]:
        baseline_target = baseline_profile_id or ""
        treatment = self.inspect_recent_records_with_source(
            limit=limit,
            calibration_profile_id=profile_id,
            created_after=created_after,
        )
        baseline = self.inspect_recent_records_with_source(
            limit=limit,
            calibration_profile_id=baseline_target if baseline_profile_id else None,
            created_after=created_after,
        )

        def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
            by_outcome: dict[str, int] = {}
            for row in rows:
                outcome = str((row.get("decision") or {}).get("outcome") or "")
                if not outcome:
                    continue
                by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
            return {"count": len(rows), "by_outcome": dict(sorted(by_outcome.items()))}

        return {
            "source": {"treatment": treatment["source"], "baseline": baseline["source"]},
            "profile_id": profile_id,
            "baseline_profile_id": baseline_profile_id,
            "treatment": _summary(treatment["records"]),
            "baseline": _summary(baseline["records"]),
            "created_after": created_after.isoformat() if created_after else None,
            "fallback_used": bool(treatment.get("fallback_used") or baseline.get("fallback_used")),
        }

    def inspect_operator_debug_surface(
        self,
        *,
        runtime_limit: int = 25,
        runtime_invocation_surface: str | None = None,
        runtime_workflow_type: str | None = None,
        runtime_outcome: str | None = None,
        runtime_audit_status: str | None = None,
        runtime_subject_ref: str | None = None,
        runtime_anchor_scope: str | None = None,
        runtime_mentor_invoked: bool | None = None,
        runtime_execution_success: bool | None = None,
        runtime_calibration_profile_id: str | None = None,
        runtime_created_after: datetime | None = None,
        audit_limit: int = 50,
        audit_profile_id: str | None = None,
        audit_event_type: str | None = None,
        audit_previous_profile_id: str | None = None,
        audit_operator_id: str | None = None,
        audit_created_after: datetime | None = None,
        profile_id: str | None = None,
        comparison_profile_id: str | None = None,
        comparison_baseline_profile_id: str | None = None,
        comparison_limit: int = 300,
        comparison_created_after: datetime | None = None,
    ) -> dict[str, Any]:
        runtime = self.inspect_recent_records_with_source(
            limit=max(1, min(500, runtime_limit)),
            invocation_surface=runtime_invocation_surface,
            workflow_type=runtime_workflow_type,
            outcome=runtime_outcome,
            audit_status=runtime_audit_status,
            subject_ref=runtime_subject_ref,
            anchor_scope=runtime_anchor_scope,
            mentor_invoked=runtime_mentor_invoked,
            execution_success=runtime_execution_success,
            calibration_profile_id=runtime_calibration_profile_id,
            created_after=runtime_created_after,
        )
        calibration_state = self.inspect_calibration_profile_state_with_source(profile_id=profile_id)
        calibration_audit = self.inspect_calibration_audit_with_source(
            limit=max(1, min(500, audit_limit)),
            profile_id=audit_profile_id,
            event_type=audit_event_type,
            previous_profile_id=audit_previous_profile_id,
            operator_id=audit_operator_id,
            created_after=audit_created_after,
        )

        comparisons: dict[str, Any] = {}
        active_profile_id = (calibration_state.get("active_profile") or {}).get("profile_id")
        previous_profile_id = calibration_state.get("active_previous_profile_id")
        selected_profile_id = comparison_profile_id or active_profile_id

        if selected_profile_id:
            comparisons["baseline_vs_active"] = self.compare_runtime_profile_outcomes(
                profile_id=str(selected_profile_id),
                baseline_profile_id=None,
                limit=max(1, min(500, comparison_limit)),
                created_after=comparison_created_after,
            )
        else:
            comparisons["baseline_vs_active"] = {
                "status": "insufficient_data",
                "reason": "no_active_or_selected_profile",
                "source": {"treatment": "none", "baseline": "none"},
            }

        if active_profile_id and previous_profile_id:
            comparisons["previous_vs_current"] = self.compare_runtime_profile_outcomes(
                profile_id=str(active_profile_id),
                baseline_profile_id=str(previous_profile_id),
                limit=max(1, min(500, comparison_limit)),
                created_after=comparison_created_after,
            )
        else:
            comparisons["previous_vs_current"] = {
                "status": "insufficient_data",
                "reason": "no_previous_profile_linkage",
                "source": {"treatment": "none", "baseline": "none"},
            }

        if comparison_profile_id and comparison_baseline_profile_id:
            comparisons["selected_profile_vs_selected_baseline"] = self.compare_runtime_profile_outcomes(
                profile_id=comparison_profile_id,
                baseline_profile_id=comparison_baseline_profile_id,
                limit=max(1, min(500, comparison_limit)),
                created_after=comparison_created_after,
            )

        return {
            "surface": "operator_debug",
            "runtime_records": runtime,
            "calibration_state": calibration_state,
            "calibration_audit": calibration_audit,
            "comparisons": comparisons,
        }

    @staticmethod
    def _policy_for_resolution(overrides: dict[str, str]) -> TriggerPolicy:
        base = TriggerPolicy()
        workflow_cooldowns = dict(base.workflow_cooldowns_seconds or {})

        def _float(key: str, fallback: float) -> float:
            raw = overrides.get(key)
            if raw is None:
                return fallback
            try:
                return max(0.0, min(1.0, float(raw)))
            except ValueError:
                return fallback

        def _int(key: str, fallback: int, *, low: int = 0, high: int = 3600) -> int:
            raw = overrides.get(key)
            if raw is None:
                return fallback
            try:
                return max(low, min(high, int(raw)))
            except ValueError:
                return fallback

        for key, value in overrides.items():
            if key.startswith("cooldown.workflow."):
                workflow = key.removeprefix("cooldown.workflow.")
                try:
                    workflow_cooldowns[workflow] = max(0, min(7200, int(value)))
                except ValueError:
                    continue

        return TriggerPolicy(
            contradiction_threshold=_float("trigger.contradiction_threshold", base.contradiction_threshold),
            concept_threshold=_float("trigger.concept_threshold", base.concept_threshold),
            autonomy_threshold=_float("trigger.autonomy_threshold", base.autonomy_threshold),
            mentor_threshold=_float("trigger.mentor_threshold", base.mentor_threshold),
            reflective_threshold=_float("trigger.reflective_threshold", base.reflective_threshold),
            workflow_cooldowns_seconds=workflow_cooldowns,
            subject_cooldown_seconds=_int("cooldown.subject_cooldown_seconds", base.subject_cooldown_seconds, low=10),
            contradiction_debounce_seconds=_int(
                "cooldown.contradiction_debounce_seconds", base.contradiction_debounce_seconds, low=10
            ),
            mentor_cooldown_seconds=_int("cooldown.mentor_cooldown_seconds", base.mentor_cooldown_seconds, low=30),
        )

    def _maybe_invoke_result(
        self,
        *,
        ctx: dict[str, Any],
        reasoning_summary: dict[str, Any],
        reflective: dict[str, Any],
        autonomy: dict[str, Any],
        concept: dict[str, Any],
    ) -> EndogenousRuntimeResultV1:
        invocation_surface = self._resolve_surface(ctx)
        base = EndogenousRuntimeAuditV1(
            enabled=self._config.enabled,
            invocation_surface=invocation_surface,
            status="disabled" if not self._config.enabled else "not_selected",
            allow_mentor_branch=self._config.allow_mentor_branch,
            allowed_workflow_types=sorted(self._config.allowed_workflow_types),
            sample_rate=self._config.sample_rate,
        )
        if not self._config.enabled:
            return EndogenousRuntimeResultV1(audit=base)
        if not invocation_surface:
            return EndogenousRuntimeResultV1(audit=base)
        if not self._surface_enabled(invocation_surface):
            return EndogenousRuntimeResultV1(audit=base.model_copy(update={"status": "surface_disabled"}))
        if not self._sampled_in(ctx):
            return EndogenousRuntimeResultV1(audit=base.model_copy(update={"status": "sampled_out"}))

        signal = self._signal_builder.build(
            ctx=ctx,
            reasoning_summary=reasoning_summary,
            reflective=reflective,
            autonomy=autonomy,
            concept=concept,
            history=self._history,
        )

        resolution = self._calibration_profiles.resolve(
            context=CalibrationRuntimeContext(
                invocation_surface=invocation_surface,
                mentor_enabled=self._config.allow_mentor_branch,
                operator_surface=invocation_surface == "operator_review",
                seed=str(ctx.get("correlation_id") or ctx.get("trace_id") or ctx.get("session_id") or signal.request.subject_ref or ""),
            )
        )
        evaluator = EndogenousTriggerEvaluator(history=self._history, policy=self._policy_for_resolution(resolution.overrides))
        planner = EndogenousWorkflowPlanner(max_actions=self._config.max_actions)
        orchestrator = EndogenousWorkflowOrchestrator(
            evaluator=evaluator,
            planner=planner,
            history=self._history,
            mentor_gateway=ctx.get("mentor_gateway"),
        )

        started = time.perf_counter()
        try:
            exec_result = orchestrator.orchestrate_runtime(
                signal.request,
                invocation_surface=invocation_surface,
                allowed_workflow_types=set(self._config.allowed_workflow_types),
                allow_mentor_execution=self._config.allow_mentor_branch,
                execute_actions=True,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            record = EndogenousRuntimeExecutionRecordV1(
                invocation_surface=invocation_surface,
                correlation_id=str(ctx.get("correlation_id") or ctx.get("trace_id") or "") or None,
                session_id=str(ctx.get("session_id") or "") or None,
                subject_ref=signal.request.subject_ref,
                trigger_request=signal.request,
                signal_digest=signal.digest,
                decision=exec_result.decision,
                plan=exec_result.plan,
                mentor_invoked=exec_result.mentor_invoked,
                mentor_request_id=(signal.request.request_id if exec_result.mentor_invoked else None),
                materialized_artifact_ids=list(exec_result.materialized_artifact_ids),
                execution_success=True,
                audit_events=list(exec_result.audit_events),
            )
            bus_primary = self._config.store_backend == "bus_sql"
            published_record = self._publish_event(
                channel=ENDOGENOUS_RUNTIME_RECORD_CHANNEL,
                kind=ENDOGENOUS_RUNTIME_RECORD_KIND,
                payload=record.model_dump(mode="json"),
                ctx=ctx,
            )
            persisted = False
            if not bus_primary:
                persisted = self._persist_record(record)
            elif not published_record:
                persisted = self._persist_record(record)
                if persisted:
                    record.audit_events.append("bus_publish_failed_fallback_store")

            status = "ok" if (published_record or persisted) else "persist_failed"
            audit = EndogenousRuntimeAuditV1(
                enabled=True,
                invocation_surface=invocation_surface,
                status=status,
                allow_mentor_branch=self._config.allow_mentor_branch,
                allowed_workflow_types=sorted(self._config.allowed_workflow_types),
                sample_rate=self._config.sample_rate,
                duration_ms=elapsed_ms,
                runtime_record_id=record.runtime_record_id,
                decision_outcome=record.decision.outcome,
                workflow_type=record.decision.workflow_type,
                cooldown_applied=record.decision.cooldown_applied,
                debounce_applied=record.decision.debounce_applied,
                mentor_invoked=record.mentor_invoked,
                materialized_artifact_ids=list(record.materialized_artifact_ids),
                actions=[action.action_type for action in record.plan.actions]
                + [f"calibration_mode:{resolution.mode}", f"calibration_reason:{resolution.reason}"]
                + ([f"bus_publish_record:{'ok' if published_record else 'failed'}"])
                + ([f"calibration_profile_id:{resolution.profile_id}"] if resolution.profile_id else []),
                reasons=list(record.decision.reasons),
            )
            self._publish_event(
                channel=ENDOGENOUS_RUNTIME_AUDIT_CHANNEL,
                kind=ENDOGENOUS_RUNTIME_AUDIT_KIND,
                payload=audit.model_dump(mode="json"),
                ctx=ctx,
            )
            logger.info("endogenous_runtime_review %s", json.dumps(audit.model_dump(mode="json"), sort_keys=True))
            return EndogenousRuntimeResultV1(audit=audit, record=record if (published_record or persisted) else None)
        except Exception as exc:  # pragma: no cover - tested via monkeypatch
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            fallback_decision: EndogenousTriggerDecisionV1 = evaluator.evaluate(signal.request)
            fallback_plan = planner.plan(fallback_decision, signal.request)
            failure_record = EndogenousRuntimeExecutionRecordV1(
                invocation_surface=invocation_surface,
                correlation_id=str(ctx.get("correlation_id") or ctx.get("trace_id") or "") or None,
                session_id=str(ctx.get("session_id") or "") or None,
                subject_ref=signal.request.subject_ref,
                trigger_request=signal.request,
                signal_digest=signal.digest,
                decision=fallback_decision,
                plan=fallback_plan,
                mentor_invoked=False,
                execution_success=False,
                execution_error=str(exc),
                audit_events=[f"execution_error:{exc}"],
            )
            bus_primary = self._config.store_backend == "bus_sql"
            published_record = self._publish_event(
                channel=ENDOGENOUS_RUNTIME_RECORD_CHANNEL,
                kind=ENDOGENOUS_RUNTIME_RECORD_KIND,
                payload=failure_record.model_dump(mode="json"),
                ctx=ctx,
            )
            persisted = False
            if not bus_primary:
                persisted = self._persist_record(failure_record)
            elif not published_record:
                persisted = self._persist_record(failure_record)
            status = "failed" if (published_record or persisted) else "persist_failed"
            audit = EndogenousRuntimeAuditV1(
                enabled=True,
                invocation_surface=invocation_surface,
                status=status,
                allow_mentor_branch=self._config.allow_mentor_branch,
                allowed_workflow_types=sorted(self._config.allowed_workflow_types),
                sample_rate=self._config.sample_rate,
                duration_ms=elapsed_ms,
                runtime_record_id=failure_record.runtime_record_id if persisted else None,
                decision_outcome=fallback_decision.outcome,
                workflow_type=fallback_decision.workflow_type,
                cooldown_applied=fallback_decision.cooldown_applied,
                debounce_applied=fallback_decision.debounce_applied,
                error=str(exc),
                actions=[f"calibration_mode:{resolution.mode}", f"calibration_reason:{resolution.reason}"]
                + ([f"bus_publish_record:{'ok' if published_record else 'failed'}"])
                + ([f"calibration_profile_id:{resolution.profile_id}"] if resolution.profile_id else []),
                reasons=list(fallback_decision.reasons),
            )
            self._publish_event(
                channel=ENDOGENOUS_RUNTIME_AUDIT_CHANNEL,
                kind=ENDOGENOUS_RUNTIME_AUDIT_KIND,
                payload=audit.model_dump(mode="json"),
                ctx=ctx,
            )
            logger.warning("endogenous_runtime_review_failed %s", json.dumps(audit.model_dump(mode="json"), sort_keys=True))
            return EndogenousRuntimeResultV1(audit=audit, record=failure_record if (published_record or persisted) else None)

    def _resolve_surface(self, ctx: dict[str, Any]) -> str | None:
        if ctx.get("endogenous_runtime_operator_review"):
            return "operator_review"
        user_message = str(ctx.get("user_message") or "").lower()
        reflective_markers = ("reflect", "contradiction", "uncertain", "revisit", "self-review")
        if any(marker in user_message for marker in reflective_markers):
            return "chat_reflective_lane"
        return None

    def _surface_enabled(self, surface: str) -> bool:
        if surface == "operator_review":
            return self._config.surface_operator_enabled
        if surface == "chat_reflective_lane":
            return self._config.surface_chat_reflective_enabled
        return False

    def _sampled_in(self, ctx: dict[str, Any]) -> bool:
        if self._config.sample_rate >= 1.0:
            return True
        seed = str(ctx.get("correlation_id") or ctx.get("trace_id") or ctx.get("session_id") or "")
        if not seed:
            seed = str(ctx.get("user_message") or "")
        digest = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        return bucket <= self._config.sample_rate


_RUNTIME_SERVICE: EndogenousRuntimeAdoptionService | None = None


def runtime_service() -> EndogenousRuntimeAdoptionService:
    global _RUNTIME_SERVICE
    if _RUNTIME_SERVICE is None:
        _RUNTIME_SERVICE = EndogenousRuntimeAdoptionService(config=EndogenousRuntimeConfig.from_settings())
    return _RUNTIME_SERVICE


def inspect_endogenous_runtime_records(
    *,
    limit: int = 25,
    invocation_surface: str | None = None,
    workflow_type: str | None = None,
    outcome: str | None = None,
    audit_status: str | None = None,
    subject_ref: str | None = None,
    anchor_scope: str | None = None,
    mentor_invoked: bool | None = None,
    execution_success: bool | None = None,
    calibration_profile_id: str | None = None,
    correlation_id: str | None = None,
    request_id: str | None = None,
    created_after: datetime | None = None,
) -> list[dict[str, Any]]:
    return runtime_service().inspect_recent_records(
        limit=limit,
        invocation_surface=invocation_surface,
        workflow_type=workflow_type,
        outcome=outcome,
        audit_status=audit_status,
        subject_ref=subject_ref,
        anchor_scope=anchor_scope,
        mentor_invoked=mentor_invoked,
        execution_success=execution_success,
        calibration_profile_id=calibration_profile_id,
        correlation_id=correlation_id,
        request_id=request_id,
        created_after=created_after,
    )


def inspect_endogenous_runtime_records_with_source(
    *,
    limit: int = 25,
    invocation_surface: str | None = None,
    workflow_type: str | None = None,
    outcome: str | None = None,
    audit_status: str | None = None,
    subject_ref: str | None = None,
    anchor_scope: str | None = None,
    mentor_invoked: bool | None = None,
    execution_success: bool | None = None,
    calibration_profile_id: str | None = None,
    correlation_id: str | None = None,
    request_id: str | None = None,
    created_after: datetime | None = None,
) -> dict[str, Any]:
    return runtime_service().inspect_recent_records_with_source(
        limit=limit,
        invocation_surface=invocation_surface,
        workflow_type=workflow_type,
        outcome=outcome,
        audit_status=audit_status,
        subject_ref=subject_ref,
        anchor_scope=anchor_scope,
        mentor_invoked=mentor_invoked,
        execution_success=execution_success,
        calibration_profile_id=calibration_profile_id,
        correlation_id=correlation_id,
        request_id=request_id,
        created_after=created_after,
    )


def consume_endogenous_runtime_for_reflective_review(*, query: EndogenousRuntimeQueryV1) -> list[dict[str, Any]]:
    items = runtime_service().consume_for_reflective_review(query=query)
    return [item.model_dump(mode="json") for item in items]


def apply_calibration_adoption(*, request: CalibrationAdoptionRequestV1) -> dict[str, Any]:
    return runtime_service().apply_calibration_adoption(request=request)


def rollback_calibration_adoption(*, request: CalibrationRollbackRequestV1) -> dict[str, Any]:
    return runtime_service().rollback_calibration_profile(request=request)


def inspect_calibration_profiles(*, include_rolled_back: bool = True) -> list[dict[str, Any]]:
    return runtime_service().inspect_calibration_profiles(include_rolled_back=include_rolled_back)


def inspect_calibration_profile_state_with_source(*, profile_id: str | None = None) -> dict[str, Any]:
    return runtime_service().inspect_calibration_profile_state_with_source(profile_id=profile_id)


def inspect_calibration_profile_audit(
    *,
    limit: int = 50,
    profile_id: str | None = None,
    event_type: str | None = None,
    previous_profile_id: str | None = None,
    operator_id: str | None = None,
    created_after: datetime | None = None,
) -> list[dict[str, Any]]:
    return runtime_service().inspect_calibration_audit(
        limit=limit,
        profile_id=profile_id,
        event_type=event_type,
        previous_profile_id=previous_profile_id,
        operator_id=operator_id,
        created_after=created_after,
    )


def inspect_calibration_profile_audit_with_source(
    *,
    limit: int = 50,
    profile_id: str | None = None,
    event_type: str | None = None,
    previous_profile_id: str | None = None,
    operator_id: str | None = None,
    created_after: datetime | None = None,
) -> dict[str, Any]:
    return runtime_service().inspect_calibration_audit_with_source(
        limit=limit,
        profile_id=profile_id,
        event_type=event_type,
        previous_profile_id=previous_profile_id,
        operator_id=operator_id,
        created_after=created_after,
    )


def resolve_runtime_calibration_profile(*, invocation_surface: str | None, ctx: dict[str, Any]) -> dict[str, Any]:
    return runtime_service().resolve_calibration_profile(invocation_surface=invocation_surface, ctx=ctx)


def compare_endogenous_runtime_profile_outcomes(
    *,
    profile_id: str,
    baseline_profile_id: str | None = None,
    limit: int = 300,
    created_after: datetime | None = None,
) -> dict[str, Any]:
    return runtime_service().compare_runtime_profile_outcomes(
        profile_id=profile_id,
        baseline_profile_id=baseline_profile_id,
        limit=limit,
        created_after=created_after,
    )


def inspect_endogenous_operator_debug_surface(
    *,
    runtime_limit: int = 25,
    runtime_invocation_surface: str | None = None,
    runtime_workflow_type: str | None = None,
    runtime_outcome: str | None = None,
    runtime_audit_status: str | None = None,
    runtime_subject_ref: str | None = None,
    runtime_anchor_scope: str | None = None,
    runtime_mentor_invoked: bool | None = None,
    runtime_execution_success: bool | None = None,
    runtime_calibration_profile_id: str | None = None,
    runtime_created_after: datetime | None = None,
    audit_limit: int = 50,
    audit_profile_id: str | None = None,
    audit_event_type: str | None = None,
    audit_previous_profile_id: str | None = None,
    audit_operator_id: str | None = None,
    audit_created_after: datetime | None = None,
    profile_id: str | None = None,
    comparison_profile_id: str | None = None,
    comparison_baseline_profile_id: str | None = None,
    comparison_limit: int = 300,
    comparison_created_after: datetime | None = None,
) -> dict[str, Any]:
    service = runtime_service()
    try:
        return service.inspect_operator_debug_surface(
            runtime_limit=runtime_limit,
            runtime_invocation_surface=runtime_invocation_surface,
            runtime_workflow_type=runtime_workflow_type,
            runtime_outcome=runtime_outcome,
            runtime_audit_status=runtime_audit_status,
            runtime_subject_ref=runtime_subject_ref,
            runtime_anchor_scope=runtime_anchor_scope,
            runtime_mentor_invoked=runtime_mentor_invoked,
            runtime_execution_success=runtime_execution_success,
            runtime_calibration_profile_id=runtime_calibration_profile_id,
            runtime_created_after=runtime_created_after,
            audit_limit=audit_limit,
            audit_profile_id=audit_profile_id,
            audit_event_type=audit_event_type,
            audit_previous_profile_id=audit_previous_profile_id,
            audit_operator_id=audit_operator_id,
            audit_created_after=audit_created_after,
            profile_id=profile_id,
            comparison_profile_id=comparison_profile_id,
            comparison_baseline_profile_id=comparison_baseline_profile_id,
            comparison_limit=comparison_limit,
            comparison_created_after=comparison_created_after,
        )
    except Exception as exc:  # pragma: no cover
        return {
            "surface": "operator_debug",
            "source": "operator_debug:error",
            "fallback_used": True,
            "error": str(exc),
            "runtime_records": {"source": "none", "records": [], "count": 0, "fallback_used": True},
            "calibration_state": {
                "source": "none",
                "fallback_used": True,
                "active_profile": None,
                "staged_profiles": [],
                "profile": None,
                "profile_found": False,
            },
            "calibration_audit": {"source": "none", "records": [], "count": 0, "fallback_used": True},
            "comparisons": {
                "baseline_vs_active": {"status": "insufficient_data", "reason": "operator_debug_failure"},
                "previous_vs_current": {"status": "insufficient_data", "reason": "operator_debug_failure"},
            },
        }


def run_endogenous_offline_evaluation(*, request: EndogenousEvaluationRequestV1) -> dict[str, Any]:
    service = runtime_service()
    fetch_limit = max(request.limit, min(5000, request.min_sample_size * 5))
    invocation_surface = request.invocation_surfaces[0] if len(request.invocation_surfaces) == 1 else None
    workflow_type = request.workflow_types[0] if len(request.workflow_types) == 1 else None
    outcome = request.outcomes[0] if len(request.outcomes) == 1 else None
    rows = service.inspect_recent_records(
        limit=fetch_limit,
        invocation_surface=invocation_surface,
        workflow_type=workflow_type,
        outcome=outcome,
        subject_ref=request.subject_ref,
        mentor_invoked=request.mentor_invoked,
        created_after=request.created_after,
    )
    records = [EndogenousRuntimeExecutionRecordV1.model_validate(item) for item in rows]
    evaluator = EndogenousOfflineEvaluator()
    calibration = EndogenousCalibrationEngine()
    evaluated = evaluator.evaluate(request, records)
    calibrated = calibration.recommend(evaluated)
    return {
        "result": calibrated.model_dump(mode="json"),
        "report_markdown": render_evaluation_report(calibrated),
        "profile_export": calibrated.generated_profile.model_dump(mode="json") if calibrated.generated_profile else None,
    }

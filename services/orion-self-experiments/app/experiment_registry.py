"""Deterministic experiment type → context-exec compile registry."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from orion.cognition.skills_manifest import load_skill_manifest
from orion.schemas.context_exec import (
    ContextExecBudgetV1,
    ContextExecPermissionV1,
    ContextExecRequestV1,
)
from orion.schemas.self_experiments import (
    SelfExperimentCreateRequestV1,
    SelfExperimentMutationPolicy,
    SelfExperimentRecordV1,
    SelfExperimentSource,
    SelfExperimentSpecV1,
    SelfExperimentType,
)

EXPERIMENT_REGISTRY: dict[str, dict[str, str]] = {
    "skill_probe": {
        "context_exec_mode": "investigation_v2",
        "expected_artifact_type": "InvestigationReportV2",
        "permission_profile": "read_only_skill_probe",
        "budget_profile": "small",
        "mutation_policy": "forbidden",
    },
    "runtime_drift_check": {
        "context_exec_mode": "investigation_v2",
        "expected_artifact_type": "InvestigationReportV2",
        "permission_profile": "runtime_read_only",
        "budget_profile": "standard",
        "mutation_policy": "forbidden",
    },
    "belief_origin_check": {
        "context_exec_mode": "belief_provenance",
        "expected_artifact_type": "BeliefProvenanceReportV1",
        "permission_profile": "memory_trace_read_only",
        "budget_profile": "standard",
        "mutation_policy": "forbidden",
    },
    "trace_failure_autopsy": {
        "context_exec_mode": "trace_autopsy",
        "expected_artifact_type": "TraceAutopsyReportV1",
        "permission_profile": "trace_runtime_read_only",
        "budget_profile": "standard",
        "mutation_policy": "forbidden",
    },
    "repo_change_probe": {
        "context_exec_mode": "repo_impact_analysis",
        "expected_artifact_type": "RepoImpactAnalysisReportV1",
        "permission_profile": "repo_read_only",
        "budget_profile": "standard",
        "mutation_policy": "forbidden",
    },
    "daily_focus_grounding_check": {
        "context_exec_mode": "investigation_v2",
        "expected_artifact_type": "InvestigationReportV2",
        "permission_profile": "recall_memory_read_only",
        "budget_profile": "small",
        "mutation_policy": "forbidden",
    },
    "memory_correction_candidate": {
        "context_exec_mode": "memory_correction_proposal",
        "expected_artifact_type": "ProposalEnvelopeV1",
        "permission_profile": "memory_trace_read_only",
        "budget_profile": "standard",
        "mutation_policy": "proposal_only",
    },
    "patch_proposal_candidate": {
        "context_exec_mode": "patch_proposal",
        "expected_artifact_type": "ProposalEnvelopeV1",
        "permission_profile": "repo_read_only",
        "budget_profile": "standard",
        "mutation_policy": "proposal_only",
    },
    "manual_review_candidate": {
        "context_exec_mode": "investigation_v2",
        "expected_artifact_type": "InvestigationReportV2",
        "permission_profile": "minimal_read_only",
        "budget_profile": "small",
        "mutation_policy": "forbidden",
    },
}

PERMISSION_PROFILES: dict[str, dict[str, bool]] = {
    "minimal_read_only": {
        "read_memory": False,
        "read_graph": False,
        "read_recall": True,
        "read_repo": False,
        "read_runtime_logs": False,
        "read_redis_traces": False,
        "write_memory": False,
        "write_graph": False,
        "write_repo": False,
        "mutate_runtime": False,
        "network_enabled": False,
        "shell_enabled": False,
    },
    "read_only_skill_probe": {
        "read_memory": False,
        "read_graph": False,
        "read_recall": True,
        "read_repo": False,
        "read_runtime_logs": False,
        "read_redis_traces": False,
        "write_memory": False,
        "write_graph": False,
        "write_repo": False,
        "mutate_runtime": False,
        "network_enabled": False,
        "shell_enabled": False,
    },
    "runtime_read_only": {
        "read_memory": False,
        "read_graph": False,
        "read_recall": True,
        "read_repo": False,
        "read_runtime_logs": True,
        "read_redis_traces": True,
        "write_memory": False,
        "write_graph": False,
        "write_repo": False,
        "mutate_runtime": False,
        "network_enabled": False,
        "shell_enabled": False,
    },
    "memory_trace_read_only": {
        "read_memory": True,
        "read_graph": True,
        "read_recall": True,
        "read_repo": False,
        "read_runtime_logs": False,
        "read_redis_traces": True,
        "write_memory": False,
        "write_graph": False,
        "write_repo": False,
        "mutate_runtime": False,
        "network_enabled": False,
        "shell_enabled": False,
    },
    "trace_runtime_read_only": {
        "read_memory": False,
        "read_graph": False,
        "read_recall": True,
        "read_repo": False,
        "read_runtime_logs": True,
        "read_redis_traces": True,
        "write_memory": False,
        "write_graph": False,
        "write_repo": False,
        "mutate_runtime": False,
        "network_enabled": False,
        "shell_enabled": False,
    },
    "repo_read_only": {
        "read_memory": False,
        "read_graph": False,
        "read_recall": True,
        "read_repo": True,
        "read_runtime_logs": False,
        "read_redis_traces": False,
        "write_memory": False,
        "write_graph": False,
        "write_repo": False,
        "mutate_runtime": False,
        "network_enabled": False,
        "shell_enabled": False,
    },
    "recall_memory_read_only": {
        "read_memory": True,
        "read_graph": True,
        "read_recall": True,
        "read_repo": False,
        "read_runtime_logs": False,
        "read_redis_traces": False,
        "write_memory": False,
        "write_graph": False,
        "write_repo": False,
        "mutate_runtime": False,
        "network_enabled": False,
        "shell_enabled": False,
    },
}

BUDGET_PROFILES: dict[str, dict[str, float | int]] = {
    "small": {
        "max_seconds": 30,
        "max_hops": 4,
        "max_subcalls": 4,
        "max_depth": 1,
    },
    "standard": {
        "max_seconds": 60,
        "max_hops": 8,
        "max_subcalls": 8,
        "max_depth": 1,
    },
    "deep_read_only": {
        "max_seconds": 120,
        "max_hops": 12,
        "max_subcalls": 12,
        "max_depth": 2,
    },
}

_MUTATION_RANK: dict[str, int] = {
    "forbidden": 0,
    "dry_run_only": 1,
    "proposal_only": 2,
}

_DAILY_SOURCES: frozenset[str] = frozenset({"daily_pulse_v1", "daily_metacog_v1"})


class ExperimentValidationError(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def compute_dedupe_key(
    *,
    experiment_type: str,
    question: str,
    source: str,
    source_ref: str | None,
) -> str:
    payload = {
        "experiment_type": experiment_type,
        "question": question.strip().lower(),
        "source": source,
        "source_ref": source_ref or "",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resolve_source(req: SelfExperimentCreateRequestV1) -> SelfExperimentSource:
    if req.source is not None:
        return req.source
    provenance_source = str((req.provenance or {}).get("source") or "").strip()
    if provenance_source in _DAILY_SOURCES:
        return provenance_source  # type: ignore[return-value]
    if provenance_source == ACTION_DAILY_PULSE:
        return "daily_pulse_v1"
    if provenance_source == ACTION_DAILY_METACOG:
        return "daily_metacog_v1"
    return "manual"


ACTION_DAILY_PULSE = "daily_pulse_v1"
ACTION_DAILY_METACOG = "daily_metacog_v1"


def normalize_create_request(
    req: SelfExperimentCreateRequestV1,
    *,
    experiment_id: str,
    created_at_utc: str,
    allow_non_read_only: bool,
) -> tuple[SelfExperimentSpecV1, str | None]:
    """Return (spec, rejection_reason). rejection_reason is set when invalid."""
    if req.skill_id and not req.experiment_type:
        skill_id = req.skill_id.strip()
        manifest = load_skill_manifest()
        entries = {item.skill_id: item for item in manifest}
        entry = entries.get(skill_id)
        if entry is None:
            raise ExperimentValidationError("unknown_skill_id")
        if not entry.read_only and not allow_non_read_only:
            raise ExperimentValidationError("non_read_only_skill_rejected")
        spec = SelfExperimentSpecV1(
            experiment_id=experiment_id,
            experiment_type="skill_probe",
            question=req.question or f"Run read-only skill probe: {skill_id}",
            rationale=req.rationale,
            source=_resolve_source(req),
            source_ref=req.source_ref or str((req.provenance or {}).get("date") or "") or None,
            correlation_id=req.correlation_id or str((req.provenance or {}).get("correlation_id") or "") or None,
            session_id=req.session_id or "orion_self_experiments",
            user_id=req.user_id or "juniper_primary",
            priority=req.priority,
            requested_skill_id=skill_id,
            requested_context_exec_mode=req.requested_context_exec_mode,
            scopes=dict(req.scopes or {}),
            args=dict(req.args or {}),
            provenance=dict(req.provenance or {}),
            mutation_policy="forbidden",
            created_at_utc=created_at_utc,
        )
        return spec, None

    experiment_type = req.experiment_type or "manual_review_candidate"
    if experiment_type not in EXPERIMENT_REGISTRY:
        raise ExperimentValidationError("unknown_experiment_type")

    question = (req.question or "").strip()
    if not question:
        raise ExperimentValidationError("question_required")

    source = _resolve_source(req)
    registry_policy = EXPERIMENT_REGISTRY[experiment_type]["mutation_policy"]
    if source in _DAILY_SOURCES:
        if registry_policy != "forbidden":
            raise ExperimentValidationError("daily_proposal_type_forbidden")
        if req.mutation_policy and req.mutation_policy != "forbidden":
            raise ExperimentValidationError("daily_mutation_forbidden")

    requested_policy = req.mutation_policy or registry_policy

    if _MUTATION_RANK[requested_policy] > _MUTATION_RANK[registry_policy]:
        raise ExperimentValidationError("mutation_policy_widen_rejected")

    skill_id = (req.requested_skill_id or req.skill_id or "").strip() or None
    if skill_id:
        manifest = load_skill_manifest()
        entries = {item.skill_id: item for item in manifest}
        entry = entries.get(skill_id)
        if entry is None:
            raise ExperimentValidationError("unknown_skill_id")
        is_proposal = registry_policy == "proposal_only"
        if not entry.read_only and not allow_non_read_only and not is_proposal:
            raise ExperimentValidationError("non_read_only_skill_rejected")

    spec = SelfExperimentSpecV1(
        experiment_id=experiment_id,
        experiment_type=experiment_type,  # type: ignore[arg-type]
        question=question,
        rationale=req.rationale,
        source=source,
        source_ref=req.source_ref or str((req.provenance or {}).get("date") or "") or None,
        correlation_id=req.correlation_id or str((req.provenance or {}).get("correlation_id") or "") or None,
        session_id=req.session_id or "orion_self_experiments",
        user_id=req.user_id or "juniper_primary",
        priority=req.priority,
        requested_skill_id=skill_id,
        requested_context_exec_mode=req.requested_context_exec_mode,
        scopes=dict(req.scopes or {}),
        args=dict(req.args or {}),
        provenance=dict(req.provenance or {}),
        mutation_policy=requested_policy,  # type: ignore[arg-type]
        created_at_utc=created_at_utc,
    )
    return spec, None


def validate_spec_for_compile(spec: SelfExperimentSpecV1) -> None:
    if spec.experiment_type not in EXPERIMENT_REGISTRY:
        raise ExperimentValidationError("unknown_experiment_type")

    config = EXPERIMENT_REGISTRY[spec.experiment_type]
    registry_mode = config["context_exec_mode"]
    if spec.requested_context_exec_mode and spec.requested_context_exec_mode != registry_mode:
        raise ExperimentValidationError("context_exec_mode_override_rejected")

    registry_policy = config["mutation_policy"]
    if _MUTATION_RANK[spec.mutation_policy] > _MUTATION_RANK[registry_policy]:
        raise ExperimentValidationError("mutation_policy_widen_rejected")


def compile_experiment_to_context_exec_request(
    record: SelfExperimentRecordV1,
) -> ContextExecRequestV1:
    validate_spec_for_compile(record.spec)
    config = EXPERIMENT_REGISTRY[record.spec.experiment_type]
    permissions = ContextExecPermissionV1(**PERMISSION_PROFILES[config["permission_profile"]])
    budget = ContextExecBudgetV1(**BUDGET_PROFILES[config["budget_profile"]])

    return ContextExecRequestV1(
        request_id=record.experiment_id,
        text=record.spec.question,
        mode=config["context_exec_mode"],  # type: ignore[arg-type]
        correlation_id=record.spec.correlation_id or record.experiment_id,
        session_id=record.spec.session_id,
        user_id=record.spec.user_id,
        scopes={
            **record.spec.scopes,
            "experiment_id": record.experiment_id,
            "experiment_type": record.spec.experiment_type,
            "source": record.spec.source,
            "source_ref": record.spec.source_ref,
            "requested_skill_id": record.spec.requested_skill_id,
        },
        permissions=permissions,
        budget=budget,
        expected_artifact_type=config["expected_artifact_type"],
        llm_profile=record.spec.args.get("llm_profile", "agent"),
    )


def registry_config_for_type(experiment_type: str) -> dict[str, str]:
    return dict(EXPERIMENT_REGISTRY[experiment_type])


def parse_context_exec_result(run_payload: dict[str, Any]) -> dict[str, Any]:
    runtime_debug = run_payload.get("runtime_debug") if isinstance(run_payload.get("runtime_debug"), dict) else {}
    operator_summary = run_payload.get("operator_summary")
    summary_text: str | None = None
    if isinstance(operator_summary, dict):
        summary_text = str(operator_summary.get("summary") or operator_summary.get("headline") or "")
    elif operator_summary is not None:
        summary_text = str(operator_summary)

    proposal_id = runtime_debug.get("proposal_id") or runtime_debug.get("ledger_proposal_id")
    ledger_status = runtime_debug.get("ledger_status") or runtime_debug.get("proposal_status")
    attention_required = bool(runtime_debug.get("attention_required", False))

    artifact = run_payload.get("artifact") if isinstance(run_payload.get("artifact"), dict) else {}
    if not proposal_id and isinstance(artifact, dict):
        proposal_id = artifact.get("proposal_id")

    return {
        "context_exec_run_id": run_payload.get("run_id"),
        "status": run_payload.get("status"),
        "artifact_type": run_payload.get("artifact_type"),
        "operator_summary": summary_text,
        "runtime_debug": runtime_debug,
        "proposal_id": proposal_id,
        "ledger_status": ledger_status,
        "attention_required": attention_required,
        "artifact_payload": artifact if artifact else None,
    }

"""Deterministic experiment-type registry and create-request validation.

Experiments used to compile to a ContextExecRequestV1 and dispatch to
orion-context-exec; that service and the dispatch path were retired 2026-10-10.
What remains is intake: type/mutation-policy validation and dedupe.
"""

from __future__ import annotations

import hashlib
import json

from orion.cognition.skills_manifest import load_skill_manifest
from orion.schemas.self_experiments import (
    SelfExperimentCreateRequestV1,
    SelfExperimentSource,
    SelfExperimentSpecV1,
)

EXPERIMENT_REGISTRY: dict[str, dict[str, str]] = {
    "skill_probe": {
        "mutation_policy": "forbidden",
    },
    "runtime_drift_check": {
        "mutation_policy": "forbidden",
    },
    "belief_origin_check": {
        "mutation_policy": "forbidden",
    },
    "trace_failure_autopsy": {
        "mutation_policy": "forbidden",
    },
    "repo_change_probe": {
        "mutation_policy": "forbidden",
    },
    "daily_focus_grounding_check": {
        "mutation_policy": "forbidden",
    },
    "memory_correction_candidate": {
        "mutation_policy": "proposal_only",
    },
    "patch_proposal_candidate": {
        "mutation_policy": "proposal_only",
    },
    "manual_review_candidate": {
        "mutation_policy": "forbidden",
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

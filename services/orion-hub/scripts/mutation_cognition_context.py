from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

from orion.substrate.mutation_control_surface import inspect_chat_reflective_lane_threshold
from orion.substrate.mutation_queue import SubstrateMutationStore


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = str(os.getenv(name, "true" if default else "false")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _latest_routing_artifact(store: SubstrateMutationStore, kind: str) -> dict[str, Any] | None:
    if kind == "proposal":
        rows = [row for row in store._proposals.values() if row.mutation_class == "routing_threshold_patch"]
    elif kind == "decision":
        rows = [
            row
            for row in store._decisions.values()
            if (
                (proposal := store.get_proposal(row.proposal_id)) is not None
                and proposal.mutation_class == "routing_threshold_patch"
            )
        ]
    elif kind == "adoption":
        rows = [
            row
            for row in store._adoptions.values()
            if (
                (proposal := store.get_proposal(row.proposal_id)) is not None
                and proposal.mutation_class == "routing_threshold_patch"
            )
        ]
    else:
        rows = [
            row
            for row in store._rollbacks.values()
            if (
                (proposal := store.get_proposal(row.proposal_id)) is not None
                and proposal.mutation_class == "routing_threshold_patch"
            )
        ]
    if not rows:
        return None
    latest = sorted(rows, key=lambda item: getattr(item, "created_at", datetime.min), reverse=True)[0]
    return latest.model_dump(mode="json")


def build_mutation_cognition_context(*, store: SubstrateMutationStore | None = None) -> Dict[str, Any]:
    mutation_store = store or SubstrateMutationStore()
    live_surface = inspect_chat_reflective_lane_threshold()
    routing_trials = [row for row in mutation_store._trials.values() if row.mutation_class == "routing_threshold_patch"]
    latest_trial = sorted(routing_trials, key=lambda item: item.created_at, reverse=True)[0] if routing_trials else None
    metrics = latest_trial.metrics if latest_trial is not None else {}
    active_shadow_profile = mutation_store.active_recall_shadow_profile()
    latest_recall_proposal = next(
        (
            item
            for item in sorted(mutation_store._proposals.values(), key=lambda row: row.created_at, reverse=True)
            if str(item.mutation_class).startswith("recall_") and str(item.mutation_class).endswith("_candidate")
        ),
        None,
    )
    latest_staged_profile = next(
        (
            item
            for item in sorted(mutation_store._recall_strategy_profiles.values(), key=lambda row: row.updated_at, reverse=True)
            if item.status in {"staged", "shadow_active"}
        ),
        None,
    )
    latest_eval_run = next(
        (
            item
            for item in sorted(
                mutation_store._recall_shadow_eval_runs.values(),
                key=lambda row: row.completed_at,
                reverse=True,
            )
            if active_shadow_profile is None or item.profile_id == active_shadow_profile.profile_id
        ),
        None,
    )
    latest_candidate_review = next(
        (
            item
            for item in sorted(
                mutation_store._recall_production_candidate_reviews.values(),
                key=lambda row: row.updated_at,
                reverse=True,
            )
            if active_shadow_profile is None or item.profile_id == active_shadow_profile.profile_id
        ),
        None,
    )
    readiness = dict((active_shadow_profile.readiness_snapshot if active_shadow_profile is not None else {}))
    active_stance_notes = [
        item
        for item in sorted(mutation_store._cognitive_stance_notes.values(), key=lambda row: row.updated_at, reverse=True)
        if item.status == "active"
    ][:8]
    bounded_notes = [
        {
            "stance_note_id": row.stance_note_id,
            "source_proposal_id": row.source_proposal_id,
            "source_draft_id": row.source_draft_id,
            "summary": str(row.summary)[:280],
            "note": str(row.note)[:800],
            "visibility": row.visibility,
            "ttl_turns": row.ttl_turns,
            "context_role": "operator_accepted_cognitive_draft_context",
            "authoritative": False,
            "lineage": dict(row.lineage or {}),
            "updated_at": row.updated_at.isoformat(),
        }
        for row in active_stance_notes
    ]
    return {
        "mutation_scope": "routing_threshold_patch_only",
        "live_ramp_active": bool(
            _env_flag("SUBSTRATE_AUTONOMY_ENABLED", default=False)
            and _env_flag("SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED", default=True)
        ),
        "routing_proposals_enabled": _env_flag("SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED", default=True),
        "routing_apply_enabled": bool(
            _env_flag("SUBSTRATE_AUTONOMY_APPLY_ENABLED", default=False)
            and _env_flag("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", default=False)
        ),
        "live_surface": live_surface,
        "latest_routing_proposal": _latest_routing_artifact(mutation_store, "proposal"),
        "latest_routing_decision": _latest_routing_artifact(mutation_store, "decision"),
        "latest_routing_adoption": _latest_routing_artifact(mutation_store, "adoption"),
        "latest_routing_rollback": _latest_routing_artifact(mutation_store, "rollback"),
        "evaluator_confidence": metrics.get("evaluator_confidence"),
        "corpus_coverage": metrics.get("corpus_coverage"),
        "active_shadow_recall_profile_id": active_shadow_profile.profile_id if active_shadow_profile is not None else None,
        "recall_strategy_readiness_recommendation": readiness.get("recommendation"),
        "recall_readiness_gates_blocked": list(readiness.get("gates_blocked") or [])[:16],
        "recall_corpus_coverage": readiness.get("corpus_coverage"),
        "recall_irrelevant_cousin_rate": readiness.get("irrelevant_cousin_rate"),
        "recall_explainability_completeness": readiness.get("explainability_completeness"),
        "latest_recall_proposal_id": latest_recall_proposal.proposal_id if latest_recall_proposal is not None else None,
        "latest_staged_profile_status": latest_staged_profile.status if latest_staged_profile is not None else None,
        "last_recall_shadow_eval_run_status": latest_eval_run.status if latest_eval_run is not None else None,
        "last_recall_readiness_recommendation": readiness.get("recommendation"),
        "latest_production_candidate_review_recommendation": (
            latest_candidate_review.recommendation if latest_candidate_review is not None else None
        ),
        "latest_production_candidate_review_status": (
            latest_candidate_review.status if latest_candidate_review is not None else None
        ),
        "cognitive_stance_notes": bounded_notes,
        "cognitive_stance_notes_summary": {
            "active_count": len(active_stance_notes),
            "included_count": len(bounded_notes),
            "authoritative": False,
        },
        "production_recall_mode": "v1",
        "recall_live_apply_enabled": False,
    }

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("orion-hub.autonomy_constitution")

PRODUCTION_RECALL_MODE = "v1"
RECALL_LIVE_APPLY_ENABLED = False
COGNITIVE_LIVE_APPLY_ENABLED = False


class AutonomyPolicySurfaceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface: str
    category: str
    propose: str
    trial: str
    apply: str
    rollback: str
    human_required: bool = True
    status: str
    required_gates: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    description: str = ""
    rationale: str = ""
    risk_tier: Literal["low", "medium", "high", "critical"] = "high"
    live_apply_allowed: bool = False
    autonomous_apply_allowed: bool = False
    production_write_allowed: bool = False


class AutonomyConstitutionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["autonomy_constitution.v1"] = "autonomy_constitution.v1"
    loaded_at: str
    source: str = "services/orion-hub/scripts/autonomy_constitution.py"
    surfaces: list[AutonomyPolicySurfaceV1] = Field(default_factory=list)
    safety_invariants: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _base_surfaces() -> list[AutonomyPolicySurfaceV1]:
    return [
        AutonomyPolicySurfaceV1(
            surface="routing_threshold_patch",
            category="routing",
            propose="auto_when_gated",
            trial="auto_replay",
            apply="gated_auto",
            rollback="auto",
            human_required=False,
            status="live_narrow",
            required_gates=[
                "SUBSTRATE_AUTONOMY_ENABLED",
                "SUBSTRATE_AUTONOMY_APPLY_ENABLED",
                "SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED",
            ],
            forbidden=[],
            description="Narrow bounded live routing threshold control.",
            rationale="Only surface permitted for gated live apply.",
            risk_tier="medium",
            live_apply_allowed=True,
            autonomous_apply_allowed=True,
            production_write_allowed=False,
        ),
        AutonomyPolicySurfaceV1(
            surface="recall_strategy_profile",
            category="recall",
            propose="auto",
            trial="shadow_canary_manual_eval",
            apply="manual_review_only",
            rollback="disable_shadow_or_canary",
            human_required=True,
            status="shadow_canary_review_only",
            forbidden=[
                "autonomous_recall_production_promotion",
                "recall_weighting_patch_live_apply",
                "silent_recall_v2_default_switch",
            ],
            description="Recall strategy candidate lifecycle in shadow/canary/review lanes.",
            rationale="Recall production defaults remain stable unless explicit out-of-band operator process.",
            risk_tier="high",
            live_apply_allowed=False,
            autonomous_apply_allowed=False,
            production_write_allowed=False,
        ),
        AutonomyPolicySurfaceV1(
            surface="recall_weighting_patch",
            category="recall",
            propose="allowed_if_existing",
            trial="shadow_only",
            apply="forbidden",
            rollback="n_a",
            human_required=True,
            status="blocked_live_apply",
            forbidden=["live_apply", "autonomous_apply"],
            description="Recall weighting mutation class remains non-live-applicable.",
            rationale="Tripwire-protected safety posture for recall production.",
            risk_tier="critical",
            live_apply_allowed=False,
            autonomous_apply_allowed=False,
            production_write_allowed=False,
        ),
        AutonomyPolicySurfaceV1(
            surface="cognitive_self_model",
            category="cognitive",
            propose="auto_or_operator_gated",
            trial="draft_context_only",
            apply="forbidden",
            rollback="archive_or_supersede_draft",
            human_required=True,
            status="proposal_draft_context_only",
            forbidden=[
                "identity_kernel_rewrite",
                "production_self_model_rewrite",
                "policy_override",
                "freeform_prompt_self_rewrite",
                "live_cognitive_apply",
            ],
            description="Cognitive proposal review lane for operator decisions and bounded draft artifacts.",
            rationale="No authoritative cognitive write surface is exposed.",
            risk_tier="critical",
            live_apply_allowed=False,
            autonomous_apply_allowed=False,
            production_write_allowed=False,
        ),
        AutonomyPolicySurfaceV1(
            surface="cognitive_stance_note",
            category="cognitive",
            propose="operator_review_only",
            trial="bounded_context",
            apply="context_only",
            rollback="archive_note",
            human_required=True,
            status="bounded_non_authoritative_context",
            forbidden=[
                "identity_kernel_write",
                "policy_write",
                "production_prompt_write",
                "authoritative_self_model_write",
            ],
            description="Bounded non-authoritative context notes for cognition lane.",
            rationale="Stance notes remain informational and expire/archive safely.",
            risk_tier="high",
            live_apply_allowed=False,
            autonomous_apply_allowed=False,
            production_write_allowed=False,
        ),
        AutonomyPolicySurfaceV1(
            surface="identity_kernel",
            category="identity",
            propose="forbidden_or_manual_only",
            trial="none",
            apply="forbidden",
            rollback="n_a",
            human_required=True,
            status="protected",
            forbidden=["autonomous_rewrite", "mutation_runtime_write", "prompt_self_rewrite"],
            description="Identity kernel is protected from mutation-runtime write surfaces.",
            rationale="Identity continuity must remain externally governed.",
            risk_tier="critical",
            live_apply_allowed=False,
            autonomous_apply_allowed=False,
            production_write_allowed=False,
        ),
        AutonomyPolicySurfaceV1(
            surface="policy_safety",
            category="policy",
            propose="forbidden_or_manual_only",
            trial="none",
            apply="forbidden",
            rollback="n_a",
            human_required=True,
            status="protected",
            forbidden=["autonomous_policy_override", "mutation_runtime_write"],
            description="Policy safety constraints cannot be rewritten via mutation runtime.",
            rationale="Constitution-level safety controls remain immutable in this lane.",
            risk_tier="critical",
            live_apply_allowed=False,
            autonomous_apply_allowed=False,
            production_write_allowed=False,
        ),
        AutonomyPolicySurfaceV1(
            surface="production_prompt",
            category="prompting",
            propose="forbidden_or_manual_only",
            trial="none",
            apply="forbidden",
            rollback="n_a",
            human_required=True,
            status="protected",
            forbidden=[
                "autonomous_prompt_rewrite",
                "freeform_prompt_self_rewrite",
                "mutation_runtime_write",
            ],
            description="Production prompting surfaces are protected from autonomy runtime writes.",
            rationale="Prevents unsafe prompt self-rewrite loops.",
            risk_tier="critical",
            live_apply_allowed=False,
            autonomous_apply_allowed=False,
            production_write_allowed=False,
        ),
    ]


def validate_autonomy_constitution(constitution: AutonomyConstitutionV1) -> list[str]:
    warnings: list[str] = []
    by_surface = {row.surface: row for row in constitution.surfaces}
    live_apply = [row.surface for row in constitution.surfaces if row.live_apply_allowed]
    if live_apply != ["routing_threshold_patch"]:
        warnings.append(f"live_apply_surface_violation:{','.join(live_apply) or 'none'}")
    for row in constitution.surfaces:
        if row.category == "cognitive" and row.live_apply_allowed:
            warnings.append(f"cognitive_live_apply_violation:{row.surface}")
        if row.surface.startswith("recall_") and row.autonomous_apply_allowed:
            warnings.append(f"recall_autonomous_apply_violation:{row.surface}")
        if row.category in {"identity", "policy", "prompting"} and row.production_write_allowed:
            warnings.append(f"protected_surface_write_violation:{row.surface}")
    if "cognitive_self_model" in by_surface and by_surface["cognitive_self_model"].apply != "forbidden":
        warnings.append("cognitive_self_model_apply_must_be_forbidden")
    if "recall_weighting_patch" in by_surface and by_surface["recall_weighting_patch"].apply != "forbidden":
        warnings.append("recall_weighting_patch_apply_must_be_forbidden")
    logger.info(
        "event=autonomy_constitution_validation_completed surface_count=%s live_apply_surface_count=%s warning_count=%s validation_passed=%s",
        len(constitution.surfaces),
        len(live_apply),
        len(warnings),
        str(not warnings).lower(),
    )
    return warnings


def constitution_summary(constitution: AutonomyConstitutionV1) -> dict[str, list[str]]:
    surfaces = constitution.surfaces
    return {
        "live_apply_surfaces": [row.surface for row in surfaces if row.live_apply_allowed],
        "blocked_surfaces": [row.surface for row in surfaces if row.apply == "forbidden"],
        "protected_surfaces": [row.surface for row in surfaces if row.status == "protected"],
        "human_required_surfaces": [row.surface for row in surfaces if row.human_required],
    }


def load_autonomy_constitution() -> AutonomyConstitutionV1:
    constitution = AutonomyConstitutionV1(
        loaded_at=datetime.now(timezone.utc).isoformat(),
        surfaces=_base_surfaces(),
        safety_invariants=[
            "routing_threshold_patch is the only live apply surface",
            "recall production default remains v1 unless explicit external operator process changes it",
            "recall autonomous production promotion is forbidden",
            "recall_weighting_patch live apply is forbidden",
            "cognitive live apply is forbidden",
            "identity kernel rewrite is forbidden from substrate mutation runtime",
            "policy override is forbidden from substrate mutation runtime",
            "production prompt rewrite is forbidden from substrate mutation runtime",
            "stance notes are bounded non-authoritative context only",
            "status/list/readiness endpoints are read-only",
            "canary/review endpoints must not trigger mutation execute-once",
        ],
    )
    warnings = validate_autonomy_constitution(constitution)
    constitution = constitution.model_copy(update={"warnings": warnings})
    summary = constitution_summary(constitution)
    logger.info(
        "event=autonomy_constitution_loaded surface_count=%s live_apply_surface_count=%s blocked_surface_count=%s protected_surface_count=%s warning_count=%s",
        len(constitution.surfaces),
        len(summary["live_apply_surfaces"]),
        len(summary["blocked_surfaces"]),
        len(summary["protected_surfaces"]),
        len(constitution.warnings),
    )
    return constitution

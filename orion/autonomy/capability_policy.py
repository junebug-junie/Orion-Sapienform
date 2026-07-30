from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from orion.autonomy.models import (
    CapabilityDecisionV1,
    CapabilityPolicyRuleV1,
    CapabilityPolicyV1,
)
from orion.schemas.field_goal import FieldGoalProvenanceV1

_TRUTHY = {"1", "true", "yes", "on"}
_GOAL_STATUS_ORDER = {"none": 0, "proposed": 1, "planned": 2, "executing": 3}
_PLANNED_STATUS_LEVEL = _GOAL_STATUS_ORDER["planned"]
_EPISODE_JOURNAL_CAPABILITY = "journal.compose.episode"
_DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "autonomy" / "capability_policy.v1.yaml"
)


@dataclass
class CapabilityEvaluationContext:
    # `predictive_pressure` (DriveStateV1.pressures["predictive"]) was removed
    # 2026-07-30 alongside the rest of the drive-pressure system (chore/
    # delete-orion-drives Wave 1 deleted its sole producer, DriveEngine).
    # `curiosity_strength` below is the real, field-native replacement for
    # "is there genuine motivation for this auto-execute" -- it is sourced
    # from FrontierInvocationSignalV1.signal_strength on real
    # world_coverage_gap signals, independent of drives. No new metric was
    # minted here per CLAUDE.md's metric-quality-gate; the existing
    # field-native signal already covers the readonly-fetch gate. The
    # episode-journal gate (_layer_a_episode_journal_enabled) had no such
    # substitute available and is now gated by its two feature flags alone
    # -- see that function's docstring.
    curiosity_strength: float
    signal_kinds: list[str]
    # SSP §6 Objective 6 (2026-07-30): real, field-native active goal from
    # orion.autonomy.goal_state.get_active_goal(), not the old GoalProposalV1
    # synthetic-per-call stub. None means "no real goal currently dominant" --
    # an honest absence (missing_goal reason code below), not a fabricated one.
    goal: FieldGoalProvenanceV1 | None
    budget_used: dict[str, int] = field(default_factory=dict)
    # Real, ambient, mesh-wide surprise signal -- see
    # docs/superpowers/specs/2026-07-24-efe-capability-gate-design.md's 2026-07-28 re-scope.
    # Sourced from bus_synaptic_prediction_error() (orion/substrate/bus_synaptic_surprise.py),
    # not a per-domain judgment-call mapping -- the same real value regardless of which
    # capability is being evaluated. `None` means "not available this call" (caller didn't
    # supply a domain_surprise_source, or the real read failed/was stale) -- honest absence,
    # never silently coerced to 0.0 here.
    domain_surprise_score: float | None = None
    domain_surprise_source: str | None = None  # always "bus_synaptic" when score is present


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUTHY


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _goal_status_level(status: str) -> int:
    return _GOAL_STATUS_ORDER.get(str(status or "").strip().lower(), 0)


def _decision(
    capability_id: str,
    *,
    outcome: str,
    reason_code: str,
    auto_execute: bool = False,
    notes: list[str] | None = None,
) -> CapabilityDecisionV1:
    return CapabilityDecisionV1(
        capability_id=capability_id,
        outcome=outcome,  # type: ignore[arg-type]
        reason_code=reason_code,
        auto_execute=auto_execute,
        notes=notes or [],
    )


@lru_cache(maxsize=1)
def load_capability_policy() -> CapabilityPolicyV1:
    data = yaml.safe_load(_DEFAULT_POLICY_PATH.read_text(encoding="utf-8")) or {}
    return CapabilityPolicyV1.model_validate(data)


def _find_rule(policy: CapabilityPolicyV1, capability_id: str) -> CapabilityPolicyRuleV1 | None:
    for rule in policy.rules:
        if rule.capability_id == capability_id:
            return rule
    return None


def _layer_a_readonly_auto_enabled(ctx: CapabilityEvaluationContext) -> tuple[bool, str]:
    if not _env_bool("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", default=False):
        return False, "policy_auto_disabled"
    min_curiosity = _env_float("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", 0.5)
    if ctx.curiosity_strength < min_curiosity:
        return False, "curiosity_strength_insufficient"
    return True, "layer_a_satisfied"


def _layer_a_episode_journal_enabled(ctx: CapabilityEvaluationContext) -> tuple[bool, str]:
    # No curiosity_strength (or other real) threshold gates this path -- the
    # caller (maybe_compose_autonomy_episode_after_fetch) has always passed
    # curiosity_strength=0.0 here (never a real value), and the removed
    # predictive_pressure check was the only threshold this gate ever had.
    # Left as two explicit feature flags, both off by default, rather than
    # inventing a replacement threshold with no theory anchor per CLAUDE.md's
    # metric-quality-gate. Flagged in the Wave 2a PR report as worth a
    # follow-up decision if a real gate is wanted here.
    del ctx  # kept for signature symmetry with _layer_a_readonly_auto_enabled
    if not _env_bool("ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED", default=False):
        return False, "episode_journal_disabled"
    if not _env_bool("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", default=False):
        return False, "policy_auto_disabled"
    return True, "layer_a_satisfied"


def _domain_surprise_gate(
    ctx: CapabilityEvaluationContext, rule: CapabilityPolicyRuleV1
) -> tuple[bool, str]:
    """Real, independent condition on the ambient bus_synaptic surprise signal -- never
    paired with, compared against, or gated on `required_drive_origins`. Absence of the
    field is deliberately not "no threshold, so pass" -- when a rule DOES require a
    threshold and the signal wasn't supplied, that's an honest denial, not a silent pass.
    """
    if rule.required_domain_surprise_below is None:
        return True, "no_surprise_gate"
    if ctx.domain_surprise_score is None:
        return False, "domain_surprise_unavailable"
    if ctx.domain_surprise_score >= rule.required_domain_surprise_below:
        return False, "domain_surprise_insufficient"
    return True, "domain_surprise_satisfied"


def _domain_surprise_note(ctx: CapabilityEvaluationContext) -> str | None:
    """Advisory-only observability (Acceptance check #5): surface the real value in
    CapabilityDecisionV1.notes on every decision, regardless of whether any rule actually
    gates on it yet -- lets the signal be observed against real traffic before anything
    flips to a hard gate.
    """
    if ctx.domain_surprise_score is None:
        return None
    return (
        f"domain_surprise_score={ctx.domain_surprise_score:.4f} "
        f"source={ctx.domain_surprise_source or 'unknown'}"
    )


def evaluate_capability(capability_id: str, ctx: CapabilityEvaluationContext) -> CapabilityDecisionV1:
    # Computed once, up front, and attached to EVERY returned decision below --
    # Acceptance check #5 ("observe against real traffic before hard-gating") needs the
    # real value visible on every real call, not just the ones that reach the final
    # allowed/surprise-gate return. Fixed 2026-07-28 after review found the note was
    # previously unreachable on requires_promote/earlier-denial paths, silently biasing
    # any correlation analysis built on it toward already-readonly-eligible traffic.
    surprise_note = _domain_surprise_note(ctx)
    notes = [surprise_note] if surprise_note else None

    policy = load_capability_policy()
    rule = _find_rule(policy, capability_id)
    if rule is None:
        return _decision(capability_id, outcome="denied", reason_code="unknown_capability", notes=notes)

    if rule.budget_per_cycle > 0 and ctx.budget_used.get(capability_id, 0) >= rule.budget_per_cycle:
        return _decision(
            capability_id, outcome="denied", reason_code="capability_budget_exhausted", notes=notes
        )

    requires_goal = _goal_status_level(rule.requires_goal_status) > 0
    if requires_goal and ctx.goal is None:
        return _decision(capability_id, outcome="denied", reason_code="missing_goal", notes=notes)

    if rule.required_signal_kinds:
        present = set(ctx.signal_kinds)
        if not set(rule.required_signal_kinds).issubset(present):
            return _decision(
                capability_id, outcome="denied", reason_code="missing_signal_kinds", notes=notes
            )

    if ctx.goal is not None:
        required_level = _goal_status_level(rule.requires_goal_status)
        if _goal_status_level(ctx.goal.proposal_status) < required_level:
            return _decision(
                capability_id, outcome="denied", reason_code="goal_status_insufficient", notes=notes
            )

    if rule.side_effect_class == "external":
        goal_level = _goal_status_level(ctx.goal.proposal_status) if ctx.goal is not None else 0
        if goal_level < _PLANNED_STATUS_LEVEL:
            return _decision(
                capability_id, outcome="requires_promote", reason_code="requires_promote", notes=notes
            )
    elif rule.side_effect_class == "write" and capability_id != _EPISODE_JOURNAL_CAPABILITY:
        goal_level = _goal_status_level(ctx.goal.proposal_status) if ctx.goal is not None else 0
        if goal_level < _PLANNED_STATUS_LEVEL:
            return _decision(
                capability_id, outcome="requires_promote", reason_code="requires_promote", notes=notes
            )

    if rule.auto_execute and rule.side_effect_class == "readonly":
        ok, reason = _layer_a_readonly_auto_enabled(ctx)
        if not ok:
            return _decision(capability_id, outcome="denied", reason_code=reason, notes=notes)
    elif rule.auto_execute and capability_id == _EPISODE_JOURNAL_CAPABILITY:
        ok, reason = _layer_a_episode_journal_enabled(ctx)
        if not ok:
            return _decision(capability_id, outcome="denied", reason_code=reason, notes=notes)

    ok, reason = _domain_surprise_gate(ctx, rule)
    if not ok:
        return _decision(capability_id, outcome="denied", reason_code=reason, notes=notes)

    return _decision(
        capability_id,
        outcome="allowed",
        reason_code="allowed",
        auto_execute=rule.auto_execute,
        notes=notes,
    )

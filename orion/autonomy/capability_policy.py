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
from orion.schemas.attention_candidate import is_world_source_id
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
) -> CapabilityDecisionV1:
    return CapabilityDecisionV1(
        capability_id=capability_id,
        outcome=outcome,  # type: ignore[arg-type]
        reason_code=reason_code,
        auto_execute=auto_execute,
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


def evaluate_capability(capability_id: str, ctx: CapabilityEvaluationContext) -> CapabilityDecisionV1:
    policy = load_capability_policy()
    rule = _find_rule(policy, capability_id)
    if rule is None:
        return _decision(capability_id, outcome="denied", reason_code="unknown_capability")

    if rule.budget_per_cycle > 0 and ctx.budget_used.get(capability_id, 0) >= rule.budget_per_cycle:
        return _decision(
            capability_id, outcome="denied", reason_code="capability_budget_exhausted"
        )

    requires_goal = _goal_status_level(rule.requires_goal_status) > 0
    if requires_goal and ctx.goal is None:
        return _decision(capability_id, outcome="denied", reason_code="missing_goal")

    if rule.required_signal_kinds:
        present = set(ctx.signal_kinds)
        if not set(rule.required_signal_kinds).issubset(present):
            return _decision(
                capability_id, outcome="denied", reason_code="missing_signal_kinds"
            )

    if ctx.goal is not None:
        required_level = _goal_status_level(rule.requires_goal_status)
        if _goal_status_level(ctx.goal.proposal_status) < required_level:
            return _decision(
                capability_id, outcome="denied", reason_code="goal_status_insufficient"
            )

    if rule.requires_world_goal and (
        ctx.goal is None or not is_world_source_id(ctx.goal.field_target_id)
    ):
        return _decision(
            capability_id, outcome="denied", reason_code="goal_not_about_world"
        )

    if rule.side_effect_class == "external":
        goal_level = _goal_status_level(ctx.goal.proposal_status) if ctx.goal is not None else 0
        if goal_level < _PLANNED_STATUS_LEVEL:
            return _decision(
                capability_id, outcome="requires_promote", reason_code="requires_promote"
            )
    elif rule.side_effect_class == "write" and capability_id != _EPISODE_JOURNAL_CAPABILITY:
        goal_level = _goal_status_level(ctx.goal.proposal_status) if ctx.goal is not None else 0
        if goal_level < _PLANNED_STATUS_LEVEL:
            return _decision(
                capability_id, outcome="requires_promote", reason_code="requires_promote"
            )

    if rule.auto_execute and rule.side_effect_class == "readonly":
        ok, reason = _layer_a_readonly_auto_enabled(ctx)
        if not ok:
            return _decision(capability_id, outcome="denied", reason_code=reason)
    elif rule.auto_execute and capability_id == _EPISODE_JOURNAL_CAPABILITY:
        ok, reason = _layer_a_episode_journal_enabled(ctx)
        if not ok:
            return _decision(capability_id, outcome="denied", reason_code=reason)

    return _decision(
        capability_id,
        outcome="allowed",
        reason_code="allowed",
        auto_execute=rule.auto_execute,
    )

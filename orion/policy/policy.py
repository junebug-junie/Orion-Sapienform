from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AutonomyConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_tier: str = "observe_only"
    max_tier_without_operator: str = "read_only"
    allow_execution_without_operator: bool = False


class PolicyThresholdsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approve_read_only_max_risk: float = 0.15
    defer_above_risk: float = 0.60
    reject_above_risk: float = 0.85
    require_review_above_risk: float = 0.20
    require_review_below_reversibility: float = 0.50
    require_review_below_confidence: float = 0.50


class ProposalKindRuleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_scope: str
    default_decision: str
    max_autonomy_tier: str
    # 2026-08-12: whether a low confidence_score should bounce this kind to a
    # human. True (the default) preserves every existing kind's behaviour
    # exactly. False opts a kind out of the CONFIDENCE branch only -- every
    # other review trigger (risk, reversibility, prepare_action, an explicit
    # operator_review gate) still applies. Only set False for a kind that has
    # its own measured gate downstream; see evaluator.py's comment for the
    # measured reason `maintain` has it.
    confidence_gates_review: bool = True
    # 2026-08-12: whether candidates of this kind change the world. Declared
    # per kind in config rather than inferred, because inference is exactly
    # what went wrong: is_read_only_candidate() used to read `proposed_effect`
    # (what the action is FOR) as a proxy for read-only-ness (what the action
    # DOES), so a prune whose effect is `preserve_stability` -- a perfectly
    # reasonable goal for a mutating action -- classified as read-only.
    # Defaults False so every existing kind keeps its current meaning.
    mutating: bool = False


class SubstratePolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate_policy.v1"] = "substrate_policy.v1"
    policy_id: str = "substrate_policy.v1"

    autonomy: AutonomyConfigV1 = Field(default_factory=AutonomyConfigV1)
    thresholds: PolicyThresholdsV1 = Field(default_factory=PolicyThresholdsV1)
    proposal_kind_rules: dict[str, ProposalKindRuleV1] = Field(default_factory=dict)
    hard_blocks: list[str] = Field(default_factory=list)
    read_only_effects: list[str] = Field(default_factory=list)


def load_substrate_policy(path: str | Path) -> SubstratePolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return SubstratePolicyV1.model_validate(data)

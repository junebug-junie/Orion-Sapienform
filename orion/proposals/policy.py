from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.schemas.action_prediction import EffectDirection, PredictableSignal

_VALID_SIGNALS = frozenset(PredictableSignal.__args__)
_VALID_DIRECTIONS = frozenset(EffectDirection.__args__)


class ProposalLimitsV1(BaseModel):
    max_candidates: int = 10
    max_suppressed: int = 10


class ProposalThresholdsV1(BaseModel):
    min_priority: float = 0.10
    suppress_below: float = 0.05
    policy_required_above_risk: float = 0.20

    # 2026-08-12: the tick-level gate, see orion/field/action_warrant.py.
    #
    # 0.5 is NOT a tuned value. It is the definitional midpoint of the
    # statistic: `action_warrant` is a combined tail probability, so 0.5 means
    # "exactly a median normal day for this machine" by construction. The
    # threshold reads as "act when the state is busier than a median day",
    # which is a decision about tolerance rather than a guess about scale.
    #
    # That distinction is the whole point. `min_priority: 0.10` was chosen
    # against an absolute pressure whose measured floor turned out to be
    # 0.3035, so it could never bind and never did -- 100.00% of ticks cleared
    # it across the arena's entire recorded history. A threshold is only
    # meaningful on a scale whose rest point is defined.
    action_warrant_min: float = 0.50


class ProposalTemplateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    target_kind: str
    target_id: str
    proposed_effect: str
    required_policy_gate: str
    base_priority: float = 0.0
    base_risk: float = 0.0
    reversibility: float = 1.0
    dimensions: dict[str, float] = Field(default_factory=dict)
    # Optional attention-binding path. Only one recognized literal in v1:
    # "attention.dominant_targets[0]" (see orion/proposals/builder.py's
    # ATTENTION_FIRST_TARGET_BINDING; renamed 2026-07-22 from
    # "self_state.dominant_attention_targets[0]" -- attention targets were
    # always FieldAttentionFrameV1 underneath, self_state was a pass-through).
    # No general binding-expression DSL -- matched exactly, nothing fancier.
    target_binding: str | None = None

    # 2026-08-21 (action-outcome ledger): the template author's claim about
    # what this action does to a measured signal. Both must be set together
    # or both left unset -- a signal with no direction is not a prediction.
    # Enforced in ProposalPolicyV1's validator below, not just documented.
    expected_signal: str | None = None
    expected_direction: str | None = None

    @model_validator(mode="after")
    def _check_expected_effect(self) -> "ProposalTemplateV1":
        """A half-declared prediction is worse than none -- it looks wired.

        Rejecting at load time (rather than skipping the template at build
        time) is deliberate: a typo'd signal name would otherwise silently
        turn a scored action back into an unscored one, which is exactly the
        failure this whole patch exists to make impossible to hide.
        """
        if (self.expected_signal is None) != (self.expected_direction is None):
            raise ValueError(
                "expected_signal and expected_direction must be set together "
                f"(signal={self.expected_signal!r}, direction={self.expected_direction!r})"
            )
        if self.expected_signal is not None and self.expected_signal not in _VALID_SIGNALS:
            raise ValueError(
                f"expected_signal={self.expected_signal!r} is not a measured signal; "
                f"valid: {sorted(_VALID_SIGNALS)}"
            )
        if (
            self.expected_direction is not None
            and self.expected_direction not in _VALID_DIRECTIONS
        ):
            raise ValueError(
                f"expected_direction={self.expected_direction!r} invalid; "
                f"valid: {sorted(_VALID_DIRECTIONS)}"
            )
        return self


class ProposalPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["proposal_policy.v1"] = "proposal_policy.v1"
    policy_id: str = "proposal_policy.v1"

    limits: ProposalLimitsV1 = Field(default_factory=ProposalLimitsV1)
    thresholds: ProposalThresholdsV1 = Field(default_factory=ProposalThresholdsV1)

    dimension_weights: dict[str, float] = Field(default_factory=dict)
    proposal_templates: dict[str, ProposalTemplateV1] = Field(default_factory=dict)


def load_proposal_policy(path: str | Path) -> ProposalPolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ProposalPolicyV1.model_validate(data)

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProposalLimitsV1(BaseModel):
    max_candidates: int = 10
    max_suppressed: int = 10


class ProposalThresholdsV1(BaseModel):
    min_priority: float = 0.10
    suppress_below: float = 0.05
    policy_required_above_risk: float = 0.20


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

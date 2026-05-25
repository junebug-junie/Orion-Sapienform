from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DispatchModeConfigV1(BaseModel):
    default_dispatch_mode: str = "dry_run"
    allow_dispatch_read_only: bool = False
    allow_mutating_dispatch: bool = False


class CortexRouteTemplateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_kind: str
    cortex_verb: str
    cortex_mode: str = "brain"
    allowed_scope: str


class DispatchLimitsV1(BaseModel):
    max_dispatch_candidates: int = 5
    max_dispatches_per_tick: int = 1


class ExecutionDispatchPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["execution_dispatch_policy.v1"] = "execution_dispatch_policy.v1"
    policy_id: str = "execution_dispatch_policy.v1"

    mode: DispatchModeConfigV1 = Field(default_factory=DispatchModeConfigV1)
    allowed_policy_decisions: list[str] = Field(default_factory=list)
    blocked_policy_decisions: list[str] = Field(default_factory=list)
    proposal_kind_to_cortex: dict[str, CortexRouteTemplateV1] = Field(default_factory=dict)
    hard_blocks: list[str] = Field(default_factory=list)
    limits: DispatchLimitsV1 = Field(default_factory=DispatchLimitsV1)


def load_execution_dispatch_policy(path: str | Path) -> ExecutionDispatchPolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ExecutionDispatchPolicyV1.model_validate(data)

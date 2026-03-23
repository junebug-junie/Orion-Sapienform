from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialShakedownSourceKind = Literal["scenario_replay", "live_run"]
SocialShakedownBehaviorCategory = Literal[
    "repair_tone",
    "bridge_summary",
    "clarifying_question",
    "handoff_or_closure",
    "stale_context",
    "calibration_or_freshness",
    "reentry_or_snapshot",
    "safety_boundary",
]
SocialShakedownSeverity = Literal["low", "medium", "high", "critical"]
SocialShakedownPriority = Literal["p3", "p2", "p1", "p0"]
SocialShakedownFixStatus = Literal["open", "tuned", "verified"]


class SocialShakedownIssueV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    issue_id: str = Field(default_factory=lambda: f"social-shakedown-issue-{uuid4()}")
    source_kind: SocialShakedownSourceKind = "scenario_replay"
    source_ref: str
    scenario_id: Optional[str] = None
    behavior_category: SocialShakedownBehaviorCategory
    observed_behavior: str
    expected_behavior: str
    severity: SocialShakedownSeverity = "medium"
    priority: SocialShakedownPriority = "p2"
    fix_status: SocialShakedownFixStatus = "open"
    linked_regression_scenario: Optional[str] = None
    linked_regression_test: Optional[str] = None
    rationale: str = ""
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialShakedownFixV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    fix_id: str = Field(default_factory=lambda: f"social-shakedown-fix-{uuid4()}")
    issue_id: str
    summary: str
    changed_surfaces: List[str] = Field(default_factory=list)
    status: SocialShakedownFixStatus = "tuned"
    linked_regression_scenario: Optional[str] = None
    linked_regression_test: Optional[str] = None
    rationale: str = ""
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)

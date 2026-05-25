from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1
from orion.schemas.feedback_frame import FeedbackFrameV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1
from orion.schemas.proposal_frame import ProposalFrameV1
from orion.schemas.self_state import SelfStateV1


@dataclass(frozen=True)
class ConsolidationWindowData:
    window_start: datetime
    window_end: datetime
    self_states: list[SelfStateV1]
    attention_frames: list[FieldAttentionFrameV1]
    proposal_frames: list[ProposalFrameV1]
    policy_frames: list[PolicyDecisionFrameV1]
    dispatch_frames: list[ExecutionDispatchFrameV1]
    feedback_frames: list[FeedbackFrameV1]


def compute_consolidation_window(
    *,
    now: datetime | None = None,
    lookback_minutes: int,
) -> tuple[datetime, datetime]:
    end = now or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)
    return start, end


def stable_consolidation_frame_id(
    *,
    window_start: datetime,
    window_end: datetime,
    policy_id: str,
) -> str:
    ws = window_start.astimezone(timezone.utc).isoformat()
    we = window_end.astimezone(timezone.utc).isoformat()
    return f"consolidation.frame:{ws}:{we}:{policy_id}"

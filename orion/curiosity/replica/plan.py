"""The plan check Hub runs before a plan may set the offer order, and the plan row (design section 4).

``validate_plan`` is deterministic and rejects the WHOLE plan on any failure. A partial plan is never
repaired: a repaired plan is a blend of two arms. The replica runs the same function on its own answer, so
it knows (without asking Hub) whether its plan could be used.

Every replica-arm run gets exactly one plan row, whatever happened (``PlanRowV1``). A fallback (any status but
``ok``) means the run used the uncertainty order and still counts as a replica-arm run (intention to treat).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

from orion.schemas.curiosity_replica import (
    PICKS_MAX,
    PICKS_MIN,
    PLAN_STATUS_INVALID_PREFIX,
    WHY_MAX,
    CuriosityOfferPickV1,
    CuriositySignalStanceV1,
    plan_row_status_allowed,
)

# Hub's own fallback reasons (the plan row's fallback_reason when Hub never got a usable answer).
REASON_TIMEOUT = "timeout"
REASON_NO_CANDIDATES = "no_candidates"
REASON_REQUEST_FAILED = "request_failed"
REASON_MALFORMED_REPLY = "malformed_reply"


def validate_plan(
    picks: Sequence[CuriosityOfferPickV1],
    *,
    candidate_ids: Iterable[str],
    fresh_ids: Iterable[str],
    signal_stance: Optional[CuriositySignalStanceV1] = None,
    live_signal_id: Optional[str] = None,
    max_picks: int = PICKS_MAX,
) -> Optional[str]:
    """None when the plan may be used, else the first reason it may not (short, for ``invalid:<reason>``).

    ``candidate_ids``: the ids Hub sent. ``fresh_ids``: the fresh live priors in the worldview read Hub is
    about to show (a candidate gone or gone stale since the request is refused). ``live_signal_id``: the
    regulate signal on curiosity in force now, if any. ``max_picks``: at most the ordered slots the offer has
    (Hub passes ``min(PICKS_MAX, sample - coverage_slots)``), so no pick is ever cut silently."""
    n = len(picks)
    if n < PICKS_MIN:
        return f"too_few_picks:{n}"
    if n > min(PICKS_MAX, max_picks):
        return f"too_many_picks:{n}"
    cands, fresh = set(candidate_ids), set(fresh_ids)
    seen: set[str] = set()
    for i, pick in enumerate(picks, start=1):
        pid = str(pick.prior_id)
        if pid in seen:
            return f"duplicate_pick:{i}"
        seen.add(pid)
        if pid not in cands:
            return f"not_a_candidate:{i}"
        if pid not in fresh:
            return f"not_fresh:{i}"
        why = str(pick.why or "").strip()
        if not 1 <= len(why) <= WHY_MAX:
            return f"why_length:{i}"
    if signal_stance is not None and (live_signal_id is None or signal_stance.signal_id != live_signal_id):
        return "unknown_signal"
    return None


@dataclass(frozen=True)
class PlanRowV1:
    """One ``curiosity_offer_plans`` row. ``fallback_reason`` is None exactly when the plan was used."""

    run_id: str
    status: str
    requested_at: datetime
    fallback_reason: Optional[str] = None
    plan_id: Optional[str] = None
    answered_at: Optional[datetime] = None
    picks: list[dict[str, Any]] = field(default_factory=list)
    focus_stance: Optional[dict[str, Any]] = None
    signal_stance: Optional[dict[str, Any]] = None
    act_id: Optional[str] = None
    latency_ms: Optional[int] = None
    frame_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if not plan_row_status_allowed(self.status):
            raise ValueError(f"plan row status {self.status!r} is not allowed")
        if (self.status == "ok") != (self.fallback_reason is None):
            raise ValueError("fallback_reason is None exactly when the status is 'ok'")

    @property
    def used(self) -> bool:
        return self.status == "ok"

    @property
    def head_ids(self) -> list[str]:
        """The picks in order, only when the plan was used (else the rule order runs: no head)."""
        return [str(p["prior_id"]) for p in self.picks] if self.used else []


def invalid_status(reason: str) -> str:
    return f"{PLAN_STATUS_INVALID_PREFIX}{reason}"


__all__ = [
    "PlanRowV1",
    "REASON_MALFORMED_REPLY",
    "REASON_NO_CANDIDATES",
    "REASON_REQUEST_FAILED",
    "REASON_TIMEOUT",
    "invalid_status",
    "validate_plan",
]

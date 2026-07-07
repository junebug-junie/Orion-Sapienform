from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from orion.schemas.embodiment import EmbodimentIntentV1


@dataclass
class ArbiterState:
    deliberate_hold_until: Optional[datetime] = None


@dataclass(frozen=True)
class ArbiterDecision:
    accept: bool
    status: str  # "accepted" | "preempted"
    reason: str


def decide(
    intent: EmbodimentIntentV1,
    state: ArbiterState,
    *,
    now: datetime,
    hold_sec: float,
) -> ArbiterDecision:
    """Pure arbitration. Mutates only ``state.deliberate_hold_until`` on accept."""
    if intent.source == "deliberate":
        state.deliberate_hold_until = now + timedelta(seconds=float(hold_sec))
        return ArbiterDecision(accept=True, status="accepted", reason="deliberate accepted")

    hold = state.deliberate_hold_until
    if hold is not None and now < hold:
        remaining = (hold - now).total_seconds()
        return ArbiterDecision(
            accept=False,
            status="preempted",
            reason=f"deliberate hold active {remaining:.1f}s remaining",
        )
    return ArbiterDecision(accept=True, status="accepted", reason="involuntary accepted")

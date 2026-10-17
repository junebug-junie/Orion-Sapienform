"""The situation a piece of personality evidence happened in. Pure.

CAPS (spec 2026-10-16 rev 2): personality is an if-then signature ("when X, Orion tends to Y").
The graph knows WHAT Orion keeps returning to; this records WHEN, at the evidence row's own
time, so situation subnodes can later be built from data instead of guessed. Recorded only,
from 2026-10-16: nothing reads it back yet.

Inputs are signals that already passed the metric gate and are stored by Temporal Self:

- arousal: the level from the latest ``arousal_transition`` at or before the row's time
  (``orion/regulation/arousal.py``; engaged / idle / strained / unknown). "unknown" when no
  transition is on record before it -- never guessed as idle.
- day_phase: ``orion.temporal_self.day.day_phase_for`` in Orion's local day.
- company: someone at the room camera (an open Temporal Self ``company`` arc). No identity:
  company arcs carry no confirmed label today. ``None`` (unknown) for a row newer than
  ``company_known_until`` (the Temporal Self chronicle's watermark): the chronicle writes an
  arc's end ~5 min late, so before it catches up a closed arc still reads as open (live
  2026-10-17: rows at 00:24 stamped "with company" for an arc that ended at 00:20:40).

Attention is left out on purpose: it nearly always crowns a prediction-error node
(2026-10-02), so it would not tell situations apart.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

from orion.temporal_self.day import day_phase_for

UNKNOWN = "unknown"


@dataclass(frozen=True)
class SituationInputs:
    """Arousal transitions as (at, level), sorted by time, and company spans as (began, ended|None)."""

    arousal: Sequence[tuple[datetime, str]] = field(default_factory=tuple)
    company: Sequence[tuple[datetime, Optional[datetime]]] = field(default_factory=tuple)
    # Company spans are trustworthy only up to here; None = no limit.
    company_known_until: Optional[datetime] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arousal", tuple(sorted(self.arousal, key=lambda t: t[0])))
        object.__setattr__(self, "_times", tuple(t for t, _ in self.arousal))


def situation_at(at: datetime, inputs: SituationInputs) -> dict:
    """``{"arousal", "day_phase", "company"}`` at ``at`` (aware). Never raises on good inputs."""
    i = bisect.bisect_right(inputs._times, at)  # type: ignore[attr-defined]
    arousal = inputs.arousal[i - 1][1] if i > 0 else UNKNOWN
    if inputs.company_known_until is not None and at > inputs.company_known_until:
        company: Optional[bool] = None
    else:
        company = any(began <= at and (ended is None or at < ended) for began, ended in inputs.company)
    return {"arousal": arousal or UNKNOWN, "day_phase": day_phase_for(at), "company": company}

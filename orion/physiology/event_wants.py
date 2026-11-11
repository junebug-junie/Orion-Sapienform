"""Event wants from a source's own rows: salience and the want (attention as the want competition, patch 2).

Design: docs/superpowers/specs/2026-11-04-attention-as-want-competition-design.md, "1. The menu: event
wants". A loop that sees an event worth effort posts a ``WantV1`` with ``event_ref`` (the row it is about)
and ``salience`` (how unusual the event is for its own source), through ``orion.physiology.client``.

**Salience** is the source's own percentile, computed by the source against its own history: the share of
the source's past events (its last 30 days, before this one) whose magnitude is STRICTLY below this
event's. It is the same strict-below percentile world-first uses for eligibility
(``orion/attention/world_first.py``). Its rest point is 0: an event no bigger than the source's smallest
usual event (for reading: a read that made no link, 82% of reads) reads 0, never "a little unusual".
The rest point is only REACHED often when many events tie at the bottom (reading: 78% of reads made no
link). For a near-continuous magnitude (camera sightings) the percentile is close to uniform, so about
half the events read above 0.5 (patch 2 PR, open question to Juniper). Fewer than ``MIN_SOURCE_HISTORY`` past events: there is no reference, so salience is unknown (None) and
the source posts nothing for that event (an unknown is not a calm 0, and not a spike either).

Pure and stdlib-only, so Hub and sql-writer both import it.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional, Sequence
from uuid import NAMESPACE_URL, uuid5

from orion.schemas.grammar import ActCostV2
from orion.schemas.physiology_want import WantV1, WantWindowV1

SOURCE_HISTORY = timedelta(days=30)
MIN_SOURCE_HISTORY = 20


def source_salience(magnitude: float, history: Sequence[float]) -> Optional[float]:
    """Share of ``history`` strictly below ``magnitude`` (0..1). None under MIN_SOURCE_HISTORY values."""
    vals = [float(h) for h in history if isinstance(h, (int, float)) and math.isfinite(h)]
    if len(vals) < MIN_SOURCE_HISTORY or not isinstance(magnitude, (int, float)) or not math.isfinite(magnitude):
        return None
    return sum(1 for h in vals if h < magnitude) / len(vals)


def saliences_over_own_history(events: Sequence[tuple[datetime, float]]) -> list[Optional[float]]:
    """For each (at, magnitude), its salience against the source's events in the 30 days before it
    (``source_salience``). Same order as ``events``. The one rule both producers and the replay use."""
    order = sorted(range(len(events)), key=lambda i: events[i][0])
    out: list[Optional[float]] = [None] * len(events)
    for i in order:
        at, mag = events[i]
        history = [m for t, m in events if at - SOURCE_HISTORY <= t < at]
        out[i] = source_salience(mag, history)
    return out


def event_want_id(source: str, key: str) -> str:
    """Deterministic per event, so a restart re-finds (and does not re-post) the same want."""
    return str(uuid5(NAMESPACE_URL, f"orion:event_want:{source}:{key}"))


def event_want(*, source: str, key: str, family: str, producer: str, verb: str, tier: int, event_ref: str,
               salience: float, opened_at: datetime, latest: Optional[datetime] = None,
               object_ref: Optional[str] = None, cost: Optional[ActCostV2] = None) -> WantV1:
    """The event want a source posts. ``latest``: when the event expires (the want's window end).
    ``cost``: what the act this want would run costs, when the source knows it better than the
    family's history (patch 2: an event want carries "the act it would run ... and cost")."""
    want_id = event_want_id(source, key)
    return WantV1(
        want_id=want_id, family=family, producer=producer, verb=verb, tier=tier,  # type: ignore[arg-type]
        object_ref=(object_ref or None) and object_ref[:500], opened_at=opened_at,
        window=WantWindowV1(latest=latest), correlation_id=want_id, event_ref=event_ref[:500],
        salience=min(1.0, max(0.0, float(salience))), cost=cost,
    )


__all__ = ["MIN_SOURCE_HISTORY", "SOURCE_HISTORY", "event_want", "event_want_id", "saliences_over_own_history",
           "source_salience"]

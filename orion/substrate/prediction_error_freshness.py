"""Omit-when-stale rule for a domain's prediction_error reading.

A domain whose Falkor node ``temporal.observed_at`` is older than the shared
horizon has "no current reading". It is LEFT OUT of aggregates (not faded to
zero -- a faded value reads as calm, the same decayed-to-zero lie as
``orion-field-digester``'s NODE_DECAY_CHANNELS incident).

Horizon: the same ``PressureConfig().prediction_error_decay_horizon_seconds``
(1800 s) that ``pressure.py`` and ``endogenous_curiosity.py`` use -- imported,
not copied, so it cannot drift.

Missing/unparseable ``observed_at`` -> OMITTED with age ``None`` (status
``unknown_age``). Rationale: we cannot show it is fresh, and treating it as
fresh (curiosity's choice) or zero would hide exactly the failure this rule
exists to expose. Real nodes always carry ``temporal``; this only fires on
corrupt/foreign nodes, and it is traced in the omitted map.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.substrate.pressure import PressureConfig

PE_STALENESS_HORIZON_SEC: float = float(PressureConfig().prediction_error_decay_horizon_seconds)

STATUS_FRESH = "fresh"
STATUS_STALE = "stale"
STATUS_UNKNOWN_AGE = "unknown_age"


def _as_dt(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def reading_age_sec(node: Any, *, now: datetime) -> float | None:
    """Seconds since the node's ``temporal.observed_at``; None if unknown."""
    observed = _as_dt(getattr(getattr(node, "temporal", None), "observed_at", None))
    if observed is None:
        return None
    return max(0.0, (now - observed).total_seconds())


def classify_reading(
    node: Any, *, now: datetime, horizon_sec: float = PE_STALENESS_HORIZON_SEC
) -> tuple[str, float | None]:
    """Return ``(status, age_sec)``. Age exactly at the horizon is still fresh."""
    age = reading_age_sec(node, now=now)
    if age is None:
        return STATUS_UNKNOWN_AGE, None
    if age > horizon_sec:
        return STATUS_STALE, age
    return STATUS_FRESH, age


def describe_omitted(omitted: dict[str, float | None], n_total: int) -> str:
    """Human/gate-readable trace, e.g. ``omitted 1 of 5: chat(7200s)``."""
    parts = [
        f"{d}({'age unknown' if a is None else f'{int(a)}s'})" for d, a in sorted(omitted.items())
    ]
    return f"omitted {len(omitted)} of {n_total}: " + ", ".join(parts)

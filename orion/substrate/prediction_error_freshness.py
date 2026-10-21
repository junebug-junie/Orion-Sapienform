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

Missing input (2026-10-21). A producer that writes on a clock (the camera
surprise nodes) still writes a 0.0 placeholder when its own input is missing
-- camera embeddings stale, baseline still warming, or no scored frame since
the last write -- because the field digester's receipt path needs a write every
tick (worker.py ``_publish_perception_group``). That 0.0 is not a reading. The
producer says so in ``metadata[PREDICTION_ERROR_MISSING_REASON_KEY]``, and
``classify_reading`` then returns ``STATUS_MISSING_INPUT`` however fresh
``observed_at`` is, so every caller that omits non-fresh readings omits it too.
Seen live 2026-10-18 04:06 -> 10-20 05:51: no embeddings for 49 h, the room
camera node wrote 0.0 every 30 s, and those zeros (31% of its 7-day window)
pulled its p90 to 0.137 against 0.205 over live hours only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.substrate.pressure import PressureConfig

PE_STALENESS_HORIZON_SEC: float = float(PressureConfig().prediction_error_decay_horizon_seconds)

STATUS_FRESH = "fresh"
STATUS_STALE = "stale"
STATUS_UNKNOWN_AGE = "unknown_age"
STATUS_MISSING_INPUT = "missing_input"

# Node metadata key: a non-empty string means "the prediction_error on this
# node is a placeholder, its input is missing", and says why. Absent/empty =
# a real reading.
PREDICTION_ERROR_MISSING_REASON_KEY = "prediction_error_missing_reason"


def missing_input_reason(node: Any) -> str | None:
    """The producer's own "this value is a placeholder" reason, or None."""
    metadata = getattr(node, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    reason = metadata.get(PREDICTION_ERROR_MISSING_REASON_KEY)
    if reason is None:
        return None
    reason = str(reason).strip()
    return reason or None


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
    """Return ``(status, age_sec)``. Age exactly at the horizon is still fresh.
    A placeholder written for missing input is ``STATUS_MISSING_INPUT``."""
    age = reading_age_sec(node, now=now)
    if missing_input_reason(node) is not None:
        return STATUS_MISSING_INPUT, age
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

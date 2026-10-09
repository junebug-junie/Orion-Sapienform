"""Energy as a stake for curiosity's scheduled spend (ORION_ENERGY_STAKES_ENABLED, default off).

Reads the latest energy_stakes_snapshot row orion-energy materializes. Holds only
when that row is fresh, comes from a healthy importer, AND says the house's
projected bill is at/over Rocky Mountain Power's own forecast. Unknown, stale,
unhealthy, or unreadable never holds -- a broken meter must not silence curiosity.
Never reads house_share_cost_usd.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping, Optional
from uuid import NAMESPACE_URL, uuid5

from orion.schemas.attention_schema import MAX_NARRATIVE_CHARS, AttentionSchemaV1, clip

HOLD_REASON = "held_off:energy_stakes"
HOLD_PRESSURES = frozenset({"near_forecast", "over_forecast"})
# A row stamped further ahead than this is a clock fault, not fresh evidence.
_MAX_FUTURE_SKEW_SEC = 300.0

_SNAPSHOT_SQL = """
SELECT as_of, cycle_start, pressure, pressure_reason, projected_to_forecast_ratio,
       orion_projected_total_usd, forecast_total_usd, marginal_usd_per_kwh, importer_state
FROM energy_stakes_snapshot
ORDER BY as_of DESC
LIMIT 1
"""


@dataclass(frozen=True)
class EnergyHold:
    as_of: datetime
    pressure: str
    pressure_reason: str
    ratio: Optional[float]
    projected_total_usd: Optional[float]
    forecast_total_usd: Optional[float]
    marginal_usd_per_kwh: Optional[float]
    cycle_start: Optional[date] = None


def _aware(value: Any) -> Optional[datetime]:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def energy_stakes_hold(
    snapshot: Optional[Mapping[str, Any]], *, now: datetime, max_age_sec: float
) -> Optional[EnergyHold]:
    if not snapshot:
        return None
    as_of = _aware(snapshot.get("as_of"))
    if as_of is None:
        return None
    age_sec = (now - as_of).total_seconds()
    if age_sec > max_age_sec or age_sec < -_MAX_FUTURE_SKEW_SEC:
        return None
    if snapshot.get("importer_state") != "healthy":
        return None
    pressure = snapshot.get("pressure")
    if not isinstance(pressure, str) or pressure not in HOLD_PRESSURES:
        return None
    return EnergyHold(
        as_of=as_of, pressure=pressure, pressure_reason=str(snapshot.get("pressure_reason") or pressure),
        ratio=snapshot.get("projected_to_forecast_ratio"),
        projected_total_usd=snapshot.get("orion_projected_total_usd"),
        forecast_total_usd=snapshot.get("forecast_total_usd"),
        marginal_usd_per_kwh=snapshot.get("marginal_usd_per_kwh"),
        cycle_start=_date(snapshot.get("cycle_start")),
    )


async def read_latest_energy_stakes(database_url: str) -> Optional[dict[str, Any]]:
    if not database_url:
        return None
    import asyncpg

    conn = await asyncpg.connect(dsn=database_url)
    try:
        row = await conn.fetchrow(_SNAPSHOT_SQL)
    finally:
        await conn.close()
    return dict(row) if row else None


def _usd(value: Optional[float]) -> str:
    return "unknown" if value is None else f"${value:.2f}"


def hold_entry_id(hold: EnergyHold) -> str:
    """One row per hold episode (billing cycle + pressure), not per 5-minute snapshot.

    Repeats collapse on the attention table's ON CONFLICT no-op; a pressure change or
    a new cycle is a new episode. No cycle on the row -> the snapshot's UTC day, so an
    unknown cycle still can't pin one row forever.
    """
    period = (hold.cycle_start or hold.as_of.astimezone(timezone.utc).date()).isoformat()
    return f"curiosity:{HOLD_REASON}:{period}:{hold.pressure}"


def hold_attention_row(hold: EnergyHold, *, now: datetime) -> AttentionSchemaV1:
    rate = "unknown" if hold.marginal_usd_per_kwh is None else f"${hold.marginal_usd_per_kwh:.4f}/kWh"
    entry_id = hold_entry_id(hold)
    return AttentionSchemaV1(
        entry_id=entry_id,
        correlation_id=str(uuid5(NAMESPACE_URL, entry_id)),
        generated_at=now,
        process="curiosity",
        attended_id=None,
        attention_reason=HOLD_REASON,
        reason_narrative=clip(
            f"Held a scheduled investigation: projected bill {_usd(hold.projected_total_usd)} vs "
            f"Rocky Mountain Power forecast {_usd(hold.forecast_total_usd)} "
            f"({hold.pressure}, {hold.pressure_reason}); the next kWh costs {rate}.",
            MAX_NARRATIVE_CHARS,
        ),
        narrative_kind="computed",
    )

"""Is house usage actually arriving?

Precedence: reauth_required > degraded > stale > healthy. The portal fetcher only
reports what it tried; freshness is judged from the usage the ledger really holds,
so a portal that says "ok" but delivers nothing still reads stale.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Optional

from orion.schemas.energy import EnergyImporterStatusV1

PortalState = Literal["ok", "reauth_required", "error"]
_PORTAL_STATES = ("ok", "reauth_required", "error")
DEFAULT_STALE_AFTER_HOURS = 48.0


@dataclass(frozen=True)
class PortalStatus:
    state: PortalState
    reason: str
    last_attempt_at: datetime
    last_success_at: Optional[datetime]


def _ts(raw: Any) -> Optional[datetime]:
    if raw in (None, ""):
        return None
    value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def parse_portal_status(raw: Mapping[str, Any]) -> PortalStatus:
    state = raw.get("state")
    if state not in _PORTAL_STATES:
        raise ValueError(f"unknown portal state {state!r}")
    attempt = _ts(raw.get("last_attempt_at"))
    if attempt is None:
        raise ValueError("portal status missing last_attempt_at")
    return PortalStatus(
        state=state, reason=str(raw.get("reason") or state),
        last_attempt_at=attempt, last_success_at=_ts(raw.get("last_success_at")),
    )


def portal_status_dict(status: PortalStatus) -> dict[str, Any]:
    return {
        "state": status.state,
        "reason": status.reason,
        "last_attempt_at": status.last_attempt_at.isoformat(),
        "last_success_at": status.last_success_at.isoformat() if status.last_success_at else None,
    }


def compute_importer_status(
    *,
    portal_enabled: bool,
    portal: Optional[PortalStatus],
    portal_interval_hours: float,
    latest_interval_end: Optional[datetime],
    last_file_at: Optional[datetime],
    now: datetime,
    stale_after_hours: float,
) -> EnergyImporterStatusV1:
    lag = None if latest_interval_end is None else max(0.0, (now - latest_interval_end).total_seconds() / 3600.0)
    state: Optional[str] = None
    reason: Optional[str] = None
    if portal_enabled:
        if portal is None:
            state, reason = "degraded", "portal_status_missing"
        elif portal.state == "reauth_required":
            state, reason = "reauth_required", portal.reason
        elif portal.state == "error":
            state, reason = "degraded", portal.reason
        elif (now - portal.last_attempt_at).total_seconds() > 2 * portal_interval_hours * 3600.0:
            state, reason = "degraded", "portal_not_running"
    if state is None:
        if lag is None:
            state, reason = "stale", "no_usage_yet"
        elif lag > stale_after_hours:
            state, reason = "stale", f"usage_lag_hours={lag:.1f}"
        else:
            state, reason = "healthy", "usage_fresh"
    if portal_enabled:
        success = portal.last_success_at if portal else None
        attempt = portal.last_attempt_at if portal else None
    else:
        success = attempt = last_file_at
    return EnergyImporterStatusV1(
        state=state, reason=reason, source="portal" if portal_enabled else "file_drop",
        last_success_at=success, last_attempt_at=attempt,
        latest_interval_end=latest_interval_end, usage_lag_hours=lag, as_of=now,
    )

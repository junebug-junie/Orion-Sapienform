"""Reverie visual-chain staleness health monitor.

Unlike `reverie_health_monitor.py`'s metacog-timeout check -- which is called
FROM INSIDE a completed reverie tick, and so requires a tick to complete at
all to report anything -- this check must run independently of
`visual_chain.py`'s own worker loop. A fully wedged worker (confirmed live
2026-09-04: `run_visual_chain_worker`'s tick stopped entirely, no ticks, no
errors, for 24+ hours) can never call anything about its own staleness, so
the only place this can be observed from is outside it: a periodic watchdog
that asks Postgres directly how old the newest `reverie_visual_chain` row is
(`store.visual_chain_age_minutes()`) and decides for itself whether that is
too old.

Same edge-triggered orion-notify attention pattern as `reverie_health_
monitor.py` (this service) -- fires once on a healthy->unhealthy transition,
a lower-severity recovery note once it clears, never once per check. Severity
is "critical" (not "error" like the metacog check): per orion-notify's own
README convention, "unhealable failures use severity=critical... transient
uses error and escalates if unacked" -- this wedge is proven non-self-healing
(the 2026-08-31 precedent needed a container restart), so it gets the
immediate-email tier rather than the wait-for-an-unacked-deadline tier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

import requests

from orion.notify.client import NotifyClient

from .settings import ThoughtSettings
from .settings import settings as _default_settings

logger = logging.getLogger("orion-thought.visual_chain_health_monitor")

_SOURCE_SERVICE = "orion-thought"
_CHECK_KEY = "visual_chain_stale"
Severity = Literal["info", "critical"]


@dataclass(frozen=True)
class HealthCheck:
    key: str
    healthy: bool
    severity: Severity
    message: str = ""


def _check(*, age_min: float | None, threshold_min: float) -> HealthCheck:
    """age_min=None (table empty) is NOT flagged as stale -- matching
    orion-field-digester/app/health_monitor.py's own precedent of only
    flagging a real age value past its stall threshold, never an absent one.
    """
    stale = age_min is not None and age_min > threshold_min
    return HealthCheck(
        key=_CHECK_KEY,
        healthy=not stale,
        severity="critical",
        message=(
            f"reverie's visual chain has produced nothing in {age_min:.1f} "
            f"minutes (threshold {threshold_min:.0f}) -- the background worker "
            "is likely wedged (see visual_chain.py's single-flight lock and "
            "run_visual_chain_worker); a restart has cleared this before "
            "(2026-08-31 precedent)."
            if stale
            else ""
        ),
    )


class VisualChainHealthMonitor:
    """Edge-triggered single-check monitor: healthy unless the newest
    `reverie_visual_chain` row is older than the configured threshold.

    A transition is only considered "handled" (in-memory state updated) once
    orion-notify actually confirms delivery -- if it is unreachable at the
    exact moment of a transition, the transition is retried on every
    subsequent call instead of being silently dropped.
    """

    def __init__(self, settings_obj: ThoughtSettings | None = None) -> None:
        self._settings = settings_obj or _default_settings
        self._client = NotifyClient(
            base_url=self._settings.notify_base_url,
            api_token=self._settings.notify_api_token,
            timeout=10,
        )
        self._last_healthy: bool | None = None

    def record_check(self, *, age_min: float | None) -> None:
        """Call once per watchdog tick with the current DB-reported age.
        Never raises."""
        try:
            self._run_tick_for_check(
                _check(
                    age_min=age_min,
                    threshold_min=self._settings.visual_chain_staleness_threshold_min,
                )
            )
        except Exception:
            logger.exception("visual_chain_health_check_failed")

    def _run_tick_for_check(self, check: HealthCheck) -> None:
        previous = self._last_healthy

        if previous is None:
            if check.healthy:
                self._last_healthy = True
                return
            # First observation since this process started, and already
            # unhealthy: consult orion-notify itself (not just local memory,
            # which a restart would have wiped) for an already-open alert.
            if self._has_open_alert() or self._publish(check, recovered=False):
                self._last_healthy = False
            # else: leave unset so the next tick retries.
            return

        if previous and not check.healthy:
            if self._publish(check, recovered=False):
                self._last_healthy = False
            # else: leave `previous=True` so the next tick retries the alert.
        elif not previous and check.healthy:
            if self._publish(check, recovered=True):
                self._last_healthy = True
            # else: leave `previous=False` so the next tick retries the note.
        else:
            self._last_healthy = check.healthy

    def _has_open_alert(self) -> bool:
        headers = {}
        if self._settings.notify_api_token:
            headers["X-Orion-Notify-Token"] = self._settings.notify_api_token
        try:
            response = requests.get(
                f"{self._settings.notify_base_url}/attention",
                params={"status": "pending", "limit": 200},
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            items = response.json()
        except Exception:
            logger.exception("visual_chain_health_pending_lookup_failed")
            # Fail open: if we can't confirm an existing alert, prefer
            # attempting a possibly duplicate one over silently missing a
            # real incident.
            return False
        if not isinstance(items, list):
            return False
        return any(
            isinstance(item, dict)
            and item.get("source_service") == _SOURCE_SERVICE
            and item.get("reason") == _CHECK_KEY
            for item in items
        )

    def _publish(self, check: HealthCheck, *, recovered: bool) -> bool:
        if recovered:
            message = f"[Orion reverie] recovered: {check.key}"
            severity: Severity = "info"
        else:
            message = f"[Orion reverie] {check.message}"
            severity = check.severity
        try:
            result = self._client.attention_request(
                message=message,
                severity=severity,
                require_ack=True,
                context={
                    "source_service": _SOURCE_SERVICE,
                    "reason": check.key,
                    "event_kind": "orion.reverie.visual_chain_stale.health.attention.v1",
                    "correlation_id": str(uuid4()),
                },
            )
            return bool(getattr(result, "ok", False))
        except Exception:
            logger.exception("visual_chain_health_attention_publish_failed")
            return False


_MONITOR: VisualChainHealthMonitor | None = None


def check_visual_chain_staleness(age_min: float | None) -> None:
    """Module-level singleton entrypoint called from the watchdog loop in
    visual_chain.py. Never raises."""
    global _MONITOR
    try:
        if _MONITOR is None:
            _MONITOR = VisualChainHealthMonitor()
        _MONITOR.record_check(age_min=age_min)
    except Exception:
        logger.exception("visual_chain_health_check_failed")


def reset_monitor_for_tests() -> None:
    global _MONITOR
    _MONITOR = None

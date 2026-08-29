"""Reverie metacog-timeout health monitor.

`ORION_REVERIE_METACOG_BACKGROUND_ENABLED` (settings.py) ships default-off on purpose: Juniper
wants to see reverie's real, unmitigated failure rate on the metacog lane under load before the
`metacog_background` yielding route takes effect. That period is only actually observable if a
timeout surfaces somewhere Juniper looks -- previously a timed-out reverie tick just logged
"reverie tick failed" and returned None, indistinguishable from any other dropped tick.

Same edge-triggered orion-notify attention pattern as `resonance_monitor.py` (this service) and
`orion-field-digester/app/health_monitor.py`'s `HealthMonitor`, simplified to one fixed check
instead of several store-derived ones: fires once on a healthy->unhealthy transition (the most
recent reverie tick's cortex-exec call to metacog timing out), a lower-severity recovery note
once it stops, never once per tick -- reverie ticks every ~90s, so per-tick paging on a
persistent outage would be immediate spam.
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

logger = logging.getLogger("orion-thought.reverie_health_monitor")

_SOURCE_SERVICE = "orion-thought"
_CHECK_KEY = "reverie_metacog_timeout"
Severity = Literal["info", "error"]


@dataclass(frozen=True)
class HealthCheck:
    key: str
    healthy: bool
    severity: Severity
    message: str = ""


def _check(*, timed_out: bool) -> HealthCheck:
    return HealthCheck(
        key=_CHECK_KEY,
        healthy=not timed_out,
        severity="error",
        message=(
            "reverie's cortex-exec call to metacog timed out "
            "(STANCE_REACT_TIMEOUT_SEC) -- the lane may be saturated. "
            "ORION_REVERIE_METACOG_BACKGROUND_ENABLED is currently off, so this call "
            "competes evenly rather than yielding for /slots slack; see "
            "orion/llm/routes.py's BACKGROUND_LLM_ROUTES."
            if timed_out
            else ""
        ),
    )


class ReverieMetacogHealthMonitor:
    """Edge-triggered single-check monitor: healthy unless the most recent reverie tick's
    metacog call timed out.

    A transition is only considered "handled" (in-memory state updated) once orion-notify
    actually confirms delivery -- if it is unreachable at the exact moment of a transition, the
    transition is retried on every subsequent call instead of being silently dropped.
    """

    def __init__(self, settings_obj: ThoughtSettings | None = None) -> None:
        self._settings = settings_obj or _default_settings
        self._client = NotifyClient(
            base_url=self._settings.notify_base_url,
            api_token=self._settings.notify_api_token,
            timeout=10,
        )
        self._last_healthy: bool | None = None

    def record_tick(self, timed_out: bool) -> None:
        """Call once per reverie tick that actually reached the metacog call, with whether
        that specific call timed out. Never raises."""
        try:
            self._run_tick_for_check(_check(timed_out=timed_out))
        except Exception:
            logger.exception("reverie_metacog_health_check_failed")

    def _run_tick_for_check(self, check: HealthCheck) -> None:
        previous = self._last_healthy

        if previous is None:
            if check.healthy:
                self._last_healthy = True
                return
            # First observation since this process started, and already unhealthy: consult
            # orion-notify itself (not just local memory, which a restart would have wiped)
            # for an already-open alert.
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
            logger.exception("reverie_metacog_health_pending_lookup_failed")
            # Fail open: if we can't confirm an existing alert, prefer attempting a possibly
            # duplicate one over silently missing a real incident.
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
                    "event_kind": "orion.reverie.metacog_timeout.health.attention.v1",
                    "correlation_id": str(uuid4()),
                },
            )
            return bool(getattr(result, "ok", False))
        except Exception:
            logger.exception("reverie_metacog_health_attention_publish_failed")
            return False


_MONITOR: ReverieMetacogHealthMonitor | None = None


def check_reverie_metacog_timeout(timed_out: bool) -> None:
    """Module-level singleton entrypoint called from reverie.py. Never raises."""
    global _MONITOR
    try:
        if _MONITOR is None:
            _MONITOR = ReverieMetacogHealthMonitor()
        _MONITOR.record_tick(timed_out)
    except Exception:
        logger.exception("reverie_metacog_health_check_failed")


def reset_monitor_for_tests() -> None:
    global _MONITOR
    _MONITOR = None

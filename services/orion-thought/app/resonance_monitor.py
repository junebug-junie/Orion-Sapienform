"""Phase H+ — resonance health monitor.

The resonance tripwire (`orion.reverie.resonance.detect_resonance`, called from
`chain.py::_maybe_emit_resonance_alert`) is observation-only: it persists a
`ResonanceAlertV1` row and publishes to the bus, but nobody gets paged. This
module closes that gap the same way field-digester/attention-runtime/
self-state-runtime's health monitors already do (all three merged, running in
production): an edge-triggered check that fires an `orion-notify` attention
request only on a healthy->unhealthy transition, retries a failed delivery
until it actually succeeds, and checks orion-notify's own pending list before
suppressing a first-observation alert (so a process restart mid-incident
can't go permanently silent).

The one thing that differs from the prior three ports: "unhealthy" here is
NOT "an alert exists." A 2026-07-12 investigation confirmed a real historical
resonance burst (2026-07-07 through 2026-07-12 03:10) had already self-resolved
-- the reverie chain refractory cooldown was holding correctly at investigation
time -- but the detector kept re-reporting the SAME old violation_count/
min_gap_sec for ~20 hours after the fact, because those old (theme_key,
created_at) rows hadn't yet aged out of `detect_resonance`'s 200-row lookback
window. Paging on "an alert exists" would have paged Juniper for ~20 hours
about an already-resolved problem. Paging on "violation_count is INCREASING
across the last 2 persisted samples for this theme" only fires while the
loop is actually getting worse, and recovers once it stops.
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
from .store import load_recent_resonance_alerts

logger = logging.getLogger("orion-thought.resonance_monitor")

_SOURCE_SERVICE = "orion-thought"
Severity = Literal["info", "error"]


@dataclass(frozen=True)
class HealthCheck:
    key: str
    healthy: bool
    severity: Severity
    message: str = ""


def _check(key: str, *, healthy: bool, severity: Severity, message_if_unhealthy: str) -> HealthCheck:
    return HealthCheck(
        key=key,
        healthy=healthy,
        severity=severity,
        message=message_if_unhealthy if not healthy else "",
    )


def _is_worsening(theme_key: str) -> HealthCheck:
    """Healthy unless this theme's violation_count strictly increased between
    its last 2 persisted resonance-alert samples (newest vs. previous)."""
    key = f"reverie_resonance_worsening:{theme_key}"
    samples = load_recent_resonance_alerts(theme_key, limit=2)
    if len(samples) < 2:
        # Nothing to compare yet -- a single sample can't show a trend.
        return _check(key, healthy=True, severity="error", message_if_unhealthy="")

    newest, previous = samples[0], samples[1]
    worsening = int(newest["violation_count"]) > int(previous["violation_count"])
    return _check(
        key,
        healthy=not worsening,
        severity="error",
        message_if_unhealthy=(
            f"theme {theme_key} resonance is worsening: violation_count "
            f"{previous['violation_count']} -> {newest['violation_count']} "
            f"(min_gap_sec={float(newest['min_gap_sec']):.1f}, "
            f"occurrences={newest['occurrences']}, "
            f"refractory_sec={float(newest['refractory_sec']):.0f}) -- the reverie "
            "chain refractory cooldown may not be holding for this theme."
        ),
    )


class ResonanceHealthMonitor:
    """Edge-triggered per-theme resonance monitor.

    Tracks every theme it has ever seen flagged (not just the current tick's
    most-severe theme, since `detect_resonance()` only names one theme per
    call) so a theme that stops being "most severe" still gets its recovery
    note fired once its own trend has calmed, rather than staying silently
    "open" forever.
    """

    def __init__(self, settings_obj: ThoughtSettings | None = None) -> None:
        self._settings = settings_obj or _default_settings
        self._client = NotifyClient(
            base_url=self._settings.notify_base_url,
            api_token=self._settings.notify_api_token,
            timeout=10,
        )
        self._last_healthy: dict[str, bool] = {}
        self._tracked_themes: set[str] = set()

    def check(self, alert) -> None:
        """Call once per completed chain, after `_maybe_emit_resonance_alert`
        persists (or decides not to persist) an alert. `alert` is the
        `ResonanceAlertV1` just persisted, or `None` if no theme is currently
        resonant this tick. Never raises."""
        try:
            themes_to_check = set(self._tracked_themes)
            if alert is not None:
                theme_key = str(getattr(alert, "theme_key", "") or "")
                if theme_key:
                    themes_to_check.add(theme_key)
                    self._tracked_themes.add(theme_key)

            for theme_key in themes_to_check:
                self._run_tick_for_check(_is_worsening(theme_key))
        except Exception:
            logger.exception("resonance_health_check_failed")

    def _run_tick_for_check(self, check: HealthCheck) -> None:
        previous = self._last_healthy.get(check.key)

        if previous is None:
            if check.healthy:
                self._last_healthy[check.key] = True
                return
            # First observation since this process started, and already
            # unhealthy: consult orion-notify itself (not just local memory,
            # which a restart would have wiped) for an already-open alert.
            if self._has_open_alert(check.key) or self._publish(check, recovered=False):
                self._last_healthy[check.key] = False
            # else: leave unset so the next tick retries.
            return

        if previous and not check.healthy:
            if self._publish(check, recovered=False):
                self._last_healthy[check.key] = False
            # else: leave `previous=True` so the next tick retries the alert.
        elif not previous and check.healthy:
            if self._publish(check, recovered=True):
                self._last_healthy[check.key] = True
            # else: leave `previous=False` so the next tick retries the note.
        else:
            self._last_healthy[check.key] = check.healthy

    def _has_open_alert(self, reason: str) -> bool:
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
            logger.exception("resonance_health_pending_lookup_failed reason=%s", reason)
            # Fail open: prefer attempting a possibly-duplicate alert over
            # silently missing a real, worsening incident.
            return False
        if not isinstance(items, list):
            return False
        return any(
            isinstance(item, dict)
            and item.get("source_service") == _SOURCE_SERVICE
            and item.get("reason") == reason
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
                    "event_kind": "orion.reverie.resonance.health.attention.v1",
                    "correlation_id": str(uuid4()),
                },
            )
            return bool(getattr(result, "ok", False))
        except Exception:
            logger.exception("resonance_health_attention_publish_failed key=%s", check.key)
            return False


_MONITOR: ResonanceHealthMonitor | None = None


def check_resonance_worsening(alert) -> None:
    """Module-level singleton entrypoint called from chain.py. Never raises."""
    global _MONITOR
    try:
        if _MONITOR is None:
            _MONITOR = ResonanceHealthMonitor()
        _MONITOR.check(alert)
    except Exception:
        logger.exception("resonance_health_check_failed")


def reset_monitor_for_tests() -> None:
    global _MONITOR
    _MONITOR = None

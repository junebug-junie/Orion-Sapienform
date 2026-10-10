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

Second, independent check (2026-10-10): `visual_painting_gap`. The staleness
check above only proves the worker loop is alive -- deferral/failure rows
(resource_deferred, run_deadline_exceeded, generation_failed, ...) count as
fresh. 2026-10-09 02:12 -> 10-10 06:02 the GPU lane controller refused every
swap and Orion produced no painting for 27.6 h while the staleness check
stayed green the whole time. This check reads `store.visual_last_painting_
age_hours()` (production receipts only) instead. Severity "error", not
"critical": a gap can be heat or a held GPU that clears on its own, so it
escalates by email only if left unacked. Each check key has its own edge-
triggered state, so one check going unhealthy never flips the other.
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
_PAINTING_GAP_CHECK_KEY = "visual_painting_gap"
_EVENT_KINDS = {
    _CHECK_KEY: "orion.reverie.visual_chain_stale.health.attention.v1",
    _PAINTING_GAP_CHECK_KEY: "orion.reverie.visual_painting_gap.health.attention.v1",
}
Severity = Literal["info", "error", "critical"]


@dataclass(frozen=True)
class HealthCheck:
    key: str
    healthy: bool
    severity: Severity
    message: str = ""
    # Empty -> the generic "recovered: <key>" note (the staleness check's
    # original wording, kept byte-identical).
    recovery_message: str = ""


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


def _painting_gap_check(*, age_hours: float | None, threshold_hours: float) -> HealthCheck:
    """age_hours=None (no painting ever, or the DB read failed) is NOT
    flagged -- same "only judge a real number" rule as `_check` above.

    Calibration (21 days live, 175 paintings): p95 gap 2.9 h; the only gaps
    over 8 h were the three real outages (27.6 h, 17.8 h, 16.0 h), so the
    12 h default fires on exactly those. See evals/test_painting_gap_replay.py.
    """
    gap = age_hours is not None and age_hours > threshold_hours
    return HealthCheck(
        key=_PAINTING_GAP_CHECK_KEY,
        healthy=not gap,
        severity="error",
        message=(
            f"Orion has not produced a painting in {age_hours:.1f} hours "
            f"(threshold {threshold_hours:g} h). The worker may still be "
            "writing deferral rows, so the visual-chain staleness check can "
            "stay green through this. Likely causes, most likely first: "
            "(1) the GPU lane controller is refusing swaps -- check the "
            "mesh-guardian GPU cards or run scripts/gpu_pool_actuator_probe.py "
            "(2026-10-09 precedent, 27.6 h); (2) long thermal refusals on the "
            "painting GPU; (3) the visual-chain worker is wedged."
            if gap
            else ""
        ),
        recovery_message=(
            f"[Orion reverie] recovered: {_PAINTING_GAP_CHECK_KEY} -- a painting "
            f"landed (latest {age_hours:.1f} hours ago)."
            if (not gap and age_hours is not None)
            else ""
        ),
    )


class VisualChainHealthMonitor:
    """Edge-triggered monitor with independent state per check key:
    `visual_chain_stale` (newest `reverie_visual_chain` row older than the
    staleness threshold) and `visual_painting_gap` (no produced painting in
    the gap threshold).

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
        # Per check key; a missing key means "no observation yet".
        self._last_healthy: dict[str, bool] = {}

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

    def record_painting_gap(self, *, age_hours: float | None) -> None:
        """Call once per watchdog tick with hours since the last produced
        painting. Never raises."""
        try:
            self._run_tick_for_check(
                _painting_gap_check(
                    age_hours=age_hours,
                    threshold_hours=self._settings.visual_painting_gap_threshold_hours,
                )
            )
        except Exception:
            logger.exception("visual_painting_gap_check_failed")

    def _run_tick_for_check(self, check: HealthCheck) -> None:
        key = check.key
        previous = self._last_healthy.get(key)

        if previous is None:
            if check.healthy:
                self._last_healthy[key] = True
                return
            # First observation since this process started, and already
            # unhealthy: consult orion-notify itself (not just local memory,
            # which a restart would have wiped) for an already-open alert.
            if self._has_open_alert(key) or self._publish(check, recovered=False):
                self._last_healthy[key] = False
            # else: leave unset so the next tick retries.
            return

        if previous and not check.healthy:
            if self._publish(check, recovered=False):
                self._last_healthy[key] = False
            # else: leave `previous=True` so the next tick retries the alert.
        elif not previous and check.healthy:
            if self._publish(check, recovered=True):
                self._last_healthy[key] = True
            # else: leave `previous=False` so the next tick retries the note.
        else:
            self._last_healthy[key] = check.healthy

    def _has_open_alert(self, key: str = _CHECK_KEY) -> bool:
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
            and item.get("reason") == key
            for item in items
        )

    def _publish(self, check: HealthCheck, *, recovered: bool) -> bool:
        if recovered:
            message = check.recovery_message or f"[Orion reverie] recovered: {check.key}"
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
                    "event_kind": _EVENT_KINDS[check.key],
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


def check_visual_painting_gap(age_hours: float | None) -> None:
    """Module-level entrypoint for the painting-gap check, sharing the same
    singleton (state is per key, so the two checks stay independent). Called
    from the watchdog loop in visual_chain.py. Never raises."""
    global _MONITOR
    try:
        if _MONITOR is None:
            _MONITOR = VisualChainHealthMonitor()
        _MONITOR.record_painting_gap(age_hours=age_hours)
    except Exception:
        logger.exception("visual_painting_gap_check_failed")


def reset_monitor_for_tests() -> None:
    global _MONITOR
    _MONITOR = None

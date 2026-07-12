from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

import requests

from orion.notify.client import NotifyClient

from app.settings import Settings
from app.store import SelfStateRuntimeStore

logger = logging.getLogger("orion.self_state.runtime.health")

_SOURCE_SERVICE = "orion-self-state-runtime"
Severity = Literal["info", "error", "critical"]


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


def run_checks(store: SelfStateRuntimeStore, settings: Settings) -> list[HealthCheck]:
    checks: list[HealthCheck] = []

    retention = settings.self_state_retention_hours
    if retention > 0:
        age_hours = store.self_state_oldest_age_hours()
        stall_limit = retention * settings.self_state_stall_multiplier
        stalled = age_hours is not None and age_hours > stall_limit
        # age_hours is only formatted when stalled is True, at which point it is
        # guaranteed non-None (an empty table can never be "stalled").
        message = ""
        if stalled:
            message = (
                f"substrate_self_state oldest row is {age_hours:.1f}h old, exceeding "
                f"{stall_limit:.1f}h ({settings.self_state_stall_multiplier}x the "
                f"{retention:.1f}h retention window) -- the hourly pruner may have stopped "
                "running, and this table is what felt_state_reader.py and phi both depend "
                "on for freshness."
            )
        checks.append(
            _check(
                "self_state_prune_stalled",
                healthy=not stalled,
                severity="critical",
                message_if_unhealthy=message,
            )
        )

    return checks


class HealthMonitor:
    """Edge-triggered health monitor: fires an orion-notify attention request only
    on healthy->unhealthy transitions (and a lower-severity note on recovery), so a
    condition that persists across many poll ticks does not spam a fresh attention
    item every cycle.

    A transition is only considered "handled" (and the in-memory state updated)
    once orion-notify actually confirms delivery. If orion-notify is unreachable
    at the exact moment of a transition -- plausible, since it may be under the
    same host pressure that triggered the alert -- the transition is retried on
    every subsequent tick until it is actually delivered, instead of being
    silently dropped.
    """

    def __init__(self, store: SelfStateRuntimeStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        self._client = NotifyClient(
            base_url=settings.notify_base_url,
            api_token=settings.notify_api_token,
            timeout=10,
        )
        self._last_healthy: dict[str, bool] = {}

    def run_tick(self) -> None:
        for check in run_checks(self._store, self._settings):
            previous = self._last_healthy.get(check.key)

            if previous is None:
                if check.healthy:
                    self._last_healthy[check.key] = True
                    continue
                # First observation since this process started, and already
                # unhealthy: consult orion-notify itself (not just local memory,
                # which a restart would have wiped) for an already-open alert.
                if self._has_open_alert(check.key) or self._publish(check, recovered=False):
                    self._last_healthy[check.key] = False
                # else: leave unset so the next tick retries.
                continue

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
            logger.exception("self_state_health_pending_lookup_failed reason=%s", reason)
            # Fail open: if we can't confirm an existing alert, prefer attempting
            # to fire a possibly-duplicate one over silently missing a real
            # incident (if orion-notify itself is down, _publish will also fail
            # and this transition retries next tick regardless).
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
            message = f"[Orion self-state-runtime] recovered: {check.key}"
            severity: Severity = "info"
        else:
            message = f"[Orion self-state-runtime] {check.message}"
            severity = check.severity
        try:
            result = self._client.attention_request(
                message=message,
                severity=severity,
                require_ack=True,
                context={
                    "source_service": _SOURCE_SERVICE,
                    "reason": check.key,
                    "event_kind": "orion.self_state_runtime.health.attention.v1",
                    "correlation_id": str(uuid4()),
                },
            )
            return bool(getattr(result, "ok", False))
        except Exception:
            logger.exception("self_state_health_attention_publish_failed key=%s", check.key)
            return False

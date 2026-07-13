from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

import requests

from orion.notify.client import NotifyClient

from app.grammar_truth import build_substrate_grammar_truth
from app.settings import Settings
from app.store import BiometricsSubstrateStore

logger = logging.getLogger("orion.substrate.runtime.health")

_SOURCE_SERVICE = "orion-substrate-runtime"
Severity = Literal["info", "error", "critical"]

# Juniper's operating timezone. Storage and the /grammar/truth API stay UTC
# (the source of truth); only this human-facing alert message is localized,
# so an operator reading it doesn't have to mentally convert from UTC.
_REPORT_TZ = ZoneInfo("America/Denver")


def _format_local(iso_timestamp: str | None) -> str | None:
    if not iso_timestamp:
        return None
    try:
        dt = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_REPORT_TZ).strftime("%Y-%m-%d %H:%M %Z")


def _annotate_reason(reason: str, cursor_by_name: dict[str, dict]) -> str:
    if not reason.startswith("cursor_lag:"):
        return reason
    cursor_name = reason.split(":", 1)[1]
    last_at = cursor_by_name.get(cursor_name, {}).get("last_event_created_at")
    local = _format_local(last_at)
    if not local:
        return reason
    return f"{reason} (last event {local})"


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


def run_checks(store: BiometricsSubstrateStore, settings: Settings) -> list[HealthCheck]:
    checks: list[HealthCheck] = []

    truth = build_substrate_grammar_truth(store)
    degraded = bool(truth.get("degraded"))
    reasons = truth.get("degraded_reasons") or []
    message = ""
    if degraded:
        cursor_by_name = {
            row["cursor_name"]: row for row in (truth.get("cursor_positions") or [])
        }
        annotated = [_annotate_reason(reason, cursor_by_name) for reason in reasons]
        message = f"substrate-runtime grammar production degraded: {', '.join(annotated)}"
    checks.append(
        _check(
            "substrate_grammar_degraded",
            healthy=not degraded,
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

    Before paging on a *fresh* unhealthy transition (one not already backed by
    an existing open orion-notify alert), it waits ``health_recheck_delay_sec``
    and rechecks once. A single degraded reducer-health tick can be a
    self-healing blip (e.g. one cursor commit losing a race with transient DB
    load, corrected on the very next poll); the recheck absorbs that class of
    false alarm without meaningfully delaying a genuinely sustained incident,
    which will still be unhealthy on recheck.
    """

    def __init__(self, store: BiometricsSubstrateStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        self._client = NotifyClient(
            base_url=settings.notify_base_url,
            api_token=settings.notify_api_token,
            timeout=10,
        )
        self._last_healthy: dict[str, bool] = {}
        # Keys that have already passed their once-per-streak recheck. Consulted
        # so a transition stuck retrying (e.g. orion-notify unreachable) doesn't
        # pay the recheck delay and a duplicate DB read on every retry tick --
        # cleared as soon as the key is observed healthy again.
        self._recheck_confirmed: set[str] = set()

    def run_tick(self) -> None:
        for check in run_checks(self._store, self._settings):
            previous = self._last_healthy.get(check.key)
            if check.healthy:
                self._recheck_confirmed.discard(check.key)

            if previous is None:
                if check.healthy:
                    self._last_healthy[check.key] = True
                    continue
                # First observation since this process started, and already
                # unhealthy: consult orion-notify itself (not just local memory,
                # which a restart would have wiped) for an already-open alert.
                # An existing open alert is already-confirmed real -- no need
                # to re-delay behind a recheck.
                if self._has_open_alert(check.key):
                    self._last_healthy[check.key] = False
                    continue
                self._page_if_confirmed(check)
                continue

            if previous and not check.healthy:
                self._page_if_confirmed(check)
            elif not previous and check.healthy:
                if self._publish(check, recovered=True):
                    self._last_healthy[check.key] = True
                # else: leave `previous=False` so the next tick retries the note.
            else:
                self._last_healthy[check.key] = check.healthy

    def _page_if_confirmed(self, check: HealthCheck) -> None:
        if not self._recheck_confirmed_or_confirm(check):
            # Transient blip (or the recheck delay hasn't elapsed yet on the
            # first attempt) -- leave `_last_healthy` untouched so the next
            # tick re-evaluates rather than wrongly recording healthy.
            return
        if self._publish(check, recovered=False):
            self._last_healthy[check.key] = False
        # else: leave state as-is so the next tick retries the publish (without
        # paying the recheck delay again -- `_recheck_confirmed` already has it).

    def _recheck_confirmed_or_confirm(self, check: HealthCheck) -> bool:
        if check.key in self._recheck_confirmed:
            return True
        if not self._confirm_still_unhealthy(check.key):
            return False
        self._recheck_confirmed.add(check.key)
        return True

    def _confirm_still_unhealthy(self, key: str) -> bool:
        delay = float(self._settings.health_recheck_delay_sec)
        if delay > 0:
            time.sleep(delay)
        try:
            rechecked = {c.key: c for c in run_checks(self._store, self._settings)}
        except Exception:
            logger.exception("substrate_runtime_health_recheck_failed key=%s", key)
            # Fail toward alerting, mirroring `_has_open_alert`'s own fail-open
            # bias: if the same DB pressure that caused the degradation also
            # breaks this recheck query, that is not evidence of recovery --
            # don't let an unconfirmable recheck silently swallow a real,
            # already-observed incident.
            return True
        check = rechecked.get(key)
        return bool(check is not None and not check.healthy)

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
            logger.exception("substrate_runtime_health_pending_lookup_failed reason=%s", reason)
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
            message = f"[Orion substrate-runtime] recovered: {check.key}"
            severity: Severity = "info"
        else:
            message = f"[Orion substrate-runtime] {check.message}"
            severity = check.severity
        try:
            result = self._client.attention_request(
                message=message,
                severity=severity,
                require_ack=True,
                context={
                    "source_service": _SOURCE_SERVICE,
                    "reason": check.key,
                    "event_kind": "orion.substrate_runtime.health.attention.v1",
                    "correlation_id": str(uuid4()),
                },
            )
            return bool(getattr(result, "ok", False))
        except Exception:
            logger.exception("substrate_runtime_health_attention_publish_failed key=%s", check.key)
            return False

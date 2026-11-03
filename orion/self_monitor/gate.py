"""E1, the act gate: a failing act stops being retried blindly. Pure, plus one small Redis read.

Spec: docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md, "Effects" E1 and "Decisions" 1
(Juniper 2026-11-03): an act whose faculty is held ``broken`` or ``unreliable`` in a fresh view is
refused with ``track_record_failing``, except one re-test probe per faculty per probe interval
(6 h), so a healed faculty comes back. Never a permanent zero.

**One direction only.** The gate can only refuse. An unknown, stale or absent view, an untracked
faculty, or any status other than ``broken``/``unreliable`` returns ``admit`` with no probe mark:
the caller behaves exactly as if the gate did not exist.

**Probe bookkeeping is derived from the act ledger, not kept here.** A probe is an attempt that
really started work, and every real attempt opens a ``grammar_acts`` row (the doer opens its act
when work starts). So "was there a probe in the last 6 h" is "did this faculty open an act in the
last 6 h" -- recorded truth that survives restarts. A refused attempt opens no act, and a probe the
caller admitted but that never started (held back, budget spent) opens none either, so the next
decision may probe again. The ledger is written over the bus, so a caller also remembers its own
admitted probes in-process (``ProbeMemo``): the probe slot is spent when the gate admits it, which
also covers the write lag. If the caller later drops the admitted probe (held back, budget spent)
the slot stays spent for this window; the next window probes again. After a restart only the
ledger counts (real attempts), so a restart can never cost a probe. When the ledger cannot be read
the memo alone decides: the gate fails toward "admit" (gate-off behaviour), never toward a
permanent refusal.

Faculty names come from ``adapters.act_faculty(actor, verb, object_ref)``, the same function the
organ grades with, so the gate and the organ always agree on what a faculty is called.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional

from orion.self_monitor.adapters import act_faculty
from orion.self_monitor.view import (
    DEFAULT_MAX_AGE_SEC,
    SELF_MONITOR_REDIS_KEY,
    TRACK_RECORD_FAILING_REASON,
    SelfMonitorView,
    self_monitor_view,
)

logger = logging.getLogger("orion.self_monitor.gate")

FAILING_STATUSES = frozenset({"broken", "unreliable"})
DEFAULT_PROBE_INTERVAL_SEC = 6 * 3600.0

# The act ledger read every caller runs to find the last real attempt (both drivers name params the
# same way through the caller's own placeholder style; see last_attempt_sql).
_LAST_ATTEMPT_SQL = (
    "SELECT object_ref, opened_at FROM grammar_acts "
    "WHERE actor = {actor} AND verb = {verb} AND opened_at IS NOT NULL "
    "AND opened_at > now() - make_interval(secs => {secs}) "
    "ORDER BY opened_at DESC LIMIT 500"
)


def last_attempt_sql(style: str = "sqlalchemy") -> str:
    """The ledger read, in the caller's placeholder style: ``sqlalchemy`` (:name) or ``pyformat``
    (psycopg2's %(name)s). Params: actor, verb, secs."""
    if style == "pyformat":
        return _LAST_ATTEMPT_SQL.format(actor="%(actor)s", verb="%(verb)s", secs="%(secs)s")
    return _LAST_ATTEMPT_SQL.format(actor=":actor", verb=":verb", secs=":secs")


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def latest_attempt_at(rows: Iterable[Mapping[str, Any]], *, actor: str, verb: str, faculty: str) -> Optional[datetime]:
    """The newest ``opened_at`` among ledger rows whose faculty is ``faculty``. ``object_ref`` is
    re-mapped with ``act_faculty`` so ``journal_trigger:walkway_grade`` never counts as an attempt
    of ``journal_trigger:town_episode``."""
    best: Optional[datetime] = None
    for row in rows:
        opened = row.get("opened_at")
        if opened is None:
            continue
        if act_faculty(actor, verb, str(row.get("object_ref") or "")) != faculty:
            continue
        opened = _utc(opened)
        if best is None or opened > best:
            best = opened
    return best


@dataclass(frozen=True)
class GateDecision:
    faculty: str
    admit: bool
    probe: bool
    status: str
    # not_failing | unknown:<view reason> | probe | track_record_failing | disabled
    reason: str
    last_attempt_at: Optional[datetime] = None

    def label(self) -> str:
        """One frame/audit-visible token: what the gate did and why."""
        if self.probe:
            when = self.last_attempt_at.isoformat() if self.last_attempt_at else "none"
            return f"self_monitor:probe:{self.faculty}:{self.status}:last_attempt={when}"
        if not self.admit:
            return f"self_monitor:{TRACK_RECORD_FAILING_REASON}:{self.faculty}:{self.status}"
        return f"self_monitor:admit:{self.faculty}:{self.reason}"


def act_gate(view: SelfMonitorView, faculty: str, *, last_attempt_at: Optional[datetime], now: datetime,
             probe_interval_sec: float = DEFAULT_PROBE_INTERVAL_SEC, enabled: bool = True) -> GateDecision:
    """Admit, admit as a probe, or refuse ``track_record_failing``.

    ``last_attempt_at`` is the newest real attempt of this faculty the caller knows of (ledger or
    its own memo), or None. A failing faculty with no attempt inside ``probe_interval_sec`` is
    admitted as the window's probe.
    """
    if not enabled:
        return GateDecision(faculty, True, False, "unknown", "disabled")
    if not view.known:
        return GateDecision(faculty, True, False, "unknown", f"unknown:{view.reason}")
    status = view.status(faculty)
    if status not in FAILING_STATUSES:
        return GateDecision(faculty, True, False, status, "not_failing")
    interval = max(0.0, float(probe_interval_sec))
    if last_attempt_at is None or (_utc(now) - _utc(last_attempt_at)).total_seconds() >= interval:
        return GateDecision(faculty, True, True, status, "probe", last_attempt_at)
    return GateDecision(faculty, False, False, status, TRACK_RECORD_FAILING_REASON, last_attempt_at)


class ProbeMemo:
    """This process's own admitted probes, faculty -> when. Covers the bus lag between a probe
    starting and its act row landing; the ledger covers restarts. Thread-safe (orion-actions reads
    the ledger on a worker thread)."""

    def __init__(self) -> None:
        self._at: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def note(self, faculty: str, at: datetime) -> None:
        with self._lock:
            self._at[faculty] = _utc(at)

    def get(self, faculty: str) -> Optional[datetime]:
        with self._lock:
            return self._at.get(faculty)


def newest(*values: Optional[datetime]) -> Optional[datetime]:
    present = [_utc(v) for v in values if v is not None]
    return max(present) if present else None


LedgerRows = Callable[[str, str, float], Optional[list]]


class ActGate:
    """The E1 gate one service holds: reads the view once per decision batch (``begin``), then
    decides per act. ``ledger_rows(actor, verb, secs)`` returns ``grammar_acts`` rows
    (object_ref, opened_at) opened in the last ``secs``, or None when the read failed."""

    def __init__(self, *, enabled: bool, probe_interval_sec: float,
                 read_view: Callable[[datetime], SelfMonitorView], ledger_rows: LedgerRows) -> None:
        self.enabled = bool(enabled)
        self.probe_interval_sec = max(0.0, float(probe_interval_sec))
        self._read_view = read_view
        self._ledger_rows = ledger_rows
        self._memo = ProbeMemo()
        self._view: Optional[SelfMonitorView] = None

    def begin(self, now: datetime) -> SelfMonitorView:
        """Read the published state once for this batch. Disabled: an unknown view, no Redis read."""
        if not self.enabled:
            self._view = SelfMonitorView(known=False, reason="disabled")
        else:
            try:
                self._view = self._read_view(now)
            except Exception:  # noqa: BLE001 -- unknown changes nothing
                self._view = SelfMonitorView(known=False, reason="read_failed")
        return self._view

    def decide(self, actor: str, verb: str, object_ref: str, *, now: datetime) -> GateDecision:
        faculty = act_faculty(actor, verb, object_ref)
        if not self.enabled:
            return act_gate(SelfMonitorView(known=False, reason="disabled"), faculty, last_attempt_at=None,
                            now=now, enabled=False)
        view = self._view if self._view is not None else self.begin(now)
        if not view.known or view.status(faculty) not in FAILING_STATUSES:
            # The common case: no ledger read at all.
            return act_gate(view, faculty, last_attempt_at=None, now=now,
                            probe_interval_sec=self.probe_interval_sec)
        ledger_at: Optional[datetime] = None
        try:
            rows = self._ledger_rows(actor, verb, self.probe_interval_sec)
        except Exception:  # noqa: BLE001 -- unreadable ledger: the memo alone decides
            logger.warning("self_monitor_gate_ledger_unreadable faculty=%s", faculty, exc_info=True)
            rows = None
        if rows is not None:
            ledger_at = latest_attempt_at(rows, actor=actor, verb=verb, faculty=faculty)
        decision = act_gate(view, faculty, last_attempt_at=newest(ledger_at, self._memo.get(faculty)), now=now,
                            probe_interval_sec=self.probe_interval_sec)
        if decision.probe:
            self._memo.note(faculty, now)
            logger.info("self_monitor_probe_admitted faculty=%s status=%s last_attempt=%s ledger_read=%s",
                        faculty, decision.status, decision.last_attempt_at, rows is not None)
        elif not decision.admit:
            logger.info("self_monitor_refused reason=%s faculty=%s status=%s last_attempt=%s",
                        decision.reason, faculty, decision.status, decision.last_attempt_at)
        return decision


def read_view_sync(redis_url: str, *, now: datetime, max_age_sec: float = DEFAULT_MAX_AGE_SEC,
                   timeout_sec: float = 1.0) -> SelfMonitorView:
    """GET the published state with a short timeout. Any failure is an unknown view (changes nothing)."""
    try:
        import redis  # inside: a missing driver degrades to unknown

        client = redis.Redis.from_url(redis_url, socket_timeout=timeout_sec, socket_connect_timeout=timeout_sec)
        try:
            raw = client.get(SELF_MONITOR_REDIS_KEY)
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001 -- unknown changes nothing
        logger.info("self_monitor_view_read_failed err=%s", type(exc).__name__)
        return SelfMonitorView(known=False, reason="read_failed")
    return self_monitor_view(raw, now=now, max_age_sec=max_age_sec)


__all__ = [
    "ActGate",
    "DEFAULT_PROBE_INTERVAL_SEC",
    "FAILING_STATUSES",
    "GateDecision",
    "ProbeMemo",
    "act_gate",
    "last_attempt_sql",
    "latest_attempt_at",
    "newest",
    "read_view_sync",
]

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
probes in-process (``ProbeMemo``), which covers the write lag. ``decide`` only OFFERS a probe; the
caller spends it with ``claim_probe`` at the moment the attempt really starts (journal: right before
its act opens; dispatch: when the candidate is actually sent). A probe offered but then dropped
(cooldown, budget, holdback) spends nothing, so the window can still probe. After a restart only
the ledger counts (real attempts), so a restart can never cost a probe. When the ledger cannot be read
the memo alone decides: the gate fails toward "admit" (gate-off behaviour), never toward a
permanent refusal.

**Retries of the same job are never refused (Juniper, option A, 2026-11-03).** The gate stops NEW
attempts at a failing faculty, not the retries of a job that already started. A caller that has a
job identity passes it as ``job_key`` (orion-actions: the journal's dedupe key, which is the act's
``test_before = journal:<dedupe_key>``; Prime: the concern's ``object_ref``). When the ledger or the
memo shows this same job already opened an act inside the ledger window (``probe_interval_sec``), the
attempt is admitted as ``same_job_retry``: not a probe, it spends no probe, logged
``self_monitor_same_job_retry``. Live reason: walkway_grade's nightly job fails while the GPU pool is
busy around 04:30Z and retries every few minutes until one attempt runs; with one probe per 6 h the
gate would have lost 3 of 7 nights. A job's first attempt is still refused unless a probe is due, so
a job refused before it ever started stays refused (it never opened an act). The chain is bounded by
the job key itself (walkway and daily keys carry the date; the others are content hashes of one
trigger), by the ledger window (a job whose last attempt is older than the window is a new job), and
by ``MAX_SAME_JOB_ATTEMPTS`` tries per job inside the window (then the normal rule applies).
The dispatch-runtime seam passes no ``job_key``: a sent dispatch leaves the pending set for good
(``dispatched_candidates``), and a refused one never started, so there is no "retry of an attempted
dispatch" to admit there.

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
SAME_JOB_RETRY_REASON = "same_job_retry"
# A job with this many attempts inside the window gets no more same-job admits: a stable key (or a
# chain of retries under 6 h apart) cannot keep a failing faculty running forever. Live max: 13
# walkway tries in one night (10-30).
MAX_SAME_JOB_ATTEMPTS = 24

# The act ledger read every caller runs to find the last real attempt (both drivers name params the
# same way through the caller's own placeholder style; see last_attempt_sql).
_LAST_ATTEMPT_SQL = (
    "SELECT object_ref, test_before, opened_at FROM grammar_acts "
    "WHERE actor = {actor} AND verb = {verb} AND opened_at IS NOT NULL "
    "AND opened_at > now() - make_interval(secs => {secs}) "
    "ORDER BY opened_at DESC LIMIT 500"
)


def last_attempt_sql(style: str = "sqlalchemy") -> str:
    """The ledger read, in the caller's placeholder style: ``sqlalchemy`` (:name), ``pyformat``
    (psycopg2's %(name)s) or ``asyncpg`` ($1 actor, $2 verb, $3 secs as float). Columns: object_ref,
    test_before, opened_at."""
    if style == "asyncpg":
        return _LAST_ATTEMPT_SQL.format(actor="$1", verb="$2", secs="$3::float8")
    if style == "pyformat":
        return _LAST_ATTEMPT_SQL.format(actor="%(actor)s", verb="%(verb)s", secs="%(secs)s")
    return _LAST_ATTEMPT_SQL.format(actor=":actor", verb=":verb", secs=":secs")


def utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


_utc = utc


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


JobKeyOf = Callable[[Mapping[str, Any]], Optional[str]]


def job_attempts(rows: Iterable[Mapping[str, Any]], *, actor: str, verb: str, faculty: str, job_key: str,
                 job_key_of: JobKeyOf) -> int:
    """How many ledger rows of this faculty belong to ``job_key`` (the caller's own job identity,
    read from the row by ``job_key_of``)."""
    n = 0
    for row in rows:
        if row.get("opened_at") is None:
            continue
        if act_faculty(actor, verb, str(row.get("object_ref") or "")) != faculty:
            continue
        if job_key_of(row) == job_key:
            n += 1
    return n


def job_attempted(rows: Iterable[Mapping[str, Any]], *, actor: str, verb: str, faculty: str, job_key: str,
                  job_key_of: JobKeyOf) -> bool:
    return job_attempts(rows, actor=actor, verb=verb, faculty=faculty, job_key=job_key, job_key_of=job_key_of) > 0


@dataclass(frozen=True)
class GateDecision:
    faculty: str
    admit: bool
    probe: bool
    status: str
    # not_failing | unknown:<view reason> | probe | same_job_retry | track_record_failing | disabled
    reason: str
    last_attempt_at: Optional[datetime] = None

    def label(self) -> str:
        """One frame/audit-visible token: what the gate did and why."""
        if self.probe:
            when = self.last_attempt_at.isoformat() if self.last_attempt_at else "none"
            return f"self_monitor:probe:{self.faculty}:{self.status}:last_attempt={when}"
        if not self.admit:
            return f"self_monitor:{TRACK_RECORD_FAILING_REASON}:{self.faculty}:{self.status}"
        if self.reason == SAME_JOB_RETRY_REASON:
            return f"self_monitor:{SAME_JOB_RETRY_REASON}:{self.faculty}:{self.status}"
        return f"self_monitor:admit:{self.faculty}:{self.reason}"

    @property
    def same_job_retry(self) -> bool:
        return self.reason == SAME_JOB_RETRY_REASON



def act_gate(view: SelfMonitorView, faculty: str, *, last_attempt_at: Optional[datetime], now: datetime,
             probe_interval_sec: float = DEFAULT_PROBE_INTERVAL_SEC, enabled: bool = True,
             same_job_attempted: bool = False) -> GateDecision:
    """Admit, admit as a probe, admit as a same-job retry, or refuse ``track_record_failing``.

    ``last_attempt_at`` is the newest real attempt of this faculty the caller knows of (ledger or
    its own memo), or None. A failing faculty with no attempt inside ``probe_interval_sec`` is
    admitted as the window's probe. ``same_job_attempted``: this attempt's own job already opened an
    act inside the window, so it is a retry of a started job and always admitted (no probe spent).
    """
    if not enabled:
        return GateDecision(faculty, True, False, "unknown", "disabled")
    if not view.known:
        return GateDecision(faculty, True, False, "unknown", f"unknown:{view.reason}")
    status = view.status(faculty)
    if status not in FAILING_STATUSES:
        return GateDecision(faculty, True, False, status, "not_failing")
    if same_job_attempted:
        return GateDecision(faculty, True, False, status, SAME_JOB_RETRY_REASON, last_attempt_at)
    interval = max(0.0, float(probe_interval_sec))
    if last_attempt_at is None or (_utc(now) - _utc(last_attempt_at)).total_seconds() >= interval:
        return GateDecision(faculty, True, True, status, "probe", last_attempt_at)
    return GateDecision(faculty, False, False, status, TRACK_RECORD_FAILING_REASON, last_attempt_at)


class ProbeMemo:
    """This process's own started probes, faculty -> when. Covers the bus lag between a probe
    starting and its act row landing; the ledger covers restarts. ``claim`` is an atomic
    check-and-set, so two concurrent callers (orion-actions runs handlers concurrently) can never
    both start a probe in one window."""

    def __init__(self) -> None:
        self._at: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def claim(self, faculty: str, at: datetime, interval_sec: float) -> bool:
        at = _utc(at)
        with self._lock:
            prev = self._at.get(faculty)
            if prev is not None and (at - prev).total_seconds() < interval_sec:
                return False
            self._at[faculty] = at
            return True

    def get(self, faculty: str) -> Optional[datetime]:
        with self._lock:
            return self._at.get(faculty)


class JobMemo:
    """This process's own started jobs, (faculty, job_key) -> when: a retry right after a start whose
    act row has not landed yet (bus lag) is still known as the same job. Entries older than the
    window are dropped on read."""

    def __init__(self) -> None:
        self._at: dict[tuple[str, str], datetime] = {}
        self._lock = threading.Lock()

    def note(self, faculty: str, job_key: str, at: datetime) -> None:
        with self._lock:
            self._at[(faculty, job_key)] = _utc(at)

    def seen(self, faculty: str, job_key: str, *, now: datetime, window_sec: float) -> bool:
        now = _utc(now)
        with self._lock:
            for key in [k for k, at in self._at.items() if (now - at).total_seconds() >= window_sec]:
                del self._at[key]
            return (faculty, job_key) in self._at


def newest(*values: Optional[datetime]) -> Optional[datetime]:
    present = [_utc(v) for v in values if v is not None]
    return max(present) if present else None


LedgerRows = Callable[[str, str, float], Optional[list]]


class ActGate:
    """The E1 gate one service holds: reads the view once per decision batch (``begin``), then
    decides per act. ``ledger_rows(actor, verb, secs)`` returns ``grammar_acts`` rows
    (object_ref, test_before, opened_at) opened in the last ``secs``, or None when the read failed.
    ``job_key_of(row)`` reads the caller's job identity off a ledger row (None: the caller has no
    same-job notion, and ``decide(job_key=...)`` is ignored)."""

    def __init__(self, *, enabled: bool, probe_interval_sec: float,
                 read_view: Callable[[datetime], SelfMonitorView], ledger_rows: LedgerRows,
                 job_key_of: Optional[JobKeyOf] = None) -> None:
        self.enabled = bool(enabled)
        self.probe_interval_sec = max(0.0, float(probe_interval_sec))
        self._read_view = read_view
        self._ledger_rows = ledger_rows
        self._job_key_of = job_key_of
        self._memo = ProbeMemo()
        self._jobs = JobMemo()
        self._view: Optional[SelfMonitorView] = None
        # Faculties already offered a probe in this batch (one tick): a second candidate of the same
        # faculty in the same batch is refused. Cleared by begin().
        self._offered: set[str] = set()

    def begin(self, now: datetime) -> SelfMonitorView:
        """Read the published state once for this batch. Disabled: an unknown view, no Redis read."""
        self._offered = set()
        if not self.enabled:
            self._view = SelfMonitorView(known=False, reason="disabled")
        else:
            try:
                self._view = self._read_view(now)
            except Exception:  # noqa: BLE001 -- unknown changes nothing
                self._view = SelfMonitorView(known=False, reason="read_failed")
        return self._view

    def decide(self, actor: str, verb: str, object_ref: str, *, now: datetime,
               job_key: Optional[str] = None) -> GateDecision:
        faculty = act_faculty(actor, verb, object_ref)
        if self._job_key_of is None:
            job_key = None
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
        tries = 0
        if job_key and rows is not None and self._job_key_of is not None:
            tries = job_attempts(rows, actor=actor, verb=verb, faculty=faculty, job_key=job_key,
                                 job_key_of=self._job_key_of)
        seen = bool(job_key) and (tries > 0 or self._jobs.seen(faculty, job_key, now=now,
                                                               window_sec=self.probe_interval_sec))
        same_job = seen and tries < MAX_SAME_JOB_ATTEMPTS
        if seen and not same_job:
            logger.info("self_monitor_same_job_cap faculty=%s job=%s tries=%d", faculty, job_key, tries)
        decision = act_gate(view, faculty, last_attempt_at=newest(ledger_at, self._memo.get(faculty)), now=now,
                            probe_interval_sec=self.probe_interval_sec, same_job_attempted=same_job)
        if decision.same_job_retry:
            logger.info("self_monitor_same_job_retry faculty=%s status=%s job=%s tries=%d",
                        faculty, decision.status, job_key, tries)
            return decision
        if decision.probe and faculty in self._offered:
            decision = GateDecision(faculty, False, False, decision.status, TRACK_RECORD_FAILING_REASON,
                                    decision.last_attempt_at)
        if decision.probe:
            self._offered.add(faculty)
            logger.info("self_monitor_probe_offered faculty=%s status=%s last_attempt=%s ledger_read=%s",
                        faculty, decision.status, decision.last_attempt_at, rows is not None)
        elif not decision.admit:
            logger.info("self_monitor_refused reason=%s faculty=%s status=%s last_attempt=%s",
                        decision.reason, faculty, decision.status, decision.last_attempt_at)
        return decision

    def claim_probe(self, faculty: str, *, now: datetime, job_key: Optional[str] = None) -> bool:
        """Spend the window's probe at the moment the attempt really starts. False: another caller
        already started one this window, so this attempt must be refused instead. ``job_key``: the
        probe's job is now started, so its own retries are same-job retries."""
        ok = self._memo.claim(faculty, now, self.probe_interval_sec)
        logger.info("self_monitor_probe_%s faculty=%s", "started" if ok else "lost_race", faculty)
        if ok and job_key and self._job_key_of is not None:
            self._jobs.note(faculty, job_key, now)
        return ok

    def note_job_started(self, faculty: str, job_key: Optional[str], *, now: datetime) -> None:
        """An admitted (not probe) attempt really started: remember its job for the bus lag."""
        if job_key and self._job_key_of is not None:
            self._jobs.note(faculty, job_key, now)


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
    "JobMemo",
    "MAX_SAME_JOB_ATTEMPTS",
    "ProbeMemo",
    "SAME_JOB_RETRY_REASON",
    "act_gate",
    "job_attempted",
    "job_attempts",
    "last_attempt_sql",
    "latest_attempt_at",
    "newest",
    "read_view_sync",
]

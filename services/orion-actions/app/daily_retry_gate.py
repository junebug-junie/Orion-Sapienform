"""Bounded retries for the nightly daily_pulse_v1 / daily_metacog_v1 jobs.

Before this, the scheduler loop (every 45s) re-ran a failed daily job on every
tick until local midnight, because only a success sets the job's done-today
cursor. A deterministic failure (daily_metacog_prompt_over_limit, 2026-09-03 to
09-30) therefore ran 230-280 times a night.

Retries now depend on what kind of failure it was (``classify_daily_failure``):

- deterministic (the same input will fail the same way: prompt over limit,
  JSON parse/validation, truncated output): at most 3 tries per scheduled local
  date, 10 then 30 minutes apart, then give up for that date;
- transient (timeouts, GPU pool / gateway capacity refusals, notify down):
  retry every 60 minutes until the local date ends -- an outage at 20:15 should
  not cost the whole night;
- unknown: paced like transient but capped at 6 tries.

Give-up is recorded once per date (the caller's ``on_gave_up``: audit + warning
notification). A date that ends with failures but no give-up (transient all
evening) is recorded as given up on the first tick of the next date. The job is
never marked completed by any of this.

State (counts, last failure time, gave_up) is persisted to a small JSON file
next to the scheduler cursors, so a restart neither resets the counts nor
reopens retries for a date that already gave up. Timestamps are wall-clock
epoch seconds for that reason.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal

logger = logging.getLogger(__name__)

FailureClass = Literal["deterministic", "transient", "unknown"]

MAX_DETERMINISTIC_ATTEMPTS = 3
MAX_UNKNOWN_ATTEMPTS = 6
# Wait after deterministic failure #1, #2 (seconds).
DETERMINISTIC_BACKOFF_SECONDS: tuple[float, ...] = (600.0, 1800.0)
# Wait after any transient or unknown failure.
SLOW_RETRY_SECONDS = 3600.0
DAILY_ATTEMPT_STATE_FILENAME = "daily_attempts.json"

# Lower-cased substrings, from the real error strings: cognition_traces step
# errors (30 days to 2026-10-01) and the RuntimeErrors raised in
# orion-actions' _execute_daily / _run_plan.
_DETERMINISTIC_MARKERS = (
    "prompt_over_limit",
    "truncated_generation",
    "daily_json_parse_unavailable",
    "llm output must be a json object",
    "could not parse json object",
    "jsondecodeerror",
    "expecting value",
    "validation error",
    "cortex_exec_decode_failed",
)
_TRANSIENT_MARKERS = (
    "timeout",  # "RPC timeout waiting on ...", TimeoutError, timeout:caller_budget_exhausted
    "gpu_pool_unavailable",
    "gpu_pool_recalled",
    "gateway_capacity_rejected",
    "resource_lease_rejected",
    "notify_failed",
    "connection",
    "temporarily unavailable",
)


def classify_daily_failure(reason: str | None) -> FailureClass:
    text = str(reason or "").lower()
    if any(m in text for m in _DETERMINISTIC_MARKERS):
        return "deterministic"
    if any(m in text for m in _TRANSIENT_MARKERS):
        return "transient"
    return "unknown"


@dataclass
class DailyAttemptResult:
    completed: bool
    # False ONLY when the run was skipped without doing work (deduped because
    # another run holds the lock). A skip does not spend an attempt.
    attempted: bool = True
    failure_reason: str | None = None


@dataclass
class _JobState:
    local_date: str
    deterministic: int = 0
    transient: int = 0
    unknown: int = 0
    last_failure_at: float | None = None
    last_failure_class: str | None = None
    last_failure_reason: str | None = None
    gave_up: bool = False

    @property
    def attempts(self) -> int:
        return self.deterministic + self.transient + self.unknown


@dataclass
class GaveUp:
    local_date: str
    attempts: int
    reason: str | None
    failure_class: str | None


class DailyAttemptGate:
    def __init__(self, *, state_path: Path | None = None) -> None:
        self._path = state_path
        self._jobs: dict[str, _JobState] = {}
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text() or "{}")
            for job, data in (raw or {}).items():
                self._jobs[str(job)] = _JobState(**data)
        except Exception as exc:
            logger.warning("daily_attempt_gate_load_failed path=%s error=%s", self._path, exc)

    def _persist(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: asdict(v) for k, v in self._jobs.items()}, indent=2, sort_keys=True))
            os.replace(tmp, self._path)
        except Exception as exc:
            logger.warning("daily_attempt_gate_persist_failed path=%s error=%s", self._path, exc)

    # -- state ---------------------------------------------------------------
    def attempts(self, job_key: str, local_date: str) -> int:
        state = self._jobs.get(job_key)
        return state.attempts if state and state.local_date == local_date else 0

    def is_gave_up(self, job_key: str, local_date: str) -> bool:
        state = self._jobs.get(job_key)
        return bool(state and state.local_date == local_date and state.gave_up)

    def roll_over(self, job_key: str, local_date: str) -> GaveUp | None:
        """Start ``local_date``. Returns the previous date's give-up if it ended
        with failures that were never recorded (e.g. transient all evening)."""
        state = self._jobs.get(job_key)
        if state is None or state.local_date == local_date:
            return None
        expired = None
        if state.attempts and not state.gave_up:
            expired = GaveUp(state.local_date, state.attempts, state.last_failure_reason, state.last_failure_class)
        del self._jobs[job_key]
        self._persist()
        return expired

    def may_attempt(self, job_key: str, local_date: str, now: float) -> tuple[bool, str]:
        state = self._jobs.get(job_key)
        if state is None or state.local_date != local_date:
            return True, "ok"
        if state.gave_up:
            return False, "gave_up"
        if state.last_failure_at is not None:
            if state.last_failure_class == "deterministic":
                idx = min(state.deterministic - 1, len(DETERMINISTIC_BACKOFF_SECONDS) - 1)
                wait = DETERMINISTIC_BACKOFF_SECONDS[max(0, idx)]
            else:
                wait = SLOW_RETRY_SECONDS
            if now - state.last_failure_at < wait:
                return False, "backoff"
        return True, "ok"

    def record_failure(self, job_key: str, local_date: str, now: float, reason: str | None) -> GaveUp | None:
        """Count one failure. Returns a GaveUp when this failure ends the date."""
        state = self._jobs.get(job_key)
        if state is None or state.local_date != local_date:
            state = _JobState(local_date=local_date)
            self._jobs[job_key] = state
        klass = classify_daily_failure(reason)
        setattr(state, klass, getattr(state, klass) + 1)
        state.last_failure_at = now
        state.last_failure_class = klass
        state.last_failure_reason = (reason or "")[:1000] or None
        if not state.gave_up and (
            state.deterministic >= MAX_DETERMINISTIC_ATTEMPTS or state.unknown >= MAX_UNKNOWN_ATTEMPTS
        ):
            state.gave_up = True
            self._persist()
            return GaveUp(local_date, state.attempts, state.last_failure_reason, klass)
        self._persist()
        return None

    def record_success(self, job_key: str) -> None:
        if self._jobs.pop(job_key, None) is not None:
            self._persist()


async def run_gated_daily_tick(
    *,
    job_key: str,
    due: bool,
    local_date: str,
    now: float,
    gate: DailyAttemptGate,
    execute: Callable[[], Awaitable[DailyAttemptResult]],
    on_completed: Callable[[str], Awaitable[None]],
    on_gave_up: Callable[[GaveUp], Awaitable[None]],
) -> str:
    """One scheduler tick for one daily job. Returns what happened, for logs/tests.

    ``due`` and ``local_date`` come from the caller's ``should_run_daily`` (plus
    its run-on-startup rule); ``on_completed`` sets the done-today cursor.
    """
    expired = gate.roll_over(job_key, local_date)
    if expired is not None:
        await on_gave_up(expired)
    if not due:
        return "not_due"
    allowed, why = gate.may_attempt(job_key, local_date, now)
    if not allowed:
        return why
    result = await execute()
    if result.completed:
        gate.record_success(job_key)
        await on_completed(local_date)
        return "completed"
    if not result.attempted:
        return "skipped"
    gave_up = gate.record_failure(job_key, local_date, now, result.failure_reason)
    if gave_up is not None:
        await on_gave_up(gave_up)
        return "gave_up"
    return "failed"

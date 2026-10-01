"""Bounded retries for the nightly daily_pulse_v1 / daily_metacog_v1 jobs.

Before this, the scheduler loop (every 45s) re-ran a failed daily job on every
tick until local midnight, because only a success sets the job's done-today
cursor. A deterministic failure (daily_metacog_prompt_over_limit, 2026-09-03 on)
therefore ran 230-280 times a night.

Now each job gets at most ``max_attempts`` tries per scheduled local date, with
backoff between them. When the last try fails, the job "gives up" for that date:
the caller records a visible failure (audit + warning notification) and the
give-up date is persisted, so a restart does not reopen the storm. The job is
NOT marked completed; the next local date starts fresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

MAX_DAILY_ATTEMPTS = 3
# Wait after failure #1, failure #2, ... (seconds, monotonic clock).
DAILY_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (600.0, 1800.0)
GAVE_UP_CURSOR_SUFFIX = ".gave_up"


def gave_up_cursor_key(job_key: str) -> str:
    return f"{job_key}{GAVE_UP_CURSOR_SUFFIX}"


@dataclass
class DailyAttemptResult:
    completed: bool
    # False when the run was skipped without doing work (e.g. deduped because a
    # manual run holds the lock). A skip does not spend an attempt.
    attempted: bool = True
    failure_reason: str | None = None


@dataclass
class _JobState:
    local_date: str
    attempts: int = 0
    last_failure_at: float | None = None
    last_failure_reason: str | None = None
    gave_up: bool = False


class DailyAttemptGate:
    def __init__(
        self,
        *,
        max_attempts: int = MAX_DAILY_ATTEMPTS,
        backoff_seconds: tuple[float, ...] = DAILY_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_seconds = tuple(float(x) for x in backoff_seconds)
        self._jobs: dict[str, _JobState] = {}

    def _state(self, job_key: str, local_date: str) -> _JobState:
        state = self._jobs.get(job_key)
        if state is None or state.local_date != local_date:
            state = _JobState(local_date=local_date)
            self._jobs[job_key] = state
        return state

    def mark_gave_up(self, job_key: str, local_date: str) -> None:
        """Seed a persisted give-up (startup) so a restart cannot restart the storm."""
        state = self._state(job_key, local_date)
        state.gave_up = True
        state.attempts = max(state.attempts, self.max_attempts)

    def attempts(self, job_key: str, local_date: str) -> int:
        state = self._jobs.get(job_key)
        return state.attempts if state and state.local_date == local_date else 0

    def may_attempt(self, job_key: str, local_date: str, now_monotonic: float) -> tuple[bool, str]:
        state = self._state(job_key, local_date)
        if state.gave_up or state.attempts >= self.max_attempts:
            return False, "gave_up"
        if state.attempts and state.last_failure_at is not None:
            idx = min(state.attempts - 1, len(self.backoff_seconds) - 1)
            wait = self.backoff_seconds[idx] if self.backoff_seconds else 0.0
            if now_monotonic - state.last_failure_at < wait:
                return False, "backoff"
        return True, "ok"

    def record_failure(
        self, job_key: str, local_date: str, now_monotonic: float, reason: str | None
    ) -> tuple[int, bool]:
        """Returns (attempts_so_far, gave_up_now)."""
        state = self._state(job_key, local_date)
        state.attempts += 1
        state.last_failure_at = now_monotonic
        state.last_failure_reason = reason
        if state.attempts >= self.max_attempts and not state.gave_up:
            state.gave_up = True
            return state.attempts, True
        return state.attempts, False

    def record_success(self, job_key: str) -> None:
        self._jobs.pop(job_key, None)


async def run_gated_daily_tick(
    *,
    job_key: str,
    due: bool,
    local_date: str,
    now_monotonic: float,
    gate: DailyAttemptGate,
    execute: Callable[[], Awaitable[DailyAttemptResult]],
    on_completed: Callable[[str], Awaitable[None]],
    on_gave_up: Callable[[str, int, str | None], Awaitable[None]],
) -> str:
    """One scheduler tick for one daily job. Returns what happened, for logs/tests.

    ``due`` and ``local_date`` come from the caller's ``should_run_daily`` (plus
    its run-on-startup rule); ``on_completed`` sets the done-today cursor.
    """
    if not due:
        return "not_due"
    allowed, why = gate.may_attempt(job_key, local_date, now_monotonic)
    if not allowed:
        return why
    result = await execute()
    if result.completed:
        gate.record_success(job_key)
        await on_completed(local_date)
        return "completed"
    if not result.attempted:
        return "skipped"
    attempts, gave_up_now = gate.record_failure(job_key, local_date, now_monotonic, result.failure_reason)
    if gave_up_now:
        await on_gave_up(local_date, attempts, result.failure_reason)
        return "gave_up"
    return "failed"

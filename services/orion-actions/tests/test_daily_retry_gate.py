"""Nightly daily jobs stop retrying after 3 failures (redesign Stage 0B).

Regression: a deterministic daily_metacog_v1 failure re-ran on every 45s
scheduler tick from 20:15 to midnight Denver, 230-280 times a night.
These tests drive the real should_run_daily + SchedulerCursorStore cursor logic
through run_gated_daily_tick, the same helper the scheduler loop uses.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.daily_retry_gate import (
    MAX_DAILY_ATTEMPTS,
    DailyAttemptGate,
    DailyAttemptResult,
    gave_up_cursor_key,
    run_gated_daily_tick,
)
from app.main import (
    ACTION_DAILY_METACOG_V1,
    ACTION_DAILY_PULSE_V1,
    _daily_skills_catalog_context,
    _plan_failure_detail,
    daily_failure_notify_request,
    should_run_daily,
)
from app.scheduler_cursor_store import SchedulerCursorStore, scheduler_cursor_completed_local_date

TZ = "America/Denver"
TICK = timedelta(seconds=45)
OVER_LIMIT = (
    "cortex_exec_missing_final_text step=draft_daily_metacog error=LLMGatewayService: "
    "daily_metacog_prompt_over_limit chars=8467 limit=8192"
)


def _simulate(
    *,
    store: SchedulerCursorStore,
    gate: DailyAttemptGate,
    start_utc: datetime,
    end_utc: datetime,
    outcome,
    job: str = ACTION_DAILY_METACOG_V1,
    hour: int = 20,
    minute: int = 15,
):
    """Replays the scheduler loop's per-tick logic for one job."""
    calls: list[datetime] = []
    gave_up: list[tuple[str, int, str | None]] = []
    last_daily_run = dict(store.all())
    t0 = start_utc

    async def _run() -> None:
        now = start_utc
        while now < end_utc:
            due, local_date = should_run_daily(
                now_utc=now, tz_name=TZ, hour_local=hour, minute_local=minute,
                last_ran_date=last_daily_run.get(job),
            )

            async def _execute(now=now) -> DailyAttemptResult:
                calls.append(now)
                return outcome(now)

            async def _on_completed(scheduled: str) -> None:
                cursor = scheduler_cursor_completed_local_date(
                    forced_date=None, window_request_date="unused", scheduled_local_date=scheduled
                )
                last_daily_run[job] = cursor
                store.set_last_completed(job, cursor)

            async def _on_gave_up(scheduled: str, attempts: int, reason: str | None) -> None:
                store.set_last_completed(gave_up_cursor_key(job), scheduled)
                gave_up.append((scheduled, attempts, reason))

            await run_gated_daily_tick(
                job_key=job,
                due=due,
                local_date=local_date,
                now_monotonic=(now - t0).total_seconds(),
                gate=gate,
                execute=_execute,
                on_completed=_on_completed,
                on_gave_up=_on_gave_up,
            )
            now += TICK

    asyncio.run(_run())
    return calls, gave_up, last_daily_run


def _always_fail(_now):
    return DailyAttemptResult(completed=False, failure_reason=OVER_LIMIT)


def test_deterministic_failure_runs_three_times_a_night_not_250(tmp_path) -> None:
    store = SchedulerCursorStore(tmp_path / "cursors.json")
    gate = DailyAttemptGate()
    # 2026-09-30 02:00 UTC (20:00 MDT, 09-29) through 2026-10-01 07:00 UTC: two nights.
    calls, gave_up, last = _simulate(
        store=store, gate=gate,
        start_utc=datetime(2026, 9, 30, 2, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 10, 1, 7, 0, tzinfo=timezone.utc),
        outcome=_always_fail,
    )
    per_night: dict[str, int] = {}
    for c in calls:
        local = c.astimezone(ZoneInfo(TZ)).date().isoformat()
        per_night[local] = per_night.get(local, 0) + 1
    assert per_night == {"2026-09-29": MAX_DAILY_ATTEMPTS, "2026-09-30": MAX_DAILY_ATTEMPTS}
    # Backoff: 2nd try >= 10 min after the 1st, 3rd >= 30 min after the 2nd.
    assert calls[1] - calls[0] >= timedelta(minutes=10)
    assert calls[2] - calls[1] >= timedelta(minutes=30)
    # One visible failure per night, naming the real cause.
    assert [g[:2] for g in gave_up] == [("2026-09-29", 3), ("2026-09-30", 3)]
    assert all("daily_metacog_prompt_over_limit" in (g[2] or "") for g in gave_up)
    # Never marked completed.
    assert ACTION_DAILY_METACOG_V1 not in last
    assert store.get(ACTION_DAILY_METACOG_V1) is None
    assert store.get(gave_up_cursor_key(ACTION_DAILY_METACOG_V1)) == "2026-09-30"


def test_transient_failure_then_success_sets_cursor_once(tmp_path) -> None:
    store = SchedulerCursorStore(tmp_path / "cursors.json")
    gate = DailyAttemptGate()
    seen = {"n": 0}

    def _fail_once(_now):
        seen["n"] += 1
        if seen["n"] == 1:
            return DailyAttemptResult(completed=False, failure_reason="timeout")
        return DailyAttemptResult(completed=True)

    calls, gave_up, last = _simulate(
        store=store, gate=gate,
        start_utc=datetime(2026, 9, 30, 2, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 9, 30, 6, 30, tzinfo=timezone.utc),
        outcome=_fail_once,
    )
    assert len(calls) == 2
    assert gave_up == []
    assert last[ACTION_DAILY_METACOG_V1] == "2026-09-29"
    assert store.get(ACTION_DAILY_METACOG_V1) == "2026-09-29"


def test_dedupe_skip_does_not_spend_an_attempt(tmp_path) -> None:
    store = SchedulerCursorStore(tmp_path / "cursors.json")
    gate = DailyAttemptGate()
    calls, gave_up, _ = _simulate(
        store=store, gate=gate,
        start_utc=datetime(2026, 9, 30, 2, 14, tzinfo=timezone.utc),
        end_utc=datetime(2026, 9, 30, 2, 30, tzinfo=timezone.utc),
        outcome=lambda _now: DailyAttemptResult(completed=False, attempted=False),
    )
    assert len(calls) > MAX_DAILY_ATTEMPTS  # skips retried every tick, as before
    assert gave_up == []
    assert gate.attempts(ACTION_DAILY_METACOG_V1, "2026-09-29") == 0


def test_persisted_give_up_survives_restart(tmp_path) -> None:
    path = tmp_path / "cursors.json"
    store = SchedulerCursorStore(path)
    store.set_last_completed(gave_up_cursor_key(ACTION_DAILY_METACOG_V1), "2026-09-29")
    # Fresh process: reload store, seed gate the way main.py's lifespan does.
    store = SchedulerCursorStore(path)
    gate = DailyAttemptGate()
    gate.mark_gave_up(ACTION_DAILY_METACOG_V1, store.get(gave_up_cursor_key(ACTION_DAILY_METACOG_V1)))
    calls, _, _ = _simulate(
        store=store, gate=gate,
        start_utc=datetime(2026, 9, 30, 4, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 9, 30, 5, 59, tzinfo=timezone.utc),
        outcome=_always_fail,
    )
    assert calls == []


def test_pulse_gets_the_same_cap(tmp_path) -> None:
    store = SchedulerCursorStore(tmp_path / "cursors.json")
    calls, gave_up, _ = _simulate(
        store=store, gate=DailyAttemptGate(),
        start_utc=datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 10, 1, 5, 59, tzinfo=timezone.utc),
        outcome=_always_fail,
        job=ACTION_DAILY_PULSE_V1, hour=8, minute=30,
    )
    assert len(calls) == MAX_DAILY_ATTEMPTS
    assert len(gave_up) == 1


def test_plan_failure_detail_recovers_step_error() -> None:
    payload = {
        "result": {
            "status": "fail",
            "steps": [
                {"step_name": "recall", "error": None},
                {
                    "step_name": "draft_daily_metacog",
                    "error": "LLMGatewayService: daily_metacog_prompt_over_limit chars=8467 limit=8192",
                },
            ],
        }
    }
    detail = _plan_failure_detail(payload)
    assert detail == (
        "step=draft_daily_metacog error=LLMGatewayService: daily_metacog_prompt_over_limit chars=8467 limit=8192"
    )
    assert _plan_failure_detail({"result": {"steps": [], "error": "boom"}}) == "error=boom"
    assert _plan_failure_detail({}) is None


def test_failure_notification_is_warning_and_names_error() -> None:
    req = daily_failure_notify_request(
        action_name=ACTION_DAILY_METACOG_V1,
        report_date="2026-09-29",
        scheduled_local_date="2026-09-29",
        attempts=3,
        error=OVER_LIMIT,
        correlation_id="00000000-0000-0000-0000-000000000001",
    )
    assert req.severity == "warning"
    assert req.event_kind == "orion.daily.failed"
    assert "daily_metacog_prompt_over_limit" in req.body_text
    assert req.context["attempts"] == 3
    assert req.dedupe_key == "actions:daily_metacog_v1:failed:2026-09-29"


def test_metacog_gets_bounded_catalog_and_pulse_keeps_json() -> None:
    meta = _daily_skills_catalog_context(ACTION_DAILY_METACOG_V1)
    pulse = _daily_skills_catalog_context(ACTION_DAILY_PULSE_V1)
    assert pulse["skills_catalog_compact"].startswith("[")  # unchanged JSON for pulse
    assert not meta["skills_catalog_compact"].startswith("[")
    assert len(meta["skills_catalog_compact"]) < len(pulse["skills_catalog_compact"]) / 2
    assert meta["skills_catalog_count"] == len(meta["skills_catalog_compact"].splitlines())

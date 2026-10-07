"""Nightly daily jobs stop retrying all night (redesign Stage 0B).

Regression: a deterministic daily_metacog_v1 failure re-ran on every 45s
scheduler tick from 20:15 to midnight Denver, 230-280 times a night.
The simulation tests drive the real should_run_daily + SchedulerCursorStore
cursor logic through run_gated_daily_tick; the lifespan tests drive the real
_scheduled_daily_tick -> _execute_daily -> _run_plan path with only the cortex
RPC stubbed.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from app import main as actions_main
from app.daily_retry_gate import (
    MAX_DETERMINISTIC_ATTEMPTS,
    MAX_UNKNOWN_ATTEMPTS,
    DailyAttemptGate,
    DailyAttemptResult,
    classify_daily_failure,
    run_gated_daily_tick,
)
from app.main import (
    ACTION_DAILY_METACOG_V1,
    ACTION_DAILY_PULSE_V1,
    _daily_skills_catalog_context,
    _normalize_daily_skill_selection,
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
RPC_TIMEOUT = (
    "cortex_exec_missing_final_text step=draft_daily_metacog error=LLMGatewayService: "
    "RPC timeout waiting on orion:exec:result:LLMGatewayService:abc"
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
    gave_up: list = []
    last_daily_run = dict(store.all())

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

            async def _on_gave_up(g) -> None:
                gave_up.append(g)

            await run_gated_daily_tick(
                job_key=job,
                due=due,
                local_date=local_date,
                now=now.timestamp(),
                gate=gate,
                execute=_execute,
                on_completed=_on_completed,
                on_gave_up=_on_gave_up,
            )
            now += TICK

    asyncio.run(_run())
    return calls, gave_up, last_daily_run


def _fail_with(reason: str):
    return lambda _now: DailyAttemptResult(completed=False, failure_reason=reason)


def _per_local_night(calls: list[datetime]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in calls:
        local = c.astimezone(ZoneInfo(TZ)).date().isoformat()
        out[local] = out.get(local, 0) + 1
    return out


# --- classification ---------------------------------------------------------

@pytest.mark.parametrize(
    "reason,expected",
    [
        (OVER_LIMIT, "deterministic"),
        ("truncated_generation", "deterministic"),
        ("Could not parse JSON object from LLM text: 'x'", "deterministic"),
        ("1 validation error for DailyMetacogV1", "deterministic"),
        (RPC_TIMEOUT, "transient"),
        ("TimeoutError", "transient"),
        ("cortex_exec_missing_final_text step=llm error=gpu_pool_unavailable:deadline", "transient"),
        ("cortex_exec_missing_final_text step=llm error=gateway_capacity_rejected:capacity_wait_budget_exhausted", "transient"),
        ("timeout:caller_budget_exhausted", "transient"),
        ("notify_failed:503", "transient"),
        ("cortex_exec_missing_final_text", "unknown"),
        ("something new", "unknown"),
    ],
)
def test_classification_uses_real_error_strings(reason: str, expected: str) -> None:
    assert classify_daily_failure(reason) == expected


# --- per-night behavior, real should_run_daily + cursor store -----------------

def test_deterministic_failure_runs_three_times_a_night_not_250(tmp_path) -> None:
    store = SchedulerCursorStore(tmp_path / "cursors.json")
    gate = DailyAttemptGate()
    # 2026-09-30 02:00 UTC (20:00 MDT, 09-29) through 2026-10-01 07:00 UTC: two nights.
    calls, gave_up, last = _simulate(
        store=store, gate=gate,
        start_utc=datetime(2026, 9, 30, 2, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 10, 1, 7, 0, tzinfo=timezone.utc),
        outcome=_fail_with(OVER_LIMIT),
    )
    assert _per_local_night(calls) == {"2026-09-29": MAX_DETERMINISTIC_ATTEMPTS, "2026-09-30": MAX_DETERMINISTIC_ATTEMPTS}
    assert calls[1] - calls[0] >= timedelta(minutes=10)
    assert calls[2] - calls[1] >= timedelta(minutes=30)
    assert [(g.local_date, g.attempts, g.failure_class) for g in gave_up] == [
        ("2026-09-29", 3, "deterministic"),
        ("2026-09-30", 3, "deterministic"),
    ]
    assert all("daily_metacog_prompt_over_limit" in (g.reason or "") for g in gave_up)
    assert ACTION_DAILY_METACOG_V1 not in last
    assert store.get(ACTION_DAILY_METACOG_V1) is None


def test_transient_failure_keeps_hourly_retry_until_the_date_ends(tmp_path) -> None:
    store = SchedulerCursorStore(tmp_path / "cursors.json")
    gate = DailyAttemptGate()
    calls, gave_up, last = _simulate(
        store=store, gate=gate,
        start_utc=datetime(2026, 9, 30, 2, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 9, 30, 6, 30, tzinfo=timezone.utc),  # past local midnight
        outcome=_fail_with(RPC_TIMEOUT),
    )
    # 20:15, 21:15, 22:15, 23:15 local: a 20:15 outage does not cost the night.
    assert len(calls) == 4
    assert all(b - a >= timedelta(minutes=60) for a, b in zip(calls, calls[1:]))
    # The date ended with no success: recorded once, on the next date's first tick.
    assert [(g.local_date, g.attempts, g.failure_class) for g in gave_up] == [("2026-09-29", 4, "transient")]
    assert ACTION_DAILY_METACOG_V1 not in last


def test_transient_outage_then_recovery_completes_same_night(tmp_path) -> None:
    store = SchedulerCursorStore(tmp_path / "cursors.json")

    def _outage_until_2130_local(now):
        if now < datetime(2026, 9, 30, 3, 30, tzinfo=timezone.utc):
            return DailyAttemptResult(completed=False, failure_reason=RPC_TIMEOUT)
        return DailyAttemptResult(completed=True)

    calls, gave_up, last = _simulate(
        store=store, gate=DailyAttemptGate(),
        start_utc=datetime(2026, 9, 30, 2, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 9, 30, 6, 30, tzinfo=timezone.utc),
        outcome=_outage_until_2130_local,
    )
    assert len(calls) == 3  # 20:15 fail, 21:15 fail, 22:15 success
    assert gave_up == []
    assert last[ACTION_DAILY_METACOG_V1] == "2026-09-29"


def test_unknown_failures_are_capped_at_six(tmp_path) -> None:
    store = SchedulerCursorStore(tmp_path / "cursors.json")
    calls, gave_up, _ = _simulate(
        store=store, gate=DailyAttemptGate(),
        start_utc=datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc),  # pulse at 08:30 local
        end_utc=datetime(2026, 10, 1, 5, 59, tzinfo=timezone.utc),
        outcome=_fail_with("cortex_exec_missing_final_text"),
        job=ACTION_DAILY_PULSE_V1, hour=8, minute=30,
    )
    assert len(calls) == MAX_UNKNOWN_ATTEMPTS
    assert [(g.attempts, g.failure_class) for g in gave_up] == [(6, "unknown")]


def test_dedupe_skip_does_not_spend_an_attempt(tmp_path) -> None:
    store = SchedulerCursorStore(tmp_path / "cursors.json")
    gate = DailyAttemptGate()
    calls, gave_up, _ = _simulate(
        store=store, gate=gate,
        start_utc=datetime(2026, 9, 30, 2, 14, tzinfo=timezone.utc),
        end_utc=datetime(2026, 9, 30, 2, 30, tzinfo=timezone.utc),
        outcome=lambda _now: DailyAttemptResult(completed=False, attempted=False),
    )
    assert len(calls) > MAX_DETERMINISTIC_ATTEMPTS
    assert gave_up == []
    assert gate.attempts(ACTION_DAILY_METACOG_V1, "2026-09-29") == 0


def test_attempt_counts_and_give_up_survive_restart(tmp_path) -> None:
    state = tmp_path / "daily_attempts.json"
    store = SchedulerCursorStore(tmp_path / "cursors.json")
    # Two deterministic failures, then the process restarts.
    calls1, _, _ = _simulate(
        store=store, gate=DailyAttemptGate(state_path=state),
        start_utc=datetime(2026, 9, 30, 2, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 9, 30, 2, 50, tzinfo=timezone.utc),
        outcome=_fail_with(OVER_LIMIT),
    )
    assert len(calls1) == 2
    reloaded = DailyAttemptGate(state_path=state)
    assert reloaded.attempts(ACTION_DAILY_METACOG_V1, "2026-09-29") == 2
    calls2, gave_up, _ = _simulate(
        store=store, gate=reloaded,
        start_utc=datetime(2026, 9, 30, 2, 50, tzinfo=timezone.utc),
        end_utc=datetime(2026, 9, 30, 5, 59, tzinfo=timezone.utc),
        outcome=_fail_with(OVER_LIMIT),
    )
    assert len(calls2) == 1  # the third and last try, not three fresh ones
    assert len(gave_up) == 1
    # A second restart after giving up: nothing more that date.
    calls3, _, _ = _simulate(
        store=store, gate=DailyAttemptGate(state_path=state),
        start_utc=datetime(2026, 9, 30, 4, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 9, 30, 5, 59, tzinfo=timezone.utc),
        outcome=_fail_with(OVER_LIMIT),
    )
    assert calls3 == []


# --- the real glue: lifespan -> _scheduled_daily_tick -> _execute_daily --------

def _failing_plan_payload(error: str) -> dict:
    return {"result": {"status": "fail", "final_text": None, "steps": [
        {"step_name": "draft_daily_metacog", "status": "fail", "error": error},
    ]}}


async def _run_real_ticks(tmp_path, *, step_error: str, ticks: list[float]):
    hunters: list = []

    class _NoopHunter:
        def __init__(self, *args, **kwargs) -> None:
            self.bus = AsyncMock()
            self.bus.codec = MagicMock()
            self.bus.codec.decode.return_value = SimpleNamespace(
                ok=True, error=None, envelope=SimpleNamespace(payload=_failing_plan_payload(step_error))
            )
            hunters.append(self)

        async def start(self) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass

    real_create_task = asyncio.create_task

    def _noop_create_task(coro, *args, **kwargs):
        coro.close()

        async def _idle() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass

        return real_create_task(_idle())

    notify_client = MagicMock()
    rpc = AsyncMock(return_value={"data": b"stub"})
    clock = {"now": 0.0}
    outcomes: list[str] = []
    with (
        patch.object(actions_main, "Hunter", _NoopHunter),
        patch("orion.notify.client.NotifyClient", return_value=notify_client),
        patch.object(actions_main.asyncio, "create_task", side_effect=_noop_create_task),
        patch.object(actions_main, "_rpc_request_with_retry", rpc),
        patch.object(actions_main.settings, "actions_scheduler_cursor_store_path", str(tmp_path / "cursors.json")),
        patch.object(actions_main.time, "time", lambda: clock["now"]),
    ):
        async with actions_main.lifespan(actions_main.app):
            tick = actions_main.app.state.scheduled_daily_tick
            for t in ticks:
                clock["now"] = t
                outcomes.append(await tick(
                    action_name=ACTION_DAILY_METACOG_V1,
                    due=True,
                    local_date="2026-09-29",
                    now_utc=datetime(2026, 9, 30, 2, 15, tzinfo=timezone.utc),
                    forced_date=None,
                ))
    audits = [
        c.args[1].payload for c in hunters[0].bus.publish.call_args_list
        if getattr(c.args[1], "kind", "") == "actions.audit.v1"
    ]
    return outcomes, rpc, notify_client, audits


def test_real_path_deterministic_failure_gives_up_after_three(tmp_path) -> None:
    t0 = 1_800_000_000.0
    ticks = [t0, t0 + 45, t0 + 601, t0 + 700, t0 + 601 + 1801, t0 + 9000]
    outcomes, rpc, notify_client, audits = asyncio.run(
        _run_real_ticks(tmp_path, step_error="LLMGatewayService: daily_metacog_prompt_over_limit chars=8467 limit=8192", ticks=ticks)
    )
    assert outcomes == ["failed", "backoff", "failed", "backoff", "gave_up", "gave_up"]
    assert rpc.await_count == 3
    gave_up = [a for a in audits if a.get("status") == "gave_up"]
    assert len(gave_up) == 1
    assert "daily_metacog_prompt_over_limit" in gave_up[0]["reason"]
    assert gave_up[0]["failure_class"] == "deterministic"
    assert not [a for a in audits if a.get("status") == "completed"]
    sent = [c.args[0] for c in notify_client.send.call_args_list]
    assert [r.event_kind for r in sent] == ["orion.daily.failed"]
    assert "daily_metacog_prompt_over_limit" in sent[0].body_text
    assert sent[0].context["report_date"] == "2026-09-28"


def test_real_path_transient_failure_is_retried_hourly(tmp_path) -> None:
    t0 = 1_800_000_000.0
    ticks = [t0, t0 + 601, t0 + 3601, t0 + 7202]
    outcomes, rpc, notify_client, _ = asyncio.run(
        _run_real_ticks(tmp_path, step_error="LLMGatewayService: RPC timeout waiting on orion:exec:result:X", ticks=ticks)
    )
    assert outcomes == ["failed", "backoff", "failed", "failed"]
    assert rpc.await_count == 3
    assert notify_client.send.call_count == 0


def _execute_daily_ast() -> ast.AsyncFunctionDef:
    tree = ast.parse(Path(inspect.getsourcefile(actions_main)).read_text())
    found = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_execute_daily"]
    assert len(found) == 1
    return found[0]


def test_every_execute_daily_exit_is_an_attempted_failure_unless_proven_otherwise() -> None:
    """A future early return must not be able to restore the 245/night storm."""
    fn = _execute_daily_ast()
    returns = sorted((n for n in ast.walk(fn) if isinstance(n, ast.Return)), key=lambda n: n.lineno)
    # Exactly two exits: the dedupe skip, and the single `return outcome`.
    assert len(returns) == 2, [ast.unparse(r) for r in returns]
    skip, final = returns
    assert ast.unparse(final) == "return outcome"
    assert "attempted=False" in ast.unparse(skip)
    # The skip lives under the dedupe-lock check and nowhere else.
    skip_parent = next(
        n for n in ast.walk(fn) if isinstance(n, ast.If) and skip in n.body
    )
    assert "deduper.try_acquire" in ast.unparse(skip_parent.test)
    # `outcome` starts as an attempted failure, and only a delivered report sets completed=True.
    assigns = sorted(
        (n for n in ast.walk(fn) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "outcome"),
        key=lambda n: n.lineno,
    )
    first = ast.unparse(assigns[0].value)
    assert "completed=False" in first and "attempted=True" in first
    attempted_false = [n for n in assigns if "attempted=False" in ast.unparse(n.value)]
    assert attempted_false == []
    successes = [n for n in assigns if "completed=True" in ast.unparse(n.value)]
    assert len(successes) == 1
    # No shared failure dict may come back.
    assert "daily_last_failure" not in ast.unparse(fn)


# --- other seams ----------------------------------------------------------

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
        failure_class="deterministic",
    )
    assert req.severity == "warning"
    assert req.event_kind == "orion.daily.failed"
    assert "daily_metacog_prompt_over_limit" in req.body_text
    assert req.context["attempts"] == 3
    assert req.context["failure_class"] == "deterministic"
    assert req.dedupe_key == "actions:daily_metacog_v1:failed:2026-09-29"


def test_metacog_gets_bounded_catalog_and_pulse_keeps_json() -> None:
    meta = _daily_skills_catalog_context(ACTION_DAILY_METACOG_V1)
    pulse = _daily_skills_catalog_context(ACTION_DAILY_PULSE_V1)
    assert pulse["skills_catalog_compact"].startswith("[")  # unchanged JSON for pulse
    assert not meta["skills_catalog_compact"].startswith("[")
    assert len(meta["skills_catalog_compact"]) < len(pulse["skills_catalog_compact"]) / 2
    assert meta["skills_catalog_count"] == len(meta["skills_catalog_compact"].splitlines())


@pytest.mark.parametrize("skill_id", ["skills.docker.compose_service_bringup.v1", "skills.imagination.render_scene.v1"])
@pytest.mark.parametrize("action,field", [
    (ACTION_DAILY_METACOG_V1, "tomorrow_experiment_skill_id"),
    (ACTION_DAILY_PULSE_V1, "focus_skill_id"),
])
def test_daily_selectors_reject_world_changing_skills(skill_id: str, action: str, field: str) -> None:
    parsed, invalid = _normalize_daily_skill_selection({field: skill_id}, action_name=action)
    assert parsed[field] is None
    assert invalid == [field]

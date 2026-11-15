"""The walkway journal night: a retry cap, a night that outlives midnight, a landed record.

Three live failures these pin (day letters of 2026-11-12 and 11-13):

  #3384  a failed compose was retried on every 45 s scheduler tick, each try a new graded act
         (59 failed walkway_day acts in 81 minutes on 11-13);
  #3385  a night still open at local midnight was dropped, and nothing recorded that it had
         no entry (11-12: no day or grade entry);
  #3386  is in orion/journaler/tests/test_code_built_title.py.

The pure tests drive run_walkway_tick / run_walkway_schedule with a real DailyAttemptGate on a
real file. The lifespan tests drive the real scheduler closure (app.state.walkway_journal_tick ->
_dispatch_journal_attempt -> _run_journal) with only the bus and the cortex RPC faked.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import app.walkway_forecast as wf
from app import main as actions_main
from app.daily_retry_gate import DAILY_ATTEMPT_STATE_FILENAME, DailyAttemptGate, DailyAttemptResult
from test_act_emit import _Harness, _grammar_events

TZ = "America/Denver"
# Thursday 2026-11-12 22:30 MST. Local midnight is 07:00Z on the 13th.
NIGHT = date(2026, 11, 12)
T0 = datetime(2026, 11, 13, 5, 30, tzinfo=timezone.utc)
TICK = timedelta(seconds=45)


def _job(kind: str, for_date: str = "2026-11-12") -> wf.WalkwayJournalJob:
    return wf.WalkwayJournalJob(
        trigger=wf.build_walkway_trigger(kind, {"for_date": for_date}, stream_id="walkway"),
        audit_action=f"journal.{kind}", dedupe_key=f"actions:journal:{kind}:walkway:{for_date}:athena",
    )


class _Night:
    """The scheduler's side of run_walkway_schedule: cursors, a clock, and what was dispatched."""

    def __init__(self, tmp_path, *, kinds=("walkway_day",), outcome=None, skips=()):
        self.path = tmp_path / DAILY_ATTEMPT_STATE_FILENAME
        self.gate = DailyAttemptGate(state_path=self.path)
        self.kinds, self.skips = kinds, list(skips)
        self.outcome = outcome or (lambda kind, now: DailyAttemptResult(False, failure_reason="RPC timeout"))
        self.cursors: dict[str, str] = {}
        self.now = T0
        self.dispatched: list[tuple[str, datetime]] = []
        self.collected_for: list[date] = []
        self.records: list[dict] = []
        self.closed: list[tuple[str, bool]] = []
        self.read_attempts: dict[str, int] = {}

    def restart(self) -> None:
        """A process restart: in-memory state is gone, the files are not."""
        self.gate = DailyAttemptGate(state_path=self.path)
        self.read_attempts = {}

    async def _collect(self, night):
        self.collected_for.append(night)
        return [_job(k, night.isoformat()) for k in self.kinds], list(self.skips)

    async def _dispatch(self, job):
        kind = job.trigger.trigger_kind
        self.dispatched.append((kind, self.now))
        return self.outcome(kind, self.now)

    async def _record(self, rec):
        self.records.append(rec)

    def _close(self, local_date, clean):
        self.cursors["walkway_journal"] = local_date
        self.closed.append((local_date, clean))

    def tick(self, hour_local=22, minute_local=30) -> str:
        return asyncio.run(wf.run_walkway_schedule(
            now_utc=self.now, tz_name=TZ, hour_local=hour_local, minute_local=minute_local,
            night_closed_on=self.cursors.get("walkway_journal"),
            collect=self._collect, dispatch=self._dispatch,
            job_done_on=self.cursors.get, mark_job_done=self.cursors.__setitem__,
            read_attempts=self.read_attempts, gate=self.gate, clock=self.now.timestamp,
            record=self._record, close_night=self._close,
        ))

    def run(self, ticks: int, *, restart_at: int | None = None, **kw) -> list[str]:
        words = []
        for i in range(ticks):
            if i == restart_at:
                self.restart()
            words.append(self.tick(**kw))
            self.now += TICK
        return words


# --- #3384: the retry cap ------------------------------------------------------------------------

def test_a_failing_job_is_tried_four_times_a_night_not_on_every_tick(tmp_path) -> None:
    night = _Night(tmp_path)
    words = night.run(120)  # 90 minutes of 45 s ticks, to the local midnight of the 13th
    assert len(night.dispatched) == wf.WALKWAY_ATTEMPT_POLICY.max_attempts == 4
    # Waits after failure 1, 2, 3 are 1, 5 and 15 minutes (to the next 45 s tick).
    gaps = [(b[1] - a[1]).total_seconds() for a, b in zip(night.dispatched, night.dispatched[1:])]
    assert gaps == [90.0, 315.0, 900.0]
    # The fourth failure closes the night; nothing runs after it.
    assert words.count("closed") == 1 and set(words[words.index("closed") + 1:]) == {"idle"}


def test_the_cap_survives_restarts(tmp_path) -> None:
    night = _Night(tmp_path)
    for start in range(0, 120, 3):  # a restart every third tick
        night.run(3, restart_at=0)
    assert len(night.dispatched) == 4
    assert json.loads(night.path.read_text())["walkway:walkway_day"]["gave_up"] is True


def test_each_job_has_its_own_cap(tmp_path) -> None:
    def outcome(kind, now):
        return DailyAttemptResult(kind == "walkway_grade", failure_reason="RPC timeout")

    night = _Night(tmp_path, kinds=("walkway_day", "walkway_grade", "walkway_forecast"), outcome=outcome)
    night.run(120)
    sent = [k for k, _ in night.dispatched]
    assert (sent.count("walkway_day"), sent.count("walkway_grade"), sent.count("walkway_forecast")) == (4, 1, 4)


def test_a_dispatch_that_did_no_work_spends_no_attempt(tmp_path) -> None:
    """A refusal (self-monitor gate, cooldown) opens no act, so it must not use up the four tries."""
    def outcome(kind, now):
        if now < T0 + 10 * TICK:
            return DailyAttemptResult(False, attempted=False, failure_reason="self_monitor_track_record_failing")
        return DailyAttemptResult(True)

    night = _Night(tmp_path, outcome=outcome)
    words = night.run(12)
    assert words[:10] == ["open"] * 10 and words[10] == "closed"
    assert night.cursors["walkway_day"] == "2026-11-12"
    assert night.records == [
        {"night": "2026-11-12", "kind": "walkway_day", "landed": True, "reason": "landed", "failed_attempts": 0},
        {"night": "2026-11-12", "kind": "walkway_grade", "landed": False, "reason": "no_job:none", "failed_attempts": 0},
        {"night": "2026-11-12", "kind": "walkway_forecast", "landed": False, "reason": "no_job:none", "failed_attempts": 0},
    ]


def test_the_wait_is_counted_from_when_the_compose_failed_not_when_it_began(tmp_path) -> None:
    """A compose can hang for the whole 420 s timeout. The clock is read after it."""
    gate = DailyAttemptGate(state_path=tmp_path / "a.json")
    clock = {"now": 1000.0}

    async def collect():
        return [_job("walkway_day")], []

    async def dispatch(job):
        clock["now"] += 420.0
        return DailyAttemptResult(False, failure_reason="RPC timeout")

    asyncio.run(wf.run_walkway_tick(
        collect=collect, dispatch=dispatch, job_done_on={}.get, mark_job_done=lambda k, v: None,
        local_date="2026-11-12", read_attempts={}, gate=gate, clock=lambda: clock["now"]))
    assert gate.may_attempt("walkway:walkway_day", "2026-11-12", 1420.0 + 59, wf.WALKWAY_ATTEMPT_POLICY) == (False, "backoff")
    assert gate.may_attempt("walkway:walkway_day", "2026-11-12", 1420.0 + 60, wf.WALKWAY_ATTEMPT_POLICY) == (True, "ok")


def test_the_daily_jobs_keep_their_own_policy(tmp_path) -> None:
    """The flat cap is opt-in: without a policy a transient failure is still retried hourly, uncapped."""
    gate = DailyAttemptGate(state_path=tmp_path / "a.json")
    for i in range(10):
        assert gate.record_failure("daily_metacog_v1", "d", i * 3600.0, "RPC timeout") is None
    assert gate.may_attempt("daily_metacog_v1", "d", 10 * 3600.0) == (True, "ok")


# --- #3385: a night that outlives local midnight --------------------------------------------------

def test_walkway_night_is_named_by_its_scheduled_date() -> None:
    def at(*hm):
        return wf.walkway_night(now_utc=datetime(2026, 11, 13, *hm, tzinfo=timezone.utc), tz_name=TZ,
                                hour_local=22, minute_local=30)

    assert at(5, 29) == (date(2026, 11, 11), timedelta(hours=23, minutes=59))  # 22:29 on the 12th
    assert at(5, 30) == (NIGHT, timedelta(0))
    assert at(7, 10) == (NIGHT, timedelta(hours=1, minutes=40))                # 00:10 on the 13th


def test_a_night_still_open_at_local_midnight_finishes_under_its_own_date(tmp_path) -> None:
    """11-12: the last try hung past midnight and the night was dropped. Scheduled 23:50 here so
    the fourth try (23:50 + 1 + 5 + 15 min) falls at 00:11 on the 13th."""
    midnight = datetime(2026, 11, 13, 7, 0, tzinfo=timezone.utc)

    def outcome(kind, now):
        return DailyAttemptResult(now > midnight, failure_reason="RPC timeout")

    night = _Night(tmp_path, kinds=("walkway_day", "walkway_grade"), outcome=outcome)
    night.now = datetime(2026, 11, 13, 6, 50, tzinfo=timezone.utc)
    words = night.run(60, hour_local=23, minute_local=50)
    assert "closed" in words
    landed_at = [t for _, t in night.dispatched][-1]
    assert landed_at > midnight
    # Read and stored for the 12th, never the 13th.
    assert set(night.collected_for) == {NIGHT}
    assert night.cursors == {"walkway_day": "2026-11-12", "walkway_grade": "2026-11-12", "walkway_journal": "2026-11-12"}
    assert [(r["kind"], r["landed"]) for r in night.records] == [
        ("walkway_day", True), ("walkway_grade", True), ("walkway_forecast", False)]
    assert night.closed == [("2026-11-12", True)]


def test_collect_reads_the_night_it_is_given_not_the_clock(monkeypatch) -> None:
    asked: list[tuple[str, dict]] = []

    async def fake(dsn, sql, params, *, label):
        asked.append((label, params))
        return []

    monkeypatch.setattr(wf, "fetch_rows", fake)
    asyncio.run(wf.collect_walkway_jobs(
        dsn="x", stream_id="walkway", tz_name=TZ, now_utc=datetime(2026, 11, 13, 7, 10, tzinfo=timezone.utc),
        min_support_days=5, node="athena", night=NIGHT))
    params = dict(asked)
    assert params["walkway_rollup_day"]["period_start"].date() == NIGHT
    assert params["walkway_grade"]["day_start"].date() == NIGHT
    assert params["walkway_forecast"]["day_kind"] == wf.day_kind_for(date(2026, 11, 13))


def test_a_night_that_runs_out_of_time_is_closed_with_a_no_for_each_job(tmp_path) -> None:
    """Nothing ran (the service was down, or every dispatch was refused) for longer than the
    night stays open. It is closed once, with a record, and tonight still runs."""
    night = _Night(tmp_path, kinds=("walkway_day",),
                   outcome=lambda k, n: DailyAttemptResult(False, attempted=False, failure_reason="journal_cooldown"))
    assert night.run(3) == ["open"] * 3
    night.now = T0 + wf.WALKWAY_NIGHT_MAX_OPEN + timedelta(seconds=1)
    before = len(night.dispatched)
    assert night.tick() == "expired"
    assert len(night.dispatched) == before
    assert [(r["kind"], r["landed"], r["reason"]) for r in night.records] == [
        ("walkway_day", False, "night_expired"),
        ("walkway_grade", False, "night_expired"),
        ("walkway_forecast", False, "night_expired"),
    ]
    assert night.closed == [("2026-11-12", False)] and night.tick() == "idle"
    night.now = T0 + timedelta(days=1)
    assert night.tick() == "open" and night.collected_for[-1] == date(2026, 11, 13)


# --- #3385: one landed yes/no record per night per job ---------------------------------------------

def test_one_record_per_job_says_landed_yes_or_no_and_why(tmp_path) -> None:
    def outcome(kind, now):
        return DailyAttemptResult(kind == "walkway_forecast", failure_reason="journal_compose_failed:exec_down")

    night = _Night(tmp_path, kinds=("walkway_day", "walkway_forecast"), outcome=outcome,
                   skips=["grade_no_expectations_today"])
    night.run(120)
    assert night.records == [
        {"night": "2026-11-12", "kind": "walkway_day", "landed": False, "failed_attempts": 4,
         "reason": "gave_up_after_4_attempts:journal_compose_failed:exec_down"},
        {"night": "2026-11-12", "kind": "walkway_grade", "landed": False, "failed_attempts": 0,
         "reason": "no_job:grade_no_expectations_today"},
        {"night": "2026-11-12", "kind": "walkway_forecast", "landed": True, "failed_attempts": 0,
         "reason": "landed"},
    ]
    assert night.closed == [("2026-11-12", False)]


def test_week_and_month_get_a_record_on_their_nights() -> None:
    assert wf.expected_walkway_kinds(date(2026, 11, 12)) == ["walkway_day", "walkway_grade", "walkway_forecast"]
    assert wf.expected_walkway_kinds(date(2026, 11, 15))[:2] == ["walkway_day", "walkway_week"]   # Sunday
    assert wf.expected_walkway_kinds(date(2026, 11, 30))[:2] == ["walkway_day", "walkway_month"]  # last night


# --- the real glue: lifespan -> _walkway_journal_tick -> _dispatch_journal_attempt -----------------

def _audits(hunter) -> list[dict]:
    return [c.args[1].payload for c in hunter.bus.publish.call_args_list
            if getattr(c.args[1], "kind", "") == "actions.audit.v1"]


def _real_night(tmp_path, harness, *, start: datetime, ticks: int, hour: int, minute: int, before_tick=None):
    async def _collect(**kw):
        return [_job("walkway_day", kw["night"].isoformat())], ["grade_no_expectations_today", "forecast_no_sightings"]

    clock = {"now": start}
    words: list[str] = []

    async def body(app):
        for _ in range(ticks):
            if before_tick is not None:
                before_tick(clock["now"])
            words.append(await app.state.walkway_journal_tick(now_utc=clock["now"]))
            clock["now"] += TICK

    with (
        patch.object(actions_main, "collect_walkway_jobs", _collect),
        patch.object(actions_main.settings, "actions_walkway_forecast_enabled", True),
        patch.object(actions_main.settings, "actions_walkway_forecast_hour_local", hour),
        patch.object(actions_main.settings, "actions_walkway_forecast_minute_local", minute),
        patch.object(actions_main.settings, "actions_daily_timezone", TZ),
        patch.object(actions_main.time, "time", lambda: clock["now"].timestamp()),
    ):
        events = asyncio.run(harness.run(tmp_path, body))
    return words, events, _audits(harness.hunters[0])


def test_real_path_forced_compose_failure_opens_at_most_four_acts_that_night(tmp_path) -> None:
    """The #3384 acceptance check: force a compose failure; at most 4 acts for that job that night."""
    h = _Harness(orch_payload={"ok": False, "error": "exec_down"})
    words, events, audits = _real_night(tmp_path, h, start=T0, ticks=120, hour=22, minute=30)
    opened = [e for e in events if e["event_kind"] == "act_opened"]
    closed = [e for e in events if e["event_kind"] == "act_closed"]
    assert len(opened) == len(closed) == 4
    assert {e["act"]["outcome"]["value"] for e in closed} == {0.0}
    assert len(h.sent) == 4  # four compose requests reached cortex, not 120
    assert words.count("closed") == 1 and words[-1] == "idle"

    records = [a for a in audits if a["action_name"] == wf.NIGHT_RECORD_ACTION]
    assert [(r["trigger_kind"], r["status"], r["landed"]) for r in records] == [
        ("walkway_day", "not_landed", False),
        ("walkway_grade", "not_landed", False),
        ("walkway_forecast", "not_landed", False),
    ]
    assert records[0]["reason"].startswith("gave_up_after_4_attempts:RuntimeError:journal_compose_failed:exec_down")
    assert records[0]["event_id"] == "walkway_night:walkway_day:walkway:2026-11-12:athena"
    assert records[1]["reason"] == "no_job:grade_no_expectations_today,forecast_no_sightings"
    # The same records are kept in a file beside the cursors, and the night is closed there too.
    lines = [json.loads(x) for x in (tmp_path / wf.NIGHT_RECORDS_FILENAME).read_text().splitlines()]
    assert [(x["kind"], x["landed"]) for x in lines] == [
        ("walkway_day", False), ("walkway_grade", False), ("walkway_forecast", False)]
    assert json.loads((tmp_path / "c.json").read_text())["walkway_journal"] == "2026-11-12"


def test_real_path_night_held_past_midnight_lands_and_is_recorded(tmp_path) -> None:
    """The #3385 acceptance check: cortex is down at 23:50 and back after midnight; the entry
    still lands for the 12th, and the yes/no record exists."""
    h = _Harness(orch_payload={"ok": True}, rpc_error=RuntimeError("RPC timeout waiting on orion:cortex:result"))
    midnight = datetime(2026, 11, 13, 7, 0, tzinfo=timezone.utc)

    def before_tick(now):
        if now > midnight:
            h.rpc_error = None

    words, events, audits = _real_night(
        tmp_path, h, start=datetime(2026, 11, 13, 6, 50, tzinfo=timezone.utc), ticks=60, hour=23, minute=50,
        before_tick=before_tick)
    closed = [e for e in events if e["event_kind"] == "act_closed"]
    assert [e["act"]["outcome"]["value"] for e in closed] == [0.0, 0.0, 0.0, 1.0]
    assert "closed" in words
    records = {a["trigger_kind"]: a for a in audits if a["action_name"] == wf.NIGHT_RECORD_ACTION}
    assert records["walkway_day"]["status"] == "landed" and records["walkway_day"]["night"] == "2026-11-12"
    assert json.loads((tmp_path / "c.json").read_text())["walkway_day"] == "2026-11-12"


def test_real_path_idle_before_the_scheduled_time_writes_nothing(tmp_path) -> None:
    """Last night is closed; before tonight's time the tick does nothing at all."""
    h = _Harness(orch_payload={"ok": True})
    (tmp_path / "c.json").write_text(json.dumps({"walkway_journal": "2026-11-11"}))
    words, events, audits = _real_night(
        tmp_path, h, start=datetime(2026, 11, 13, 3, 0, tzinfo=timezone.utc), ticks=3, hour=22, minute=30)
    assert words == ["idle"] * 3 and events == [] and audits == []

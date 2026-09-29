"""Every urgent run ends in a critical Hub + email report -- never silently.

Covers the composer (pure), delivery (Hub's own dedupe + backoff, since
orion-notify never enforces `dedupe_key`), the in-process watchdog (no GPU by
the grant wait, no terminal state by the overall deadline), the run-state
reader, and `_handle_run_state` routing urgent terminals to a report instead of
reach-out while ordinary runs keep their old path.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.durable_run import DurableRunStateV1
from orion.schemas.notify import NotificationAccepted
from scripts.curiosity_investigation import (
    URGENT_INCIDENTS_KEY,
    CuriosityInvestigation,
    urgent_open_key,
)
from scripts.urgent_report import (
    MISSED_DETAIL_RETRY_SEC,
    UrgentReporter,
    compose_urgent_report,
    read_urgent_run_progress,
    retry_delay,
    urgent_sent_key,
)

NOW = datetime(2026, 9, 28, 20, 0, tzinfo=timezone.utc)
SOURCE = ServiceRef(name="orion-hub", version="0.1.0", node="athena")
INCIDENT = "b2" * 16
RUN = "abcdef123456"
EVIDENCE = {"hosts": {"athena": {"measurements": {"temp_c_max": 88.0}}}, "pool": {"queued": 1}}
REPORT = {
    "incident_id": INCIDENT,
    "is_real": "real",
    "likely_cause": "radiator fan stalled",
    "evidence": ["athena temp_c_max 88.0", "fan rpm 0"],
    "severity": "critical",
    "operator_action": "power down athena and check the fan",
    "confidence": 0.82,
}


def _incident(**over) -> dict:
    base = {
        "incident_id": INCIDENT,
        "run_id": RUN,
        "question": "Why is athena at 88C?",
        "trigger": "manual",
        "subject": "athena",
        "requested_at": NOW.isoformat(),
        "requested_by": "juniper",
        "evidence": EVIDENCE,
        "status": "dispatched",
    }
    base.update(over)
    return base


def _detail(**over) -> dict:
    base = {
        "urgent": {"incident_id": INCIDENT, "trigger": "manual", "subject": "athena",
                   "question": "Why is athena at 88C?", "requested_at": NOW.isoformat()},
        "incident_report": dict(REPORT),
        "report_flag": None,
        "finding_text": "The fan on athena reads zero rpm while load is flat.",
    }
    base.update(over)
    return base


# --- composer -----------------------------------------------------------------


def _common(req, kind: str) -> None:
    assert req.severity == "critical"
    assert req.channels_requested == ["in_app", "email"]
    assert req.source_service == "orion-hub"
    assert req.event_kind == "curiosity.urgent.report"
    assert req.dedupe_key == f"urgent:{INCIDENT}:{kind}"
    assert req.correlation_id == RUN


def _order(body: str, *needles: str) -> None:
    positions = [body.index(n) for n in needles]
    assert positions == sorted(positions), list(zip(needles, positions))


def test_final_report_leads_with_the_verdict_in_the_documented_order() -> None:
    req = compose_urgent_report(_incident(), kind="final", detail=_detail())
    _common(req, "final")
    assert req.title == "URGENT: real / critical — athena"
    body = req.body_text
    _order(
        body,
        "Verdict: real / severity critical / confidence 0.82",
        "Operator action: power down athena and check the fan",
        "Likely cause: radiator fan stalled",
        "- fan rpm 0",
        "The fan on athena reads zero rpm",
        "Why is athena at 88C?",
    )
    assert "no_structured_verdict" not in body and "INCOMPLETE" not in body


def test_final_without_a_structured_verdict_flags_it_and_keeps_the_prose() -> None:
    req = compose_urgent_report(
        _incident(), kind="final", detail=_detail(incident_report=None, report_flag="no_structured_verdict")
    )
    _common(req, "final")
    assert req.title == "URGENT: no structured verdict — athena"
    body = req.body_text
    _order(body, "no_structured_verdict", "The fan on athena reads zero rpm", "Why is athena at 88C?")
    assert "Verdict:" not in body
    # The raw readings go along, since there is no cited evidence to lean on.
    assert '"temp_c_max": 88.0' in body


def test_failed_report_names_the_reason_and_attaches_the_bundle() -> None:
    req = compose_urgent_report(_incident(), kind="failed", reason="HarnessTurnFailed: empty_generation")
    _common(req, "failed")
    assert req.title.startswith("Urgent investigation failed")
    body = req.body_text
    _order(body, "investigation failed: HarnessTurnFailed: empty_generation", '"temp_c_max": 88.0',
           "Why is athena at 88C?")


def test_timeout_report_is_incomplete_with_evidence() -> None:
    req = compose_urgent_report(_incident(), kind="timeout", reason="no result after 1200 s")
    _common(req, "timeout")
    assert req.title == "URGENT: incomplete — athena"
    _order(req.body_text, "INCOMPLETE: no result after 1200 s", '"temp_c_max": 88.0', "Why is athena at 88C?")
    assert "final report" in req.body_text  # says the run keeps going


def test_no_gpu_report_says_not_investigated_with_trigger_question_and_bundle() -> None:
    req = compose_urgent_report(_incident(trigger="gpu_temp"), kind="no_gpu", reason="still waiting for a GPU")
    _common(req, "no_gpu")
    assert req.title == "URGENT: not investigated — athena"
    _order(req.body_text, "not investigated: still waiting for a GPU", '"temp_c_max": 88.0',
           "Trigger: gpu_temp", "Why is athena at 88C?")


def test_no_gpu_after_an_unconfirmed_dispatch_says_so() -> None:
    req = compose_urgent_report(
        _incident(status="dispatch_unconfirmed"), kind="no_gpu", reason="no run record found"
    )
    assert "dispatch was unconfirmed" in req.body_text.lower()


def test_title_falls_back_to_the_question_when_there_is_no_subject() -> None:
    long_q = "Q" * 100
    req = compose_urgent_report(_incident(subject=None, question=long_q), kind="timeout", reason="late")
    assert req.title == f"URGENT: incomplete — {'Q' * 60}"


def test_only_incident_and_detail_fields_reach_the_notice() -> None:
    detail = _detail(chat_history="SECRET-CHAT", memory_cards=["SECRET-MEMORY"], journal="SECRET-JOURNAL")
    incident = _incident(session_id="SECRET-SESSION", transcript="SECRET-TRANSCRIPT")
    for kind in ("final", "failed", "timeout", "no_gpu"):
        req = compose_urgent_report(incident, kind=kind, detail=detail, reason="r")
        dumped = req.model_dump_json()
        assert "SECRET" not in dumped, kind
        assert req.session_id is None


def test_evidence_bundle_is_capped() -> None:
    huge = {"hosts": {"athena": {"trend": ["x" * 50] * 2000}}}
    req = compose_urgent_report(_incident(evidence=huge), kind="failed", reason="r")
    assert len(req.body_text) < 12000
    assert "truncated" in req.body_text


# --- delivery -----------------------------------------------------------------


class _Redis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.hashes: dict[str, dict[str, str]] = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, *, nx=False, ex=None):
        if nx and key in self.values:
            return None
        self.values[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    async def delete(self, *keys):
        return sum(self.values.pop(k, None) is not None for k in keys)

    async def hget(self, key, field):
        value = self.hashes.get(key, {}).get(field)
        return value.encode() if isinstance(value, str) else value

    async def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value
        return 1

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def hlen(self, key):
        return len(self.hashes.get(key, {}))

    async def hdel(self, key, *fields):
        h = self.hashes.get(key, {})
        return sum(h.pop(f, None) is not None for f in fields)


class _Notify:
    def __init__(self, results=None) -> None:
        self.results = list(results or [])
        self.sent: list = []

    def send(self, request):
        self.sent.append(request)
        ok = self.results.pop(0) if self.results else True
        return NotificationAccepted(ok=ok, detail=None if ok else "notify down")


class _Clock:
    """Fake time: `sleep` advances `now`; gated delays wait until released."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []
        self.gates: dict[float, asyncio.Event] = {}

    def clock(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.slept.append(delay)
        gate = self.gates.get(delay)
        if gate is not None:
            await gate.wait()
        self.now += delay


SETTINGS = SimpleNamespace(HUB_CURIOSITY_URGENT_GRANT_WAIT_SEC=120.0, HUB_CURIOSITY_URGENT_TIMEOUT_SEC=1200.0)


def _reporter(*, notify=None, redis=None, reader=None, clock=None) -> UrgentReporter:
    clock = clock or _Clock()

    async def _none(run_id):
        return None

    return UrgentReporter(
        notify=notify or _Notify(), redis=redis if redis is not None else _Redis(), settings=SETTINGS,
        run_state_reader=reader or _none, sleep=clock.sleep, clock=clock.clock,
    )


async def _none_async():
    return None


def _seed_hash(redis: _Redis, **over) -> None:
    redis.hashes.setdefault(URGENT_INCIDENTS_KEY, {})[INCIDENT] = json.dumps(_incident(**over))


def test_retry_delays_double_then_hold_at_sixty() -> None:
    assert [retry_delay(n) for n in range(1, 10)] == [2, 4, 8, 16, 32, 60, 60, 60, 60]


def test_two_refusals_then_accept_delivers_once_and_marks_sent(caplog) -> None:
    redis, notify, clock = _Redis(), _Notify([False, False, True]), _Clock()
    _seed_hash(redis)
    reporter = _reporter(notify=notify, redis=redis, clock=clock)
    req = compose_urgent_report(_incident(), kind="final", detail=_detail())
    with caplog.at_level(logging.WARNING):
        ok = asyncio.run(reporter.deliver(_incident(), req, kind="final"))
    assert ok is True
    assert len(notify.sent) == 3 and all(r is req for r in notify.sent)
    assert clock.slept == [2, 4]
    assert f"urgent_report_retry incident_id={INCIDENT} kind=final attempt=1" in caplog.text
    assert redis.values[urgent_sent_key(INCIDENT, "final")] == RUN
    assert redis.ttls[urgent_sent_key(INCIDENT, "final")] == 7 * 24 * 3600
    assert json.loads(redis.hashes[URGENT_INCIDENTS_KEY][INCIDENT])["status"] == "reported_final"


def test_already_sent_is_not_sent_again() -> None:
    redis, notify = _Redis(), _Notify()
    redis.values[urgent_sent_key(INCIDENT, "final")] = RUN.encode()
    reporter = _reporter(notify=notify, redis=redis)
    req = compose_urgent_report(_incident(), kind="final", detail=_detail())
    assert asyncio.run(reporter.deliver(_incident(), req, kind="final")) is True
    assert notify.sent == []


def test_a_retried_run_for_the_same_incident_is_still_reported() -> None:
    # Dispatch raised for the first run (failed report sent, key released);
    # the retry's own failure must not be swallowed by the first one's sent key.
    redis, notify = _Redis(), _Notify()
    reporter = _reporter(notify=notify, redis=redis)
    first = _incident(run_id="111111111111")
    second = _incident(run_id="222222222222")
    asyncio.run(reporter.deliver(first, compose_urgent_report(first, kind="failed", reason="a"), kind="failed"))
    asyncio.run(reporter.deliver(second, compose_urgent_report(second, kind="failed", reason="b"), kind="failed"))
    assert [r.correlation_id for r in notify.sent] == ["111111111111", "222222222222"]


def test_a_retried_run_is_still_watched_after_an_earlier_run_failed() -> None:
    notify = _Notify()

    async def reader(run_id):
        return _progress(past=False)

    reporter = _reporter(notify=notify, reader=reader)
    first = _incident(run_id="111111111111")
    asyncio.run(reporter.deliver(first, compose_urgent_report(first, kind="failed", reason="a"), kind="failed"))
    _run_watch(reporter, _incident(run_id="222222222222"))
    assert any(r.dedupe_key.endswith(":no_gpu") and r.correlation_id == "222222222222" for r in notify.sent)


def test_duplicate_concurrent_deliveries_send_once() -> None:
    redis, notify = _Redis(), _Notify()
    reporter = _reporter(notify=notify, redis=redis)
    req = compose_urgent_report(_incident(), kind="final", detail=_detail())

    async def scenario():
        return await asyncio.gather(
            reporter.deliver(_incident(), req, kind="final"), reporter.deliver(_incident(), req, kind="final")
        )

    asyncio.run(scenario())
    assert len(notify.sent) == 1


def test_thirty_minutes_of_refusals_is_undelivered_and_says_so(caplog) -> None:
    redis, clock = _Redis(), _Clock()
    _seed_hash(redis)
    notify = _Notify([False] * 1000)
    reporter = _reporter(notify=notify, redis=redis, clock=clock)
    req = compose_urgent_report(_incident(), kind="failed", reason="boom")
    with caplog.at_level(logging.WARNING):
        ok = asyncio.run(reporter.deliver(_incident(), req, kind="failed"))
    assert ok is False
    assert clock.slept[:7] == [2, 4, 8, 16, 32, 60, 60]
    assert sum(clock.slept) <= 1800
    assert sum(clock.slept) + 60 > 1800  # kept trying until the window ran out
    assert any(r.levelno == logging.ERROR and "urgent_report_undelivered" in r.getMessage() for r in caplog.records)
    assert json.loads(redis.hashes[URGENT_INCIDENTS_KEY][INCIDENT])["status"] == "report_undelivered"
    assert urgent_sent_key(INCIDENT, "failed") not in redis.values


def test_a_late_watchdog_report_does_not_overwrite_a_terminal_status() -> None:
    redis = _Redis()
    _seed_hash(redis, status="reported_final")
    reporter = _reporter(redis=redis)
    req = compose_urgent_report(_incident(), kind="timeout", reason="late")
    asyncio.run(reporter.deliver(_incident(), req, kind="timeout"))
    assert json.loads(redis.hashes[URGENT_INCIDENTS_KEY][INCIDENT])["status"] == "reported_final"


def test_dispatch_failed_delivers_a_failed_report() -> None:
    redis, notify = _Redis(), _Notify()
    reporter = _reporter(notify=notify, redis=redis)

    async def scenario():
        await reporter.dispatch_failed(_incident(status="dispatch_failed"), "RuntimeError: boom")
        await asyncio.gather(*list(reporter._tasks))

    asyncio.run(scenario())
    [req] = notify.sent
    assert req.dedupe_key == f"urgent:{INCIDENT}:failed"
    assert "investigation failed: RuntimeError: boom" in req.body_text


def test_dispatch_failed_still_delivers_without_redis() -> None:
    """start_urgent reports a redis_unavailable refusal through this same reporter."""
    notify = _Notify()
    reporter = UrgentReporter(notify=notify, redis=None, settings=SETTINGS, run_state_reader=lambda r: None)

    async def scenario():
        await reporter.dispatch_failed(_incident(run_id="", status="refused:redis_unavailable"),
                                       "refused: redis_unavailable")
        await asyncio.gather(*list(reporter._tasks))

    asyncio.run(scenario())
    [req] = notify.sent
    assert "investigation failed: refused: redis_unavailable" in req.body_text


def test_main_wires_the_reporter_before_the_loop_starts_and_gives_it_the_release() -> None:
    """start() launches the run-state listener; a terminal it hears before the
    reporter exists would only log `urgent_reporter_missing`."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "scripts" / "main.py").read_text()
    wired = source.index("curiosity_investigation.urgent_reporter = UrgentReporter(")
    assert wired < source.index("await curiosity_investigation.start(bus")
    assert "release_open_key=curiosity_investigation.release_urgent_open_key_for" in source[wired:wired + 800]


# --- watchdog -----------------------------------------------------------------


def _progress(*, past=False, terminal=None, detail=None) -> dict:
    return {"past_resource_wait": past, "terminal": terminal, "detail": detail or {}}


def _run_watch(reporter: UrgentReporter, incident: dict, *, between=None) -> None:
    async def scenario():
        reporter.watch(incident)
        if between is not None:
            await between()
        for _ in range(50):
            if not reporter._tasks:
                break
            await asyncio.gather(*list(reporter._tasks))

    asyncio.run(scenario())


def test_run_still_waiting_for_a_gpu_at_the_grant_wait_sends_no_gpu() -> None:
    notify = _Notify()

    async def reader(run_id):
        return _progress(past=False)

    reporter = _reporter(notify=notify, reader=reader)
    _run_watch(reporter, _incident())
    kinds = [r.dedupe_key.rsplit(":", 1)[1] for r in notify.sent]
    assert "no_gpu" in kinds
    [no_gpu] = [r for r in notify.sent if r.dedupe_key.endswith(":no_gpu")]
    assert "not investigated" in no_gpu.body_text


def test_run_past_resource_wait_sends_no_no_gpu() -> None:
    notify = _Notify()

    async def reader(run_id):
        return _progress(past=True)

    _run_watch(_reporter(notify=notify, reader=reader), _incident())
    assert not any(r.dedupe_key.endswith(":no_gpu") for r in notify.sent)


def test_confirmed_run_missing_at_the_grant_wait_is_not_investigated() -> None:
    notify = _Notify()
    _run_watch(_reporter(notify=notify), _incident(status="dispatched"))
    [no_gpu] = [r for r in notify.sent if r.dedupe_key.endswith(":no_gpu")]
    assert "no run record found" in no_gpu.body_text


def _released_loop(held: bytes) -> tuple[CuriosityInvestigation, "_Redis"]:
    bus = _Bus()
    bus.redis.values[urgent_open_key(INCIDENT)] = held
    return _loop(bus), bus.redis


def test_unconfirmed_dispatch_with_no_run_record_is_a_failed_report_and_frees_the_incident() -> None:
    """Cortex rejected the kickoff and nothing registered: never 'still queued'
    or 'final report will follow' -- one failed report, then silence."""
    notify, clock = _Notify(), _Clock()
    loop, redis = _released_loop(RUN.encode())
    _seed_hash(redis, status="dispatch_unconfirmed")
    reporter = UrgentReporter(
        notify=notify, redis=redis, settings=SETTINGS, run_state_reader=lambda r: _none_async(),
        sleep=clock.sleep, clock=clock.clock, release_open_key=loop.release_urgent_open_key_for,
    )
    _run_watch(reporter, _incident(status="dispatch_unconfirmed"))
    [failed] = notify.sent
    assert failed.dedupe_key == f"urgent:{INCIDENT}:failed"
    assert "cortex never registered the run" in failed.body_text
    assert "still queued" not in failed.body_text and "final report" not in failed.body_text
    assert urgent_open_key(INCIDENT) not in redis.values
    assert json.loads(redis.hashes[URGENT_INCIDENTS_KEY][INCIDENT])["status"] == "reported_failed"
    assert 1200.0 in clock.slept  # the deadline check woke and stayed quiet


def test_the_unconfirmed_release_never_frees_another_runs_key() -> None:
    notify = _Notify()
    loop, redis = _released_loop(b"ffffff000000")
    reporter = UrgentReporter(
        notify=notify, redis=redis, settings=SETTINGS, run_state_reader=lambda r: _none_async(),
        sleep=_Clock().sleep, release_open_key=loop.release_urgent_open_key_for,
    )
    _run_watch(reporter, _incident(status="dispatch_unconfirmed"))
    assert redis.values[urgent_open_key(INCIDENT)] == b"ffffff000000"


def test_unconfirmed_dispatch_with_unreadable_state_still_says_not_investigated() -> None:
    notify = _Notify()

    async def reader(run_id):
        raise RuntimeError("no_postgres_pool")

    _run_watch(_reporter(notify=notify, reader=reader), _incident(status="dispatch_unconfirmed"))
    [no_gpu] = [r for r in notify.sent if r.dedupe_key.endswith(":no_gpu")]
    assert "dispatch was unconfirmed" in no_gpu.body_text.lower()


def test_unreadable_run_state_is_reported_rather_than_silent() -> None:
    notify = _Notify()

    async def reader(run_id):
        raise RuntimeError("no_postgres_pool")

    _run_watch(_reporter(notify=notify, reader=reader), _incident())
    [no_gpu] = [r for r in notify.sent if r.dedupe_key.endswith(":no_gpu")]
    assert "unreadable" in no_gpu.body_text


def test_no_terminal_state_by_the_deadline_sends_timeout() -> None:
    notify, clock = _Notify(), _Clock()

    async def reader(run_id):
        return _progress(past=True)

    _run_watch(_reporter(notify=notify, reader=reader, clock=clock), _incident())
    [timeout] = [r for r in notify.sent if r.dedupe_key.endswith(":timeout")]
    assert "INCOMPLETE" in timeout.body_text
    assert 120.0 in clock.slept and 1200.0 in clock.slept


def test_terminal_report_before_the_deadline_suppresses_timeout() -> None:
    notify, clock = _Notify(), _Clock()

    async def reader(run_id):
        return _progress(past=True)

    reporter = _reporter(notify=notify, reader=reader, clock=clock)

    async def scenario():
        gate = asyncio.Event()
        clock.gates[1200.0] = gate
        reporter.watch(_incident())
        await asyncio.sleep(0)
        req = compose_urgent_report(_incident(), kind="final", detail=_detail())
        await reporter.deliver(_incident(), req, kind="final")
        gate.set()
        await asyncio.gather(*list(reporter._tasks))

    asyncio.run(scenario())
    kinds = [r.dedupe_key.rsplit(":", 1)[1] for r in notify.sent]
    assert kinds == ["final"]


def test_terminal_in_the_run_store_but_missed_on_the_bus_still_reports() -> None:
    notify = _Notify()

    async def reader(run_id):
        return _progress(past=True, terminal="completed", detail=_detail())

    _run_watch(_reporter(notify=notify, reader=reader), _incident())
    kinds = [r.dedupe_key.rsplit(":", 1)[1] for r in notify.sent]
    assert kinds == ["final"]
    assert notify.sent[0].title == "URGENT: real / critical — athena"


def test_missed_completed_run_without_detail_is_reread_before_the_final() -> None:
    """A "no structured verdict" final now would dedupe-block the real one."""
    notify, clock = _Notify(), _Clock()
    loop, redis = _released_loop(RUN.encode())
    reads = iter([_progress(past=True), _progress(past=True, terminal="completed"),
                  _progress(past=True, terminal="completed", detail=_detail())])

    async def reader(run_id):
        return next(reads)

    reporter = UrgentReporter(
        notify=notify, redis=redis, settings=SETTINGS, run_state_reader=reader,
        sleep=clock.sleep, clock=clock.clock, release_open_key=loop.release_urgent_open_key_for,
    )
    _run_watch(reporter, _incident())
    [final] = notify.sent
    assert final.dedupe_key == f"urgent:{INCIDENT}:final"
    assert final.title == "URGENT: real / critical — athena"
    assert MISSED_DETAIL_RETRY_SEC in clock.slept
    assert urgent_open_key(INCIDENT) not in redis.values


def test_missed_completed_run_still_unreadable_reports_failed_not_an_empty_final(caplog) -> None:
    notify, clock = _Notify(), _Clock()
    loop, redis = _released_loop(RUN.encode())
    reads = iter([_progress(past=True)] + [_progress(past=True, terminal="completed")] * 2)

    async def reader(run_id):
        return next(reads)

    reporter = UrgentReporter(
        notify=notify, redis=redis, settings=SETTINGS, run_state_reader=reader,
        sleep=clock.sleep, clock=clock.clock, release_open_key=loop.release_urgent_open_key_for,
    )
    with caplog.at_level(logging.ERROR):
        _run_watch(reporter, _incident())
    [failed] = notify.sent
    assert failed.dedupe_key == f"urgent:{INCIDENT}:failed"
    assert failed.severity == "critical"
    assert "could not be read" in failed.body_text
    assert "urgent_report_missed_terminal_unreadable" in caplog.text
    assert urgent_open_key(INCIDENT) not in redis.values


def test_missed_failed_run_reports_and_frees_the_incident() -> None:
    notify = _Notify()
    loop, redis = _released_loop(RUN.encode())
    reads = iter([_progress(past=True), _progress(past=True, terminal="failed", detail={"error": "workflow_deadline"})])

    async def reader(run_id):
        return next(reads)

    reporter = UrgentReporter(
        notify=notify, redis=redis, settings=SETTINGS, run_state_reader=reader,
        sleep=_Clock().sleep, release_open_key=loop.release_urgent_open_key_for,
    )
    _run_watch(reporter, _incident())
    [failed] = notify.sent
    assert failed.dedupe_key == f"urgent:{INCIDENT}:failed"
    assert "workflow_deadline" in failed.body_text
    assert urgent_open_key(INCIDENT) not in redis.values


def test_close_cancels_the_pending_timers() -> None:
    clock = _Clock()
    reporter = _reporter(clock=clock)

    async def scenario():
        clock.gates[120.0] = asyncio.Event()  # never released
        reporter.watch(_incident())
        await asyncio.sleep(0)
        tasks = list(reporter._tasks)
        assert tasks
        await reporter.close()
        return tasks

    tasks = asyncio.run(scenario())
    assert all(task.cancelled() for task in tasks)
    assert reporter._tasks == set()


# --- run-state reader ---------------------------------------------------------


class _Conn:
    def __init__(self, *, admission=None, state=None, progressed=False, terminal_event=None) -> None:
        self.admission, self.state, self.progressed = admission, state, progressed
        self.terminal_event = terminal_event
        self.queries: list[str] = []
        self.entry_ids: list[str] = []

    async def fetchrow(self, sql, *args):
        self.queries.append(sql)
        if "durable_admission_runs" in sql:
            return self.admission
        if "substrate_durable_run_state" in sql:
            return self.state
        if "entry_id" in sql:
            self.entry_ids.append(args[0])
            return self.terminal_event
        if "durable_resource_events" in sql:
            return {"one": 1} if self.progressed else None
        raise AssertionError(sql)


class _Pool:
    def __init__(self, conn) -> None:
        self.conn = conn

    def acquire(self):
        conn = self.conn

        class _Ctx:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


def test_reader_no_rows_anywhere_is_none() -> None:
    assert asyncio.run(read_urgent_run_progress(_Pool(_Conn()), RUN)) is None


def test_reader_admitted_but_waiting_is_not_past_resource_wait() -> None:
    got = asyncio.run(read_urgent_run_progress(_Pool(_Conn(admission={"terminal": None, "control": None})), RUN))
    assert got == {"past_resource_wait": False, "terminal": None, "detail": {}}


def test_reader_granted_event_is_past_resource_wait() -> None:
    conn = _Conn(admission={"terminal": None, "control": None}, progressed=True)
    got = asyncio.run(read_urgent_run_progress(_Pool(conn), RUN))
    assert got["past_resource_wait"] is True and got["terminal"] is None


def test_reader_terminal_state_row_carries_its_detail() -> None:
    conn = _Conn(
        admission={"terminal": "completed", "control": None},
        state={"node": "finish", "status": "completed", "detail": json.dumps(_detail())},
    )
    got = asyncio.run(read_urgent_run_progress(_Pool(conn), RUN))
    assert got["terminal"] == "completed" and got["past_resource_wait"] is True
    assert got["detail"]["incident_report"]["is_real"] == "real"
    assert conn.entry_ids == []


def test_reader_terminal_without_a_state_row_takes_detail_from_the_terminal_event() -> None:
    conn = _Conn(
        admission={"terminal": "completed", "control": None},
        terminal_event={"detail": json.dumps(_detail())},
    )
    got = asyncio.run(read_urgent_run_progress(_Pool(conn), RUN))
    assert got["terminal"] == "completed"
    assert got["detail"]["incident_report"]["is_real"] == "real"
    assert conn.entry_ids == [f"{RUN}:terminal:completed"]


def test_reader_terminal_with_a_lagging_state_row_takes_detail_from_the_terminal_event() -> None:
    conn = _Conn(
        admission={"terminal": "failed", "control": None},
        state={"node": "admitted_turn", "status": "running", "detail": json.dumps({"stale": True})},
        terminal_event={"detail": {"error": "workflow_deadline"}},
    )
    got = asyncio.run(read_urgent_run_progress(_Pool(conn), RUN))
    assert got["terminal"] == "failed"
    assert got["detail"] == {"error": "workflow_deadline"}


def test_reader_without_a_pool_raises() -> None:
    try:
        asyncio.run(read_urgent_run_progress(None, RUN))
    except RuntimeError as exc:
        assert "no_postgres_pool" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


# --- _handle_run_state --------------------------------------------------------


class _Bus:
    def __init__(self) -> None:
        from orion.core.bus.codec import OrionCodec

        self.codec = OrionCodec()
        self.redis = _Redis()


class _FakeReporter:
    def __init__(self) -> None:
        self.delivered: list[tuple[dict, object, str]] = []

    async def deliver(self, incident, request, *, kind):
        self.delivered.append((incident, request, kind))
        return True


def _loop(bus) -> CuriosityInvestigation:
    loop = CuriosityInvestigation(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=14400.0, daily_cap=3,
        timeout_sec=8840.0, session_id="orion_curiosity", pool_provider=lambda: None, source_ref=SOURCE,
        kickoff_via_cortex=True, durable_admission_enabled=True, urgent_enabled=True,
    )
    loop._bus = bus
    loop.urgent_reporter = _FakeReporter()
    loop.reached_out = []

    async def _reach(**kwargs):
        loop.reached_out.append(kwargs)

    async def _no_help(run_id):
        return 0

    loop._maybe_reach_out = _reach  # type: ignore[assignment]
    loop._enqueue_help_requests_after_run = _no_help  # type: ignore[assignment]
    return loop


def _state_msg(bus, *, status, node, detail) -> dict:
    state = DurableRunStateV1(
        run_id=RUN, workflow="curiosity.investigate", thread_id=RUN, node=node, status=status, correlation_id="c",
        detail=detail,
    )
    env = BaseEnvelope(kind="durable.run.state.v1", source=SOURCE, payload=state.model_dump(mode="json"))
    return {"data": bus.codec.encode(env)}


def _handle(loop, msg) -> None:
    async def scenario():
        await loop._handle_run_state(msg)
        await asyncio.gather(*list(loop._urgent_report_tasks))

    asyncio.run(scenario())


def test_urgent_completed_sends_the_final_report_and_never_reaches_out() -> None:
    bus = _Bus()
    _seed_hash(bus.redis)
    bus.redis.values[urgent_open_key(INCIDENT)] = RUN.encode()
    loop = _loop(bus)
    _handle(loop, _state_msg(bus, status="completed", node="finish", detail=_detail(reach_out=True)))

    [(incident, req, kind)] = loop.urgent_reporter.delivered
    assert kind == "final" and req.title == "URGENT: real / critical — athena"
    assert incident["evidence"] == EVIDENCE  # the bundle came from the incident hash
    assert loop.reached_out == []
    assert urgent_open_key(INCIDENT) not in bus.redis.values


def test_urgent_failed_sends_the_failed_report_with_the_runner_reason() -> None:
    bus = _Bus()
    _seed_hash(bus.redis)
    bus.redis.values[urgent_open_key(INCIDENT)] = RUN.encode()
    loop = _loop(bus)
    detail = {"error": "gpu_pool_unavailable:no_lanes", "urgent": _detail()["urgent"]}
    _handle(loop, _state_msg(bus, status="failed", node="failed", detail=detail))

    [(_, req, kind)] = loop.urgent_reporter.delivered
    assert kind == "failed"
    assert "investigation failed: gpu_pool_unavailable:no_lanes" in req.body_text
    assert urgent_open_key(INCIDENT) not in bus.redis.values


def test_urgent_cancelled_is_reported_as_failed() -> None:
    bus = _Bus()
    _seed_hash(bus.redis)
    loop = _loop(bus)
    _handle(loop, _state_msg(bus, status="cancelled", node="finish",
                             detail={"error": "cancelled", "urgent": _detail()["urgent"]}))
    [(_, req, kind)] = loop.urgent_reporter.delivered
    assert kind == "failed" and "investigation failed: cancelled" in req.body_text


def test_urgent_non_terminal_failed_is_ignored() -> None:
    bus = _Bus()
    loop = _loop(bus)
    _handle(loop, _state_msg(bus, status="failed", node="harness_turn",
                             detail={"error": "x", "urgent": _detail()["urgent"]}))
    assert loop.urgent_reporter.delivered == []


def test_urgent_terminal_does_not_release_another_runs_open_key() -> None:
    bus = _Bus()
    _seed_hash(bus.redis)
    bus.redis.values[urgent_open_key(INCIDENT)] = b"ffffff000000"
    loop = _loop(bus)
    _handle(loop, _state_msg(bus, status="completed", node="finish", detail=_detail()))
    assert bus.redis.values[urgent_open_key(INCIDENT)] == b"ffffff000000"


def test_a_terminal_for_another_run_never_overwrites_the_incident_record() -> None:
    """A retry of the incident owns the record now; an older run's terminal reports from
    its own event and leaves that record exactly as it was."""
    bus = _Bus()
    _seed_hash(bus.redis, run_id="ffffff000000", status="dispatched")
    before = bus.redis.hashes[URGENT_INCIDENTS_KEY][INCIDENT]
    bus.redis.values[urgent_open_key(INCIDENT)] = b"ffffff000000"
    loop = _loop(bus)
    _handle(loop, _state_msg(bus, status="completed", node="finish", detail=_detail()))
    assert bus.redis.hashes[URGENT_INCIDENTS_KEY][INCIDENT] == before
    [(incident, req, kind)] = loop.urgent_reporter.delivered
    assert kind == "final" and incident["run_id"] == RUN and req.correlation_id == RUN
    assert incident["evidence"] is None  # the other run's bundle is not borrowed
    assert bus.redis.values[urgent_open_key(INCIDENT)] == b"ffffff000000"


def test_delivery_never_marks_another_runs_record() -> None:
    redis = _Redis()
    _seed_hash(redis, run_id="ffffff000000", status="dispatched")
    before = redis.hashes[URGENT_INCIDENTS_KEY][INCIDENT]
    reporter = _reporter(redis=redis)
    req = compose_urgent_report(_incident(), kind="final", detail=_detail())
    assert asyncio.run(reporter.deliver(_incident(), req, kind="final")) is True
    assert redis.hashes[URGENT_INCIDENTS_KEY][INCIDENT] == before


def test_urgent_terminal_without_a_stored_incident_still_reports() -> None:
    bus = _Bus()
    loop = _loop(bus)
    _handle(loop, _state_msg(bus, status="completed", node="finish", detail=_detail()))
    [(incident, req, kind)] = loop.urgent_reporter.delivered
    assert kind == "final" and incident["run_id"] == RUN and incident["incident_id"] == INCIDENT
    assert req.correlation_id == RUN


def test_ordinary_completed_keeps_the_reach_out_path_and_sends_no_report() -> None:
    bus = _Bus()
    loop = _loop(bus)
    detail = {"line": "investigate", "reach_out": True, "reach_out_why": "worth saying",
              "finding_text": "found a thing"}
    _handle(loop, _state_msg(bus, status="completed", node="finish", detail=detail))
    assert loop.urgent_reporter.delivered == []
    [call] = loop.reached_out
    assert call["run_id"] == RUN and call["finding_text"] == "found a thing"


def test_ordinary_failed_sends_nothing() -> None:
    bus = _Bus()
    loop = _loop(bus)
    _handle(loop, _state_msg(bus, status="failed", node="failed", detail={"error": "x"}))
    assert loop.urgent_reporter.delivered == [] and loop.reached_out == []

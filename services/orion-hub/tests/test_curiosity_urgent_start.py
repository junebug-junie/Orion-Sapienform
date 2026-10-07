"""Hub's urgent entry: one seeded curiosity run per incident, at urgent GPU priority.

Urgent runs bypass the curiosity gates (lock, cooldown, daily cap) and never
spend the daily budget. They refuse rather than run at background priority when
durable admission is off, and a failed dispatch is still handed to the reporter
so it is never silent.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.curiosity.urgent_prompt import build_urgent_prompt
from orion.schemas.curiosity_urgent import URGENT_REQUEST_CHANNEL, CuriosityUrgentSeedV1
from orion.schemas.durable_run import DurableRunRequestV1
from scripts.curiosity_investigation import (
    URGENT_INCIDENTS_KEY,
    URGENT_INCIDENTS_MAX,
    CuriosityInvestigation,
    urgent_open_key,
)
from scripts.curiosity_urgent import handle_urgent_request, urgent_request_loop

NOW = datetime(2026, 9, 28, 20, 0, tzinfo=timezone.utc)
SOURCE = ServiceRef(name="orion-hub", version="0.1.0", node="athena")
INCIDENT = "a1" * 16


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


class _Bus:
    def __init__(self, *, reply_status: str = "accepted", raise_on_rpc: bool = False) -> None:
        from orion.core.bus.codec import OrionCodec

        self.codec = OrionCodec()
        self.redis = _Redis()
        self.rpc_calls: list = []
        self.published: list = []
        self.reply_status = reply_status
        self.raise_on_rpc = raise_on_rpc

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec=60.0):
        self.rpc_calls.append((channel, envelope))
        if self.raise_on_rpc:
            raise RuntimeError("cortex down")
        out = BaseEnvelope(kind="cortex.orch.result", source=SOURCE, correlation_id=envelope.correlation_id,
                           payload={"status": self.reply_status})
        return {"channel": reply_channel, "data": self.codec.encode(out)}

    def dispatched(self) -> list[DurableRunRequestV1]:
        return [
            DurableRunRequestV1.model_validate(env.payload["context"]["metadata"]["durable_run"])
            for _, env in self.rpc_calls
        ]


class _Reporter:
    def __init__(self) -> None:
        self.watched: list[dict] = []
        self.failed: list[tuple[dict, str]] = []

    def watch(self, incident: dict) -> None:
        self.watched.append(incident)

    async def dispatch_failed(self, incident: dict, reason: str) -> None:
        self.failed.append((incident, reason))


def _loop(bus, **over) -> CuriosityInvestigation:
    kwargs = dict(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=14400.0, daily_cap=3,
        timeout_sec=8840.0, session_id="orion_curiosity",
        pool_provider=lambda: None, source_ref=SOURCE,
        kickoff_via_cortex=True, durable_admission_enabled=True,
        hub_url="http://host.docker.internal:8080", graph_own="orion_worldview",
        urgent_enabled=True, urgent_turn_timeout_sec=900.0, urgent_timeout_sec=1200.0,
    )
    kwargs.update(over)
    loop = CuriosityInvestigation(**kwargs)
    loop._bus = bus
    loop._harness_rpc_bus = bus
    loop.urgent_reporter = _Reporter()

    async def _no_record(*a, **k):  # the daily budget must never be spent by an urgent run
        loop.recorded = True

    loop.recorded = False
    loop._record_investigation = _no_record  # type: ignore[assignment]
    return loop


def _seed(incident_id: str = INCIDENT, **over) -> CuriosityUrgentSeedV1:
    base = dict(
        incident_id=incident_id, question="Why is athena at 88C?", trigger="manual",
        subject="athena", evidence={"hosts": {"athena": {"measurements": {"temp_c_max": 88.0}}}},
        requested_at=NOW, requested_by="juniper",
    )
    base.update(over)
    return CuriosityUrgentSeedV1(**base)


def test_urgent_dispatch_asks_for_urgent_priority_with_the_seed_and_urgent_prompt() -> None:
    bus = _Bus()
    loop = _loop(bus)
    result = asyncio.run(loop.start_urgent(_seed()))

    assert result["ok"] is True and result["incident_id"] == INCIDENT
    run_id = result["run_id"]
    [request] = bus.dispatched()
    assert request.run_id == run_id and request.workflow == "curiosity.investigate"
    assert request.admission is not None and request.admission.priority == "urgent"
    assert request.brief.urgent is not None and request.brief.urgent.incident_id == INCIDENT
    assert request.brief.timeout_sec == 900.0
    assert request.brief.line == "investigate"
    assert request.brief.prompt == build_urgent_prompt(
        _seed(), run_id=run_id, own_graph="orion_worldview", graph_enabled=False,
    )
    # The stance appraisal is the question itself.
    assert loop._mind_appraisal_by_run_id[run_id] == "Why is athena at 88C?"


def test_urgent_run_holds_one_open_key_and_records_the_incident() -> None:
    bus = _Bus()
    loop = _loop(bus)
    result = asyncio.run(loop.start_urgent(_seed()))

    assert bus.redis.values[urgent_open_key(INCIDENT)] == result["run_id"]
    # The admission deadline, two turn attempts and the 10 s retry past it, plus grace.
    assert bus.redis.ttls[urgent_open_key(INCIDENT)] == 1200 + 2 * 900 + 10 + 600
    stored = json.loads(bus.redis.hashes[URGENT_INCIDENTS_KEY][INCIDENT])
    assert stored["run_id"] == result["run_id"] and stored["status"] == "dispatched"
    assert stored["question"] == "Why is athena at 88C?" and stored["trigger"] == "manual"
    assert stored["subject"] == "athena" and stored["requested_at"] == NOW.isoformat()
    assert loop.urgent_reporter.watched == [stored]
    assert loop.urgent_reporter.failed == []


def test_urgent_run_is_not_blocked_by_the_run_lock_and_spends_no_budget() -> None:
    bus = _Bus()
    loop = _loop(bus)

    async def scenario():
        async with loop._run_lock:  # an ordinary curiosity turn is running
            return await asyncio.wait_for(loop.start_urgent(_seed()), timeout=5)

    result = asyncio.run(scenario())
    assert result["ok"] is True
    assert loop.recorded is False
    assert bus.redis.values.get("orion:curiosity:last_investigation_at") is None


def test_a_second_request_for_the_same_incident_is_refused() -> None:
    bus = _Bus()
    loop = _loop(bus)
    assert asyncio.run(loop.start_urgent(_seed()))["ok"] is True
    again = asyncio.run(loop.start_urgent(_seed()))
    assert again == {"ok": False, "reason": "incident_already_open", "incident_id": INCIDENT}
    assert len(bus.rpc_calls) == 1


def _assert_refusal_reported(loop, bus, reason: str) -> None:
    """A refusal is never silent: a failed report for the incident, and a record."""
    [(stub, why)] = loop.urgent_reporter.failed
    assert why == f"refused: {reason}"
    assert stub["incident_id"] == INCIDENT and stub["run_id"] == ""
    assert stub["status"] == f"refused:{reason}"
    assert stub["question"] == "Why is athena at 88C?" and stub["evidence"] == _seed().evidence
    assert loop.urgent_reporter.watched == []
    if bus.redis is not None:
        stored = json.loads(bus.redis.hashes[URGENT_INCIDENTS_KEY][INCIDENT])
        assert stored == stub


def test_admission_off_refuses_rather_than_running_at_background_priority() -> None:
    bus = _Bus()
    loop = _loop(bus, durable_admission_enabled=False)
    result = asyncio.run(loop.start_urgent(_seed()))
    assert result["ok"] is False and result["reason"] == "durable_admission_disabled"
    assert bus.rpc_calls == [] and bus.redis.values == {}
    _assert_refusal_reported(loop, bus, "durable_admission_disabled")


def test_urgent_disabled_refuses() -> None:
    bus = _Bus()
    loop = _loop(bus, urgent_enabled=False)
    result = asyncio.run(loop.start_urgent(_seed()))
    assert result["ok"] is False and result["reason"] == "urgent_disabled"
    assert bus.rpc_calls == []
    _assert_refusal_reported(loop, bus, "urgent_disabled")


def test_curiosity_loop_off_refuses_since_nothing_would_serve_the_turn() -> None:
    bus = _Bus()
    loop = _loop(bus, enabled=False)
    result = asyncio.run(loop.start_urgent(_seed()))
    assert result["ok"] is False and result["reason"] == "curiosity_disabled"
    assert bus.rpc_calls == [] and bus.redis.values == {}
    _assert_refusal_reported(loop, bus, "curiosity_disabled")


def test_no_redis_refuses_and_still_reports() -> None:
    bus = _Bus()
    bus.redis = None
    loop = _loop(bus)
    result = asyncio.run(loop.start_urgent(_seed()))
    assert result == {"ok": False, "reason": "redis_unavailable", "incident_id": INCIDENT}
    assert bus.rpc_calls == []
    _assert_refusal_reported(loop, bus, "redis_unavailable")


def test_a_refusal_while_a_run_is_open_neither_overwrites_its_record_nor_claims_failure() -> None:
    bus = _Bus()
    loop = _loop(bus)
    started = asyncio.run(loop.start_urgent(_seed()))
    loop.urgent_enabled = False
    assert asyncio.run(loop.start_urgent(_seed()))["reason"] == "urgent_disabled"
    stored = json.loads(bus.redis.hashes[URGENT_INCIDENTS_KEY][INCIDENT])
    assert stored["run_id"] == started["run_id"] and stored["status"] == "dispatched"
    # The open run still reports; a "failed" notice now would be false.
    assert loop.urgent_reporter.failed == []


def test_a_hung_role_check_is_bounded_and_never_strands_the_incident(monkeypatch) -> None:
    monkeypatch.setitem(CuriosityInvestigation.start_urgent.__globals__, "URGENT_PG_ROLE_CHECK_SEC", 0.05)
    bus = _Bus()
    loop = _loop(bus, pg_readonly_role="orion_readonly")

    async def _hang():
        await asyncio.sleep(3600)

    loop._pg_role_missing = _hang  # type: ignore[assignment]
    result = asyncio.run(asyncio.wait_for(loop.start_urgent(_seed()), timeout=5))
    assert result["ok"] is True
    assert bus.redis.values[urgent_open_key(INCIDENT)] == result["run_id"]
    assert len(bus.rpc_calls) == 1


def test_incident_already_open_sends_no_second_notice() -> None:
    bus = _Bus()
    loop = _loop(bus)
    asyncio.run(loop.start_urgent(_seed()))
    asyncio.run(loop.start_urgent(_seed()))
    assert loop.urgent_reporter.failed == []


def test_unconfirmed_dispatch_keeps_the_key_and_is_watched_not_failed() -> None:
    # No `accepted` from cortex does not mean nothing registered: the receipt
    # can be lost. The watchdog reports "not investigated" if no run appears.
    for bus in (_Bus(reply_status="rejected"), _Bus(raise_on_rpc=True)):
        loop = _loop(bus)
        result = asyncio.run(loop.start_urgent(_seed()))

        assert result["ok"] is True and result["unconfirmed"] is True
        assert result["incident_id"] == INCIDENT and result["run_id"]
        assert bus.redis.values[urgent_open_key(INCIDENT)] == result["run_id"]
        stored = json.loads(bus.redis.hashes[URGENT_INCIDENTS_KEY][INCIDENT])
        assert stored["status"] == "dispatch_unconfirmed"
        assert loop.urgent_reporter.watched == [stored]
        assert loop.urgent_reporter.failed == []
        # Still open, so a retry cannot start a second run for the same incident.
        assert asyncio.run(loop.start_urgent(_seed()))["reason"] == "incident_already_open"


def test_confirmed_dispatch_has_no_unconfirmed_flag() -> None:
    result = asyncio.run(_loop(_Bus()).start_urgent(_seed()))
    assert result == {"ok": True, "run_id": result["run_id"], "incident_id": INCIDENT}


def test_dispatch_exception_tells_the_reporter_and_releases_the_open_key() -> None:
    bus = _Bus()
    loop = _loop(bus)

    async def _boom(**kwargs):
        raise RuntimeError("brief would not build")

    loop._dispatch_durable_run = _boom  # type: ignore[assignment]
    result = asyncio.run(loop.start_urgent(_seed()))

    assert result["ok"] is False and result["reason"] == "dispatch_failed"
    assert urgent_open_key(INCIDENT) not in bus.redis.values
    [(incident, reason)] = loop.urgent_reporter.failed
    assert incident["incident_id"] == INCIDENT and incident["status"] == "dispatch_failed"
    assert reason == "RuntimeError: brief would not build"
    stored = json.loads(bus.redis.hashes[URGENT_INCIDENTS_KEY][INCIDENT])
    assert stored["status"] == "dispatch_failed"
    assert loop.urgent_reporter.watched == []
    # Released, so the incident can be retried.
    del loop._dispatch_durable_run
    assert asyncio.run(loop.start_urgent(_seed()))["ok"] is True


def test_missing_reporter_is_logged_and_the_run_still_starts(caplog) -> None:
    bus = _Bus()
    loop = _loop(bus)
    loop.urgent_reporter = None
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(loop.start_urgent(_seed()))
    assert result["ok"] is True
    assert f"urgent_reporter_missing incident_id={INCIDENT}" in caplog.text


def test_incident_hash_keeps_only_the_newest() -> None:
    bus = _Bus()
    loop = _loop(bus)
    for i in range(URGENT_INCIDENTS_MAX + 3):
        seed = _seed(incident_id=f"{i:032x}", requested_at=NOW.replace(minute=i % 60, second=i // 60))
        assert asyncio.run(loop.start_urgent(seed))["ok"] is True
        del bus.redis.values[urgent_open_key(seed.incident_id)]  # that run has ended
    kept = bus.redis.hashes[URGENT_INCIDENTS_KEY]
    assert len(kept) == URGENT_INCIDENTS_MAX
    assert f"{0:032x}" not in kept and f"{URGENT_INCIDENTS_MAX + 2:032x}" in kept


def _full_incident_hash(bus, oldest: dict[str, str]) -> None:
    rows = bus.redis.hashes.setdefault(URGENT_INCIDENTS_KEY, {})
    for i in range(URGENT_INCIDENTS_MAX - len(oldest)):
        rows[f"f{i:031x}"] = json.dumps({"requested_at": "2026-09-29T20:00:00+00:00"})
    for field, requested_at in oldest.items():
        rows[field] = json.dumps({"requested_at": requested_at})


def test_incident_eviction_orders_by_time_not_by_text() -> None:
    bus = _Bus()
    # 20:00+02:00 is 18:00Z, older than 19:00Z although it sorts later as text.
    offset, utc = "e1" * 16, "e2" * 16
    _full_incident_hash(bus, {offset: "2026-09-28T20:00:00+02:00", utc: "2026-09-28T19:00:00+00:00"})
    assert asyncio.run(_loop(bus).start_urgent(_seed()))["ok"] is True
    kept = bus.redis.hashes[URGENT_INCIDENTS_KEY]
    assert offset not in kept and utc in kept and INCIDENT in kept


def test_incident_eviction_never_drops_an_incident_whose_run_is_open() -> None:
    bus = _Bus()
    held = "e3" * 16
    _full_incident_hash(bus, {held: "2026-09-27T00:00:00"})
    bus.redis.values[urgent_open_key(held)] = "0123456789ab"
    assert asyncio.run(_loop(bus).start_urgent(_seed()))["ok"] is True
    assert held in bus.redis.hashes[URGENT_INCIDENTS_KEY]


class _RoleMissingPool:
    def acquire(self):
        class _Ctx:
            async def __aenter__(self_inner):
                class _Conn:
                    async def fetchval(self, sql, *args):
                        return None  # pg_roles has no such role

                return _Conn()

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


def test_urgent_prompt_offers_no_psql_when_the_readonly_role_is_missing() -> None:
    bus = _Bus()
    loop = _loop(bus, pool_provider=lambda: _RoleMissingPool(), pg_readonly_role="orion_readonly")
    result = asyncio.run(loop.start_urgent(_seed()))
    assert result["ok"] is True
    [request] = bus.dispatched()
    assert "psql" not in request.brief.prompt
    assert "no Postgres history this run" in request.brief.prompt


def test_stop_cancels_urgent_deliveries_and_closes_the_reporter() -> None:
    bus = _Bus()
    loop = _loop(bus)
    closed: list[bool] = []

    async def _close():
        closed.append(True)

    loop.urgent_reporter.close = _close

    async def scenario():
        delivery = asyncio.ensure_future(asyncio.sleep(3600))
        loop._urgent_report_tasks.add(delivery)
        await loop.stop()
        return delivery

    delivery = asyncio.run(scenario())
    assert delivery.cancelled()
    assert loop._urgent_report_tasks == set()
    assert closed == [True]


def test_ordinary_dispatch_keeps_background_priority() -> None:
    from orion.curiosity.study_material import StudyMaterial

    bus = _Bus()
    loop = _loop(bus)
    ok = asyncio.run(loop._dispatch_durable_run(
        run_id="abcdef123456", correlation_id="c", prompt="p", material=StudyMaterial(generated_at=NOW),
    ))
    assert ok is True
    [request] = bus.dispatched()
    assert request.admission.priority == "background"
    assert request.brief.urgent is None and request.brief.timeout_sec == 8840.0
    assert request.admission.deadline_at is None
    durable = bus.rpc_calls[0][1].payload["context"]["metadata"]["durable_run"]
    assert "urgent" not in durable["brief"]
    assert "deadline_at" not in durable["admission"]


def test_urgent_admission_carries_the_overall_urgent_deadline() -> None:
    """durable-runs fails a run still queued or running at deadline_at, so an urgent
    run cannot outlive its report window."""
    bus = _Bus()
    before = datetime.now(timezone.utc)
    asyncio.run(_loop(bus, urgent_timeout_sec=1200.0).start_urgent(_seed()))
    after = datetime.now(timezone.utc)
    [request] = bus.dispatched()
    deadline = request.admission.deadline_at
    assert deadline is not None and deadline.tzinfo is not None
    assert before.timestamp() + 1200 <= deadline.timestamp() <= after.timestamp() + 1200


# --- the bus consumer --------------------------------------------------------


class _FakeInvestigation:
    def __init__(self) -> None:
        self.seeds: list = []
        self.urgent_reporter = None

    async def start_urgent(self, seed):
        self.seeds.append(seed)
        return {"ok": True, "run_id": "abcdef123456", "incident_id": seed.incident_id}


def _msg(bus, payload) -> dict:
    env = BaseEnvelope(kind="curiosity.urgent.request.v1", source=SOURCE, payload=payload)
    return {"data": bus.codec.encode(env)}


def test_a_valid_bus_request_starts_an_urgent_run() -> None:
    bus = _Bus()
    inv = _FakeInvestigation()
    payload = _seed().model_dump(mode="json")
    asyncio.run(handle_urgent_request(bus, inv, _msg(bus, payload)))
    [seed] = inv.seeds
    assert type(seed) is CuriosityUrgentSeedV1
    assert seed.incident_id == INCIDENT and seed.question == "Why is athena at 88C?"


def test_an_invalid_bus_request_is_dropped_without_raising(caplog) -> None:
    bus = _Bus()
    inv = _FakeInvestigation()
    with caplog.at_level(logging.WARNING):
        asyncio.run(handle_urgent_request(bus, inv, _msg(bus, {"question": ""})))
        asyncio.run(handle_urgent_request(bus, inv, {"data": b"not an envelope"}))
    assert inv.seeds == []
    assert "urgent_request_invalid" in caplog.text


def test_an_invalid_request_naming_an_incident_gets_a_failed_report() -> None:
    bus = _Bus()
    inv = _FakeInvestigation()
    inv.urgent_reporter = _Reporter()
    payload = {**_seed().model_dump(mode="json"), "trigger": "smoke", "question": "Q" * 5000}
    asyncio.run(handle_urgent_request(bus, inv, _msg(bus, payload)))
    assert inv.seeds == []
    [(stub, reason)] = inv.urgent_reporter.failed
    assert stub["incident_id"] == INCIDENT and stub["run_id"] == ""
    assert stub["status"] == "refused:invalid_request"
    assert stub["trigger"] == "smoke" and stub["subject"] == "athena"
    assert len(stub["question"]) == 2000
    assert reason.startswith("refused: invalid_request: ")


def test_an_invalid_request_for_an_incident_with_an_open_run_claims_no_failure() -> None:
    bus = _Bus()
    inv = _FakeInvestigation()
    inv.urgent_reporter = _Reporter()
    inv._bus = bus
    bus.redis.values[urgent_open_key(INCIDENT)] = "abcdef123456"
    payload = {**_seed().model_dump(mode="json"), "trigger": "smoke"}
    asyncio.run(handle_urgent_request(bus, inv, _msg(bus, payload)))
    assert inv.seeds == [] and inv.urgent_reporter.failed == []


def test_an_invalid_request_without_a_usable_incident_id_is_only_logged() -> None:
    bus = _Bus()
    inv = _FakeInvestigation()
    inv.urgent_reporter = _Reporter()
    for payload in ({"question": ""}, {"incident_id": "NOT-HEX", "question": "x"}, {"incident_id": 12345}):
        asyncio.run(handle_urgent_request(bus, inv, _msg(bus, payload)))
    assert inv.urgent_reporter.failed == []


class _SubBus(_Bus):
    def __init__(self, messages) -> None:
        super().__init__()
        self.messages = messages
        self.channels: list = []

    def subscribe(self, channel):
        self.channels.append(channel)
        bus = self

        class _Ctx:
            async def __aenter__(self_inner):
                return object()

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()

    async def iter_messages(self, pubsub):
        for msg in self.messages:
            yield msg
        await asyncio.Event().wait()  # stay subscribed until cancelled


def test_the_loop_subscribes_to_the_urgent_channel_and_serves_requests() -> None:
    inv = _FakeInvestigation()
    probe = _Bus()
    bus = _SubBus([_msg(probe, {"bad": 1}), _msg(probe, _seed().model_dump(mode="json"))])

    async def scenario():
        task = asyncio.create_task(urgent_request_loop(bus, inv))
        for _ in range(100):
            if inv.seeds:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
    assert bus.channels == [URGENT_REQUEST_CHANNEL]
    assert [s.incident_id for s in inv.seeds] == [INCIDENT]

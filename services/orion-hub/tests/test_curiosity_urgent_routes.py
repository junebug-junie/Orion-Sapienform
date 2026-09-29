"""`POST /curiosity/api/urgent` and `GET /curiosity/api/urgent` -- the Hub
"Run urgent" button and the urgent-runs list (Plan 3 Task 7).

The POST publishes a `CuriosityUrgentRequestV1` on the same bus channel the
hardware watcher uses, so there is one path into `start_urgent`. The round-trip
test feeds the published envelope through the real consumer
(`scripts.curiosity_urgent.handle_urgent_request`) to prove it decodes.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

import scripts.curiosity_routes as cr
from orion.core.bus.codec import OrionCodec
from orion.schemas.curiosity_urgent import URGENT_REQUEST_CHANNEL, URGENT_REQUEST_KIND
from scripts.curiosity_investigation import URGENT_INCIDENTS_KEY
from scripts.curiosity_urgent import handle_urgent_request


class _Redis:
    def __init__(self, rows=None, fail=False):
        self.rows = rows or {}
        self.fail = fail

    async def hgetall(self, key):
        if self.fail:
            raise RuntimeError("redis down")
        assert key == URGENT_INCIDENTS_KEY
        return dict(self.rows)


class _Bus:
    def __init__(self, *, enabled=True, redis=None, fail_publish=False):
        self.enabled = enabled
        self.redis = redis if redis is not None else _Redis()
        self.codec = OrionCodec()
        self.fail_publish = fail_publish
        self.published: list[tuple[str, object]] = []

    async def publish(self, channel, env):
        if self.fail_publish:
            raise RuntimeError("bus down")
        self.published.append((channel, env))


class _Task:
    def __init__(self, done=False):
        self._done = done

    def done(self):
        return self._done


class _Loop:
    def __init__(self, *, urgent_enabled=True, listener=None, durable_admission_enabled=True):
        self.urgent_enabled = urgent_enabled
        self.durable_admission_enabled = durable_admission_enabled
        self.urgent_listener_task = listener if listener is not None else _Task()
        self.seeds = []

    async def start_urgent(self, seed):
        self.seeds.append(seed)
        return {"ok": True, "run_id": "abc123", "incident_id": seed.incident_id}


_EVIDENCE = {"cooling": {"power_w": 410.0}, "collected_at": "2026-09-28T21:00:00+00:00"}


def _install(monkeypatch, *, loop=None, bus=None, evidence=None):
    fake_main = types.SimpleNamespace(curiosity_investigation=loop, bus=bus)
    monkeypatch.setitem(sys.modules, "scripts.main", fake_main)
    import scripts as scripts_pkg

    monkeypatch.setattr(scripts_pkg, "main", fake_main, raising=False)
    calls = []

    async def _fake_collect():
        calls.append(1)
        return dict(evidence if evidence is not None else _EVIDENCE)

    monkeypatch.setattr(cr, "_collect_urgent_evidence", _fake_collect)
    return calls


def _post(question):
    resp = asyncio.run(cr.curiosity_urgent_api({"question": question}))
    return resp.status_code, json.loads(resp.body)


# --- POST -------------------------------------------------------------------


def test_valid_question_publishes_one_manual_request_and_returns_its_incident_id(monkeypatch) -> None:
    bus, loop = _Bus(), _Loop()
    calls = _install(monkeypatch, loop=loop, bus=bus)
    status, body = _post("  why is circe hot?  ")
    assert status == 200
    assert body["ok"] is True
    incident_id = body["incident_id"]
    assert len(incident_id) == 32 and int(incident_id, 16) >= 0
    assert calls == [1]
    assert len(bus.published) == 1
    channel, env = bus.published[0]
    assert channel == URGENT_REQUEST_CHANNEL
    assert env.kind == URGENT_REQUEST_KIND
    assert env.correlation_id.hex == incident_id
    payload = env.payload
    assert payload["incident_id"] == incident_id
    assert payload["question"] == "why is circe hot?"
    assert payload["trigger"] == "manual"
    assert payload["subject"] == ""
    assert payload["requested_by"] == "juniper"
    assert payload["evidence"] == _EVIDENCE
    assert loop.seeds == [], "the route publishes; it never calls start_urgent itself"


def test_the_published_envelope_is_decoded_by_the_real_consumer(monkeypatch) -> None:
    bus, loop = _Bus(), _Loop()
    _install(monkeypatch, loop=loop, bus=bus)
    status, body = _post("is the AC keeping up?")
    assert status == 200
    _, env = bus.published[0]
    asyncio.run(handle_urgent_request(bus, loop, {"data": bus.codec.encode(env)}))
    assert len(loop.seeds) == 1
    seed = loop.seeds[0]
    assert seed.incident_id == body["incident_id"]
    assert seed.trigger == "manual" and seed.question == "is the AC keeping up?"
    assert seed.evidence == _EVIDENCE and seed.requested_by == "juniper"


def test_empty_question_is_refused(monkeypatch) -> None:
    bus = _Bus()
    calls = _install(monkeypatch, loop=_Loop(), bus=bus)
    for q in ("", "   ", None):
        status, body = _post(q)
        assert status == 400
        assert body == {"ok": False, "reason": "question_required"}
    assert bus.published == [] and calls == []


def test_question_over_2000_chars_is_refused(monkeypatch) -> None:
    bus = _Bus()
    _install(monkeypatch, loop=_Loop(), bus=bus)
    status, body = _post("x" * 2001)
    assert status == 400
    assert body == {"ok": False, "reason": "question_too_long"}
    assert bus.published == []
    status, _ = _post("x" * 2000)
    assert status == 200


def test_no_loop_is_503(monkeypatch) -> None:
    bus = _Bus()
    calls = _install(monkeypatch, loop=None, bus=bus)
    status, body = _post("why?")
    assert status == 503
    assert body == {"ok": False, "reason": "loop_not_running"}
    assert bus.published == [] and calls == []


def test_nothing_listening_is_refused_rather_than_published_into_the_void(monkeypatch) -> None:
    """A request with no consumer would read as 'incident started' and then never run."""
    cases = [
        (_Loop(urgent_enabled=False), "urgent_disabled"),
        (_Loop(listener=_Task(done=True)), "urgent_listener_not_running"),
    ]
    none_listener = _Loop()
    none_listener.urgent_listener_task = None
    cases.append((none_listener, "urgent_listener_not_running"))
    for loop, reason in cases:
        bus = _Bus()
        _install(monkeypatch, loop=loop, bus=bus)
        status, body = _post("why?")
        assert status == 503, reason
        assert body == {"ok": False, "reason": reason}
        assert bus.published == []


def test_no_bus_or_disabled_bus_is_503(monkeypatch) -> None:
    for bus in (None, _Bus(enabled=False)):
        _install(monkeypatch, loop=_Loop(), bus=bus)
        status, body = _post("why?")
        assert status == 503
        assert body == {"ok": False, "reason": "bus_unavailable"}


def test_admission_off_is_503_not_a_published_request_that_start_urgent_refuses(monkeypatch) -> None:
    bus = _Bus()
    calls = _install(monkeypatch, loop=_Loop(durable_admission_enabled=False), bus=bus)
    status, body = _post("why?")
    assert status == 503
    assert body == {"ok": False, "reason": "durable_admission_disabled"}
    assert bus.published == [] and calls == []


def test_bus_without_redis_is_503(monkeypatch) -> None:
    bus = _Bus()
    bus.redis = None
    calls = _install(monkeypatch, loop=_Loop(), bus=bus)
    status, body = _post("why?")
    assert status == 503
    assert body == {"ok": False, "reason": "redis_unavailable"}
    assert bus.published == [] and calls == []


def test_publish_failure_is_ok_false_not_a_crash(monkeypatch) -> None:
    _install(monkeypatch, loop=_Loop(), bus=_Bus(fail_publish=True))
    status, body = _post("why?")
    assert status == 500
    assert body["ok"] is False and "bus down" in body["reason"]


# --- GET --------------------------------------------------------------------


def _row(incident_id, requested_at, **over):
    row = {
        "incident_id": incident_id, "run_id": f"r{incident_id[:6]}", "question": f"q {incident_id}",
        "trigger": "manual", "subject": "", "requested_at": requested_at, "requested_by": "juniper",
        "status": "dispatched", "evidence": {"cooling": {"power_w": 1.0}, "big": "x" * 5000},
    }
    row.update(over)
    return json.dumps(row)


def test_list_is_newest_first_without_the_evidence_blob(monkeypatch) -> None:
    rows = {
        b"a" * 32: _row("a" * 32, "2026-09-28T20:00:00+00:00").encode(),
        "b" * 32: _row("b" * 32, "2026-09-28T21:30:00+00:00"),
        "c" * 32: _row("c" * 32, "2026-09-28T19:00:00-02:00", status="reported_final"),  # 21:00 UTC
        "bad": "not json",
    }
    _install(monkeypatch, loop=_Loop(), bus=_Bus(redis=_Redis(rows)))
    resp = asyncio.run(cr.curiosity_urgent_list_api())
    body = json.loads(resp.body)
    assert resp.status_code == 200
    assert body["available"] is True
    ids = [i["incident_id"] for i in body["incidents"]]
    assert ids == ["b" * 32, "c" * 32, "a" * 32]
    for item in body["incidents"]:
        assert "evidence" not in item
    assert body["incidents"][1]["status"] == "reported_final"
    assert body["incidents"][0]["question"] == "q " + "b" * 32
    assert len(resp.body) < 4000


def test_list_caps_at_twenty(monkeypatch) -> None:
    rows = {f"{i:032x}": _row(f"{i:032x}", f"2026-09-28T{i:02d}:00:00+00:00") for i in range(24)}
    _install(monkeypatch, loop=_Loop(), bus=_Bus(redis=_Redis(rows)))
    body = json.loads(asyncio.run(cr.curiosity_urgent_list_api()).body)
    assert len(body["incidents"]) == 20
    assert body["incidents"][0]["incident_id"] == f"{23:032x}"


def test_list_never_500s(monkeypatch) -> None:
    _install(monkeypatch, loop=_Loop(), bus=None)
    body = json.loads(asyncio.run(cr.curiosity_urgent_list_api()).body)
    assert body == {"available": False, "reason": "redis_unavailable", "incidents": []}
    _install(monkeypatch, loop=_Loop(), bus=_Bus(redis=_Redis(fail=True)))
    resp = asyncio.run(cr.curiosity_urgent_list_api())
    body = json.loads(resp.body)
    assert resp.status_code == 200
    assert body["available"] is False and "redis down" in body["reason"] and body["incidents"] == []

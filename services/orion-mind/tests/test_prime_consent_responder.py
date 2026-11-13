"""orion-mind's answer to a consent ask (app/prime_consent.py): every path answers, only a parsed model
answer is Orion's, the act opens only when the model was reached, and the self-monitor gate on
act.prime.infer.consent can refuse."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.prime_consent import GATE_PAUSED_REASON, PrimeConsentResponder
from orion.claude_tools.transport import CONSENT_REQUEST_KIND, CONSENT_RESULT_KIND, CONSENT_RESULT_PREFIX
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.grammar.publish import GRAMMAR_EVENT_CHANNEL
from orion.mindloop import gate as mindloop_gate
from orion.schemas.claude_tools import PrimeConsentAnswerV1, PrimeConsentRequestV1
from orion.self_monitor.view import self_monitor_view

T0 = datetime(2026, 11, 13, 4, 0, tzinfo=timezone.utc)
SETTINGS = SimpleNamespace(
    SERVICE_NAME="orion-mind", SERVICE_VERSION="t", NODE_NAME="athena",
    PRIME_CONSENT_LLM_ROUTE="prime_consent", PRIME_CONSENT_SEAT_WAIT_SEC=15.0, PRIME_CONSENT_LLM_TIMEOUT_SEC=20.0,
    PRIME_CONSENT_MAX_TOKENS=200, SELF_MONITOR_ACT_GATE_ENABLED=False, SELF_MONITOR_PROBE_INTERVAL_SEC=21600.0,
    SELF_MONITOR_LEDGER_PG_DSN="", ORION_BUS_URL="redis://x:6379/0",
)


class Redis:
    async def get(self, key):
        return None


class Bus:
    def __init__(self):
        self.redis = Redis()
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


class LLM:
    def __init__(self, parsed=None, err=None, meta=None):
        self.parsed, self.err, self.meta = parsed, err, meta or {}
        self.calls = []

    def request_json(self, **kw):
        self.calls.append(kw)
        return self.parsed, self.err, self.meta


def _req() -> PrimeConsentRequestV1:
    return PrimeConsentRequestV1(call_id="call-00000001", turn_request_id="room-claude-req-1", room_id="hub-room",
                                 claude_session_id="sess-a", tool="body_economy", category="body",
                                 tool_summary="your body economy", reason="to check a claim")


def _responder(llm, settings=SETTINGS, gate=None):
    bus = Bus()
    return PrimeConsentResponder(settings=settings, bus=bus, llm=llm, gate=gate, clock=lambda: T0), bus


def _acts(bus):
    return [e.payload for ch, e in bus.published if ch == GRAMMAR_EVENT_CHANNEL]


def test_a_parsed_yes_is_orions_answer_and_one_act_opens_and_closes():
    llm = LLM(parsed={"answer": "yes", "scope": "conversation", "reason": "my costs are fine to share"})
    r, bus = _responder(llm)
    ans = asyncio.run(r.decide(_req()))
    assert (ans.answer, ans.scope, ans.reason, ans.path) == ("yes", "conversation", "my costs are fine to share", "model")
    [call] = llm.calls
    assert call["route"] == "prime_consent" and call["max_tokens"] == 200 and call["thinking"] is False
    assert call["timeout_sec"] == 20.0 and call["pool_wait_sec"] == 15.0
    assert len(call["user_prompt"]) <= 4000
    acts = _acts(bus)
    assert [a["event_kind"] for a in acts] == ["act_opened", "act_closed"]
    assert acts[0]["act"]["object_ref"] == "claude_tools:consent:call-00000001"
    assert ans.act_id == acts[0]["act"]["act_id"]
    assert acts[1]["act"]["outcome"]["value"] == 1.0
    assert acts[0]["correlation_id"] == "room-claude-req-1", "the act shares the Claude turn's request id"


def test_an_unparsed_answer_is_a_no_and_the_act_closes_zero():
    r, bus = _responder(LLM(parsed={"answer": "maybe"}))
    ans = asyncio.run(r.decide(_req()))
    assert (ans.answer, ans.path, ans.reason) == ("no", "unparsed", "Orion's answer did not parse")
    acts = _acts(bus)
    assert [a["event_kind"] for a in acts] == ["act_opened", "act_closed"]
    assert acts[1]["act"]["outcome"]["value"] == 0.0


def test_no_model_seat_is_a_no_and_opens_no_act():
    from orion.mindloop.backoff import GATEWAY_NO_SEAT_ERRORS

    code = sorted(GATEWAY_NO_SEAT_ERRORS)[0]
    llm = LLM(parsed=None, err="empty", meta={"gateway_error": {"error": code}})
    r, bus = _responder(llm)
    ans = asyncio.run(r.decide(_req()))
    assert (ans.answer, ans.path) == ("no", "no_model") and _acts(bus) == []


def test_the_gate_refusing_the_consent_faculty_is_a_no_without_a_model_call(monkeypatch):
    now = T0
    raw = json.dumps({"kind": "self_monitor.state.v1", "computed_at": now.isoformat(), "window_days": 7.0,
                      "faculties": [{"faculty": "act.prime.infer.consent", "gradeability": "graded",
                                     "window_start": (now - timedelta(days=7)).isoformat(),
                                     "window_end": now.isoformat(), "n_bets": 30, "n_graded": 30,
                                     "n_failed_to_execute": 0, "n_expired": 0, "weight_graded": 30.0,
                                     "reliability": 0.02, "candidate_status": "broken", "status": "broken",
                                     "status_since": (now - timedelta(days=1)).isoformat()}]})
    monkeypatch.setattr(mindloop_gate, "read_view_sync", lambda url, now: self_monitor_view(raw, now=now))
    # The ledger shows a recent consent act, so no re-test probe is due.
    monkeypatch.setattr(mindloop_gate, "_ledger_rows", lambda dsn, actor, verb, secs: [
        {"object_ref": "claude_tools:consent:call-0000000x", "test_before": None, "opened_at": now - timedelta(hours=1)}])
    settings = SimpleNamespace(**{**vars(SETTINGS), "SELF_MONITOR_ACT_GATE_ENABLED": True,
                                  "SELF_MONITOR_LEDGER_PG_DSN": "postgresql://x/y"})
    llm = LLM(parsed={"answer": "yes", "reason": "x"})
    r, bus = _responder(llm, settings=settings, gate=mindloop_gate.build_act_gate(settings))
    ans = asyncio.run(r.decide(_req()))
    assert (ans.answer, ans.path, ans.reason) == ("no", "gate_refused", GATE_PAUSED_REASON)
    assert llm.calls == [] and _acts(bus) == []


def test_a_prime_thought_failing_does_not_pause_consent(monkeypatch):
    """Faculties are separate: act.prime.infer (thoughts) broken leaves act.prime.infer.consent alone."""
    raw = json.dumps({"kind": "self_monitor.state.v1", "computed_at": T0.isoformat(), "window_days": 7.0,
                      "faculties": [{"faculty": "act.prime.infer", "gradeability": "graded",
                                     "window_start": (T0 - timedelta(days=7)).isoformat(), "window_end": T0.isoformat(),
                                     "n_bets": 30, "n_graded": 30, "n_failed_to_execute": 0, "n_expired": 0,
                                     "weight_graded": 30.0, "reliability": 0.02, "candidate_status": "broken",
                                     "status": "broken", "status_since": (T0 - timedelta(days=1)).isoformat()}]})
    monkeypatch.setattr(mindloop_gate, "read_view_sync", lambda url, now: self_monitor_view(raw, now=now))
    monkeypatch.setattr(mindloop_gate, "_ledger_rows", lambda *a: [])
    settings = SimpleNamespace(**{**vars(SETTINGS), "SELF_MONITOR_ACT_GATE_ENABLED": True})
    r, _ = _responder(LLM(parsed={"answer": "yes", "reason": "ok"}), settings=settings,
                      gate=mindloop_gate.build_act_gate(settings))
    assert asyncio.run(r.decide(_req())).path == "model"


def test_handle_replies_on_the_result_channel_and_ignores_foreign_reply_to():
    r, bus = _responder(LLM(parsed={"answer": "no", "reason": "not now"}))
    env = BaseEnvelope(kind=CONSENT_REQUEST_KIND, source=ServiceRef(name="g"), payload=_req().model_dump(mode="json"))
    env = env.model_copy(update={"reply_to": f"{CONSENT_RESULT_PREFIX}{env.correlation_id}"})
    asyncio.run(r.handle(env))
    [(ch, out)] = [(c, e) for c, e in bus.published if c.startswith(CONSENT_RESULT_PREFIX)]
    assert ch == f"{CONSENT_RESULT_PREFIX}{env.correlation_id}" and out.kind == CONSENT_RESULT_KIND
    assert PrimeConsentAnswerV1.model_validate(out.payload).reason == "not now"
    foreign = env.model_copy(update={"reply_to": "orion:somewhere:else"})
    before = len(bus.published)
    asyncio.run(r.handle(foreign))
    assert len(bus.published) == before


def test_a_crash_still_answers_no():
    class Boom(LLM):
        def request_json(self, **kw):
            raise RuntimeError("kaput")

    r, bus = _responder(Boom())
    env = BaseEnvelope(kind=CONSENT_REQUEST_KIND, source=ServiceRef(name="g"), payload=_req().model_dump(mode="json"))
    env = env.model_copy(update={"reply_to": f"{CONSENT_RESULT_PREFIX}{env.correlation_id}"})
    asyncio.run(r.handle(env))
    [out] = [e for c, e in bus.published if c.startswith(CONSENT_RESULT_PREFIX)]
    assert PrimeConsentAnswerV1.model_validate(out.payload).answer == "no"


def test_lifespan_starts_the_consent_listener_on_its_own_bus_and_stops_it(monkeypatch):
    """A bus that cannot connect at boot delays the listener (retry in its task); it never blocks the
    lifespan. Off: no bus, no task."""
    import app.main as main
    from fastapi import FastAPI

    made = []

    class FailingBus:
        def __init__(self, **kw):
            made.append(self)
            self.closed = False

        async def connect(self):
            raise ConnectionError("bus down")

        async def close(self):
            self.closed = True

    async def _noop():
        return None

    monkeypatch.setattr(main, "OrionBusAsync", FailingBus)
    monkeypatch.setattr(main, "build_heartbeat_chassis", lambda: SimpleNamespace(start_background=_noop, stop=_noop))
    for key, val in (("RPC_HEALTH_PUBLISH_ENABLED", False), ("ORION_BUS_ENABLED", True), ("PRIME_ENABLED", False),
                     ("CURIOSITY_REPLICA_ENABLED", False), ("PRIME_CONSENT_ENABLED", True)):
        monkeypatch.setattr(main.settings, key, val)

    async def go():
        async with main.lifespan(FastAPI()):
            assert isinstance(main.consent_responder, main.PrimeConsentResponder)
        assert main.consent_responder is None and made[0].closed

    asyncio.run(go())
    made.clear()
    monkeypatch.setattr(main.settings, "PRIME_CONSENT_ENABLED", False)

    async def off():
        async with main.lifespan(FastAPI()):
            assert main.consent_responder is None

    asyncio.run(off())
    assert made == []


def _view(status: str, now) -> str:
    return json.dumps({"kind": "self_monitor.state.v1", "computed_at": now.isoformat(), "window_days": 7.0,
                       "faculties": [{"faculty": "act.prime.infer.consent", "gradeability": "graded",
                                      "window_start": (now - timedelta(days=7)).isoformat(),
                                      "window_end": now.isoformat(), "n_bets": 30, "n_graded": 30,
                                      "n_failed_to_execute": 0, "n_expired": 0, "weight_graded": 30.0,
                                      "reliability": 0.02 if status == "broken" else 0.9,
                                      "candidate_status": status, "status": status,
                                      "status_since": (now - timedelta(days=1)).isoformat()}]})


def test_the_gate_reads_a_fresh_view_on_every_ask(monkeypatch):
    """Lifecycle: healthy -> broken -> healthy across three asks on ONE responder. A view cached from the
    first ask would miss both changes (review finding, 2026-11-13)."""
    state = {"raw": _view("reliable", T0)}
    monkeypatch.setattr(mindloop_gate, "read_view_sync", lambda url, now: self_monitor_view(state["raw"], now=now))
    monkeypatch.setattr(mindloop_gate, "_ledger_rows", lambda dsn, actor, verb, secs: [
        {"object_ref": "claude_tools:consent:call-0000000x", "test_before": None, "opened_at": T0 - timedelta(hours=1)}])
    settings = SimpleNamespace(**{**vars(SETTINGS), "SELF_MONITOR_ACT_GATE_ENABLED": True,
                                  "SELF_MONITOR_LEDGER_PG_DSN": "postgresql://x/y"})
    r, _ = _responder(LLM(parsed={"answer": "yes", "reason": "ok"}), settings=settings,
                      gate=mindloop_gate.build_act_gate(settings))
    assert asyncio.run(r.decide(_req())).path == "model"
    state["raw"] = _view("broken", T0)
    assert asyncio.run(r.decide(_req())).path == "gate_refused"
    state["raw"] = _view("reliable", T0)
    assert asyncio.run(r.decide(_req())).path == "model"


def test_a_probe_is_claimed_before_the_model_call_and_only_once(monkeypatch):
    """Broken faculty, no recent attempt: the first ask is the re-test probe; the next ask in the same
    window is refused, even though each ask reads a fresh view."""
    monkeypatch.setattr(mindloop_gate, "read_view_sync", lambda url, now: self_monitor_view(_view("broken", T0), now=now))
    monkeypatch.setattr(mindloop_gate, "_ledger_rows", lambda *a: [])
    settings = SimpleNamespace(**{**vars(SETTINGS), "SELF_MONITOR_ACT_GATE_ENABLED": True,
                                  "SELF_MONITOR_LEDGER_PG_DSN": "postgresql://x/y"})
    llm = LLM(parsed={"answer": "no", "reason": "testing myself"})
    r, _ = _responder(llm, settings=settings, gate=mindloop_gate.build_act_gate(settings))
    assert asyncio.run(r.decide(_req())).path == "model"
    assert asyncio.run(r.decide(_req())).path == "gate_refused"
    assert len(llm.calls) == 1

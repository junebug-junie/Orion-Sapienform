"""The broker's bus wiring in the governor (app/claude_tools_listener.py): replies only on the exact
reply channel, a malformed request still gets a "not shared" answer, and the broker is built with
the governor's settings."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from orion.claude_tools.transport import TOOL_REQUEST_KIND, TOOL_RESULT_KIND, TOOL_RESULT_PREFIX
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.claude_tools import ClaudeToolBindingV1, ClaudeToolRequestV1, ClaudeToolResultV1


class Bus:
    def __init__(self):
        self.published = []
        self.redis = SimpleNamespace()

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


class Broker:
    def __init__(self):
        self.seen = []

    async def handle(self, req):
        self.seen.append(req)
        return ClaudeToolResultV1(call_id=req.call_id, tool=req.tool, shared=False, decision_path="standing_deny",
                                  text="Orion does not share body with you.")


def _env(payload, *, reply=True):
    env = BaseEnvelope(kind=TOOL_REQUEST_KIND, source=ServiceRef(name="orion-room-companion"), payload=payload)
    if reply:
        env = env.model_copy(update={"reply_to": f"{TOOL_RESULT_PREFIX}{env.correlation_id}"})
    return env


def _payload():
    b = ClaudeToolBindingV1(turn_request_id="t", room_id="r", platform="hub", claude_session_id="s")
    return ClaudeToolRequestV1(call_id="call-00000001", binding=b, tool="body_economy",
                               reason="checking").model_dump(mode="json")


def test_answers_on_the_exact_reply_channel():
    from app.claude_tools_listener import handle_envelope

    bus, broker = Bus(), Broker()
    env = _env(_payload())
    asyncio.run(handle_envelope(bus, broker, env))
    [(channel, out)] = bus.published
    assert channel == f"{TOOL_RESULT_PREFIX}{env.correlation_id}" and out.kind == TOOL_RESULT_KIND
    assert str(out.correlation_id) == str(env.correlation_id)
    assert ClaudeToolResultV1.model_validate(out.payload).decision_path == "standing_deny"


def test_a_request_without_its_reply_channel_is_ignored():
    from app.claude_tools_listener import handle_envelope

    bus, broker = Bus(), Broker()
    asyncio.run(handle_envelope(bus, broker, _env(_payload(), reply=False)))
    assert bus.published == [] and broker.seen == []


def test_a_malformed_request_is_answered_not_shared():
    from app.claude_tools_listener import handle_envelope

    bus, broker = Bus(), Broker()
    bad = _payload()
    bad["binding"]["room_id"] = ""
    asyncio.run(handle_envelope(bus, broker, _env(bad)))
    [(_, out)] = bus.published
    res = ClaudeToolResultV1.model_validate(out.payload)
    assert not res.shared and res.decision_path == "invalid" and broker.seen == []


def test_broker_is_built_from_governor_settings(monkeypatch):
    from app import claude_tools_listener as l

    monkeypatch.setattr(l.settings, "claude_tools_enabled", False)
    monkeypatch.setattr(l.settings, "claude_tools_max_asks_per_turn", 2)
    broker = l.build_broker(Bus())
    assert broker.config.enabled is False and broker.config.max_asks_per_turn == 2
    assert broker.config.consent_deadline_sec == 40.0 and set(broker.readers) == {"prime_now", "body_economy"}

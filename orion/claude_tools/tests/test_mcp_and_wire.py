"""The MCP server's one ability (a bus request), its failure answers, the call log the footer reads, the
consent RPC adapter, the footer words, the consent report and the grants view."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from orion.claude_tools import broker as b
from orion.claude_tools import mcp_server as m
from orion.claude_tools.grants import append_grant, grants_view, make_grant, settings_line
from orion.claude_tools.render import render_footer
from orion.claude_tools.report import consent_summary
from orion.claude_tools.transport import (
    CONSENT_RESULT_KIND,
    TOOL_REQUEST_CHANNEL,
    TOOL_RESULT_KIND,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.claude_tools import (
    ClaudeToolBindingV1,
    ClaudeToolRequestV1,
    ClaudeToolResultV1,
    PrimeConsentAnswerV1,
    PrimeConsentRequestV1,
)

T0 = datetime(2026, 11, 13, 4, 0, tzinfo=timezone.utc)


def _binding(tmp_path=None) -> ClaudeToolBindingV1:
    return ClaudeToolBindingV1(turn_request_id="room-claude-req-1", room_id="hub-room", platform="hub",
                               claude_session_id="sess-a",
                               call_log_path=str(tmp_path / "calls.jsonl") if tmp_path else None)


class FakeBus:
    def __init__(self, reply=None, exc=None):
        self.codec = OrionCodec()
        self.reply = reply
        self.exc = exc
        self.sent = []

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec, require_listener=False):
        self.sent.append((channel, envelope, reply_channel, timeout_sec, require_listener))
        if self.exc:
            raise self.exc
        payload, kind, corr = self.reply(envelope)
        env = BaseEnvelope(kind=kind, correlation_id=corr or envelope.correlation_id,
                           source=ServiceRef(name="x"), payload=payload)
        return {"data": self.codec.encode(env)}


def _broker_reply(envelope):
    req = ClaudeToolRequestV1.model_validate(envelope.payload)
    res = ClaudeToolResultV1(call_id=req.call_id, tool=req.tool, category="body", shared=True,
                             decision_path="asked", text='Orion said yes: "fine"\n\ndata', orion_answer="yes",
                             orion_reason="fine")
    return res.model_dump(mode="json"), TOOL_RESULT_KIND, None


def test_tool_list_is_the_registry_with_reason_required():
    tools = {t["name"]: t for t in m.tool_list()}
    assert set(tools) == {"prime_now", "body_economy"}
    assert "reason" in tools["body_economy"]["inputSchema"]["required"]


def test_one_call_is_one_bus_request_and_one_log_line(tmp_path):
    bus = FakeBus(reply=_broker_reply)
    client = m.ToolClient(bus, _binding(tmp_path))
    res = asyncio.run(client.call("body_economy", {"reason": "checking the meter", "section": "limits"}))
    assert res.shared and res.text.startswith("Orion said yes")
    [(channel, env, reply, timeout, require_listener)] = bus.sent
    assert channel == TOOL_REQUEST_CHANNEL and require_listener and timeout == m.RPC_TIMEOUT_SEC
    req = ClaudeToolRequestV1.model_validate(env.payload)
    assert req.reason == "checking the meter" and req.arguments == {"section": "limits"}
    assert req.binding.room_id == "hub-room", "the binding is the companion's, never Claude's"
    [line] = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert line["tool"] == "body_economy" and line["orion_answer"] == "yes"


def test_broker_down_is_not_shared(tmp_path):
    for bus in (FakeBus(exc=TimeoutError("RPC timeout")), FakeBus(exc=ConnectionError("no listener"))):
        res = asyncio.run(m.ToolClient(bus, _binding(tmp_path)).call("prime_now", {"reason": "why not"}))
        assert not res.shared and res.text == m.NO_BROKER_TEXT


def test_a_reply_for_another_call_or_kind_is_refused(tmp_path):
    def wrong_kind(env):
        payload, _, _ = _broker_reply(env)
        return payload, "other.kind.v1", None

    def other_call(env):
        payload, kind, _ = _broker_reply(env)
        payload["call_id"] = "someone-else"
        return payload, kind, None

    for reply in (wrong_kind, other_call):
        res = asyncio.run(m.ToolClient(FakeBus(reply=reply), _binding()).call("prime_now", {"reason": "why not"}))
        assert not res.shared and res.decision_path == "error"


def test_consent_asker_reads_the_answer_and_refuses_silence():
    from orion.core.bus.async_service import RpcNoListener

    creq = PrimeConsentRequestV1(call_id="call-00000001", turn_request_id="t", room_id="r", claude_session_id="s",
                                 tool="body_economy", category="body", tool_summary="x")

    def reply(env):
        ans = PrimeConsentAnswerV1(call_id="call-00000001", answer="yes", reason="ok", path="model")
        return ans.model_dump(mode="json"), CONSENT_RESULT_KIND, None

    bus = FakeBus(reply=reply)
    ans = asyncio.run(b.bus_consent_asker(bus, ServiceRef(name="g"))(creq, 40.0))
    assert ans.answer == "yes" and bus.sent[0][4] is True and bus.sent[0][3] == 40.0
    nobody = FakeBus(exc=RpcNoListener("nobody"))
    try:
        asyncio.run(b.bus_consent_asker(nobody, ServiceRef(name="g"))(creq, 40.0))
    except b.ConsentUnavailable as exc:
        assert "not listening" in str(exc)
    else:
        raise AssertionError("no listener must be ConsentUnavailable")


def test_footer_words():
    footer = render_footer([
        {"tool": "body_economy", "decision_path": "asked", "orion_answer": "yes", "orion_reason": "my costs are fine to share"},
        {"tool": "prime_now", "decision_path": "asked", "orion_answer": "no", "orion_reason": "not now"},
        {"tool": "prime_now", "decision_path": "asked", "error": "timeout"},
        {"tool": "prime_now", "decision_path": "asked", "error": "orion-mind is not listening for consent asks"},
        {"tool": "body_economy", "decision_path": "standing_allow"},
        {"tool": "body_economy", "decision_path": "limit"},
    ])
    assert footer == ('Looked at: body economy (Orion: yes, "my costs are fine to share"); prime now (Orion: no, '
                      '"not now"); prime now (Orion did not answer); prime now (Orion could not be asked); body economy (Orion allows it); body economy '
                      '(not available).')
    assert render_footer([]) is None


def test_consent_summary_rates_latency_flips_and_rubber_stamp():
    def row(i, ans, *, path="model", lat=1000, reason="r", tool="body_economy", cat="body"):
        return {"phase": "answer", "category": cat, "tool": tool, "orion_answer": ans, "consent_path": path,
                "consent_latency_ms": lat, "orion_reason": reason, "created_at": T0 + timedelta(hours=i)}

    rows = [row(0, "yes"), row(1, "no", reason=""), row(2, None, path=None, lat=40000)]
    s = consent_summary(rows)["body"]
    assert s["asks"] == 3 and s["model_answers"] == 2 and s["yes_rate"] == 0.5
    assert s["unanswered_rate"] == round(1 / 3, 3) and s["reason_share"] == 0.5 and s["flips_24h"] == 1
    assert s["latency_ms_p50"] == 1000 and s["rubber_stamp"] is None
    stamp = consent_summary([row(i, "yes", cat="prime", tool="prime_now") for i in range(20)])["prime"]
    assert stamp["rubber_stamp"] == "yes 20 of 20"


def test_grants_view_and_index_line(tmp_path):
    path = tmp_path / "self" / "claude_grants.jsonl"
    append_grant(path, make_grant("body", "allow", "my costs are fine to share", now=T0, dig="2026-11-13"))
    append_grant(path, make_grant("people", "deny", "neighbors did not agree", now=T0))
    view = grants_view(path)
    rows = {r["category"]: r for r in view["categories"]}
    assert rows["body"]["setting"] == "allow" and rows["body"]["reason"] == "my costs are fine to share"
    assert rows["prime"]["setting"] == "ask" and rows["prime"]["default"]
    line = settings_line(path)
    assert line.startswith("- Orion's tool settings for you (allow: body; ask: conversation, turns, prime")
    assert line.endswith("deny: people)") and len(line) < 200

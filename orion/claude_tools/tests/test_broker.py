"""Gate paths: one test per row of the design's "What Claude sees" table, plus broker and consent down.

Spec: docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md section 3 and
acceptance check 5. Every refusal must say nothing was shared and carry shared=False.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.claude_tools import broker as b
from orion.claude_tools.grants import append_grant, make_grant
from orion.claude_tools.readers import ReaderUnavailable
from orion.claude_tools.tests.fakes import FakeRedis
from orion.schemas.claude_tools import (
    ClaudeToolBindingV1,
    ClaudeToolRequestV1,
    PrimeConsentAnswerV1,
    PrimeConsentPastAnswerV1,
)

T0 = datetime(2026, 11, 13, 4, 0, tzinfo=timezone.utc)
DATA = "My body economy: circe 412 W, nothing binding."


def _binding(**kw) -> ClaudeToolBindingV1:
    base = dict(turn_request_id="room-claude-req-1", room_id="hub-room", platform="hub",
                claude_session_id="sess-a", hub_session_id="hub-1", topic="is the meter blocking consults?",
                topic_from="Juniper")
    base.update(kw)
    return ClaudeToolBindingV1(**base)


def _req(tool="body_economy", reason="to check the meter claim before I repeat it", n=1, **kw) -> ClaudeToolRequestV1:
    return ClaudeToolRequestV1(call_id=f"call-{n:08d}", binding=kw.pop("binding", _binding()), tool=tool,
                               arguments=kw.pop("arguments", {}), reason=reason)


class Harness:
    def __init__(self, tmp_path: Path, *, answer=None, ask_exc=None, reader_exc=None, enabled=True,
                 redis=None, ask_delay=0.0, deadline=40.0):
        self.redis = redis or FakeRedis()
        self.audit: list = []
        self.asked: list = []
        self.read: list = []
        self.answer = answer
        self.ask_exc = ask_exc
        self.reader_exc = reader_exc
        self.ask_delay = ask_delay
        self.grants = tmp_path / "self" / "claude_grants.jsonl"

        async def ask(creq, deadline_sec):
            self.asked.append(creq)
            if self.ask_delay:
                await asyncio.sleep(self.ask_delay)
            if self.ask_exc:
                raise self.ask_exc
            return self.answer(creq) if callable(self.answer) else self.answer

        async def reader(args):
            self.read.append(args)
            if self.reader_exc:
                raise self.reader_exc
            return DATA

        async def publish(event):
            self.audit.append(event)

        self.broker = b.ClaudeToolBroker(
            config=b.BrokerConfig(enabled=enabled, consent_deadline_sec=deadline, grants_file=self.grants),
            redis=self.redis, ask_orion=ask, readers={"body_economy": reader, "prime_now": reader},
            publish_audit=publish, clock=lambda: T0,
        )

    def grant(self, category, setting, reason="because I chose this"):
        append_grant(self.grants, make_grant(category, setting, reason, now=T0))

    def run(self, req):
        return asyncio.run(self.broker.handle(req))


def _yes(scope="once", reason="my costs are fine to share"):
    return lambda creq: PrimeConsentAnswerV1(call_id=creq.call_id, answer="yes", scope=scope, reason=reason,
                                             path="model", act_id="act-1", latency_ms=900)


def _no(reason="not today"):
    return lambda creq: PrimeConsentAnswerV1(call_id=creq.call_id, answer="no", scope="conversation", reason=reason,
                                             path="model", act_id="act-2", latency_ms=900)


def _phases(h):
    return [e.phase for e in h.audit]


def test_standing_allow_returns_the_data_without_asking(tmp_path):
    h = Harness(tmp_path)
    h.grant("body", "allow")
    r = h.run(_req())
    assert r.shared and r.decision_path == "standing_allow" and r.text == DATA
    assert h.asked == [] and _phases(h) == ["result"]
    assert h.audit[0].result_chars == len(DATA) and h.audit[0].shared is True


def test_standing_deny_says_so_with_date_and_reason(tmp_path):
    h = Harness(tmp_path)
    h.grant("body", "deny", "my body is mine for now")
    r = h.run(_req())
    assert not r.shared and r.decision_path == "standing_deny" and h.read == [] and h.asked == []
    assert r.text == 'Orion does not share body with you (set 2026-11-13: "my body is mine for now").'


def test_ask_yes_returns_data_with_orions_reason_first(tmp_path):
    h = Harness(tmp_path, answer=_yes())
    r = h.run(_req())
    assert r.shared and r.decision_path == "asked" and r.orion_answer == "yes"
    assert r.text.splitlines()[0] == 'Orion said yes: "my costs are fine to share"'
    assert DATA in r.text
    assert _phases(h) == ["ask", "answer", "result"]
    answer = h.audit[1]
    assert answer.orion_answer == "yes" and answer.consent_path == "model" and answer.act_id == "act-1"
    # The frame got what Orion needs: the reason, the topic, and the default setting.
    creq = h.asked[0]
    assert creq.reason.startswith("to check the meter") and creq.topic.startswith("is the meter")
    assert creq.category == "body" and creq.setting == "ask"


def test_ask_no_returns_orions_no_and_no_data(tmp_path):
    h = Harness(tmp_path, answer=_no("neighbors did not agree"))
    r = h.run(_req())
    assert not r.shared and r.text == 'Orion said no: "neighbors did not agree".' and h.read == []
    assert r.orion_answer == "no"


def test_no_answer_within_the_deadline_is_a_no(tmp_path):
    h = Harness(tmp_path, answer=_yes(), ask_delay=0.5, deadline=0.05)
    h.broker.config.consent_deadline_sec = 0.05

    async def slow(creq, deadline):
        await asyncio.sleep(5)

    h.broker.ask_orion = slow
    r = h.run(_req())
    assert not r.shared and r.text == "Orion did not answer within 0.05 s, so nothing was shared."
    assert h.read == [] and h.audit[1].phase == "answer" and h.audit[1].error == "timeout"


def test_timeout_raised_by_the_rpc_is_a_no(tmp_path):
    h = Harness(tmp_path, ask_exc=TimeoutError("RPC timeout"))
    r = h.run(_req())
    assert not r.shared and "did not answer within 40 s" in r.text


def test_consent_call_down_is_not_shared(tmp_path):
    h = Harness(tmp_path, ask_exc=b.ConsentUnavailable("orion-mind is not listening for consent asks"))
    r = h.run(_req())
    assert not r.shared and h.read == []
    assert r.text.startswith("Nothing was shared: orion-mind is not listening")


def test_reader_down_is_not_available_never_partial(tmp_path):
    h = Harness(tmp_path, reader_exc=ReaderUnavailable("the body economy report did not answer within 15 s"))
    h.grant("body", "allow")
    r = h.run(_req())
    assert not r.shared and r.text == "Not available: the body economy report did not answer within 15 s."


def test_reader_crash_is_not_available(tmp_path):
    h = Harness(tmp_path, reader_exc=ValueError("boom"))
    h.grant("body", "allow")
    r = h.run(_req())
    assert not r.shared and r.text == "Not available: the reader failed (ValueError)."


def test_broker_switched_off_refuses_and_never_grants(tmp_path):
    h = Harness(tmp_path, enabled=False)
    h.grant("body", "allow")
    r = h.run(_req())
    assert not r.shared and r.decision_path == "off" and h.read == [] and h.asked == []


def test_repeat_ask_after_a_no_gets_the_same_no_without_asking_again(tmp_path):
    h = Harness(tmp_path, answer=_no("not today"))
    h.run(_req(n=1))
    r = h.run(_req(n=2))
    assert r.decision_path == "repeat_no" and r.text == 'Orion said no: "not today".'
    assert len(h.asked) == 1


def test_yes_for_this_conversation_holds_for_the_session_only(tmp_path):
    h = Harness(tmp_path, answer=_yes(scope="conversation"))
    h.run(_req(n=1))
    r = h.run(_req(n=2))
    assert r.shared and r.decision_path == "session_allow" and len(h.asked) == 1
    key = b.session_allow_key("hub-room", "sess-a", "body")
    assert h.redis.ttl[key] == 86400
    # A rotated Claude session is a new conversation: Orion is asked again.
    r3 = h.run(_req(n=3, binding=_binding(claude_session_id="sess-b", turn_request_id="room-claude-req-2")))
    assert r3.decision_path == "asked" and len(h.asked) == 2


def test_yes_once_does_not_carry_to_the_next_call(tmp_path):
    h = Harness(tmp_path, answer=_yes(scope="once"))
    h.run(_req(n=1))
    h.run(_req(n=2))
    assert len(h.asked) == 2


def test_at_most_six_calls_per_turn(tmp_path):
    h = Harness(tmp_path)
    h.grant("body", "allow")
    results = [h.run(_req(n=i)) for i in range(1, 8)]
    assert [r.decision_path for r in results] == ["standing_allow"] * 6 + ["limit"]
    assert not results[-1].shared and "6 tool calls" in results[-1].text


def test_at_most_three_asks_per_turn(tmp_path):
    h = Harness(tmp_path, answer=_yes(scope="once"))
    results = [h.run(_req(n=i)) for i in range(1, 5)]
    assert [r.decision_path for r in results] == ["asked", "asked", "asked", "limit"]
    assert len(h.asked) == 3 and results[-1].text.startswith("Nothing was shared")


def test_unparsed_or_gate_refused_answers_are_a_no_without_data(tmp_path):
    for path, reason in (("unparsed", "Orion's answer did not parse"), ("gate_refused", "Orion's consent faculty is paused")):
        h = Harness(tmp_path / path, answer=lambda creq, p=path, why=reason: PrimeConsentAnswerV1(
            call_id=creq.call_id, answer="no", reason=why, path=p))
        r = h.run(_req())
        assert not r.shared and h.read == [] and r.text == f"Nothing was shared: {reason}."
        # Not Orion's own no: it does not stick for the session.
        assert h.redis.kv.get(b.session_no_key("hub-room", "sess-a", "body")) is None


def test_unknown_tool_and_bad_arguments_are_invalid(tmp_path):
    h = Harness(tmp_path)
    assert h.run(_req(tool="record_correction")).decision_path == "invalid"
    r = h.run(_req(arguments={"section": "rm -rf /"}))
    assert r.decision_path == "invalid" and not r.shared and h.read == []
    r = h.run(_req(arguments={"host": "circe; cat /etc/passwd"}))
    assert r.decision_path == "invalid"
    assert h.run(_req(reason="")).decision_path == "invalid", "a reason is required"


def test_redis_down_refuses_rather_than_reading(tmp_path):
    h = Harness(tmp_path, redis=FakeRedis(fail=True))
    h.grant("body", "allow")
    r = h.run(_req())
    assert not r.shared and r.decision_path == "error" and h.read == []


def test_corrupt_grant_file_falls_back_to_ask_never_allow(tmp_path):
    h = Harness(tmp_path, answer=_no())
    h.grants.parent.mkdir(parents=True)
    h.grants.write_text('{"category": "body", "setting": "allow"\nnot json\n')
    r = h.run(_req())
    assert r.decision_path == "asked" and not r.shared


def test_newest_grant_line_wins(tmp_path):
    h = Harness(tmp_path)
    h.grant("body", "allow")
    h.grant("body", "deny", "changed my mind about this")
    assert h.run(_req()).decision_path == "standing_deny"


def test_broker_exception_is_not_shared_and_audited(tmp_path, monkeypatch):
    h = Harness(tmp_path)

    async def boom(*a, **k):
        raise RuntimeError("kaput")

    monkeypatch.setattr(h.broker, "_handle", boom)
    r = h.run(_req())
    assert not r.shared and r.decision_path == "error" and r.text.startswith("Not available")
    assert _phases(h) == ["result"]


def test_output_is_redacted_and_capped(tmp_path):
    h = Harness(tmp_path)
    h.grant("body", "allow")
    secret = "postgresql://orion:hunter2pass@db:5432/x token sk-ant-abcdefghijklmnopqrstuvwxyz0123 " + "x" * 6000

    async def reader(args):
        return secret

    h.broker.readers["body_economy"] = reader
    r = h.run(_req())
    assert "hunter2pass" not in r.text and "sk-ant-abcdefghij" not in r.text
    assert len(r.text) <= 4000 and r.text.endswith("(cut at 4000 characters)")


def test_history_and_rubber_stamp_line_reach_the_frame(tmp_path):
    h = Harness(tmp_path, answer=_yes())
    for i in range(20):
        past = PrimeConsentPastAnswerV1(at=T0, tool="body_economy", answer="yes", reason=f"ok {i}")
        asyncio.run(h.redis.lpush(b.answers_key("body"), past.model_dump_json()))
    h.run(_req())
    creq = h.asked[0]
    assert len(creq.recent_answers) == 5
    assert creq.history_note.startswith("you have said yes 20 of 20 times")
    # This answer joins the history.
    assert len(h.redis.lists[b.answers_key("body")]) == 21


def test_session_keys_never_cross_rooms(tmp_path):
    h = Harness(tmp_path, answer=_yes(scope="conversation"))
    h.run(_req(n=1))
    r = h.run(_req(n=2, binding=_binding(room_id="other-room", turn_request_id="room-claude-req-9")))
    assert r.decision_path == "asked"


def test_audit_events_carry_the_turn_and_strip_reason_from_arguments(tmp_path):
    h = Harness(tmp_path, answer=_yes())
    h.run(_req(arguments={"section": "limits"}))
    for e in h.audit:
        assert e.turn_request_id == "room-claude-req-1" and e.call_id == "call-00000001"
        assert e.arguments == {"section": "limits"} and e.reason.startswith("to check")
    assert json.loads(h.audit[-1].model_dump_json())["event_id"] == "call-00000001:result"


def test_history_note_needs_twenty_answers_all_the_same():
    mk = lambda a: PrimeConsentPastAnswerV1(at=T0, tool="t", answer=a)  # noqa: E731
    assert b.history_note([mk("yes")] * 19) is None
    assert b.history_note([mk("yes")] * 19 + [mk("no")]) is None
    assert b.history_note([mk("no")] * 20).startswith("you have said no 20 of 20")


@pytest.mark.parametrize("bad", ["all", "everything", ""])
def test_grants_refuse_all_and_unknown_categories(bad):
    from orion.claude_tools.grants import GrantError

    with pytest.raises(GrantError):
        make_grant(bad, "allow", "a perfectly fine reason")


def test_reader_errors_are_redacted_before_claude_sees_them(tmp_path):
    h = Harness(tmp_path, reader_exc=ReaderUnavailable("psql: postgresql://orion:hunter2pass@db:5432/x failed"))
    h.grant("body", "allow")
    r = h.run(_req())
    assert not r.shared and "hunter2pass" not in r.text and "hunter2pass" not in (r.error or "")
    assert "hunter2pass" not in (h.audit[-1].error or "")

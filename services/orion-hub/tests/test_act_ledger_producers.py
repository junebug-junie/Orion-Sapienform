"""Hub act-ledger producers (rung 2): chat reply `say`, outreach `say`/`ask`, reading `look`.

Every act carries a non-empty expectation with a data-derived horizon, is a valid
act grammar event, and publishing it never raises into the doing it records.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from orion.grammar.acts import JUNIPER_REPLY_HORIZON_SEC
from orion.schemas.grammar import GrammarEventV1

from scripts import act_emit


class _Bus:
    def __init__(self):
        self.sent = []

    async def publish(self, channel, envelope):
        self.sent.append((channel, envelope))

    def events(self):
        return [GrammarEventV1.model_validate(e.payload) for ch, e in self.sent if ch == "orion:grammar:event"]


class _BrokenBus:
    async def publish(self, *a, **k):
        raise RuntimeError("bus down")


@pytest.fixture(autouse=True)
def _flag_on(monkeypatch):
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_ACT_LEDGER_ENABLED", True, raising=False)


# -- chat reply ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_reply_opens_a_say_act_expecting_juniper():
    from orion.hub.turn_orchestrator import _publish_reply_say_act

    bus = _Bus()
    assert await _publish_reply_say_act(bus, correlation_id="7d3c6f8e-1111-4222-8333-444455556666",
                                        session_id="s1", source_label="hub_orion", client_meta=None)
    (ev,) = bus.events()
    assert ev.event_kind == "act_opened" and ev.correlation_id == "7d3c6f8e-1111-4222-8333-444455556666"
    act = ev.act
    assert (act.verb, act.actor, act.object_ref) == ("say", "hub.chat_reply", "juniper")
    assert act.expect.signal_id == "juniper_replied" and act.expect.horizon_sec == JUNIPER_REPLY_HORIZON_SEC
    assert "median time" in act.expect.basis and "turn start" in act.expect.basis


@pytest.mark.asyncio
@pytest.mark.parametrize("source_label,client_meta", [
    ("aitown", None),
    ("hub_orion", {"unsolicited": True}),
    ("hub_orion", {"unsolicited": "true"}),
])
async def test_chat_reply_skips_non_juniper_turns(source_label, client_meta):
    from orion.hub.turn_orchestrator import _publish_reply_say_act

    bus = _Bus()
    assert not await _publish_reply_say_act(bus, correlation_id="c", session_id=None,
                                            source_label=source_label, client_meta=client_meta)
    assert bus.sent == []


@pytest.mark.asyncio
async def test_chat_reply_flag_off_and_bus_failure(monkeypatch):
    from orion.hub.turn_orchestrator import _publish_reply_say_act
    from scripts.settings import settings

    assert not await _publish_reply_say_act(_BrokenBus(), correlation_id="c", session_id=None,
                                            source_label="hub_orion", client_meta=None)
    monkeypatch.setattr(settings, "HUB_ACT_LEDGER_ENABLED", False, raising=False)
    bus = _Bus()
    assert not await _publish_reply_say_act(bus, correlation_id="c", session_id=None,
                                            source_label="hub_orion", client_meta=None)
    assert bus.sent == []


# -- outreach -----------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("text,verb", [("Want to look at this together?", "ask"), ("I painted the rain.", "say")])
async def test_outreach_act_verb_and_expectation(text, verb):
    bus = _Bus()
    assert await act_emit.publish_outreach_act(bus, text=text, message_id="m1", correlation_id="c1",
                                               session_id="s1", source_tag="curiosity")
    (ev,) = bus.events()
    assert ev.act.verb == verb and ev.act.actor == "hub.outreach"
    assert ev.act.expect.signal_id == "juniper_replied"
    assert ev.act.expect.horizon_sec == act_emit.OUTREACH_REPLY_HORIZON_SEC
    assert ev.act.test_before == "outreach:curiosity"


@pytest.mark.asyncio
async def test_outreach_act_never_raises():
    assert not await act_emit.publish_outreach_act(_BrokenBus(), text="x", message_id="m", correlation_id="c",
                                                   session_id=None, source_tag=None)


# -- reading ------------------------------------------------------------------

def _seed():
    return SimpleNamespace(seed_id="seed-9", url="https://example.org/a",
                           request=SimpleNamespace(request_id="req-9"))


@pytest.mark.asyncio
async def test_read_look_act_open_and_close_share_identity():
    bus = _Bus()
    await act_emit.publish_read_opened(bus, _seed())
    await act_emit.publish_read_closed(bus, _seed(), produced=False, note="no_read_evidence")
    opened, closed = bus.events()
    assert opened.act.act_id == closed.act.act_id
    assert opened.act.verb == "look" and opened.act.expect.signal_id == "artifact_produced"
    assert opened.act.object_ref == "https://example.org/a"
    assert closed.act.outcome.value == 0.0 and closed.act.outcome.note == "no_read_evidence"


@pytest.mark.asyncio
async def test_read_failure_closes_only_when_no_retry_is_scheduled(monkeypatch):
    from scripts import world_pulse_read_pipeline as pipe

    obj = pipe.WorldPulseReadPipeline.__new__(pipe.WorldPulseReadPipeline)
    obj._bus = _Bus()
    await obj._read_act_failed(_seed(), SimpleNamespace(retry_scheduled=True), "transient")
    assert obj._bus.events() == [], "a retried seed keeps its act open"
    await obj._read_act_failed(_seed(), SimpleNamespace(retry_scheduled=False), "no_read_evidence")
    await obj._read_act_failed(_seed(), None, "empty_generation")
    closes = [e.act.outcome.value for e in obj._bus.events()]
    assert closes == [0.0, 0.0]


def test_read_act_opens_only_after_refusal_checks():
    import inspect

    from scripts import world_pulse_read_pipeline as pipe

    src = inspect.getsource(pipe.WorldPulseReadPipeline._read_claimed)
    opened = src.index("publish_read_opened(")
    for refusal in ("bad_url", "skipped_index_url", "skipped_reader_blocked"):
        assert src.index(refusal) < opened, refusal
    assert src.index("publish_read_closed(self._bus, seed, produced=True") > src.index('"stage1_completed"')


@pytest.mark.asyncio
async def test_solicited_delivery_is_not_an_outreach_act(monkeypatch):
    from scripts import act_emit as emit
    from scripts.endogenous_outreach import EndogenousOutreach

    calls = []

    async def fake(bus, **kw):
        calls.append(kw)
        return True

    monkeypatch.setattr(emit, "publish_outreach_act", fake)
    eo = EndogenousOutreach.__new__(EndogenousOutreach)
    eo._bus = _Bus()
    eo._push_to_sockets = lambda **kw: None

    async def _noop(**kw):
        return None

    eo._publish_history = _noop
    eo._publish_notification = _noop
    await eo._deliver(text="hi?", session_id="s", correlation_id="c", unsolicited=False)
    assert calls == []
    await eo._deliver(text="hi?", session_id="s", correlation_id="c", unsolicited=True)
    assert len(calls) == 1


def test_flag_ships_on():
    from app.settings import Settings

    assert Settings.model_fields["HUB_ACT_LEDGER_ENABLED"].default is True

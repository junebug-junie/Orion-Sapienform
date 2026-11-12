"""Living Prime Part 2 in the resident loop (app/prime_loop.py): cockpit hops off the bus -> the tick's fold ->
Prime's state and its own Redis record. Fake bus, no network."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

from orion.cockpit.publish import publish_cockpit_hop
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.prime.handback import HANDBACK_CHANNEL, TURNS_KEY, PrimeTurnsStateV1
from orion.prime.state import PrimeStateV1
from orion.schemas.cockpit_sighting import CockpitHopV1
from test_prime_loop import MOVE, SETTINGS, T0, runtime

PROMISE = "I'll check the walkway logs tonight and tell you what I find"


def on(rt):
    rt.s = SimpleNamespace(**{**vars(rt.s), "PRIME_TURN_HANDBACK_ENABLED": True})
    return rt


def hop(corr, stage, raw, at):
    return CockpitHopV1(correlation_id=corr, seq=0 if stage == "ingress" else 9, ts=at, stage=stage,
                        visor_line=stage, status="ok", raw=raw, producer="orion-hub")


class Capture:
    """Records what Hub's own publisher (orion.cockpit.publish) puts on the bus."""

    def __init__(self):
        self.sent = []

    async def publish(self, channel, envelope):
        self.sent.append((channel, envelope))


def wire(*hops):
    """Each hop as the bytes Prime's listener receives: Hub's publisher's envelope, JSON-encoded."""
    cap = Capture()
    for h in hops:
        asyncio.run(publish_cockpit_hop(cap, h))
    assert {c for c, _ in cap.sent} == {HANDBACK_CHANNEL}
    return [env.model_dump_json() for _, env in cap.sent]


class JsonCodec:
    def decode(self, data):
        return SimpleNamespace(ok=True, envelope=BaseEnvelope.model_validate_json(data))


def chat_turn(corr="corr-7", *, promise=True, prompt="walkway again?"):
    ing = hop(corr, "ingress", {"session_id": "sess-j", "mode": "orion", "user_message": prompt}, T0)
    fin = hop(corr, "finalize", {"final_text": f"{PROMISE}.", "compliance_verdict": "completed",
                                 "reflection": {"promises_made": [{"quote": PROMISE, "offer": False}] if promise else []}},
              T0 + timedelta(minutes=2))
    return ing, fin


def test_a_chat_turn_that_opens_a_thread_is_a_concern_citing_its_correlation_id_within_one_tick():
    rt = on(runtime(answer={"reading": "a thread with Juniper", "concerns": {"keep": []}}))
    rt.bus.codec = JsonCodec()
    for data in wire(*chat_turn()):
        rt.accept_hop(data)
    asyncio.run(rt.tick())
    turn_concerns = [c for c in rt.state.concerns if c.event_ref == "turn:corr-7"]
    assert len(turn_concerns) == 1
    # It is an ordinary concern: its bet (predict act) opens like any other.
    assert any(k == "act_opened" and a["verb"] == "predict" for k, a in rt.bus.acts())
    # The thread woke a thought, and the frame showed the turn.
    assert "opened a thread" in rt.llm.calls[0]["user_prompt"]
    assert "TURNS THAT HANDED BACK TO YOU" in rt.llm.calls[0]["user_prompt"]
    saved = PrimeTurnsStateV1.model_validate_json(rt.bus.redis.kv[TURNS_KEY])
    assert saved.recent[0].correlation_id == "corr-7" and saved.recent[0].who == "chat_juniper"
    assert PrimeStateV1.model_validate_json(rt.bus.redis.kv["orion:prime:state"]).concerns


def test_handback_state_survives_a_restart_and_dedupes_a_replayed_hop():
    rt = on(runtime(answer={"reading": "ok"}))
    rt.bus.codec = JsonCodec()
    ing, fin = wire(*chat_turn())
    rt.accept_hop(ing)
    asyncio.run(rt.tick())  # ingress pending, saved
    rt2 = on(runtime(answer={"reading": "ok"}, bus=rt.bus, now=T0 + timedelta(minutes=3)))
    rt2.bus.codec = JsonCodec()
    asyncio.run(rt2.load())
    assert [p.correlation_id for p in rt2.turn_record.pending] == ["corr-7"]
    rt2.accept_hop(fin)
    rt2.accept_hop(fin)
    asyncio.run(rt2.tick())
    assert rt2.turn_record.turns_total == 1
    assert sum(1 for c in rt2.state.concerns if c.event_ref == "turn:corr-7") == 1


def test_an_undecodable_or_off_contract_hop_is_dropped_not_raised():
    rt = on(runtime(answer={"reading": "ok"}))
    rt.bus.codec = JsonCodec()
    rt.accept_hop("not json")
    rt.accept_hop(BaseEnvelope(kind="cockpit.hop.v1", source={"name": "x"}, payload={"stage": "ingress"}).model_dump_json())
    assert len(rt._turn_inbox) == 0 and rt._turn_hops_dropped == 2


def test_flag_off_means_no_fold_no_record_no_frame_section():
    rt = runtime(answer=MOVE)
    assert not getattr(rt.s, "PRIME_TURN_HANDBACK_ENABLED", False)
    rt._turn_inbox.extend([])  # the listener never starts with the flag off
    asyncio.run(rt.tick())
    assert TURNS_KEY not in rt.bus.redis.kv
    assert "TURNS THAT HANDED BACK" not in rt.llm.calls[0]["user_prompt"]


def test_a_treated_question_links_its_curiosity_turn_to_the_act_that_started_it():
    rt = on(runtime(answer=MOVE))
    asyncio.run(rt.tick())
    nom = rt.state.open_nominations[0]
    # The curiosity turn the run ran as handed back first.
    rt.bus.codec = JsonCodec()
    for data in wire(hop("cur-1", "ingress", {"session_id": "orion_curiosity", "user_message": "investigate"},
                         T0 + timedelta(hours=1)),
                     hop("cur-1", "finalize", {"final_text": "found it", "compliance_verdict": "completed"},
                         T0 + timedelta(hours=2))):
        rt.accept_hop(data)
    run = {"run_id": "r1", "line": "investigate", "started_at_iso": (T0 + timedelta(hours=1)).isoformat(),
           "outcome_kind": "finished", "focus": {"focus_id": nom.focus_id, "disposition": "explored"},
           "harness": {"turn_correlation_id": "cur-1"}}
    rt2 = on(runtime(answer={"reading": "ok"}, bus=rt.bus, runs=[run], now=T0 + timedelta(hours=3)))
    rt2.bus.codec = JsonCodec()
    asyncio.run(rt2.load())
    rt2._turn_inbox.extend(rt._turn_inbox)
    asyncio.run(rt2.tick())
    close = [a for k, a in rt2.bus.acts() if k == "act_closed" and a["verb"] == "ask"][0]
    assert close["act_id"] == nom.act_id and "turn cur-1" in close["outcome"]["note"]
    assert rt2.turn_record.recent[0].started_by_act == nom.act_id


def test_settings_default_ships_on():
    import app.settings as s

    assert s.Settings.model_fields["PRIME_TURN_HANDBACK_ENABLED"].default is True
    assert SETTINGS is not None

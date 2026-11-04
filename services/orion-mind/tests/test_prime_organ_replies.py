"""Prime hears the organ (curiosity replica patch 4, docs/superpowers/specs/2026-11-04-curiosity-replica-design.md
section 9): the ORGAN REPLIES frame section, and settle notes that name a refusal. Prime reads only the replica's
published replies view (Redis orion:curiosity:replica:replies)."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from orion.prime.acts import organ_refusal
from orion.schemas.curiosity_replica import REPLIES_KEY, CuriosityReplicaRepliesViewV1, CuriosityReplicaReplyV1
from test_prime_loop import MOVE, SETTINGS, T0, runtime
from test_prime_regulate_loop import SLOW, rt_with

REASON = "the last three runs died on cabinet questions; this would waste the next one"


def put_view(bus, prime_act_id, *, stance="refuse", kind="focus", at=T0 + timedelta(minutes=5)):
    v = CuriosityReplicaRepliesViewV1(as_of=at, replies=[CuriosityReplicaReplyV1(
        answer_act_id="act_answer1", prime_act_id=prime_act_id, target_kind=kind, target_id="t1", stance=stance,
        reason=REASON, about="What heats the cabinet after 14:00?", at=at)])
    bus.redis.kv[REPLIES_KEY] = v.model_dump_json()


def answers_on(rt):
    rt.s.CURIOSITY_REPLICA_ENABLED = True
    rt.s.CURIOSITY_REPLICA_ANSWERS_ENABLED = True
    return rt


def closes(rt, verb):
    return [a for k, a in rt.bus.acts() if k == "act_closed" and a["verb"] == verb]


def test_a_refused_question_closes_0_with_the_organs_reason_and_stays_treated():
    rt = runtime(answer=MOVE)
    asyncio.run(rt.tick())
    [nom] = rt.state.open_nominations
    assert nom.arm == "treated" and nom.queued is True
    put_view(rt.bus, nom.act_id)
    rt.llm.answer = {"reading": "ok"}
    rt.clock_box["now"] = T0 + timedelta(minutes=6)
    asyncio.run(rt.tick())
    [closed] = closes(rt, "ask")
    assert closed["act_id"] == nom.act_id and closed["test_before"] == "holdback:treated"   # intention to treat
    assert closed["outcome"]["value"] == 0.0
    assert closed["outcome"]["note"] == f"refused by curiosity.replica: {REASON}"
    assert rt.state.open_nominations == []
    assert any(f"refused by curiosity.replica: {REASON}" in o for o in rt.state.recent_outcomes)


def test_a_challenge_does_not_settle_the_question():
    rt = runtime(answer=MOVE)
    asyncio.run(rt.tick())
    [nom] = rt.state.open_nominations
    put_view(rt.bus, nom.act_id, stance="challenge")
    rt.llm.answer = {"reading": "ok"}
    rt.clock_box["now"] = T0 + timedelta(minutes=6)
    asyncio.run(rt.tick())
    assert closes(rt, "ask") == [] and len(rt.state.open_nominations) == 1


def test_the_frame_shows_organ_replies_with_reasons_only_while_answering_is_on():
    rt = answers_on(runtime(answer=MOVE))
    asyncio.run(rt.tick())
    [nom] = rt.state.open_nominations
    put_view(rt.bus, nom.act_id, stance="challenge")
    rt.state = rt.state.model_copy(update={"pending_changes": ["something moved"], "last_thought_at": T0})
    rt.clock_box["now"] = T0 + timedelta(minutes=10)
    rt.llm.answer = {"reading": "ok"}
    asyncio.run(rt.tick())
    call = rt.llm.calls[-1]
    frame = call["user_prompt"]
    assert "ORGAN REPLIES (last 48 h, newest first):" in frame
    assert "curiosity.replica challenged it (it still went ahead)" in frame and REASON in frame
    assert "What heats the cabinet after 14:00?" in frame and "Organs can answer you." in call["system_prompt"]
    # The first thought (no view published yet) showed the section as a gap, never as calm.
    assert "(unavailable: curiosity.replica has published no replies view)" in rt.llm.calls[0]["user_prompt"]
    # Off (the default in every older test, and the recorded replay): no section, no extra prompt.
    off = runtime(answer=MOVE)
    asyncio.run(off.tick())
    assert "ORGAN REPLIES" not in off.llm.calls[0]["user_prompt"]
    assert "Organs can answer you." not in off.llm.calls[0]["system_prompt"]


def test_an_old_reply_is_not_shown_and_a_refusal_reads_as_not_shown_to_the_run():
    rt = answers_on(runtime(answer={"reading": "ok"}))
    put_view(rt.bus, "act_somebody", at=T0 - timedelta(hours=49))
    asyncio.run(rt.tick())
    assert "(no challenge or refusal from an organ in the last 48 h)" in rt.llm.calls[0]["user_prompt"]
    put_view(rt.bus, "act_somebody", at=T0 - timedelta(hours=2))
    rt.state = rt.state.model_copy(update={"pending_changes": ["x"], "last_thought_at": T0 - timedelta(hours=1)})
    rt.clock_box["now"] = T0 + timedelta(minutes=5)
    asyncio.run(rt.tick())
    assert "refused it: the question was not shown to the run" in rt.llm.calls[-1]["user_prompt"]


def test_a_regulate_settle_note_names_the_organs_refusal():
    rt = rt_with(SLOW)
    asyncio.run(rt.tick())
    [reg] = rt.state.open_regulations
    put_view(rt.bus, reg.act_id, kind="signal")
    rt.llm.answer = {"reading": "watching", "concerns": {}, "move": None}
    rt.clock_box["now"] = reg.expires_at + timedelta(minutes=5)
    asyncio.run(rt.tick())
    [closed] = closes(rt, "regulate")
    assert f"refused by curiosity.replica: {REASON}" in closed["outcome"]["note"]


def test_organ_refusal_reads_only_refusals_of_that_act():
    v = CuriosityReplicaRepliesViewV1(as_of=T0, replies=[
        CuriosityReplicaReplyV1(answer_act_id="a", prime_act_id="p1", target_kind="focus", target_id="f",
                                stance="challenge", reason="r", at=T0)])
    assert organ_refusal("p1", v) is None and organ_refusal("p1", None) is None
    assert SETTINGS.PRIME_TICK_SEC == 15   # the shared fake settings carry no replica switch (off by default)

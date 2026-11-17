"""The concern evidence clock in the resident loop (PRIME_CONCERN_EVIDENCE_CLOCK_ENABLED, #3574): a sent question
renews the concern it names, a bare keep or a refused move does not. Fake bus, model and market."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from orion.prime.state import PrimeConcernV1, PrimeNominationV1, PrimeStateV1
from test_prime_loop import T0, Bus, runtime

CID = "197c7a88cacc4b76"
OPENED = T0 - timedelta(hours=20)


def _held(**kw) -> PrimeStateV1:
    concern = PrimeConcernV1(concern_id=CID, text="a thread from my letter for 2026-11-13: walkway render break",
                             event_ref="orion_day_letter:2026-11-13#thread0", opened_at=OPENED,
                             last_evidence_at=OPENED, act_opened_at=OPENED)
    return PrimeStateV1(concerns=[concern], pending_changes=["something changed"], **kw)


def _ask(concern_id=CID):
    return {"reading": "the render break is still live", "concerns": {"keep": [CID]},
            "move": {"kind": "curiosity_focus", "question": "Why does the walkway render fail each morning?",
                     "why": "the thread is unresolved", "concern_id": concern_id}}


KEEP_ONLY = {"reading": "nothing new", "concerns": {"keep": [CID]}, "move": None}


def _run(answer, *, clock=True, state=None, **kw):
    rt = runtime(answer=answer, **kw)
    rt.s.PRIME_CONCERN_EVIDENCE_CLOCK_ENABLED = clock
    rt.state = state or _held()
    asyncio.run(rt.tick())
    return rt


def _clock_of(rt):
    return next(c for c in rt.state.concerns if c.concern_id == CID).last_evidence_at


def test_a_bare_keep_no_longer_renews_the_concern():
    rt = _run(KEEP_ONLY)
    assert len(rt.llm.calls) == 1
    assert _clock_of(rt) == OPENED
    # The model is told so, and the frame line says when something last touched it.
    call = rt.llm.calls[0]
    assert "Keeping a concern does not renew it" in call["system_prompt"]
    assert "A concern you do not keep fades after a day" not in call["system_prompt"]
    assert ", last touched " in call["user_prompt"] and ", last kept " not in call["user_prompt"]


def test_with_the_clock_off_a_keep_renews_as_before():
    rt = _run(KEEP_ONLY, clock=False)
    assert _clock_of(rt) == T0
    assert "A concern you do not keep fades after a day" in rt.llm.calls[0]["system_prompt"]


def test_a_sent_question_about_the_concern_renews_it():
    rt = _run(_ask())
    [nom] = rt.state.open_nominations
    assert nom.concern_id == CID and nom.queued is True
    assert _clock_of(rt) == T0


def test_a_held_back_question_renews_it_too_so_the_arms_age_alike():
    rt = _run(_ask(), holdback=1.0)
    [nom] = rt.state.open_nominations
    assert nom.arm == "control" and rt.bus.rpcs == []
    assert _clock_of(rt) == T0


def test_a_question_hub_refused_does_not_renew_it():
    rt = _run(_ask(), bus=Bus(focus_reply={"ok": False, "error": "a focus is already waiting"}))
    assert rt.state.open_nominations[0].queued is False
    assert _clock_of(rt) == OPENED


def test_a_refused_move_does_not_renew_it():
    open_nom = PrimeNominationV1(act_id="act_" + "0" * 24, thought_id="t0", focus_id="f0", question="q", why="w",
                                 arm="treated", opened_at=T0 - timedelta(hours=1), queued=True)
    rt = _run(_ask(), state=_held(open_nominations=[open_nom]))
    assert "still open" in rt.state.recent_outcomes[-1]
    assert _clock_of(rt) == OPENED


def test_a_question_about_no_concern_renews_nothing():
    rt = _run(_ask(concern_id=None))
    assert rt.state.open_nominations and _clock_of(rt) == OPENED


def test_an_unrenewed_concern_fades_24h_after_its_last_evidence():
    """The jam (#3574): kept on every thought, it never faded. Now the keep at T0 leaves the clock at OPENED, and
    the first tick past OPENED + 24 h closes the act faded."""
    rt = _run(KEEP_ONLY)
    rt.clock_box["now"] = OPENED + timedelta(hours=24, minutes=1)
    rt.llm.answer = None
    asyncio.run(rt.tick())
    assert all(c.concern_id != CID for c in rt.state.concerns)
    rec = [r for r in rt.state.concern_record if r.concern_id == CID]
    assert rec and rec[-1].end == "faded" and rec[-1].note == "faded: nothing renewed it for a day"
    closes = [a for k, a in rt.bus.acts() if k == "act_closed" and a["verb"] == "predict"]
    assert closes and closes[-1]["outcome"]["value"] == 0.0


@pytest.mark.parametrize("clock, damped", [(True, True), (False, False)])
def test_a_keep_only_thought_no_longer_answers_the_horizon_need(clock, damped):
    """Guard 1: with the clock on, keeping a concern is not an answer to its horizon need, so a thought that only
    keeps halves the need; with it off, the keep answers it (nothing halved)."""
    rt = runtime(answer=KEEP_ONLY)
    rt.s.PRIME_CONCERN_EVIDENCE_CLOCK_ENABLED = clock
    rt.s.PRIME_HUNGER_NEEDS_ENABLED = True
    rt.state = _held()
    asyncio.run(rt.tick())
    assert len(rt.llm.calls) == 1
    assert ("horizon" in rt.state.needs.damp) is damped


def test_an_unknown_hub_answer_counts_as_sent():
    class Silent(Bus):
        async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
            raise TimeoutError

    rt = _run(_ask(), bus=Silent())
    assert rt.state.open_nominations[0].queued is None
    assert _clock_of(rt) == T0


def test_a_move_refused_after_apply_answer_does_not_renew_it():
    """A ref no curiosity run can open refuses the move in _conclude, after apply_answer: no renewal."""
    answer = _ask()
    answer["move"]["refs"] = [{"kind": "act", "id": "act_" + "a" * 24}]
    rt = _run(answer)
    assert rt.state.open_nominations == [] and "not sent" in rt.state.recent_outcomes[-1]
    assert _clock_of(rt) == OPENED


def test_the_prompt_names_only_real_evidence():
    rt = _run(KEEP_ONLY)
    prompt = rt.llm.calls[0]["system_prompt"]
    assert "a question you sent about it (its concern_id), or taking it on;" in prompt
    assert "a turn about it" not in prompt

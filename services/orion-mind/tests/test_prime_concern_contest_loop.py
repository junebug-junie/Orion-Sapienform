"""The concern contest in the resident loop (PRIME_CONCERN_CONTEST_ENABLED; design 2026-11-17 patch 2, #3574, #3656).
Fake bus, model and market."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

from orion.prime import day_letter as dl
from orion.prime.acts import concern_act
from orion.prime.state import PrimeConcernV1, PrimeStateV1
from test_prime_day_letter import DAY, at, letter_rt
from test_prime_loop import T0, runtime

from app.settings import Settings

QUIET = T0 - timedelta(hours=20)


def _quiet(i: int) -> PrimeConcernV1:
    return PrimeConcernV1(concern_id=f"{i:016x}", text=f"an old concern {i}", opened_at=QUIET, last_evidence_at=QUIET,
                          act_opened_at=QUIET)


def test_the_flags_ship_on():
    s = Settings()
    assert s.PRIME_CONCERN_CONTEST_ENABLED is True and s.PRIME_CONCERN_CONTEST_MARGIN == 0.1


def test_a_letter_with_every_slot_quiet_gets_two_threads_held_and_says_so():
    # Live 11-23 07:33Z: "5 thread(s), 0 now held as concerns, 5 without room".
    run = {"status": "completed", "orion_day": {"persist_outcome": "written", "threads": ["a", "b", "c", "d", "e"]}}
    rt = letter_rt(run=run)
    rt.s = SimpleNamespace(**{**vars(rt.s), "PRIME_CONCERN_CONTEST_ENABLED": True})
    # Prime thought after the 3 concerns went quiet (it saw them and did not act).
    rt.state = rt.state.model_copy(update={"concerns": [_quiet(i) for i in range(3)],
                                           "last_thought_at": T0 - timedelta(hours=1)})
    asyncio.run(rt.tick())
    at(rt, 6)
    done = rt.day_letter.done[-1]
    mine = [dl.thread_concern_id(DAY, i) for i in range(5)]
    assert done.concern_ids == mine[:2] and done.threads_without_room == 3
    assert {c.concern_id for c in rt.state.concerns} >= set(mine[:2])
    assert [w.concern_id for w in rt.state.waiting] == mine[2:]
    displaced = [r for r in rt.state.concern_record if r.end == "displaced"]
    assert len(displaced) == 2 and all(r.note.startswith("displaced: by ") for r in displaced)
    assert any("2 now held as concerns, 3 waiting for a slot" in o for o in rt.state.recent_outcomes)
    # The displaced concerns' acts close (0) on the same tick, and the new threads' acts open.
    closed = {a["act_id"]: a for kind, a in rt.bus.acts() if kind == "act_closed" and a["verb"] == "predict"}
    for r in displaced:
        assert closed[concern_act(r.concern_id).act_id]["outcome"]["value"] == 0.0
    opened = {a["act_id"] for kind, a in rt.bus.acts() if kind == "act_opened" and a["verb"] == "predict"}
    assert {concern_act(c).act_id for c in mine[:2]} <= opened


def test_with_the_contest_off_the_threads_wait_and_nothing_is_displaced():
    run = {"status": "completed", "orion_day": {"persist_outcome": "written", "threads": ["a", "b"]}}
    rt = letter_rt(run=run)
    rt.state = rt.state.model_copy(update={"concerns": [_quiet(i) for i in range(3)]})
    asyncio.run(rt.tick())
    at(rt, 6)
    assert rt.day_letter.done[-1].concern_ids == [] and len(rt.state.waiting) == 2
    assert not [r for r in rt.state.concern_record if r.end == "displaced"]


def test_the_frame_shows_standings_and_the_prompt_explains_the_contest():
    rt = runtime(answer={"reading": "ok", "concerns": {"keep": []}, "move": None})
    rt.s.PRIME_CONCERN_CONTEST_ENABLED = True
    rt.state = PrimeStateV1(concerns=[_quiet(0)], pending_changes=["something changed"])
    asyncio.run(rt.tick())
    call = rt.llm.calls[0]
    assert "decided by a contest" in call["system_prompt"]
    # Prime has not thought since this concern went quiet: the frame says this thought is its chance to act.
    assert ("contest: score unscored, quiet 20.0 h, quiet past half its life; it can be displaced after this thought "
            "unless I act on it") in call["user_prompt"]
    assert "WAITING FOR A SLOT" in call["user_prompt"]

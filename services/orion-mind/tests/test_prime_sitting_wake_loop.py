"""Game company patch 4 in the running Prime loop: a sitting wakes Prime, the frame shows it, at most 2 thoughts name it.

Acceptance (spec 2026-11-16, "Acceptance checks: Patch 4"): "A sitting produces at most 2 Prime thoughts that name it
as their wake." A thought names its wake in its act's close note (``woke: ...``)."""
from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace

from orion.retro.sitting import SITTING_KEY, WATCH_KEY
from test_prime_loop import SETTINGS, T0, runtime

KEEP = {"reading": "someone is playing"}
# An answer Prime cannot use: the thought closes, its pending changes stay for the next try.
BAD = {"reading": "x", "concerns": {}, "move": {"kind": "none"}}


def _on(rt, on=True):
    rt.s = SimpleNamespace(**{**vars(rt.s), "PRIME_SITTING_WAKE_ENABLED": on})
    return rt


def _put_sitting(rt, minutes, sid="S1"):
    now = rt.clock_box["now"].timestamp()
    rt.bus.redis.kv[SITTING_KEY] = json.dumps({
        "sitting_id": sid, "retro_session_id": "r1", "started_at": now - minutes * 60, "last_input_at": now - 4,
        "game_id": "a-link-to-the-past", "game_title": "A Link to the Past", "player_stated": "juniper",
        "mask_changes": 300, "duration_sec": minutes * 60})
    rt.bus.redis.kv[WATCH_KEY] = json.dumps({
        "sitting_id": sid, "mode": "on", "set_by": None, "why": None,
        "counts": {"notes_stored": 3, "model_calls": 3, "failed_looks": 0},
        "last_note": "Link crosses a bridge.", "last_note_at": now - 50})


def _thought_notes(rt, sid="S1"):
    return [a["outcome"]["note"] for k, a in rt.bus.acts()
            if k == "act_closed" and a["verb"] == "infer" and f"retro_sitting:{sid}" in (a["outcome"] or {}).get("note", "")]


def _advance(rt, minutes):
    rt.clock_box["now"] = rt.clock_box["now"] + timedelta(minutes=minutes)


def test_a_sitting_wakes_prime_on_open_and_on_close_and_never_a_third_time():
    rt = _on(runtime(answer=KEEP))
    rt.state = rt.state.model_copy(update={"last_thought_at": T0 - timedelta(hours=1), "pending_changes": []})
    _put_sitting(rt, 10.0)
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    assert "a game sitting is open: Juniper has played A Link to the Past" in call["user_prompt"]
    assert "A GAME SITTING IS OPEN" in call["user_prompt"]
    assert "Newest note (50 s ago; a look, not a fact): Link crosses a bridge." in call["user_prompt"]
    assert len(_thought_notes(rt)) == 1
    # Play goes on: no new wake, no new thought.
    _advance(rt, 10)
    _put_sitting(rt, 20.0)
    asyncio.run(rt.tick())
    assert len(rt.llm.calls) == 1
    # The sitting ends (the gateway deletes its key): one more wake, one more thought.
    del rt.bus.redis.kv[SITTING_KEY]
    _advance(rt, 10)
    asyncio.run(rt.tick())
    assert len(rt.llm.calls) == 2 and "a game sitting ended: Juniper played" in rt.llm.calls[1]["user_prompt"]
    assert "A GAME SITTING IS OPEN" not in rt.llm.calls[1]["user_prompt"]
    for _ in range(3):
        _advance(rt, 10)
        asyncio.run(rt.tick())
    assert len(_thought_notes(rt)) == 2


def test_a_sitting_is_named_by_two_thoughts_at_most_even_when_its_wake_stays_pending():
    """The cap itself: thoughts whose answer cannot be used keep their pending changes, so without the cap every
    later thought would name the sitting again."""
    rt = _on(runtime(answer=BAD))
    rt.state = rt.state.model_copy(update={"last_thought_at": T0 - timedelta(hours=1), "pending_changes": []})
    _put_sitting(rt, 10.0)
    for k in range(6):
        asyncio.run(rt.tick())
        _advance(rt, 10)
        _put_sitting(rt, 20.0 + 10 * k)
    assert len(rt.llm.calls) >= 2
    assert len(_thought_notes(rt)) == 2


def test_a_failed_redis_read_wakes_nothing_and_closes_nothing():
    rt = _on(runtime(answer=KEEP))
    rt.state = rt.state.model_copy(update={"last_thought_at": T0 - timedelta(hours=1), "pending_changes": []})
    _put_sitting(rt, 10.0)
    asyncio.run(rt.tick())
    assert len(rt.llm.calls) == 1

    async def boom(key):
        raise ConnectionError("redis down")

    rt.bus.redis.get = boom
    _advance(rt, 10)
    asyncio.run(rt.tick())
    assert len(rt.llm.calls) == 1 and not rt.state.sittings["S1"].closed_woke


def test_off_reads_nothing_and_shows_nothing():
    rt = _on(runtime(answer=KEEP), on=False)
    rt.state = rt.state.model_copy(update={"last_thought_at": T0 - timedelta(hours=1), "pending_changes": []})
    _put_sitting(rt, 10.0)
    asyncio.run(rt.tick())
    assert rt.llm.calls == [] and rt.state.sittings == {}


def test_the_settings_flag_ships_on():
    from app.settings import Settings

    assert Settings().PRIME_SITTING_WAKE_ENABLED is True
    assert "PRIME_SITTING_WAKE_ENABLED" not in vars(SETTINGS)

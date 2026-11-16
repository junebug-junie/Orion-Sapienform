"""Prime hunger in the runtime (integration design Part B; orion/prime/hunger.py, app/prime_loop.py)."""

from __future__ import annotations

import asyncio
import urllib.parse
from datetime import timedelta
from types import SimpleNamespace

from app.prime_loop import STATE_KEY
from orion.prime import hunger as hg
from orion.prime.state import PrimeConcernV1, PrimeStateV1
from test_prime_loop import T0, runtime

CID = "a" * 16
HUNGER_URL = "http://dr:8121/temporal-self/hunger?organ=prime"
IDLE = {"reading": "nothing needs me", "concerns": {}}


def hungry(rt, on=True, **extra):
    rt.s = SimpleNamespace(**{**vars(rt.s), "PRIME_HUNGER_NEEDS_ENABLED": on, **extra})
    return rt


def held(hours_unkept: float) -> PrimeStateV1:
    kept = T0 - timedelta(hours=hours_unkept)
    return PrimeStateV1(last_thought_at=T0 - timedelta(hours=1), seen_binding=[], concerns=[
        PrimeConcernV1(concern_id=CID, text="cabinet heat", opened_at=kept, last_evidence_at=kept,
                       act_opened_at=kept)])


def surprise_page(nats=4.2):
    return {"organ": "prime", "as_of": T0.isoformat(), "surprises": [
        {"event_id": "prediction_surprise:x", "organ": "prediction", "source_kind": "prediction_surprise",
         "label": "cabinet forecast crossed 32 C", "occurred_at": T0.isoformat(), "surprise_nats": nats,
         "source_p90": 0.0123, "source_n": 138}], "juniper": {"message_times": []}}


def closed_note(rt):
    return [a for k, a in rt.bus.acts() if k == "act_closed" and a["verb"] == "infer"][0]["outcome"]["note"]


def test_a_horizon_need_wakes_prime_with_no_change_and_writes_the_trace():
    rt = hungry(runtime(answer={"reading": "I keep it", "concerns": {"keep": [CID]}}))
    rt.state = held(19)
    asyncio.run(rt.tick())
    [want] = rt.bids
    # The bid carries hunger: salience = the strongest damped need, event_ref names it.
    assert want.salience == hg.horizon_drive(timedelta(hours=19), timedelta(hours=24)) or abs(
        want.salience - (19 / 24 - 0.5) / 0.5) < 1e-3
    assert want.event_ref == f"prime_need:horizon:prime_concern:{CID}"
    [call] = rt.llm.calls
    assert "WHAT IS PULLING AT YOU" in call["user_prompt"] and "(this is what woke you)" in call["user_prompt"]
    assert "WHAT IS PULLING AT YOU" in call["system_prompt"]
    note = closed_note(rt)
    assert note.startswith(f"woke: need:horizon prime_concern:{CID} |")
    assert hg.parse_woke_need(note) == ("horizon", f"prime_concern:{CID}")
    # Kept: the need is gone; the thought counted for the budget and the line is recorded as fired.
    assert len(rt.state.needs.thoughts_at) == 1 and f"horizon:prime_concern:{CID}" in rt.state.needs.fired
    assert rt.state.needs.damp == {}  # keeping the concern answered its horizon need: nothing is halved


def test_a_thought_that_changes_nothing_halves_the_shown_need_and_it_does_not_wake_again():
    rt = hungry(runtime(answer=IDLE))
    rt.state = held(19)
    asyncio.run(rt.tick())
    assert len(rt.llm.calls) == 1
    assert rt.state.needs.damp["horizon"].factor == 0.5
    # Past the refractory interval: the same line has not risen by MIN_RISE and is halved -> no second thought.
    rt.clock_box["now"] = T0 + timedelta(minutes=10)
    asyncio.run(rt.tick())
    assert len(rt.llm.calls) == 1
    saved = PrimeStateV1.model_validate_json(rt.bus.redis.kv[STATE_KEY])
    assert saved.needs.damp["horizon"].factor == 0.5  # survives a restart


def test_below_the_threshold_nothing_wakes_and_the_flag_off_is_changes_only():
    rt = hungry(runtime(answer=IDLE))
    rt.state = held(17)  # drive ~0.42
    asyncio.run(rt.tick())
    assert rt.bids == [] and rt.llm.calls == []
    off = hungry(runtime(answer=IDLE), on=False)
    off.state = held(23)
    asyncio.run(off.tick())
    assert off.bids == [] and off.llm.calls == []


def test_the_daily_budget_holds_a_need_wake():
    rt = hungry(runtime(answer=IDLE))
    state = held(20)
    for i in range(hg.DAILY_BUDGET):
        state = hg.note_thought(state, T0 - timedelta(hours=20) + timedelta(minutes=45 * i), None)
    rt.state = state
    asyncio.run(rt.tick())
    assert rt.bids == []


def test_a_surprise_need_is_read_from_durable_runs_and_pinned_into_the_now():
    rt = hungry(runtime(answer=IDLE, extra_pages={HUNGER_URL: surprise_page()}))
    rt.state = PrimeStateV1(last_thought_at=T0 - timedelta(hours=1), seen_binding=[])
    asyncio.run(rt.tick())
    assert HUNGER_URL in rt.urls
    note = closed_note(rt)
    assert note.startswith("woke: need:surprise experience:prediction_surprise:x")
    now_urls = [u for u in rt.urls if u.startswith("http://dr:8121/temporal-self/now?")]
    assert now_urls and urllib.parse.parse_qs(urllib.parse.urlparse(now_urls[0]).query)["include"] == [
        "prediction_surprise:x"]
    assert "[experience:prediction_surprise:x]" in rt.llm.calls[0]["user_prompt"]


def test_a_calm_surprise_and_an_unread_hunger_read_wake_nothing_and_say_so():
    calm = hungry(runtime(answer=IDLE, extra_pages={HUNGER_URL: surprise_page(nats=0.22)}))
    calm.state = PrimeStateV1(last_thought_at=T0 - timedelta(hours=1), seen_binding=[])
    asyncio.run(calm.tick())
    assert calm.bids == []
    # Unread (404): the needs it carries are unknown, no wake; a change still wakes and the frame names the gap.
    rt = hungry(runtime(answer=IDLE))
    rt.state = PrimeStateV1(last_thought_at=T0 - timedelta(hours=1), seen_binding=[],
                            pending_changes=["an act of mine settled"])
    asyncio.run(rt.tick())
    assert "surprise and connection could not be read" in rt.llm.calls[0]["user_prompt"]
    assert closed_note(rt).startswith("woke: an act of mine settled")


def test_the_read_is_cached_between_ticks():
    rt = hungry(runtime(answer=IDLE, extra_pages={HUNGER_URL: surprise_page(nats=0.22)}))
    rt.state = PrimeStateV1(last_thought_at=T0 - timedelta(hours=1), seen_binding=[])
    asyncio.run(rt.tick())
    rt.clock_box["now"] = T0 + timedelta(seconds=15)
    asyncio.run(rt.tick())
    assert rt.urls.count(HUNGER_URL) == 1
    rt.clock_box["now"] = T0 + timedelta(seconds=301)
    asyncio.run(rt.tick())
    assert rt.urls.count(HUNGER_URL) == 2

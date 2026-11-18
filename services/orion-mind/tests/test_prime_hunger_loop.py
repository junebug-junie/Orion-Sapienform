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
    # The bid carries hunger: salience = the waking need's damped drive. Never an event_ref (it would list Prime's
    # own bid as an event for Prime).
    assert abs(want.salience - (19 / 24 - 0.5) / 0.5) < 1e-3
    assert want.event_ref is None
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


def test_a_change_wake_without_a_need_bids_as_a_clock_want_and_the_mixed_trace_parses():
    rt = hungry(runtime(answer=IDLE))
    rt.state = held(10).model_copy(update={"pending_changes": ["an act of mine settled"]})
    asyncio.run(rt.tick())
    [want] = rt.bids
    assert want.salience is None and want.event_ref is None
    assert closed_note(rt).startswith("woke: an act of mine settled")
    mixed = hungry(runtime(answer=IDLE))
    mixed.state = held(19).model_copy(update={"pending_changes": ["an act of mine settled"]})
    asyncio.run(mixed.tick())
    note = closed_note(mixed)
    assert note.startswith(f"woke: need:horizon prime_concern:{CID} (also: an act of mine settled) |")
    assert hg.parse_woke_need(note) == ("horizon", f"prime_concern:{CID}")


EVENTS_URL = "http://dr:8121/physiology/event-wants?hours=24&limit=8&care=1"


def _reading_page(*decided_and_ids):
    return {"schema_version": "physiology.funded_event_wants.v1", "as_of": T0.isoformat(), "considered": 1,
            "funded_open": len(decided_and_ids), "wants": [
                {"want_id": f"w{i}", "family": "hub.world_pulse_read", "event_ref": f"world_pulse_read_seed:reading:{i}",
                 "salience": 0.95, "value_rank": 0.4, "net": 0.4, "decided_at": d.isoformat(),
                 "expires_at": (d + timedelta(hours=3)).isoformat(), "label": f"a read finished: {i}"}
                for d, i in decided_and_ids]}


def test_kept_concerns_do_not_shield_an_ignored_reading_want_issue_3552(caplog):
    """#3552, live 11-16..17: Prime kept its 3 concerns on every thought and ignored the reading want that woke it.
    The reading must be halved (under its event family), so the next new reading does not wake Prime again."""
    cids = [ch * 16 for ch in "abc"]
    kept = T0 - timedelta(hours=13)  # each concern's horizon need is shown (drive ~0.08) but cannot wake Prime
    page = {EVENTS_URL: _reading_page((T0 - timedelta(hours=2, minutes=15), 1))}
    rt = hungry(runtime(answer={"reading": "two reads finished, low value", "concerns": {"keep": cids}, "move": None},
                        extra_pages=page))
    rt.s.PRIME_CONCERNS_FROM_EVENT_WANTS = True
    rt.s.PRIME_EVENT_WANTS_HOURS = 24.0
    rt.s.PRIME_EVENT_WANTS_LIMIT = 8
    rt.state = PrimeStateV1(last_thought_at=T0 - timedelta(hours=1), seen_binding=[], concerns=[
        PrimeConcernV1(concern_id=c, text=f"held {c[0]}", opened_at=kept, last_evidence_at=kept, act_opened_at=kept)
        for c in cids])
    caplog.set_level("INFO", logger="orion-mind.prime")
    asyncio.run(rt.tick())
    # The log names the key actually halved, not every shown need's name.
    assert [r.getMessage().split("needs=")[1] for r in caplog.records
            if r.getMessage().startswith("prime_needs_damped")] == ["horizon:event:hub.world_pulse_read"]
    assert len(rt.llm.calls) == 1
    note = closed_note(rt)
    assert note.startswith("woke: need:horizon world_pulse_read_seed:reading:1 |") and "kept 3" in note
    assert set(rt.state.needs.damp) == {"horizon:event:hub.world_pulse_read"}  # kept concerns: "horizon" not halved
    # The next reading, a new ref at the 0.5 cap, past the refractory interval: no second thought.
    later = T0 + timedelta(minutes=10)
    rt.clock_box["now"] = later
    second = _reading_page((T0 - timedelta(hours=2, minutes=15), 1), (later - timedelta(hours=2, minutes=15), 2))
    # runtime() copies extra_pages, so serve the new menu through get_json itself (review finding).
    orig = rt.get_json
    rt.get_json = lambda u, *a, **k: second if u == EVENTS_URL else orig(u, *a, **k)
    asyncio.run(rt.tick())
    # Prime saw the new reading at the cap, halved below the threshold, and did not think again.
    [r2] = [n for n in rt._needs if n.ref == "world_pulse_read_seed:reading:2"]
    assert r2.drive == hg.EVENT_DRIVE_CAP and r2.damped < hg.THRESHOLD
    assert len(rt.llm.calls) == 1
    saved = PrimeStateV1.model_validate_json(rt.bus.redis.kv[STATE_KEY])
    assert set(saved.needs.damp) == {"horizon:event:hub.world_pulse_read"}  # survives a restart


LEARNING_URL = "http://pred:8132/learning-progress"
LEARNING_PAGE = {"as_of": "2026-11-16", "horizon_min": 15, "window_days": 7, "alpha": 0.05, "min_effect": 0.05, "series": [
    {"series_id": "office.temp_c", "status": "ok", "reason": None, "learning_progress": 0.228, "se": 0.08,
     "t": 2.86, "t_crit": 2.18, "p_value": 0.014, "direction": "better", "roster_changed_on": None,
     "recent": {"start": "2026-11-09", "end": "2026-11-15", "days": 7, "mean_skill": 0.341},
     "prior": {"start": "2026-11-02", "end": "2026-11-08", "days": 7, "mean_skill": 0.113}}]}


def test_a_learning_move_from_orion_prediction_wakes_prime_with_the_learning_trace():
    rt = hungry(runtime(answer=IDLE, extra_pages={HUNGER_URL: surprise_page(0.01), LEARNING_URL: LEARNING_PAGE}),
                PRIME_PREDICTION_URL="http://pred:8132")
    asyncio.run(rt.tick())
    assert LEARNING_URL in rt.urls
    [call] = rt.llm.calls
    assert "[learning] Your office temperature forecasts got better" in call["user_prompt"]
    assert hg.parse_woke_need(closed_note(rt)) == ("learning", "learning:office.temp_c:better")


def test_learning_off_reads_nothing_and_unreadable_says_unknown():
    rt = hungry(runtime(answer=IDLE, extra_pages={HUNGER_URL: surprise_page(0.01)}), PRIME_PREDICTION_URL="")
    asyncio.run(rt.tick())
    assert not any("learning-progress" in u for u in rt.urls)
    rt = hungry(runtime(answer=IDLE, extra_pages={HUNGER_URL: surprise_page(0.01)}),
                PRIME_PREDICTION_URL="http://pred:8132")
    asyncio.run(rt.tick())  # 404: first failure keeps nothing (there was no good read), so the need is unknown
    assert LEARNING_URL in rt.urls and rt._learning is None and rt._learning_error == "the read is off (404)"
    assert all(n.name != "learning" for n in rt._needs)

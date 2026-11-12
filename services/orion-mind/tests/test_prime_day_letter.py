"""Living Prime Part 3 in the resident loop (app/prime_loop.py): the day letter is Prime's end-of-day turn.

Day closes -> make act -> order -> run terminal -> act closes and the letter's threads become concerns. A deferred
bid writes nothing; past the bid window the day is recorded unwritten and the next letter's order names it.
Fake bus, Hub and durable-runs; no network."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from types import SimpleNamespace

from orion.prime import day_letter as dl
from orion.prime.state import MAX_CONCERNS, PrimeConcernV1
from orion.schemas.orion_day import OrionDayLetterOrderV1
from test_prime_loop import T0, runtime

DAY = date(2026, 10, 29)  # T0 is 2026-10-30 12:00Z = 06:00 in Denver: yesterday is 10-29
CLOSED = f"http://dr:8121/temporal-self/day/{DAY.isoformat()}"
STATUS = f"http://hub:8080/api/orion-day/letters/{DAY.isoformat()}"
ORDER = "http://hub:8080/api/orion-day/letter-run"
RUN = f"http://dr:8121/runs/{dl.letter_run_id(DAY)}"


def letter_rt(*, closed=True, exists=False, decisions=None, order_status="submitted", run=None, enabled=True,
              extra=None, now=T0, bus=None):
    pages = {**(extra or {})}
    if closed:
        pages[CLOSED] = {"day_id": DAY.isoformat(), "closed_at": "2026-10-30T07:05:00+00:00"}
    pages[STATUS] = {"letter_date": DAY.isoformat(), "letter_exists": exists, "run_id": "orion-day-x-1" if exists else None}
    if run is not None:
        pages[RUN] = run
    rt = runtime(answer={"reading": "ok"}, extra_pages=pages, now=now, bus=bus)
    rt.s = SimpleNamespace(**{**vars(rt.s), "PRIME_DAY_LETTER_ENABLED": enabled,
                              "PRIME_DAY_LETTER_BID_WINDOW_HOURS": 18, "PRIME_DAY_LETTER_TURN_TIMEOUT_SEC": 1800,
                              "PRIME_DAY_LETTER_RUN_DEADLINE_HOURS": 12})
    rt.pages = pages
    plain_get = rt.get_json
    rt.get_json = lambda url: pages[url] if url in pages else plain_get(url)
    queue = list(decisions or [])
    letter_bids = []
    plain_bid = rt.bid

    async def bid(want):
        if want.verb != "make":
            return await plain_bid(want)
        letter_bids.append(want)
        d = queue.pop(0) if queue else {"outcome": "funded"}
        return SimpleNamespace(**{"reason": "net_positive", "binding_constraint": None, "retry_after_sec": None,
                                  "why": "", **d})

    rt.bid = bid
    rt.letter_bids = letter_bids
    orders = []

    def post_json(url, body):
        orders.append((url, body))
        status = order_status(len(orders)) if callable(order_status) else order_status
        if status is None:
            raise TimeoutError("hub slow")
        return {"letter_date": body["letter_date"], "run_id": body["run_id"], "status": status,
                "detail": "boom" if status == "refused" else None}

    rt.post_json = post_json
    rt.orders = orders
    return rt


def at(rt, minutes):
    rt.clock_box["now"] = T0 + timedelta(minutes=minutes)
    rt._day_letter_checked_at = None  # each step in these tests is past the check interval
    asyncio.run(rt.tick())


def make_acts(bus):
    return [(kind, a) for kind, a in bus.acts() if a["verb"] == "make"]


def test_a_closed_day_without_a_letter_opens_a_make_act_and_orders_one_turn_with_the_right_inputs():
    rt = letter_rt()
    asyncio.run(rt.tick())
    [want] = rt.letter_bids
    assert (want.family, want.verb, want.tier, want.lanes) == ("prime", "make", 1, ["agent"])
    assert want.cost.gpu_slot_s_by_lane == {"agent": 1800.0}
    assert want.object_ref == f"orion_day_letter:{DAY.isoformat()}"
    [(kind, act)] = make_acts(rt.bus)
    assert kind == "act_opened" and act["actor"] == "prime" and act["object_ref"] == want.object_ref
    [(url, body)] = rt.orders
    assert url == ORDER
    order = OrionDayLetterOrderV1.model_validate(body)
    assert order.letter_date == DAY and order.run_id == "orion-day-2026-10-29-1"
    assert order.prime_act_id == act["act_id"] and order.correlation_id == dl.letter_correlation_id(DAY)
    assert order.deadline_at == T0 + timedelta(hours=12) and order.timeout_sec == 1800.0
    assert order.missed_letters == []
    saved = dl.PrimeDayLetterStateV1.model_validate_json(rt.bus.redis.kv[dl.DAY_LETTER_KEY])
    assert saved.open.act_id == act["act_id"] and saved.open.order_status == "submitted"


def test_a_day_the_chronicle_has_not_closed_is_left_alone():
    rt = letter_rt(closed=False)
    asyncio.run(rt.tick())
    assert rt.letter_bids == [] and rt.orders == [] and make_acts(rt.bus) == []


def test_a_day_that_already_has_a_letter_is_recorded_without_a_bid():
    rt = letter_rt(exists=True)
    asyncio.run(rt.tick())
    assert rt.letter_bids == [] and rt.orders == []
    assert rt.day_letter.done[-1].outcome == "already_written"
    at(rt, 10)  # never looked at again
    assert rt.letter_bids == []


def test_a_deferred_bid_writes_nothing_and_past_the_window_the_next_letter_says_so():
    deferred = [{"outcome": "deferred", "reason": "price", "binding_constraint": "heat", "retry_after_sec": 60.0}] * 50
    rt = letter_rt(decisions=deferred)
    asyncio.run(rt.tick())
    assert len(rt.letter_bids) == 1 and rt.orders == [] and make_acts(rt.bus) == []
    op = rt.day_letter.open
    assert "deferred (price, heat)" in op.last_refusal
    assert op.next_bid_at == T0 + timedelta(seconds=dl.REBID_MIN_SEC)  # the floor beats retry_after_sec 60
    at(rt, 5)
    assert len(rt.letter_bids) == 1  # not before next_bid_at
    at(rt, 16)
    assert len(rt.letter_bids) == 2
    at(rt, 18 * 60 + 1)  # the bid window is over
    assert rt.day_letter.open is None
    done = rt.day_letter.done[-1]
    assert done.outcome == "not_written" and "never funded" in done.reason and done.act_id is None
    assert make_acts(rt.bus) == [] and rt.orders == []
    assert any("did not write my letter" in line for line in rt.state.recent_outcomes)

    # The next day closes: its order names the missed day, once.
    nxt = DAY + timedelta(days=1)
    rt.pages[f"http://dr:8121/temporal-self/day/{nxt.isoformat()}"] = {"day_id": nxt.isoformat()}
    rt.pages[f"http://hub:8080/api/orion-day/letters/{nxt.isoformat()}"] = {"letter_date": nxt.isoformat(),
                                                                             "letter_exists": False}
    rt.letter_bids.clear()
    rt.bid = _funding(rt.bid)
    at(rt, 24 * 60 + 5)
    [(_, body)] = rt.orders
    order = OrionDayLetterOrderV1.model_validate(body)
    assert [m.letter_date for m in order.missed_letters] == [DAY]
    assert "never funded" in order.missed_letters[0].reason
    assert rt.day_letter.done[-1].told is True


def _funding(_old):
    async def bid(want):
        return SimpleNamespace(outcome="funded", reason="net_positive", binding_constraint=None, retry_after_sec=None,
                               why="")
    return bid


def test_a_written_letter_closes_the_act_at_1_and_its_threads_become_concerns_within_the_caps():
    run = {"status": "completed", "orion_day": {"persist_outcome": "written", "turn_correlation_id": "turn-abc",
                                                "threads": ["Why did the walkway count drop?", "Re-test the heat",
                                                            "Ask about the retro game"]}}
    rt = letter_rt(run=run)
    asyncio.run(rt.tick())  # funded + ordered
    act_id = rt.day_letter.open.act_id
    want_id = rt.day_letter.open.want_id
    at(rt, 6)  # settles
    closes = [a for kind, a in make_acts(rt.bus) if kind == "act_closed"]
    assert [c["act_id"] for c in closes] == [act_id]
    closed_events = [e.payload for c, e in rt.bus.published if c == "orion:grammar:event"
                     and e.payload["event_kind"] == "act_closed" and e.payload["act"]["verb"] == "make"]
    assert closed_events[0]["act"]["outcome"]["value"] == 1.0
    assert "turn turn-abc" in closed_events[0]["act"]["outcome"]["note"]
    assert want_id in rt.released
    threads = [c for c in rt.state.concerns if (c.event_ref or "").startswith(f"orion_day_letter:{DAY.isoformat()}")]
    assert len(threads) == dl.LETTER_CONCERNS_MAX
    assert "Why did the walkway count drop?" in threads[0].text
    done = rt.day_letter.done[-1]
    assert done.outcome == "written" and done.turn_correlation_id == "turn-abc" and done.threads == 3
    assert done.threads_without_room == 1 and done.concern_ids == [c.concern_id for c in threads]
    assert any("my letter for 2026-10-29" in line for line in rt.state.recent_outcomes)
    # Thread concerns place the ordinary concern bet (so a Prime ask about one can settle it).
    from orion.prime.acts import concern_act

    bets = {a["act_id"] for kind, a in rt.bus.acts() if a["verb"] == "predict" and kind == "act_opened"}
    assert {concern_act(c.concern_id).act_id for c in threads} <= bets


def test_thread_concerns_never_pass_max_concerns():
    run = {"status": "completed", "orion_day": {"persist_outcome": "written", "threads": ["a", "b"]}}
    rt = letter_rt(run=run)
    full = [PrimeConcernV1(concern_id=f"{i:016x}", text=f"c{i}", opened_at=T0, last_evidence_at=T0)
            for i in range(MAX_CONCERNS)]
    rt.state = rt.state.model_copy(update={"concerns": full})
    asyncio.run(rt.tick())
    at(rt, 6)
    assert len(rt.state.concerns) <= MAX_CONCERNS
    assert rt.day_letter.done[-1].threads_without_room == 2 and rt.day_letter.done[-1].concern_ids == []


def test_a_failed_run_closes_the_act_at_0_and_the_day_is_unwritten():
    rt = letter_rt(run={"status": "failed", "error": "harness_step_failed:letter:empty_generation", "orion_day": {}})
    asyncio.run(rt.tick())
    at(rt, 6)
    done = rt.day_letter.done[-1]
    assert done.outcome == "not_written" and "failed" in done.reason and done.concern_ids == []
    assert [k for k, _ in make_acts(rt.bus)] == ["act_opened", "act_closed"]
    closed = [e.payload for c, e in rt.bus.published if c == "orion:grammar:event"
              and e.payload["event_kind"] == "act_closed" and e.payload["act"]["verb"] == "make"]
    assert closed[0]["act"]["outcome"]["value"] == 0.0


def test_a_refused_order_closes_the_act_at_0_and_releases_the_bid():
    rt = letter_rt(order_status="refused")
    asyncio.run(rt.tick())
    assert rt.day_letter.open is None
    done = rt.day_letter.done[-1]
    assert done.outcome == "not_written" and "Hub refused" in done.reason
    assert [k for k, _ in make_acts(rt.bus)] == ["act_opened", "act_closed"]
    assert rt.letter_bids[0].want_id in rt.released


def test_an_unanswered_order_is_sent_again_and_never_opens_a_second_act():
    rt = letter_rt(order_status=lambda n: None if n == 1 else "exists")
    asyncio.run(rt.tick())
    assert rt.day_letter.open.order_status == "unavailable"
    at(rt, 6)
    assert len(rt.orders) == 2 and rt.orders[0][1] == rt.orders[1][1]  # the same order, byte for byte
    assert rt.day_letter.open.order_status == "exists"
    assert len(rt.letter_bids) == 1 and [k for k, _ in make_acts(rt.bus)] == ["act_opened"]


def test_the_open_letter_survives_a_restart():
    rt = letter_rt()
    asyncio.run(rt.tick())
    rt2 = letter_rt(bus=rt.bus, now=T0 + timedelta(minutes=1))
    asyncio.run(rt2.load())
    assert rt2.day_letter.open.act_id == rt.day_letter.open.act_id


def test_flag_off_means_no_bid_no_order_no_record():
    rt = letter_rt(enabled=False)
    asyncio.run(rt.tick())
    assert rt.letter_bids == [] and rt.orders == [] and dl.DAY_LETTER_KEY not in rt.bus.redis.kv

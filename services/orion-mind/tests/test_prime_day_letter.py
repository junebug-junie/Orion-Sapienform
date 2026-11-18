"""Living Prime Part 3 in the resident loop (app/prime_loop.py): the day letter is Prime's end-of-day turn.

Day closes -> make act -> order -> run terminal -> act closes and the letter's threads become concerns. Tier 0
(Juniper 2026-11-12): no bid, no gate; a run that wrote nothing is ordered again, up to MAX_ATTEMPTS runs inside the
deadline; all failed -> the day is recorded unwritten and the next letter's order names it.
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


def run_url(attempt):
    return f"http://dr:8121/runs/{dl.letter_run_id(DAY, attempt)}"


FAILED = {"status": "failed", "error": "HarnessStepFailed: harness_step_failed:letter:no_final_frame", "orion_day": {}}
WRITTEN = {"status": "completed", "orion_day": {"persist_outcome": "written", "threads": []}}


def letter_rt(*, closed=True, exists=False, order_status="submitted", run=None, enabled=True,
              extra=None, now=T0, bus=None, max_attempts=3):
    pages = {**(extra or {})}
    if closed:
        pages[CLOSED] = {"day_id": DAY.isoformat(), "closed_at": "2026-10-30T07:05:00+00:00"}
    pages[STATUS] = {"letter_date": DAY.isoformat(), "letter_exists": exists, "run_id": "orion-day-x-1" if exists else None}
    if run is not None:
        pages[RUN] = run
    rt = runtime(answer={"reading": "ok"}, extra_pages=pages, now=now, bus=bus)
    rt.s = SimpleNamespace(**{**vars(rt.s), "PRIME_DAY_LETTER_ENABLED": enabled,
                              "PRIME_DAY_LETTER_MAX_ATTEMPTS": max_attempts, "PRIME_DAY_LETTER_TURN_TIMEOUT_SEC": 1800,
                              "PRIME_DAY_LETTER_RUN_DEADLINE_HOURS": 12})
    rt.pages = pages
    plain_get = rt.get_json
    rt.get_json = lambda url: pages[url] if url in pages else plain_get(url)
    # Tier 0: the letter never bids and never asks the self-monitor gate. Spies prove it.
    letter_bids = []
    plain_bid = rt.bid

    async def bid(want):
        if want.verb == "make":
            letter_bids.append(want)
        return await plain_bid(want)

    rt.bid = bid
    rt.letter_bids = letter_bids
    gated = []
    plain_gate = rt._gate_decide

    async def gate_decide(verb, *a, **kw):
        if verb == "make":
            gated.append(verb)
        return await plain_gate(verb, *a, **kw)

    rt._gate_decide = gate_decide
    rt.gated = gated
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
    assert rt.letter_bids == [] and rt.gated == []  # tier 0: no bid, no gate
    [(kind, act)] = make_acts(rt.bus)
    assert kind == "act_opened" and act["actor"] == "prime" and act["object_ref"] == f"orion_day_letter:{DAY.isoformat()}"
    [(url, body)] = rt.orders
    assert url == ORDER
    order = OrionDayLetterOrderV1.model_validate(body)
    assert order.letter_date == DAY and order.run_id == "orion-day-2026-10-29-1"
    assert order.prime_act_id == act["act_id"] and order.correlation_id == dl.letter_correlation_id(DAY)
    assert order.deadline_at == T0 + timedelta(hours=12) and order.timeout_sec == 1800.0
    assert order.missed_letters == []
    saved = dl.PrimeDayLetterStateV1.model_validate_json(rt.bus.redis.kv[dl.DAY_LETTER_KEY])
    assert saved.open.act_id == act["act_id"] and saved.open.order_status == "submitted" and saved.open.attempt == 1


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


def test_a_failed_run_is_ordered_again_under_the_same_act_and_a_later_success_writes_the_day():
    # Live 11-11: one failed run (E2BIG) ended the day. Now it is the next attempt.
    rt = letter_rt(run=FAILED)
    asyncio.run(rt.tick())
    act_id = rt.day_letter.open.act_id
    at(rt, 6)  # attempt 1 failed -> attempt 2 ordered
    assert [OrionDayLetterOrderV1.model_validate(b).run_id for _, b in rt.orders] == [
        "orion-day-2026-10-29-1", "orion-day-2026-10-29-2"]
    second = OrionDayLetterOrderV1.model_validate(rt.orders[1][1])
    assert second.prime_act_id == act_id and second.correlation_id == dl.letter_correlation_id(DAY, 2)
    assert second.deadline_at == T0 + timedelta(hours=12)  # every attempt shares the first deadline
    op = rt.day_letter.open
    assert op.attempt == 2 and op.order_status == "submitted" and "no_final_frame" in op.failed_attempts[0]
    assert [k for k, _ in make_acts(rt.bus)] == ["act_opened"]  # one act for the day, still open
    rt.pages[run_url(2)] = WRITTEN
    at(rt, 12)
    done = rt.day_letter.done[-1]
    assert done.outcome == "written" and done.run_id == "orion-day-2026-10-29-2" and done.act_id == act_id
    closed = [e.payload for c, e in rt.bus.published if c == "orion:grammar:event"
              and e.payload["event_kind"] == "act_closed" and e.payload["act"]["verb"] == "make"]
    assert [c["act"]["outcome"]["value"] for c in closed] == [1.0]


def test_after_max_attempts_fail_the_day_is_unwritten_and_the_next_letter_says_so():
    rt = letter_rt(run=FAILED)
    rt.pages[run_url(2)] = FAILED
    rt.pages[run_url(3)] = FAILED
    asyncio.run(rt.tick())
    at(rt, 6)
    at(rt, 12)
    assert rt.day_letter.open.attempt == 3
    at(rt, 18)
    assert rt.day_letter.open is None and len(rt.orders) == 3
    done = rt.day_letter.done[-1]
    assert done.outcome == "not_written" and "attempt 3 of 3" in done.reason and "attempt 1:" in done.reason
    assert [k for k, _ in make_acts(rt.bus)] == ["act_opened", "act_closed"]

    # The next day closes: its order names the missed day, once.
    nxt = DAY + timedelta(days=1)
    rt.pages[f"http://dr:8121/temporal-self/day/{nxt.isoformat()}"] = {"day_id": nxt.isoformat()}
    rt.pages[f"http://hub:8080/api/orion-day/letters/{nxt.isoformat()}"] = {"letter_date": nxt.isoformat(),
                                                                             "letter_exists": False}
    rt.orders.clear()
    at(rt, 24 * 60 + 5)
    [(_, body)] = rt.orders
    order = OrionDayLetterOrderV1.model_validate(body)
    assert [m.letter_date for m in order.missed_letters] == [DAY]
    assert {d.letter_date: d.told for d in rt.day_letter.done}[DAY] is False
    assert rt.day_letter.open.missed_sent == [DAY]


def test_no_retry_past_the_shared_deadline():
    rt = letter_rt()
    asyncio.run(rt.tick())
    rt.pages[RUN] = FAILED
    at(rt, 12 * 60 + 1)  # attempt 1 is read failed only after the deadline: no time for another run
    assert rt.day_letter.open is None and len(rt.orders) == 1
    assert rt.day_letter.done[-1].outcome == "not_written"


def test_a_failed_act_open_is_tried_again_without_a_bid():
    rt = letter_rt()
    plain_open = rt._open
    calls = []

    async def flaky_open(*a, **kw):
        calls.append(1)
        return False if len(calls) == 1 else await plain_open(*a, **kw)

    rt._open = flaky_open
    asyncio.run(rt.tick())
    assert rt.day_letter.open.act_id is None and rt.orders == []
    at(rt, 6)
    assert rt.day_letter.open.act_id is not None and len(rt.orders) == 1 and rt.letter_bids == []


def test_a_written_letter_closes_the_act_at_1_and_its_threads_become_concerns_within_the_caps():
    run = {"status": "completed", "orion_day": {"persist_outcome": "written", "turn_correlation_id": "turn-abc",
                                                "threads": ["Why did the walkway count drop?", "Re-test the heat",
                                                            "Ask about the retro game"]}}
    rt = letter_rt(run=run)
    asyncio.run(rt.tick())  # opened + ordered
    act_id = rt.day_letter.open.act_id
    at(rt, 6)  # settles
    closes = [a for kind, a in make_acts(rt.bus) if kind == "act_closed"]
    assert [c["act_id"] for c in closes] == [act_id]
    closed_events = [e.payload for c, e in rt.bus.published if c == "orion:grammar:event"
                     and e.payload["event_kind"] == "act_closed" and e.payload["act"]["verb"] == "make"]
    assert closed_events[0]["act"]["outcome"]["value"] == 1.0
    assert "turn turn-abc" in closed_events[0]["act"]["outcome"]["note"]
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


def test_with_one_attempt_a_failed_run_closes_the_act_at_0_and_the_day_is_unwritten():
    rt = letter_rt(run={"status": "failed", "error": "harness_step_failed:letter:empty_generation", "orion_day": {}},
                   max_attempts=1)
    asyncio.run(rt.tick())
    at(rt, 6)
    done = rt.day_letter.done[-1]
    assert done.outcome == "not_written" and "failed" in done.reason and done.concern_ids == []
    assert [k for k, _ in make_acts(rt.bus)] == ["act_opened", "act_closed"]
    closed = [e.payload for c, e in rt.bus.published if c == "orion:grammar:event"
              and e.payload["event_kind"] == "act_closed" and e.payload["act"]["verb"] == "make"]
    assert closed[0]["act"]["outcome"]["value"] == 0.0


def test_a_refused_order_closes_the_act_at_0_and_is_not_retried():
    rt = letter_rt(order_status="refused")
    asyncio.run(rt.tick())
    assert rt.day_letter.open is None
    done = rt.day_letter.done[-1]
    assert done.outcome == "not_written" and "Hub refused" in done.reason
    assert [k for k, _ in make_acts(rt.bus)] == ["act_opened", "act_closed"] and len(rt.orders) == 1


def test_an_unanswered_order_is_sent_again_and_never_opens_a_second_act():
    rt = letter_rt(order_status=lambda n: None if n == 1 else "exists")
    asyncio.run(rt.tick())
    assert rt.day_letter.open.order_status == "unavailable"
    at(rt, 6)
    assert len(rt.orders) == 2 and rt.orders[0][1] == rt.orders[1][1]  # the same order, byte for byte
    assert rt.day_letter.open.order_status == "exists"
    assert rt.letter_bids == [] and [k for k, _ in make_acts(rt.bus)] == ["act_opened"]


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


def test_a_paused_run_past_its_deadline_closes_the_day_instead_of_blocking_every_later_letter():
    rt = letter_rt(run={"status": "paused", "orion_day": {}})
    asyncio.run(rt.tick())
    at(rt, 6)
    assert rt.day_letter.open is not None  # still inside the deadline: wait
    at(rt, 13 * 60 + 5)                    # 12 h deadline + 1 h grace passed
    assert rt.day_letter.open is None
    done = rt.day_letter.done[-1]
    assert done.outcome == "not_written" and "still paused past its deadline" in done.reason
    assert [k for k, _ in make_acts(rt.bus)] == ["act_opened", "act_closed"]


def test_missed_days_stay_untold_when_the_run_that_carried_them_fails():
    prior = DAY - timedelta(days=1)
    rt = letter_rt(run={"status": "failed", "error": "boom", "orion_day": {}}, max_attempts=1)
    rt.day_letter = dl.finish(rt.day_letter, dl.PrimeDayLetterDoneV1(letter_date=prior, outcome="not_written",
                                                                     reason="never funded", closed_at=T0))
    asyncio.run(rt.tick())
    assert [m["letter_date"] for m in rt.orders[0][1]["missed_letters"]] == [str(prior)]
    at(rt, 6)  # the run failed
    told = {d.letter_date: d.told for d in rt.day_letter.done}
    assert told[prior] is False  # no letter named it, so the next order lists it again
    assert {m.letter_date for m in dl.missed_for(rt.day_letter, DAY + timedelta(days=1))} == {prior, DAY}


def test_missed_days_are_told_once_the_letter_carrying_them_is_written():
    prior = DAY - timedelta(days=1)
    rt = letter_rt(run={"status": "completed", "orion_day": {"persist_outcome": "written", "threads": []}})
    rt.day_letter = dl.finish(rt.day_letter, dl.PrimeDayLetterDoneV1(letter_date=prior, outcome="not_written",
                                                                     reason="never funded", closed_at=T0))
    asyncio.run(rt.tick())
    at(rt, 6)
    assert {d.letter_date: d.told for d in rt.day_letter.done}[prior] is True


def test_a_row_another_run_wrote_is_a_letter_not_a_missed_day():
    rt = letter_rt(run={"status": "completed", "orion_day": {"persist_outcome": "already_written"}})
    asyncio.run(rt.tick())
    at(rt, 6)
    done = rt.day_letter.done[-1]
    assert done.outcome == "already_written"
    assert dl.missed_for(rt.day_letter, DAY + timedelta(days=1)) == []


def test_an_order_whose_answer_was_lost_is_followed_when_the_run_exists():
    run = {"status": "completed", "orion_day": {"persist_outcome": "written", "threads": []}}
    rt = letter_rt(order_status=None, run=run)  # Hub never answers
    asyncio.run(rt.tick())
    assert rt.day_letter.open.order_status == "unavailable"
    at(rt, 12 * 60 + 1)  # past the run deadline: Prime looks for the run before giving up
    assert rt.day_letter.done[-1].outcome == "written"


def test_a_run_that_wrote_the_row_then_failed_is_written_and_not_retried():
    run = {"status": "failed", "error": "journal publish", "orion_day": {"persist_outcome": "written",
                                                                         "threads": ["a thread"]}}
    rt = letter_rt(run=run)
    asyncio.run(rt.tick())
    at(rt, 6)
    assert len(rt.orders) == 1 and rt.day_letter.done[-1].outcome == "written"


def test_hub_naming_our_own_earlier_attempt_closes_the_day_written_and_tells_the_missed_days():
    prior = DAY - timedelta(days=1)
    rt = letter_rt(run=FAILED)
    rt.day_letter = dl.finish(rt.day_letter, dl.PrimeDayLetterDoneV1(letter_date=prior, outcome="not_written",
                                                                     reason="boom", closed_at=T0))
    plain_post = rt.post_json

    def post_json(url, body):
        if body["run_id"].endswith("-2"):  # attempt 1 wrote the row although its run reads failed
            rt.orders.append((url, body))
            return {"letter_date": body["letter_date"], "run_id": dl.letter_run_id(DAY, 1), "status": "already_written"}
        return plain_post(url, body)

    rt.post_json = post_json
    asyncio.run(rt.tick())
    at(rt, 6)
    done = rt.day_letter.done[-1]
    assert done.outcome == "written" and done.run_id == dl.letter_run_id(DAY, 1)
    assert {d.letter_date: d.told for d in rt.day_letter.done}[prior] is True
    closed = [e.payload for c, e in rt.bus.published if c == "orion:grammar:event"
              and e.payload["event_kind"] == "act_closed" and e.payload["act"]["verb"] == "make"]
    assert [c["act"]["outcome"]["value"] for c in closed] == [1.0]


def test_a_completed_run_with_no_row_is_retried():
    rt = letter_rt(run={"status": "completed", "orion_day": {"persist_outcome": "skipped"}})
    asyncio.run(rt.tick())
    at(rt, 6)
    assert rt.day_letter.open.attempt == 2 and len(rt.orders) == 2


def test_a_cancelled_run_is_not_retried():
    rt = letter_rt(run={"status": "cancelled", "orion_day": {}})
    asyncio.run(rt.tick())
    at(rt, 6)
    assert rt.day_letter.open is None and len(rt.orders) == 1
    assert rt.day_letter.done[-1].outcome == "not_written"

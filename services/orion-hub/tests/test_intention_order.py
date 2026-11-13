"""Intentions Prime takes on: Hub's order and delivery desk (scripts/intention_order.py; choice-as-EFE Step -1).

Order: the promise adapter reads the source rows, Hub submits ONE intention.carry run whose prompt cites the concern
and whose admission deadline is the promise's horizon. Delivery: one live send into the originating thread through
the normal outreach gates; when it is held, the reply is still written into that thread's history (S-1.6).
Fake pool, durable-runs and outreach; no network."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from scripts.intention_order import DELIVERED_SQL, IntentionDesk, build_carry_request
from scripts.orion_day_order import DurableUnavailable
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.intention import IntentionDeliverV1, IntentionOrderV1, intention_correlation_id, intention_run_id

CID = "d8db9d7b1bba4cf7"
CORR = "a129b72f-e698-425a-a435-3388cd7cd759"
SESSION = "9c91d646-cbd0-4977-8cdd-d2c82045c7f9"
QUOTE = "While you're off in Hyrule I'll take a lazy browse through the last 24 hours"
PROMISED = datetime(2026, 11, 12, 7, 8, 40)          # naive, as Postgres stores it
NOW = datetime(2026, 11, 13, 8, 0, tzinfo=timezone.utc)
ROW = {"promised_at": PROMISED, "promises_made": [{"offer": False, "quote": QUOTE}], "session_id": SESSION,
       "prompt": "have a look at the PRs", "response": f"Accepted, lightly. {QUOTE}."}


class Conn:
    def __init__(self, row, delivered=False):
        self.row, self.delivered, self.calls = row, delivered, []

    async def fetchrow(self, sql, *args):
        self.calls.append(("fetchrow", args))
        return self.row

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval", sql, args))
        return 1 if self.delivered else None


class Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return pool.conn

            async def __aexit__(self, *a):
                return False
        return _Ctx()


class Durable:
    def __init__(self, run=None, submit_status=202, down=False):
        self.run, self.submit_status, self.down, self.submitted = run, submit_status, down, []

    async def get_run(self, run_id):
        if self.down:
            raise DurableUnavailable("ConnectError")
        return self.run

    async def submit(self, body):
        self.submitted.append(body)
        return self.submit_status, ""


class Outreach:
    def __init__(self, *, admit=True, sent=True, reason="sent"):
        self.admit, self.sent, self.reason = admit, sent, reason
        self.offers, self.recorded, self.done = [], [], []

    async def economy_admit(self, *, door, key, **kw):
        return self.admit, {}

    async def economy_done(self, *, door, key, sent):
        self.done.append((door, key, sent))

    async def offer_message(self, **kw):
        self.offers.append(kw)
        return {"outreach": self.sent, "reason": self.reason}

    async def record_in_thread(self, **kw):
        self.recorded.append(kw)
        return True


def desk(*, row=ROW, run=None, submit_status=202, outreach=None, delivered=False, durable=None):
    conn = Conn(row, delivered)
    d = IntentionDesk(pool_provider=lambda: Pool(conn), durable=durable or Durable(run, submit_status),
                      outreach_provider=lambda: outreach, clock=lambda: NOW)
    d.conn = conn
    return d


def order(**kw):
    base = {"concern_id": CID, "event_ref": f"turn:{CORR}", "run_id": intention_run_id(CID),
            "correlation_id": intention_correlation_id(CID), "prime_act_id": "act_take_on", "why": "I said I would"}
    return IntentionOrderV1(**{**base, **kw})


def deliver_req():
    return IntentionDeliverV1(concern_id=CID, run_id=intention_run_id(CID), correlation_id=intention_correlation_id(CID),
                              route="chat_thread", route_target=SESSION)


COMPLETED = {"status": "completed", "harness_steps": {"carry": {"text": "I browsed the PRs; two stand out."}}}


# --- order -----------------------------------------------------------------------------------------------------

def test_an_order_submits_one_carry_run_citing_the_concern_with_the_horizon_as_deadline():
    d = desk()
    res = asyncio.run(d.order(order()))
    assert res.status == "submitted" and res.run_id == f"intention-{CID}"
    assert res.route == "chat_thread" and res.route_target == SESSION and res.source == "turn_promise"
    assert res.horizon_at == datetime(2026, 11, 15, 7, 8, 40, tzinfo=timezone.utc)
    [body] = d.durable.submitted
    req = DurableRunRequestV1.model_validate(body)
    assert req.workflow == "intention.carry" and req.run_id == f"intention-{CID}"
    assert req.brief.prompt.startswith(f"prime_concern:{CID}\n\n") and f'"{QUOTE}"' in req.brief.prompt
    assert req.admission.deadline_at == res.horizon_at and req.admission.preferred_lane == "agent"
    assert req.brief.prime_act_id == "act_take_on" and d.conn.calls[0] == ("fetchrow", (CORR,))


def test_an_order_for_an_existing_run_answers_exists_and_submits_nothing():
    d = desk(run={"status": "waiting_resource"})
    res = asyncio.run(d.order(order()))
    assert res.status == "exists" and d.durable.submitted == [] and res.route_target == SESSION


def test_orders_are_refused_with_a_reason_or_unavailable_when_transient():
    assert asyncio.run(desk().order(order(event_ref="orion_day_letter:2026-11-12#thread1"))).status == "refused"
    res = asyncio.run(desk(row={**ROW, "session_id": "orion_intention"}).order(order()))
    assert res.status == "refused" and "loop guard" in res.detail
    late = IntentionDesk(pool_provider=lambda: Pool(Conn(ROW)), durable=Durable(), outreach_provider=lambda: None,
                         clock=lambda: datetime(2026, 11, 15, 6, 0, tzinfo=timezone.utc))
    assert "no room" in asyncio.run(late.order(order())).detail
    assert asyncio.run(desk(submit_status=400).order(order())).status == "refused"
    assert asyncio.run(desk(submit_status=503).order(order())).status == "unavailable"
    assert asyncio.run(desk(submit_status=422).order(order())).status == "unavailable"   # older durable-runs
    assert asyncio.run(desk(durable=Durable(down=True)).order(order())).status == "unavailable"
    no_pool = IntentionDesk(pool_provider=lambda: None, durable=Durable(), outreach_provider=lambda: None)
    assert asyncio.run(no_pool.order(order())).status == "unavailable"


# --- delivery --------------------------------------------------------------------------------------------------

def test_a_sent_reply_goes_into_the_originating_thread_through_the_normal_gates_once():
    out = Outreach()
    res = asyncio.run(desk(run=COMPLETED, outreach=out).deliver(deliver_req()))
    assert res.status == "published" and not res.stored and res.pinged and res.reply_chars > 0
    [offer] = out.offers
    assert offer["session_id"] == SESSION and offer["text"] == "I browsed the PRs; two stand out."
    assert offer["skip_rate_gates"] is True and offer.get("skip_schedule_gates", False) is False   # quiet hours hold
    assert offer["correlation_id"] == intention_correlation_id(CID) and offer["finding_key"] == f"intention:{CID}"
    assert out.recorded == [] and out.done == [("intention", CID, True)]   # the send wrote the history itself


def test_a_reply_held_by_quiet_hours_still_lands_in_the_thread_history():
    # S-1.6: a blocked ping does not lose the reply.
    out = Outreach(sent=False, reason="quiet_hours")
    res = asyncio.run(desk(run=COMPLETED, outreach=out).deliver(deliver_req()))
    assert res.status == "published" and not res.stored and not res.pinged and res.ping_reason == "quiet_hours"
    [rec] = out.recorded
    assert rec["session_id"] == SESSION and rec["text"] == "I browsed the PRs; two stand out."
    assert rec["correlation_id"] == intention_correlation_id(CID)


def test_an_economy_hold_skips_the_ping_but_still_writes_the_thread():
    out = Outreach(admit=False)
    res = asyncio.run(desk(run=COMPLETED, outreach=out).deliver(deliver_req()))
    assert res.status == "published" and not res.pinged and res.ping_reason == "economy_held" and out.offers == []
    assert len(out.recorded) == 1


def test_delivery_is_idempotent_and_waits_for_the_run():
    out = Outreach()
    d = desk(run=COMPLETED, outreach=out, delivered=True)
    res = asyncio.run(d.deliver(deliver_req()))
    assert res.status == "already_delivered" and res.stored and out.offers == [] and out.recorded == []
    assert d.conn.calls[-1] == ("fetchval", DELIVERED_SQL, (intention_correlation_id(CID), SESSION))
    assert asyncio.run(desk(run={"status": "running"}, outreach=out).deliver(deliver_req())).status == "not_ready"
    assert asyncio.run(desk(run={"status": "failed", "error": "x"}, outreach=out).deliver(deliver_req())).status == "failed"
    empty = {"status": "completed", "harness_steps": {"carry": {"text": " "}}}
    assert asyncio.run(desk(run=empty, outreach=out).deliver(deliver_req())).status == "failed"
    assert asyncio.run(desk(run=COMPLETED, outreach=None).deliver(deliver_req())).status == "unavailable"


def test_build_carry_request_is_a_valid_durable_request():
    from orion.prime.intention_sources import turn_promise

    src = turn_promise.adapt(f"turn:{CORR}", ROW, why="w")
    req = build_carry_request(order(), src)
    assert req.admission.deadline_at == src.horizon_at == datetime(2026, 11, 15, 7, 8, 40, tzinfo=timezone.utc)
    assert DurableRunRequestV1.model_validate(req.model_dump(mode="json")).brief.route_target == SESSION
    assert src.horizon_at - NOW > timedelta(hours=40)


def test_a_content_guard_refusal_is_never_written_into_the_thread():
    # Review M4: a PASS or an ungrounded-signal claim is not bypassed through the history write.
    for reason in ("orion_passed", "named_ungrounded_signal", "empty_generation"):
        out = Outreach(sent=False, reason=reason)
        res = asyncio.run(desk(run=COMPLETED, outreach=out).deliver(deliver_req()))
        assert res.status == "failed" and not res.stored and out.recorded == [] and reason in res.detail


def test_a_repeat_call_after_publishing_never_sends_or_writes_again():
    out = Outreach(sent=False, reason="quiet_hours")
    d = desk(run=COMPLETED, outreach=out)
    first = asyncio.run(d.deliver(deliver_req()))
    second = asyncio.run(d.deliver(deliver_req()))
    assert first.status == second.status == "published" and len(out.recorded) == 1 and len(out.offers) == 1
    d.conn.delivered = True                      # sql-writer wrote the row: now it is proof
    assert asyncio.run(d.deliver(deliver_req())).status == "already_delivered"


def test_a_re_sent_order_near_the_horizon_follows_the_existing_run():
    # Review m3: Hub took the first order, Prime never heard; the re-send must not be refused at the horizon.
    late = IntentionDesk(pool_provider=lambda: Pool(Conn(ROW)), durable=Durable(run={"status": "running"}),
                         outreach_provider=lambda: None, clock=lambda: datetime(2026, 11, 15, 6, 0, tzinfo=timezone.utc))
    assert asyncio.run(late.order(order())).status == "exists"

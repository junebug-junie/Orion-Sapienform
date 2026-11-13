"""Intentions Prime takes on or lets go, in the resident loop (app/prime_loop.py; choice-as-EFE Step -1).

take_on -> make act opened -> one carry run ordered from Hub -> run terminal -> reply delivered into the chat thread
-> act closed (1 = stored) and the concern settled. Let go, not before, a failed run, Hub refusing, the horizon and
the flag off. Fake bus, model, Hub and durable-runs; no network."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

from orion.prime import intention as it
from orion.prime.state import PrimeConcernV1, PrimeStateV1
from orion.schemas.intention import IntentionDeliverV1, IntentionOrderV1, intention_run_id
from test_prime_loop import T0, runtime

CID = "d8db9d7b1bba4cf7"
CORR = "a129b72f-e698-425a-a435-3388cd7cd759"
ORDER = "http://hub:8080/api/intention/run"
DELIVER = "http://hub:8080/api/intention/deliver"
RUN = f"http://dr:8121/runs/{intention_run_id(CID)}"
SESSION = "9c91d646-cbd0-4977-8cdd-d2c82045c7f9"
HORIZON = T0 + timedelta(hours=70)


def take_on(not_before=None):
    return {"reading": "I said I would browse the PRs", "concerns": {"keep": [CID]},
            "move": {"kind": "take_on", "concern_id": CID, "why": "I said I would", "not_before": not_before}}


def concern():
    opened = T0 - timedelta(hours=2)
    return PrimeConcernV1(concern_id=CID, text="a thread I opened in a chat with Juniper", event_ref=f"turn:{CORR}",
                          opened_at=opened, last_evidence_at=opened)


def rt_with(answer, *, enabled=True, order_status="submitted", run=None, deliver=None):
    pages = {}
    if run is not None:
        pages[RUN] = run
    rt = runtime(answer=answer, extra_pages=pages)
    rt.s = SimpleNamespace(**{**vars(rt.s), "PRIME_INTENTION_TAKE_ON_ENABLED": enabled})
    rt.pages = pages
    plain_get = rt.get_json
    rt.get_json = lambda url: pages[url] if url in pages else plain_get(url)
    rt.state = PrimeStateV1(concerns=[concern()], pending_changes=["a chat turn opened a thread"])
    rt.orders, rt.deliveries = [], []
    rt.deliver_answer = deliver or {"status": "delivered", "stored": True, "pinged": True, "ping_reason": "sent",
                                    "reply_chars": 420}

    def post_json(url, body):
        if url == ORDER:
            rt.orders.append(IntentionOrderV1.model_validate(body))
            status = order_status(len(rt.orders)) if callable(order_status) else order_status
            if status is None:
                raise TimeoutError("hub slow")
            return {"concern_id": CID, "run_id": body["run_id"] if status in ("submitted", "exists") else None,
                    "status": status, "detail": "no adapter" if status == "refused" else None,
                    **({"source": "turn_promise", "horizon_at": HORIZON.isoformat(), "route": "chat_thread",
                        "route_target": SESSION} if status in ("submitted", "exists") else {})}
        if url == DELIVER:
            rt.deliveries.append(IntentionDeliverV1.model_validate(body))
            return {"concern_id": CID, **rt.deliver_answer}
        raise AssertionError(url)

    rt.post_json = post_json
    return rt


def at(rt, minutes):
    rt.clock_box["now"] = T0 + timedelta(minutes=minutes)
    asyncio.run(rt.tick())


def make_acts(bus):
    return [(kind, a) for kind, a in bus.acts() if a["verb"] == "make"]


def test_take_on_opens_the_act_orders_one_run_delivers_the_reply_and_closes_it_carried():
    rt = rt_with(take_on())
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    assert '"kind": "take_on"' in call["system_prompt"] and "you can take this on" in call["user_prompt"]
    [(kind, act)] = make_acts(rt.bus)
    assert kind == "act_opened" and act["actor"] == "prime" and act["object_ref"] == f"intention:{CID}"
    assert act["test_before"] == "intention_source:turn_promise"
    [order] = rt.orders
    assert order.concern_id == CID and order.event_ref == f"turn:{CORR}" and order.prime_act_id == act["act_id"]
    assert order.run_id == f"intention-{CID}" and order.why == "I said I would"
    [op] = rt.intentions.open
    assert op.order_status == "submitted" and op.route_target == SESSION and op.horizon_at == HORIZON
    saved = it.PrimeIntentionsStateV1.model_validate_json(rt.bus.redis.kv[it.INTENTIONS_KEY])
    assert saved.open[0].act_id == act["act_id"]

    rt.pages[RUN] = {"status": "running"}
    at(rt, 2)
    assert rt.deliveries == [] and rt.intentions.open[0].run_status == "running"
    rt.pages[RUN] = {"status": "completed"}
    at(rt, 4)
    [d] = rt.deliveries
    assert d.route == "chat_thread" and d.route_target == SESSION and d.run_id == order.run_id
    kinds = [k for k, _ in make_acts(rt.bus)]
    assert kinds == ["act_opened", "act_closed"]
    closed = make_acts(rt.bus)[1][1]
    assert closed["outcome"]["value"] == 1.0 and "carried" in closed["outcome"]["note"]
    assert rt.intentions.open == [] and rt.intentions.done[-1].outcome == "carried" and rt.intentions.done[-1].pinged
    # The concern is settled (the hand-back usually does it first; this is the backstop).
    assert rt.state.concerns[0].graded == "settled"
    at(rt, 10)
    assert len(rt.orders) == 1 and len(rt.deliveries) == 1  # one run per take-on, never re-ordered


def test_a_reply_held_by_quiet_hours_still_counts_when_it_is_stored_in_the_thread():
    rt = rt_with(take_on(), run={"status": "completed"},
                 deliver={"status": "delivered", "stored": True, "pinged": False, "ping_reason": "quiet_hours",
                          "reply_chars": 300})
    asyncio.run(rt.tick())
    at(rt, 2)
    closed = make_acts(rt.bus)[-1][1]
    assert closed["outcome"]["value"] == 1.0 and "quiet_hours" in closed["outcome"]["note"]


def test_not_before_holds_the_order_until_its_time():
    rt = rt_with(take_on((T0 + timedelta(hours=3)).isoformat()))
    asyncio.run(rt.tick())
    assert [k for k, _ in make_acts(rt.bus)] == ["act_opened"] and rt.orders == []   # S-1.4
    at(rt, 60)
    assert rt.orders == []
    at(rt, 181)
    assert len(rt.orders) == 1 and rt.intentions.open[0].order_status == "submitted"


def test_dropping_a_concern_taken_on_lets_it_go_and_nothing_is_ordered_or_delivered():
    rt = rt_with(take_on((T0 + timedelta(hours=3)).isoformat()))
    asyncio.run(rt.tick())
    rt.llm.answer = {"reading": "Juniper said never mind", "concerns": {"drop": [{"concern_id": CID,
                                                                                  "why": "she said never mind"}]}}
    rt.state = rt.state.model_copy(update={"pending_changes": ["Juniper wrote"]})
    at(rt, 5)
    closed = make_acts(rt.bus)[-1]
    assert closed[0] == "act_closed" and closed[1]["outcome"]["value"] == 0.0
    assert "let go: she said never mind" in closed[1]["outcome"]["note"]
    assert rt.intentions.done[-1].outcome == "let_go" and rt.state.concerns == []
    at(rt, 200)
    assert rt.orders == [] and rt.deliveries == []


def test_a_failed_run_closes_the_act_at_zero_and_is_not_retried():
    rt = rt_with(take_on(), run={"status": "failed", "error": "harness_step_failed"})
    asyncio.run(rt.tick())
    at(rt, 2)
    closed = make_acts(rt.bus)[-1][1]
    assert closed["outcome"]["value"] == 0.0 and "failed" in closed["outcome"]["note"]
    assert rt.deliveries == [] and rt.intentions.done[-1].outcome == "not_carried"
    at(rt, 10)
    assert len(rt.orders) == 1


def test_hub_refusing_the_order_closes_the_act_and_prime_is_told():
    rt = rt_with(take_on(), order_status="refused")
    asyncio.run(rt.tick())
    closed = make_acts(rt.bus)[-1][1]
    assert closed["outcome"]["value"] == 0.0 and "Hub refused the order" in closed["outcome"]["note"]
    assert any("Hub refused" in line for line in rt.state.recent_outcomes)


def test_an_unanswered_order_is_sent_again_on_the_next_check():
    rt = rt_with(take_on(), order_status=lambda n: None if n == 1 else "submitted")
    asyncio.run(rt.tick())
    assert rt.intentions.open[0].order_status == "unavailable"
    at(rt, 2)
    assert len(rt.orders) == 2 and rt.intentions.open[0].order_status == "submitted"


def test_a_run_still_waiting_past_the_horizon_expires():
    rt = rt_with(take_on(), run={"status": "waiting_resource"})
    asyncio.run(rt.tick())
    at(rt, 70 * 60 + 61)
    closed = make_acts(rt.bus)[-1][1]
    assert closed["outcome"]["value"] == 0.0 and "past the horizon" in closed["outcome"]["note"]
    assert rt.intentions.done[-1].outcome == "expired"


def test_a_concern_taken_on_does_not_fade_while_its_run_waits():
    rt = rt_with(take_on(), run={"status": "waiting_resource"})
    asyncio.run(rt.tick())
    at(rt, 26 * 60)   # past the 24 h concern TTL with no thought keeping it
    assert [c.concern_id for c in rt.state.concerns] == [CID]


def test_flag_off_offers_no_move_orders_nothing_and_saves_nothing():
    rt = rt_with(take_on(), enabled=False)
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    assert "take_on" not in call["system_prompt"] and "take this on" not in call["user_prompt"]
    assert make_acts(rt.bus) == [] and rt.orders == [] and it.INTENTIONS_KEY not in rt.bus.redis.kv
    assert any("switched off" in line for line in rt.state.recent_outcomes)

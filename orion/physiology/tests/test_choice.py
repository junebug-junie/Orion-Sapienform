"""Organ 4 Choice (Patch C1, SHADOW): net = weight x rank - sum price x cost, placement, reservations,
the exploration floor, Tier 1's reserved share, and never funding on a missing market."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from orion.physiology import client
from orion.physiology.appetite import ActRecord
from orion.physiology.choice import (
    MarketContext,
    ReservationBook,
    build_context,
    decide,
    tier1_reserved_share,
)
from orion.physiology.families import load_families
from orion.physiology.weights import FamilyWeight, Evidence
from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R, PhysiologyStateV1
from orion.schemas.physiology_want import WantDecisionV1, WantV1

NOW = datetime(2026, 10, 27, 6, 0, tzinfo=timezone.utc)
FAMS, _ = load_families("config/physiology/families.yaml")


def price(v, state="slack", horizon=3600.0):
    return {"value": v, "ceiling": 1.0, "bind_state": state, "horizon_sec": horizon, "why": ""}


def lane(role, *, level=0.0, slots=1, p=0.0, queued=0, waiting=0.0, state="slack"):
    return R(constraint=f"lane:{role}", unit="slot_s", level=level, budget=slots * 3600.0, budget_kind="capacity",
             detail={"slots": slots, "lease_s": level, "window_sec": 3600.0, "deferrable_queued": queued,
                     "deferrable_waiting_sec": waiting}, price=price(p, state))


def body(*readings, at=NOW, juniper_sent=0.0, juniper_p=0.0):
    base = [R(constraint="heat", unit="c", level=28.0, budget=31.0, budget_kind="soft_line", price=price(0.0, horizon=21600.0)),
            R(constraint="attention:juniper", unit="messages", level=juniper_sent, budget=4.0, budget_kind="daily_cap",
              price=price(juniper_p, horizon=86400.0))]
    return PhysiologyStateV1(tick_id=f"physiology:{at.isoformat()}", generated_at=at, constraints=base + list(readings))


def history(family, n=12, *, gap_h=2.0, slot=None, last_h=3.0):
    out = []
    for i in range(n):
        closed = NOW - timedelta(hours=last_h + gap_h * i)
        cost = {"seconds": 1000.0, "gpu_slot_s_by_lane": slot} if slot else {"seconds": 30.0}
        out.append(ActRecord(act_id=f"{family}{i}", family=family, opened_at=closed - timedelta(seconds=60),
                             closed_at=closed, outcome_value=1.0, outcome_source="producer", cost=cost))
    return out


def ctx_for(acts, state, *, shares=None, weights=None):
    c = build_context(FAMS, acts, [], state, NOW, weights=weights or {})
    if shares is not None:
        c = MarketContext(families=c.families, appetite=c.appetite, weights=c.weights, tier1_share=c.tier1_share,
                          funded_share=shares)
    return c


def want(family="durable.self_inquiry", tier=2, i=0, verb="ask", **kw):
    return WantV1(want_id=f"w{i}", family=family, producer="test", verb=verb, tier=tier, **kw)


SELF = history("durable.self_inquiry", slot={"agent": 1800.0})


def test_hungry_want_at_zero_prices_is_funded_shadow_and_reserves_its_lane():
    st = body(lane("agent"), lane("agent-gpu2", slots=2))
    book = ReservationBook()
    d = decide(want(), st, ctx_for(SELF, st), book, NOW)
    assert d.shadow is True and d.outcome == "funded" and d.reason == "net_positive"
    assert d.net > 0 and d.charge == 0.0 and d.lane == "agent"
    assert d.reserved == {"lane:agent": 1800.0} and d.reservation_expires_at is not None
    assert any(w.startswith("unpriced:heat") for w in d.warnings)
    assert book.total("lane:agent", NOW) == 1800.0
    assert book.release("w0") == 1 and book.total("lane:agent", NOW) == 0.0


def test_placement_picks_the_cheapest_acceptable_lane():
    st = body(lane("agent", p=0.5), lane("agent-gpu2", slots=2, p=0.1))
    d = decide(want(), st, ctx_for(SELF, st), ReservationBook(), NOW)
    assert d.lane == "agent-gpu2" and d.cost_frac["lane:agent-gpu2"] == pytest.approx(1800 / 7200)


def test_price_deferral_names_the_binding_constraint_and_when_to_retry():
    st = body(lane("agent", p=1.0, state="binding"))
    w = {"durable.self_inquiry": FamilyWeight("durable.self_inquiry", 2, 0.2, 0.2, "evidence", Evidence(0.0, 40, None, 0))}
    d = decide(want(), st, ctx_for(SELF, st, shares={"durable.self_inquiry": 0.5}, weights=w), ReservationBook(), NOW)
    assert d.outcome == "deferred" and d.reason == "price" and d.binding_constraint == "lane:agent"
    assert d.net < 0 and d.retry_after_sec is not None and d.retry_after_sec > 0
    assert d.prices_seen["lane:agent"] == 1.0


def test_exploration_floor_funds_a_starved_family_on_price_but_never_past_capacity():
    st = body(lane("agent", p=1.0, state="binding"))
    w = {"durable.self_inquiry": FamilyWeight("durable.self_inquiry", 2, 0.2, 0.2, "evidence", Evidence(0.0, 40, None, 0))}
    d = decide(want(), st, ctx_for(SELF, st, shares={"durable.self_inquiry": 0.01}, weights=w), ReservationBook(), NOW)
    assert d.outcome == "funded" and d.reason == "exploration_floor" and d.net < 0
    full = body(lane("agent", level=3500.0, p=1.0, state="binding"))
    d2 = decide(want(), full, ctx_for(SELF, full, shares={"durable.self_inquiry": 0.01}, weights=w), ReservationBook(), NOW)
    assert d2.outcome == "deferred" and d2.reason == "capacity_reserved"


def test_r6_twenty_simultaneous_bids_at_a_zero_price_cannot_over_reserve_the_lane():
    """Replay check R6 in miniature: prices are stale at 0 inside one tick; reservations stop a burst."""
    st = body(lane("agent"))
    book = ReservationBook()
    ctx = ctx_for(SELF, st)
    ds = [decide(want(i=i), st, ctx, book, NOW) for i in range(20)]
    funded = [d for d in ds if d.outcome == "funded"]
    assert book.total("lane:agent", NOW) <= 1.2 * 3600.0
    assert len(funded) == 2 and all(d.reason == "capacity_reserved" for d in ds if d.outcome == "deferred")
    late = [d for d in ds if d.outcome == "deferred"][0]
    assert set(late.displaced_by) == {"w0", "w1"} and late.retry_after_sec > 0


def test_juniper_cap_is_a_capacity_line_for_outreach():
    acts = history("hub.outreach", slot={"chat": 30.0})
    st = body(lane("chat"), juniper_sent=4.0)
    d = decide(want("hub.outreach", verb="say"), st, ctx_for(acts, st), ReservationBook(), NOW)
    assert d.outcome == "deferred" and d.reason == "capacity_reserved" and d.binding_constraint == "attention:juniper"


def test_tier0_never_bids_unknown_family_and_missing_market_are_never_funded():
    st = body(lane("agent"))
    book = ReservationBook()
    t0 = decide(want("hub.chat_reply", tier=1), st, ctx_for([], st), book, NOW)
    assert t0.outcome == "deferred" and t0.reason == "family_never_bids" and t0.tier == 0
    unk = decide(want("made.up"), st, ctx_for([], st), book, NOW)
    assert unk.outcome == "deferred" and unk.reason == "unknown_family"
    stale = body(lane("agent"), at=NOW - timedelta(seconds=400))
    d = decide(want(), stale, ctx_for(SELF, stale), book, NOW)
    assert d.outcome == "deferred" and d.reason == "physiology_unavailable" and d.retry_after_sec == 120.0
    assert decide(want(), None, ctx_for(SELF, None), book, NOW).reason == "physiology_unavailable"
    with pytest.raises(Exception):
        WantV1(want_id="x", family="hub.chat_reply", producer="p", verb="say", tier=0)


def test_tier1_pays_no_lane_price_inside_its_reserved_share():
    journal = []
    for h in range(14 * 24):
        closed = NOW - timedelta(hours=h, minutes=30)
        journal.append(ActRecord(act_id=f"j{h}", family="actions.journal", opened_at=closed - timedelta(seconds=10),
                                 closed_at=closed, outcome_value=0.0, outcome_source="producer",
                                 cost={"seconds": 10.0, "gpu_slot_s_by_lane": {"metacog": 360.0}}))
    share = tier1_reserved_share(journal, FAMS, "metacog", 4 * 3600.0, NOW)
    assert share == pytest.approx(360.0 / 14400.0)
    st = body(lane("metacog", slots=4, p=1.0, state="binding"))
    d = decide(want("actions.journal", tier=1, verb="make"), st, ctx_for(journal, st), ReservationBook(), NOW)
    assert d.lane == "metacog" and d.charge == 0.0 and d.outcome == "funded"
    assert tier1_reserved_share(journal * 50, FAMS, "metacog", 4 * 3600.0, NOW) == 0.5   # capped


def test_unloaded_swap_seat_is_priced_on_a_nominal_slot_and_warned():
    acts = history("thought.painting", slot={"diffusion": 60.0})
    st = body(R(constraint="lane:diffusion", unit="slot_s", level=0.0, budget=0.0, budget_kind="capacity",
                detail={"slots": 0, "window_sec": 3600.0, "deferrable_queued": 1, "deferrable_waiting_sec": 900.0},
                price=price(0.0)))
    d = decide(want("thought.painting", verb="make"), st, ctx_for(acts, st), ReservationBook(), NOW)
    assert d.outcome == "funded" and d.lane == "diffusion" and "swap_seat_unloaded:diffusion" in d.warnings


def test_decision_round_trips_through_its_schema():
    st = body(lane("agent"))
    d = decide(want(), st, ctx_for(SELF, st), ReservationBook(), NOW)
    assert WantDecisionV1.model_validate(json.loads(d.model_dump_json())) == d


# --- client: never fail open to "funded" ---------------------------------------------------------

def test_client_returns_none_on_any_failure(monkeypatch):
    monkeypatch.setenv("ORION_PHYSIOLOGY_BID_URL", "http://127.0.0.1:9/physiology/bid")
    assert client.bid(want(), timeout_sec=0.2) is None
    monkeypatch.setenv("ORION_PHYSIOLOGY_BID_CLIENT_ENABLED", "false")
    assert client.bid(want()) is None


def _serve(body_bytes: bytes, status: int = 200):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body_bytes)

        def log_message(self, *a):  # noqa: D401
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/physiology/bid"


def test_client_parses_a_decision_and_refuses_malformed_or_foreign_answers():
    st = body(lane("agent"))
    d = decide(want(), st, ctx_for(SELF, st), ReservationBook(), NOW)
    srv, url = _serve(d.model_dump_json().encode())
    got = client.bid(want(), url=url, timeout_sec=2.0)
    srv.server_close()
    assert got is not None and got.shadow is True and got.outcome == "funded"
    srv, url = _serve(b'{"not": "a decision"}')
    assert client.bid(want(), url=url, timeout_sec=2.0) is None
    srv.server_close()
    srv, url = _serve(d.model_dump_json().encode())
    assert client.bid(want(i=7), url=url, timeout_sec=2.0) is None   # an answer for another want
    srv.server_close()

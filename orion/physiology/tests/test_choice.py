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
    FundedWant,
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
from pathlib import Path

FAMS, _ = load_families(Path(__file__).resolve().parents[3] / "config/physiology/families.yaml")


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


def ctx_for(acts, state, *, weights=None):
    return build_context(FAMS, acts, [], state, NOW, weights=weights or {})


def seeded_book(counts, *, at=NOW - timedelta(days=1)):
    """A book whose rolling week of funded wants (as read from want_ledger) holds ``counts`` per family."""
    book = ReservationBook()
    rows = [FundedWant(f"{f}:{i}@x", at, f, FAMS.get(f).tier) for f, n in counts.items() for i in range(n)]
    book.seed_funded(rows, NOW)
    return book


# Tier 2 funded 100 wants this week; self_inquiry got 1 of them (1%: under the 5% quota).
STARVED = {"durable.self_inquiry": 1, "durable.curiosity_investigate": 99}


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
    st = body(lane("agent", p=1.0, state="slack"))
    w = {"durable.self_inquiry": FamilyWeight("durable.self_inquiry", 2, 0.2, 0.2, "evidence", Evidence(0.0, 40, None, 0))}
    d = decide(want(), st, ctx_for(SELF, st, weights=w), ReservationBook(), NOW)
    assert d.outcome == "deferred" and d.reason == "price" and d.binding_constraint == "lane:agent"
    # A slack price decays at least ceiling / horizon: 1.0 -> needed price within the hour.
    assert d.net < 0 and d.retry_after_sec is not None and 0 < d.retry_after_sec <= 3600
    assert d.prices_seen["lane:agent"] == 1.0
    # A binding price is not falling: no retry time is promised (review finding 7).
    st2 = body(lane("agent", p=1.0, state="binding"))
    d2 = decide(want(), st2, ctx_for(SELF, st2, weights=w), ReservationBook(), NOW)
    assert d2.reason == "price" and d2.retry_after_sec is None


def test_exploration_floor_funds_a_starved_family_on_price_but_never_past_capacity():
    st = body(lane("agent", p=1.0, state="binding"))
    w = {"durable.self_inquiry": FamilyWeight("durable.self_inquiry", 2, 0.2, 0.2, "evidence", Evidence(0.0, 40, None, 0))}
    d = decide(want(), st, ctx_for(SELF, st, weights=w), seeded_book(STARVED), NOW)
    assert d.outcome == "funded" and d.reason == "exploration_floor" and d.net < 0
    full = body(lane("agent", level=3500.0, p=1.0, state="binding"))
    d2 = decide(want(), full, ctx_for(SELF, full, weights=w), seeded_book(STARVED), NOW)
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
    for h in range(1, 14 * 24):
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


def test_a_hold_longer_than_the_window_costs_one_window_and_fits_an_empty_one_slot_lane():
    """Replay finding: a 5,600 s curiosity hold priced at 1.6 of a 1-slot lane's hour could never fit
    an empty lane. One act holds one slot: inside the window it costs at most that window."""
    long = history("durable.curiosity_investigate", slot={"agent": 5600.0})
    st = body(lane("agent"))
    d = decide(want("durable.curiosity_investigate", verb="infer"), st, ctx_for(long, st), ReservationBook(), NOW)
    assert d.outcome == "funded" and d.reserved == {"lane:agent": 3600.0} and d.cost_frac["lane:agent"] == 1.0


def test_a_family_that_just_acted_is_sated_not_price_deferred():
    """Zero prices and zero hunger: net 0 is not > 0, but nothing binds. Named 'sated', never 'price'."""
    fresh = history("durable.self_inquiry", slot={"agent": 600.0}, last_h=0.0)
    st = body(lane("agent"))
    d = decide(want(), st, ctx_for(fresh, st), ReservationBook(), NOW)
    assert d.outcome == "deferred" and d.reason == "sated" and d.binding_constraint is None and d.intensity == 0.0


def test_a_live_seat_wins_over_an_unloaded_one_even_when_the_unloaded_one_is_cheaper():
    """Live 10-27 06:11 sample: curiosity was placed on agent-deep (hecate asleep, 0 slots) because its
    stale price was 0. Waking or loading a seat is not priced in C1, so a live seat that fits wins."""
    st = body(lane("agent", p=0.2), R(constraint="lane:agent-deep", unit="slot_s", level=0.0, budget=0.0,
                                      budget_kind="capacity", detail={"slots": 0, "window_sec": 3600.0},
                                      price=price(0.0)))
    d = decide(want(), st, ctx_for(SELF, st), ReservationBook(), NOW)
    assert d.lane == "agent" and d.outcome == "funded"


# --- review findings (C1 code review, 2026-10-27) ------------------------------------------------

def test_a_gpu_family_with_no_slot_history_is_priced_at_one_window_never_free():
    """Finding 1: no measured slot time used to skip placement, charge and capacity entirely."""
    bare = history("durable.self_inquiry")          # acts with seconds only, no gpu_slot_s_by_lane
    full = body(lane("agent", level=3600.0, p=1.0, state="binding"), lane("agent-gpu2", slots=2, level=7200.0, p=1.0))
    d = decide(want(), full, ctx_for(bare, full), ReservationBook(), NOW)
    assert d.outcome == "deferred" and d.reason == "capacity_reserved"
    assert "lane_cost_unmeasured:priced_at_one_window" in d.warnings


def test_a_lane_with_no_reading_is_never_an_empty_lane():
    """Finding 2: a no_reading lane kept a budget and counted as level 0."""
    r = R(constraint="lane:agent", unit="slot_s", no_reading="stale:gpu_pool_events", budget=3600.0,
          budget_kind="capacity", detail={"slots": 1}, price=price(0.0, "no_reading"))
    st = body(r)
    d = decide(want(), st, ctx_for(SELF, st), ReservationBook(), NOW)
    assert d.outcome == "deferred" and d.reason == "physiology_unavailable" and "lane_no_reading:agent" in d.warnings


def test_full_live_seat_does_not_spill_onto_an_unloaded_swap_seat():
    """Finding 3: agent full, agent-gpu2 unloaded -> was funded on agent-gpu2 (evicting diffusion)."""
    st = body(lane("agent", level=3600.0), R(constraint="lane:agent-gpu2", unit="slot_s", level=0.0, budget=0.0,
                                            budget_kind="capacity", detail={"slots": 0}, price=price(0.0)))
    d = decide(want(), st, ctx_for(SELF, st), ReservationBook(), NOW)
    assert d.outcome == "deferred" and d.reason == "capacity_reserved"


def test_tier2_line_does_not_subtract_tier1_share_twice():
    """Finding 4: metacog 4 slots, Tier 1 share 0.5 and Tier 1 already using its share in the measured
    level: Tier 2 may still use the free other half."""
    from orion.physiology.choice import MarketContext
    tf = history("topic_foundry.run", slot={"metacog": 600.0})
    st = body(lane("metacog", slots=4, level=7200.0))
    c = build_context(FAMS, tf, [], st, NOW, weights={})
    c = MarketContext(families=c.families, appetite=c.appetite, weights={}, tier1_share={"metacog": 0.5},
                      tier1_recent_use={"metacog": 7200.0})
    d = decide(want("topic_foundry.run", verb="maintain"), st, c, ReservationBook(), NOW)
    assert d.outcome == "funded" and d.lane == "metacog"
    # With Tier 1 idle, Tier 2 must leave Tier 1's 7,200 s free: 7,200 measured + 600 > 7,200 line.
    c2 = MarketContext(families=c.families, appetite=c.appetite, weights={}, tier1_share={"metacog": 0.5},
                       tier1_recent_use={})
    d2 = decide(want("topic_foundry.run", verb="maintain", i=2), st, c2, ReservationBook(), NOW)
    assert d2.reason == "capacity_reserved"


def test_exploration_floor_is_a_weekly_quota_not_once_per_window():
    """C1's R7 failure: one floor fund per lane window could not hold 5%. Now the floor funds a
    starved Tier 2 family on every hungry bid until its rolling-week share of Tier 2's funded wants
    reaches 5%, then stops by itself."""
    st = body(lane("agent", p=1.0, state="binding"), lane("agent-gpu2", slots=2, p=1.0, state="binding"),
              lane("agent-deep", p=1.0, state="binding"))
    w = {"durable.self_inquiry": FamilyWeight("durable.self_inquiry", 2, 0.2, 0.2, "evidence", Evidence(0.0, 40, None, 0))}
    ctx = ctx_for(SELF, st, weights=w)
    book = seeded_book(STARVED)
    reasons = []
    for i in range(8):
        t = NOW + timedelta(seconds=30 * i)   # inside one fresh body tick
        d = decide(want(i=i), st, ctx, book, t)
        book.release(d.want_id)          # the act ran and closed: capacity is not what binds here
        reasons.append(d.reason)
    # 1/100 -> 2/101 -> 3/102 -> 4/103 -> 5/104 (4.8%) -> 6/105 (5.7%): five floor funds, then price.
    assert reasons[:5] == ["exploration_floor"] * 5 and reasons[5:] == ["price"] * 3
    assert book.funded_share("durable.self_inquiry", 2, NOW + timedelta(hours=1)) == pytest.approx(6 / 105)


def test_exploration_floor_counts_the_rolling_week_and_dedups_ledger_rows():
    book = ReservationBook()
    old = NOW - timedelta(days=8)
    book.seed_funded([FundedWant("a@1", old, "durable.self_inquiry", 2),
                      FundedWant("b@1", NOW - timedelta(days=1), "durable.curiosity_investigate", 2)], NOW)
    # The 8-day-old row fell out of the week; self_inquiry has 0 of Tier 2's 1.
    assert book.funded_share("durable.self_inquiry", 2, NOW) == 0.0
    # The process's own decision, then the ledger catching up with the same decision_id: counted once.
    book.note_funded("c@1", "durable.self_inquiry", 2, NOW)
    book.seed_funded([FundedWant("c@1", NOW, "durable.self_inquiry", 2)], NOW)
    assert book.funded_share("durable.self_inquiry", 2, NOW) == pytest.approx(0.5)
    # Tier 1 never counts toward Tier 2, and a tier with no funded wants has no share.
    assert book.funded_share("actions.journal", 1, NOW) is None


def test_exploration_floor_is_tier2_only_and_off_until_the_ledger_was_read():
    st = body(lane("agent", p=1.0, state="binding"), lane("agent-gpu2", slots=2, p=1.0, state="binding"),
              lane("agent-deep", p=1.0, state="binding"))
    w = {"durable.self_inquiry": FamilyWeight("durable.self_inquiry", 2, 0.2, 0.2, "evidence", Evidence(0.0, 40, None, 0))}
    unread = decide(want(), st, ctx_for(SELF, st, weights=w), ReservationBook(), NOW)
    assert unread.reason == "price"
    journal = history("actions.journal", slot={"metacog": 900.0})
    st1 = body(lane("metacog", slots=4, p=1.0, state="binding"), lane("fast", slots=4, p=1.0, state="binding"),
               lane("agent", p=1.0, state="binding"), lane("agent-gpu2", slots=2, p=1.0, state="binding"))
    wj = {"actions.journal": FamilyWeight("actions.journal", 1, 0.01, 0.01, "evidence", Evidence(0.0, 40, None, 0))}
    d = decide(want("actions.journal", tier=1, verb="make"), st1, ctx_for(journal, st1, weights=wj),
               seeded_book({"durable.journal_compose": 50}), NOW)
    assert d.reason != "exploration_floor"


def test_reservation_ends_at_the_wants_window_if_no_act_opens():
    """C1 concern: a shadow-funded want whose real act never runs must not hold capacity for an hour."""
    st = body(lane("agent"), lane("agent-gpu2", slots=2))
    book = ReservationBook()
    latest = NOW + timedelta(minutes=10)
    d = decide(want(window={"latest": latest}), st, ctx_for(SELF, st), book, NOW)
    assert d.outcome == "funded" and d.reservation_expires_at == latest
    assert book.total("lane:agent", NOW + timedelta(minutes=9)) > 0
    assert book.total("lane:agent", NOW + timedelta(minutes=10, seconds=1)) == 0.0


def test_reservation_reanchors_at_the_open_and_releases_at_the_close():
    st = body(lane("agent"), lane("agent-gpu2", slots=2))
    book = ReservationBook()
    d = decide(want(window={"latest": NOW + timedelta(minutes=10)}), st, ctx_for(SELF, st), book, NOW)
    ttl = (book.holders("lane:agent", NOW)[0].ttl_sec)
    assert ttl == pytest.approx(1000.0)        # one act-length: the family's p90 act seconds (history: 1,000 s)
    opened_at = NOW + timedelta(minutes=8)
    assert book.mark_opened(d.want_id, opened_at) == 1
    # Past the want's window, but the act is running: the reservation stays one act-length from the open.
    assert book.total("lane:agent", NOW + timedelta(minutes=20)) > 0
    assert book.holders("lane:agent", NOW + timedelta(minutes=20))[0].opened is True
    assert book.release(d.want_id) == 1
    assert book.total("lane:agent", NOW + timedelta(minutes=20)) == 0.0
    # A reservation that already lapsed is not revived by a late open.
    d2 = decide(want(i=9, window={"latest": NOW + timedelta(minutes=1)}), st, ctx_for(SELF, st), book, NOW)
    assert book.mark_opened(d2.want_id, NOW + timedelta(minutes=5)) == 0


def test_a_want_whose_window_already_passed_reserves_nothing():
    st = body(lane("agent"), lane("agent-gpu2", slots=2))
    book = ReservationBook()
    d = decide(want(window={"latest": NOW - timedelta(seconds=1)}), st, ctx_for(SELF, st), book, NOW)
    assert d.outcome == "funded" and d.reserved == {} and "window_passed:no_reservation" in d.warnings
    assert book.total("lane:agent", NOW) == 0.0
def test_client_returns_none_on_a_broken_http_response():
    """Finding 5: an http.client error on read (bad status line) must not raise into a producer."""
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        conn.recv(65536)
        conn.sendall(b"NOT-HTTP garbage\r\n\r\n")
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    assert client.bid(want(), url=f"http://127.0.0.1:{port}/physiology/bid", timeout_sec=2.0) is None
    srv.close()


def test_client_release_posts_to_the_want_and_returns_none_on_any_failure(monkeypatch):
    """Patch C3a: a dispatch that returned releases its want's reservation (the act closed)."""
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["path"] = self.path
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"want_id": "x", "released": 2}')

        def log_message(self, *a):  # noqa: D401
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/physiology/bid"
    assert client.release("orch:c/1:dream_cycle", url=url, timeout_sec=2.0) == 2
    srv.server_close()
    assert seen["path"] == "/physiology/bid/orch%3Ac%2F1%3Adream_cycle/release"
    assert client.release("w", url="http://127.0.0.1:9/physiology/bid", timeout_sec=0.2) is None
    monkeypatch.setenv("ORION_PHYSIOLOGY_BID_CLIENT_ENABLED", "false")
    assert client.release("w", url=url) is None

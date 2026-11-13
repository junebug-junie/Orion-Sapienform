"""Attention patch 4b: a camera ask bids before its card opens, and obeys (app/ask_bind.py).

Lifecycle with a fake connection: a candidate bids -> funded opens; held does not open, the backoff is respected,
a later re-bid can fund and open; no answer (timeout) admits; ``shadow`` is the behaviour before 4b; Juniper's own
card never bids. Plus the no-deadline rule (issue #3285), the one-want-per-ask rule for the shadow loop, and a static
guard that every camera opener goes through the bid site.

Decision shapes are copied from a live ``want_ledger`` row of family vision.ask (read-only, 2026-11-13); the ask
shapes from live ``orion_ask`` rows (a vehicle "seen N times" card and an "Is this <name>?" card).
"""
from __future__ import annotations

import ast
import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import ask_bind as ab
from app.ask_event_wants import ASKS_SQL, GRADES_SQL, plan_asks, run_one_cycle
from app.settings import settings
from orion.grammar.camera_ask_acts import (
    BID_WANT_REF_PREFIX, CANDIDATE_SOURCE, JUNIPER_STORY_CARD_REF, REBID_CAP_SEC, REBID_FIRST_SEC, bid_want_ref,
    candidate_key, rebid_after_sec,
)
from orion.physiology.binding import BINDING_PRODUCERS
from orion.physiology.event_wants import event_want_id
from orion.schemas.ask import OrionAskV1
from orion.schemas.physiology_want import WantDecisionV1

NOW = datetime(2026, 11, 13, 14, 0, 53, tzinfo=timezone.utc)
VEHICLE = "420762ba8f4b4e28af57ffe1e66c2889"
PERSON = "10da7f9891364980bb6d3bc9e65b6340"

# A live funded vision.ask decision (want_ledger, 2026-11-13 07:05Z), want id replaced per bid.
LIVE_FUNDED = {
    "net": 1.101074, "why": "vision.ask: weight 1.10 (prior) x rank 1.00 (event:graded) - charge 0.000 = net +1.101",
    "lane": None, "tier": 2, "verb": "ask", "charge": 0.0, "family": "vision.ask", "graded": True,
    "reason": "net_positive", "shadow": True, "outcome": "funded", "reserved": {}, "cost_frac": {}, "displaced": [],
    "intensity": 1.0, "value_rank": 1.0, "decided_at": "2026-11-13T07:05:22.041219Z",
    "value_basis": "event:graded", "displaced_by": [], "weight_basis": "prior", "family_weight": 1.101074,
    "schema_version": "physiology.want.decision.v1", "retry_after_sec": None, "binding_constraint": None,
}


def _decision(want, **over):
    d = {**copy.deepcopy(LIVE_FUNDED), "want": want.model_dump(mode="json"), "want_id": want.want_id,
         "producer": want.producer, "decision_id": f"{want.want_id}@x", **over}
    return WantDecisionV1.model_validate(d)


def funded(want):
    return _decision(want)


def sated(want):
    return _decision(want, outcome="deferred", reason="sated", net=0.0, value_rank=0.0,
                     why="nothing charged, but the event is no bigger than this family's usual events")


def unavailable(want):
    return _decision(want, outcome="deferred", reason="physiology_unavailable", net=None, value_rank=None,
                     binding_constraint="physiology_unavailable", retry_after_sec=300.0)


class _Res:
    def __init__(self, rows=None, scalar=None):
        self._rows, self._scalar = rows or [], scalar

    def scalar(self):
        return self._scalar

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def mappings(self):
        return self

    def __iter__(self):
        return iter(self._rows)

    def all(self):
        return self._rows


class FakeConn:
    """Answers the four reads ask_bind makes, and records the card inserts."""

    def __init__(self, *, kinds=None, sightings=None, history_n=40, grades=None):
        self.kinds = kinds or {VEHICLE: "vehicle", PERSON: "person"}
        self.sightings = sightings or {VEHICLE: 86, PERSON: 9}
        # 30 days of Orion-made asks' sightings: 2..41, so 86 ranks 1.0 and 9 ranks 7/40.
        self.history = [{"sightings": 2 + i} for i in range(history_n)]
        self.grades = grades if grades is not None else (
            [("camera_ask:confirm", 1.0)] * 141 + [("camera_ask:confirm", 0.0)] * 4
            + [("camera_ask:vehicle", 1.0)] * 5 + [("camera_ask:vehicle", 0.0)] * 40)
        self.inserted: list[dict] = []

    def execute(self, clause, params=None):
        sql = str(clause)
        if sql == ab.KIND_SQL:
            return _Res(scalar=self.kinds.get(params["i"]))
        if sql == ab.SIGHTINGS_SQL:
            return _Res(scalar=self.sightings.get(params["i"], 0))
        if sql == ASKS_SQL:
            return _Res(rows=self.history)
        if sql == GRADES_SQL:
            return _Res(rows=self.grades)
        if "INSERT INTO orion_ask" in sql:
            if any(r["sk"] == params["sk"] and r["sr"] == params["sr"] for r in self.inserted):
                return _Res(rows=[])                     # the open-card unique index
            self.inserted.append(params)
            return _Res(rows=[(params["id"],)])
        raise AssertionError(f"unexpected SQL: {sql[:80]}")


class Market:
    """Records every bid; answers with ``answer(want)``."""

    def __init__(self, answer):
        self.answer, self.wants = answer, []

    def __call__(self, want):
        self.wants.append(want)
        return self.answer(want)


def vehicle_ask(now=NOW, ask_id="ask-5c5a0cca61174a35905ed71b5c97970c", refs=None):
    return OrionAskV1(ask_id=ask_id, question="I have seen this same vehicle 86 times over 12 days. Whose is it?",
                      evidence_refs=refs if refs is not None else [f"sighting:9b55d8b2{i}" for i in range(5)],
                      image_ref="thumb:74cd", image_refs=["thumb:74cd"], created_at=now,
                      expires_at=now + timedelta(days=7), source_kind="vision_individual", source_ref=VEHICLE)


def confirm_ask(now=NOW, ask_id="ask-0aba677f3c0043d1bf32308fbb2e753a"):
    return OrionAskV1(ask_id=ask_id, question="Is this Juniper? It is my best guess, but I am not sure.",
                      evidence_refs=["sighting:18cc2b1a", "sighting:d5bd6387"], image_ref="thumb:ccdb",
                      image_refs=["thumb:ccdb"], created_at=now, expires_at=now + timedelta(days=7),
                      source_kind="vision_person_confirm", source_ref=f"{PERSON}|Juniper")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    ab.reset_state()
    monkeypatch.setattr(settings, "orion_physiology_bid_camera_asks", "on")
    monkeypatch.setattr(settings, "sql_writer_act_camera_asks_enabled", True)
    yield
    ab.reset_state()


# --- the want -----------------------------------------------------------------------------------

def test_the_bid_is_a_binding_event_want_keyed_by_candidate_not_by_ask_id():
    conn, market = FakeConn(), Market(funded)
    ab.ask_bind(conn, vehicle_ask(ask_id="ask-a"), now=NOW, bid=market)
    ab.reset_state()
    ab.ask_bind(conn, vehicle_ask(ask_id="ask-b"), now=NOW, bid=market)
    w1, w2 = market.wants
    assert w1.want_id == w2.want_id == event_want_id(CANDIDATE_SOURCE, candidate_key("vehicle", "vision_individual",
                                                                                     VEHICLE))
    assert w1.producer == ab.PRODUCER and ab.PRODUCER in BINDING_PRODUCERS
    assert w1.family == "vision.ask" and w1.verb == "ask" and w1.tier == 2
    # Prime's menu labels and expires by the card's own ref.
    assert (w1.event_ref, w2.event_ref) == ("orion_ask:ask-a", "orion_ask:ask-b")
    assert w1.object_ref == f"vision_individual:{VEHICLE}"
    assert w1.salience == 1.0                                       # 86 sightings, above all 40 past asks
    assert w1.outcome_value == pytest.approx(6 / 47)                # vehicle: (5 + 1) / (45 + 2)
    assert w1.window.latest == NOW + timedelta(days=7)


def test_a_confirm_ask_carries_its_own_rules_expected_answer():
    market = Market(funded)
    ab.ask_bind(FakeConn(), confirm_ask(), now=NOW, bid=market)
    assert market.wants[0].outcome_value == pytest.approx(142 / 147)
    assert market.wants[0].salience == pytest.approx(7 / 40)


def test_graded_off_sends_no_expected_answer(monkeypatch):
    monkeypatch.setattr(settings, "sql_writer_act_camera_asks_enabled", False)
    market = Market(funded)
    ab.ask_bind(FakeConn(), vehicle_ask(), now=NOW, bid=market)
    assert market.wants[0].outcome_value is None


# --- the lifecycle ------------------------------------------------------------------------------

def test_funded_opens_the_card_stamped_with_its_want():
    conn, market = FakeConn(), Market(funded)
    opened = ab.open_camera_ask(conn, vehicle_ask(), now=NOW, bid=market)
    assert opened is not None and len(conn.inserted) == 1
    assert bid_want_ref(opened.evidence_refs) == market.wants[0].want_id
    assert f"{BID_WANT_REF_PREFIX}{market.wants[0].want_id}" in conn.inserted[0]["ev"]
    assert ab.held_snapshot() == {}


def test_held_does_not_open_then_backs_off_then_a_later_rebid_funds_and_opens():
    conn = FakeConn()
    answers = iter([sated, sated, funded])
    market = Market(lambda w: next(answers)(w))
    assert ab.open_camera_ask(conn, vehicle_ask(), now=NOW, bid=market) is None
    assert conn.inserted == [] and len(market.wants) == 1
    (held,) = ab.held_snapshot().values()
    assert held.retry_at == NOW + timedelta(seconds=REBID_FIRST_SEC) and held.n_held == 1
    # Every 60 s opener tick inside the backoff: no bid, no card.
    for m in range(1, 10):
        t = NOW + timedelta(minutes=m)
        assert ab.open_camera_ask(conn, vehicle_ask(now=t, ask_id=f"ask-t{m}"), now=t, bid=market) is None
    assert len(market.wants) == 1 and conn.inserted == []
    # At retry_at: re-bid, held again, the wait doubles.
    t1 = NOW + timedelta(seconds=REBID_FIRST_SEC)
    assert ab.open_camera_ask(conn, vehicle_ask(now=t1, ask_id="ask-r1"), now=t1, bid=market) is None
    (held,) = ab.held_snapshot().values()
    assert held.n_held == 2 and held.retry_at == t1 + timedelta(seconds=2 * REBID_FIRST_SEC)
    # The re-bid is the SAME want, opened when the candidate was first seen.
    assert market.wants[1].want_id == market.wants[0].want_id and market.wants[1].opened_at == NOW
    # Later: funded -> the card opens.
    t2 = held.retry_at
    opened = ab.open_camera_ask(conn, vehicle_ask(now=t2, ask_id="ask-r2"), now=t2, bid=market)
    assert opened is not None and opened.ask_id == "ask-r2" and len(conn.inserted) == 1
    assert ab.held_snapshot() == {}


def test_no_deadline_a_held_ask_is_never_opened_by_the_clock():
    """Issue #3285: no 24 h cutoff and no cooldown. Ten days of sated answers never open the card, and the
    re-bids never come closer than the cap allows once it is reached."""
    conn, market = FakeConn(), Market(sated)
    t = NOW
    while t < NOW + timedelta(days=10):
        assert ab.open_camera_ask(conn, vehicle_ask(now=t, ask_id=f"ask-{t.timestamp():.0f}"), now=t,
                                  bid=market) is None
        t += timedelta(seconds=60)
    assert conn.inserted == []
    assert 70 < len(market.wants) < 10 * 86400 / REBID_CAP_SEC + 8     # ~8 a day once at the cap, not 1440 a day
    assert all(w.want_id == market.wants[0].want_id for w in market.wants)


def test_bid_timeout_or_market_down_admits_unstamped():
    conn = FakeConn()
    opened = ab.open_camera_ask(conn, vehicle_ask(), now=NOW, bid=Market(lambda w: None))
    assert opened is not None and bid_want_ref(opened.evidence_refs) is None
    conn2 = FakeConn()
    opened = ab.open_camera_ask(conn2, vehicle_ask(), now=NOW, bid=Market(unavailable))
    assert opened is not None and bid_want_ref(opened.evidence_refs) is None and len(conn2.inserted) == 1


def test_a_market_call_that_raises_admits():
    def boom(want):
        raise RuntimeError("socket")
    conn = FakeConn()
    assert ab.open_camera_ask(conn, vehicle_ask(), now=NOW, bid=boom) is not None


def test_shadow_mode_is_the_behaviour_before_4b(monkeypatch):
    monkeypatch.setattr(settings, "orion_physiology_bid_camera_asks", "shadow")
    conn, market = FakeConn(), Market(sated)
    opened = ab.open_camera_ask(conn, vehicle_ask(), now=NOW, bid=market)
    assert opened is not None and market.wants == [] and bid_want_ref(opened.evidence_refs) is None


def test_juniper_story_card_never_bids():
    conn, market = FakeConn(), Market(sated)
    card = vehicle_ask(refs=["motion_story:f53dfa78", f"individual:{VEHICLE}", JUNIPER_STORY_CARD_REF])
    v = ab.ask_bind(conn, card, now=NOW, bid=market)
    assert v.admit and v.why == "juniper_card" and market.wants == []


def test_too_little_history_admits_without_a_want():
    conn, market = FakeConn(history_n=5), Market(sated)
    opened = ab.open_camera_ask(conn, vehicle_ask(), now=NOW, bid=market)
    assert opened is not None and market.wants == []


def test_a_funded_card_that_loses_the_insert_is_not_reported_opened():
    conn, market = FakeConn(), Market(funded)
    assert ab.open_camera_ask(conn, vehicle_ask(ask_id="ask-1"), now=NOW, bid=market) is not None
    assert ab.open_camera_ask(conn, vehicle_ask(ask_id="ask-2"), now=NOW, bid=market) is None


def test_rebid_backoff_doubles_to_the_cap_and_respects_the_markets_wait():
    assert [rebid_after_sec(n) for n in (1, 2, 3, 4, 5, 6, 50)] == [600, 1200, 2400, 4800, 9600, 10800, 10800]
    assert rebid_after_sec(1, 3000.0) == 3000.0
    assert rebid_after_sec(1, 99999.0) == REBID_CAP_SEC


# --- one want per ask: the shadow loop -----------------------------------------------------------

def _row(ask_id, refs, at):
    return {"ask_id": ask_id, "source_kind": "vision_individual", "status": "open", "created_at": at,
            "expires_at": at + timedelta(days=7), "iid": VEHICLE, "sightings": 86, "evidence_refs": refs,
            "individual_kind": "vehicle"}


def test_the_shadow_loop_posts_no_second_want_for_a_card_that_bid_at_open():
    hist = [_row(f"h{i}", [], NOW - timedelta(days=2, minutes=i)) for i in range(25)]
    stamped = _row("bid", [f"{BID_WANT_REF_PREFIX}abc"], NOW - timedelta(minutes=5))
    plain = _row("plain", ["sighting:1"], NOW - timedelta(minutes=4))
    plan = {p["ask_id"]: p for p in plan_asks(hist + [stamped, plain], now=NOW, lookback=timedelta(hours=24))}
    assert plan["bid"]["bid_want"] == "abc" and plan["plain"]["bid_want"] is None

    class Eng:
        def connect(self):
            outer = self

            class C:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def execute(self, clause, params=None):
                    sql = str(clause)
                    if sql == ASKS_SQL:
                        return _Res(rows=hist + [stamped, plain])
                    if sql == GRADES_SQL:
                        return _Res(rows=[])
                    return _Res(rows=[])
            return C()

    posted, market = set(), Market(funded)
    summary = run_one_cycle(postgres_uri="x", lookback=timedelta(hours=24), posted=posted, bid=market, now=NOW,
                            engine=Eng())
    assert summary["bid_at_open"] == 1 and summary["posted"] == 1
    assert [w.event_ref for w in market.wants] == ["orion_ask:plain"]


# --- the one bid site ----------------------------------------------------------------------------

def test_every_camera_opener_goes_through_the_bid_site():
    """No module but app/ask_bind.py may call ``_insert_ask``: an opener that inserted directly would skip the bid."""
    app_dir = Path(__file__).resolve().parents[1] / "app"
    callers = []
    for f in sorted(app_dir.glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
                if name == "_insert_ask":
                    callers.append(f.name)
    assert callers == ["ask_bind.py"]
    openers = {"vision_individuals.py": 4, "vision_handoff_loop.py": 1, "vision_office_sessions.py": 1}
    for name, n in openers.items():
        assert (app_dir / name).read_text().count("open_camera_ask(") == n, name

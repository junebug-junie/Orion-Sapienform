"""Each open camera ask posts one event want (attention as the want competition, patch 2). No database."""
from __future__ import annotations

import pytest

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.ask_event_wants import ASKS_SQL, FAMILY, GRADES_SQL, plan_asks, run_one_cycle
from orion.physiology.event_wants import event_want_id
from orion.physiology.families import load_families

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
REPO = __import__("pathlib").Path(__file__).resolve().parents[3]


def _rows():
    rows = [{"ask_id": f"old{k}", "source_kind": "vision_individual", "status": "answered",
             "created_at": NOW - timedelta(days=20) + timedelta(hours=10 * k), "expires_at": None,
             "iid": f"i{k}", "sightings": k % 10} for k in range(40)]
    rows += [
        {"ask_id": "big", "source_kind": "vision_individual", "status": "open", "created_at": NOW - timedelta(hours=5),
         "expires_at": NOW + timedelta(days=6), "iid": "ibig", "sightings": 49},
        {"ask_id": "small", "source_kind": "vision_person_confirm", "status": "open", "created_at": NOW - timedelta(hours=4),
         "expires_at": NOW + timedelta(days=6), "iid": "ismall", "sightings": 0},
        {"ask_id": "answered", "source_kind": "vision_individual", "status": "answered",
         "created_at": NOW - timedelta(hours=3), "expires_at": NOW + timedelta(days=6), "iid": "ia", "sightings": 5},
        {"ask_id": "expired", "source_kind": "vision_individual", "status": "open",
         "created_at": NOW - timedelta(hours=2), "expires_at": NOW - timedelta(minutes=1), "iid": "ie", "sightings": 5},
        {"ask_id": "stale", "source_kind": "vision_individual", "status": "open",
         "created_at": NOW - timedelta(hours=30), "expires_at": NOW + timedelta(days=5), "iid": "is", "sightings": 5},
    ]
    return rows


def test_only_open_unexpired_recent_asks_are_planned():
    plan = plan_asks(_rows(), now=NOW, lookback=timedelta(hours=24))
    by = {p["ask_id"]: p for p in plan}
    assert set(by) == {"big", "small"}
    assert by["big"]["salience"] == 1.0 and by["small"]["salience"] == 0.0


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return SimpleNamespace(all=lambda: self.rows)

    def __iter__(self):
        return iter(self.rows)

    def fetchall(self):
        return list(self.rows)


class _Conn:
    def __init__(self, ledger, grades=()):
        self.ledger = ledger
        self.grades = list(grades)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params):
        if str(stmt) == ASKS_SQL:
            return _Result(_rows())
        if str(stmt) == GRADES_SQL:
            assert params["actor"] == "vision.ask"
            return _Result(self.grades)
        return _Result([(w,) for w in self.ledger if w in params["ids"]])


class _Engine:
    def __init__(self, ledger=(), grades=()):
        self.ledger = set(ledger)
        self.grades = grades

    def connect(self):
        return _Conn(self.ledger, self.grades)


def test_a_cycle_posts_each_open_ask_once():
    bids = []

    def bid(want):
        bids.append(want)
        return SimpleNamespace(outcome="funded", reason="net_positive", value_rank=0.9)

    posted: set[str] = set()
    s = run_one_cycle(postgres_uri="x", lookback=timedelta(hours=24), posted=posted, bid=bid, now=NOW, engine=_Engine())
    assert s["posted"] == 2
    w = next(b for b in bids if b.event_ref == "orion_ask:big")
    assert w.family == FAMILY and w.verb == "ask" and w.tier == 2 and w.salience == 1.0
    assert w.object_ref == "vision_individual:ibig" and w.window.latest == NOW + timedelta(days=6)
    assert w.want_id == event_want_id("camera_ask", "big")
    run_one_cycle(postgres_uri="x", lookback=timedelta(hours=24), posted=posted, bid=bid, now=NOW, engine=_Engine())
    assert len(bids) == 2
    # A restart (empty memory) finds both in the ledger.
    again = run_one_cycle(postgres_uri="x", lookback=timedelta(hours=24), posted=set(), bid=bid, now=NOW,
                          engine=_Engine({event_want_id("camera_ask", "big"), event_want_id("camera_ask", "small")}))
    assert again["already"] == 2 and len(bids) == 2


def test_no_answer_is_retried():
    calls = []
    posted: set[str] = set()
    for _ in range(2):
        run_one_cycle(postgres_uri="x", lookback=timedelta(hours=24), posted=posted,
                      bid=lambda w: calls.append(w), now=NOW, engine=_Engine())
    assert len(calls) == 4


def test_the_family_exists_and_is_tier_2_with_no_charge():
    fams, err = load_families(REPO / "config" / "physiology" / "families.yaml")
    cfg = fams.get(FAMILY)
    assert err is None and cfg is not None and cfg.tier == 2 and cfg.bids and not cfg.lanes and not cfg.fixed_cost


def test_an_unpriced_decision_is_retried():
    from app.ask_event_wants import POSTED_SQL

    calls = []
    posted: set[str] = set()

    def bid(w):
        calls.append(w)
        return SimpleNamespace(outcome="deferred", reason="physiology_unavailable", value_rank=None)

    for _ in range(2):
        run_one_cycle(postgres_uri="x", lookback=timedelta(hours=24), posted=posted, bid=bid, now=NOW, engine=_Engine())
    assert len(calls) == 4 and not posted and "reason <> 'physiology_unavailable'" in POSTED_SQL



def test_juniper_story_cards_are_neither_asks_nor_history():
    """Patch 4a: her "named on a walkway story" cards are filtered structurally in the SQL."""
    from orion.grammar.camera_ask_acts import ORION_MADE_CAMERA_ASK_SQL

    assert ORION_MADE_CAMERA_ASK_SQL in ASKS_SQL and "named on a walkway story" not in ASKS_SQL


def test_each_want_carries_its_rules_expected_answer():
    """Patch 4a: outcome_value = the rule's answered share from Juniper's graded closes, shrunk toward the
    family's share (issue #3350)."""
    rows = _rows()
    for r in rows:
        r["individual_kind"] = "vehicle" if r["ask_id"] == "big" else "person"
        r["evidence_refs"] = ["sighting:1"]
    # Graded closes, as GRADES_SQL returns them: vehicle 1 of 10 answered; confirm 9 of 9.
    grades = [("camera_ask:vehicle", 1.0)] + [("camera_ask:vehicle", 0.0)] * 9 + [("camera_ask:confirm", 1.0)] * 9

    class _E(_Engine):
        def connect(self):
            c = _Conn(self.ledger, grades)
            c.execute_orig = c.execute

            def ex(stmt, params, _c=c):
                return _Result(rows) if str(stmt) == ASKS_SQL else _c.execute_orig(stmt, params)
            c.execute = ex
            return c

    bids = []
    run_one_cycle(postgres_uri="x", lookback=timedelta(hours=24), posted=set(), now=NOW, engine=_E(),
                  bid=lambda w: bids.append(w) or SimpleNamespace(outcome="funded", reason="net_positive", value_rank=0.5))
    by = {b.event_ref: b for b in bids}
    m = 10 / 19                                                  # the family: 10 answered of 19 closed (#3350)
    assert by["orion_ask:big"].outcome_value == pytest.approx((1 + 2 * m) / 12)     # vehicle 1 of 10
    assert by["orion_ask:small"].outcome_value == pytest.approx((9 + 2 * m) / 11)   # confirm 9 of 9
    # Off (SQL_WRITER_ACT_CAMERA_ASKS_ENABLED=false): no outcome on the want, as before patch 4a.
    bids.clear()
    run_one_cycle(postgres_uri="x", lookback=timedelta(hours=24), posted=set(), now=NOW, engine=_E(), graded=False,
                  bid=lambda w: bids.append(w) or SimpleNamespace(outcome="funded", reason="net_positive", value_rank=0.5))
    assert all(b.outcome_value is None for b in bids) and len(bids) == 2

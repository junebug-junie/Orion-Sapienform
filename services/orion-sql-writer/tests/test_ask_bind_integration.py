"""Attention patch 4b end to end against real Postgres: the real slow-rule opener (``_open_sighting_asks``) bids
each candidate before its card opens, and obeys.

Lifecycle on one throwaway schema: a vehicle candidate (Juniper answers 5 of 45) is held and gets no card while a
person candidate is funded and opens, stamped ``want:<id>``; the held vehicle is not re-bid inside its backoff;
a later re-bid funds it and it opens; the daily cap still stops a third card; the shadow loop's SQL (ASKS_SQL)
reads the stamp, so it posts no second want; Juniper's own story card is in no history and never bids; ``shadow``
opens first with no bid.

Opt-in like the other act-ledger tests: ORION_SQL_WRITER_TEST_DATABASE_URI is a scratch database. Row shapes
copied from live ``orion_ask`` / ``vision_individual`` / ``vision_individual_sighting`` (read-only, 2026-11-13).
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine, text

from app import ask_bind as ab
from app.ask_event_wants import ASKS_SQL, plan_asks, run_one_cycle
from app.models.grammar_trace import GrammarActSQL
from app.db import Base
from app.settings import settings
from app.vision_individuals import IndividualsConfig, _open_sighting_asks
from orion.grammar.camera_ask_acts import bid_want_ref
from orion.schemas.physiology_want import WantDecisionV1

pytestmark = pytest.mark.integration

DDL = """
CREATE TABLE orion_ask (
    ask_id TEXT PRIMARY KEY, asked_of TEXT NOT NULL DEFAULT 'juniper', question TEXT NOT NULL,
    evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb, image_ref TEXT, status TEXT NOT NULL DEFAULT 'open',
    answer TEXT, answered_at TIMESTAMPTZ, applied_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ, source_kind TEXT NOT NULL, source_ref TEXT NOT NULL,
    image_refs JSONB NOT NULL DEFAULT '[]'::jsonb, answer_detail JSONB);
CREATE UNIQUE INDEX orion_ask_open_source_uidx ON orion_ask (source_kind, source_ref) WHERE status = 'open';
CREATE TABLE vision_individual (individual_id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT,
    stationary BOOLEAN NOT NULL DEFAULT false, sighting_count INT NOT NULL DEFAULT 0, distinct_days INT NOT NULL DEFAULT 0);
CREATE TABLE vision_individual_sighting (sighting_id TEXT PRIMARY KEY, individual_id TEXT, started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ, thumb_ref TEXT, zone_no_embed BOOLEAN NOT NULL DEFAULT false, face_thumb_ref TEXT,
    stationary BOOLEAN NOT NULL DEFAULT false);
CREATE TABLE want_ledger (decision_id TEXT PRIMARY KEY, want_id TEXT NOT NULL, family TEXT, producer TEXT NOT NULL,
    outcome TEXT, reason TEXT NOT NULL, decided_at TIMESTAMPTZ NOT NULL, decision JSONB NOT NULL)
"""

TZ = ZoneInfo("America/Denver")
NOW = datetime(2026, 11, 13, 14, 0, 53, tzinfo=timezone.utc)      # 07:00 MST, the morning batch
VEHICLE = "420762ba8f4b4e28af57ffe1e66c2889"
PERSON = "1f0c6bfb55b5491992dcf7c77865ffc5"
OTHER_VEHICLE = "8ac2822b05494132af491d488c9ce267"
CFG = IndividualsConfig(ask_daily_cap=2, local_tz="America/Denver")


@pytest.fixture
def conn():
    uri = os.environ["ORION_SQL_WRITER_TEST_DATABASE_URI"]
    schema = f"ask_bind_test_{uuid.uuid4().hex[:10]}"
    admin = create_engine(uri)
    with admin.begin() as c:
        c.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(uri, connect_args={"options": f"-csearch_path={schema}"})
    with engine.begin() as c:
        for stmt in DDL.split(";"):
            if stmt.strip():
                c.execute(text(stmt))
    Base.metadata.create_all(bind=engine, tables=[GrammarActSQL.__table__])
    c = engine.connect()
    try:
        yield c
    finally:
        c.close()
        engine.dispose()
        with admin.begin() as a:
            a.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


def _individual(c, iid, kind, n, days):
    c.execute(text("INSERT INTO vision_individual (individual_id, kind, sighting_count, distinct_days) "
                   "VALUES (:i, :k, :n, :d)"), {"i": iid, "k": kind, "n": n, "d": days})
    for j in range(n):
        at = NOW - timedelta(days=(j % days) + 1, minutes=j)
        c.execute(text("INSERT INTO vision_individual_sighting (sighting_id, individual_id, started_at, ended_at, "
                       "thumb_ref) VALUES (:s, :i, :a, :e, :t)"),
                  {"s": f"{iid[:8]}-{j}", "i": iid, "a": at, "e": at + timedelta(seconds=40),
                   "t": f"thumb:{uuid.uuid4().hex}"})


def _past_ask(c, ask_id, iid, at, *, refs=None, detail=None, status="answered"):
    c.execute(text("INSERT INTO orion_ask (ask_id, question, evidence_refs, status, created_at, expires_at, "
                   "source_kind, source_ref, answer_detail, answered_at) VALUES (:a, 'q', CAST(:e AS jsonb), :st, :c, "
                   ":x, 'vision_individual', :r, CAST(:d AS jsonb), :c)"),
              {"a": ask_id, "e": json.dumps(refs or []), "st": status, "c": at, "x": at + timedelta(days=7),
               "r": iid, "d": None if detail is None else json.dumps(detail)})


def _seed(c):
    _individual(c, VEHICLE, "vehicle", 30, 6)
    _individual(c, OTHER_VEHICLE, "vehicle", 20, 5)
    _individual(c, PERSON, "person", 12, 5)
    # 30 days of past Orion-made asks about other individuals (the salience history; 25 >= 20).
    for i in range(25):
        hid = f"hist{i:02d}"
        _individual(c, hid, "person", 2 + i % 7, 1)
        _past_ask(c, f"ask-h{i}", hid, NOW - timedelta(days=3, minutes=i))
    # Juniper's own walkway-story card: in no history, never bids.
    _past_ask(c, "ask-juniper", "story1", NOW - timedelta(days=2),
              refs=["motion_story:f53d", "individual:story1", "rule:juniper_story_naming"],
              detail={"story_id": "f53d", "action": "name", "via": "motion_story"})
    # Juniper's grades: vehicles 5 of 45, people ("sighting") 4 of 5.
    rows = [("vehicle", 1.0)] * 5 + [("vehicle", 0.0)] * 40 + [("sighting", 1.0)] * 4 + [("sighting", 0.0)]
    for k, (rule, v) in enumerate(rows):
        c.execute(text("INSERT INTO grammar_acts (act_id, trace_id, actor, verb, level, signal_id, opened_at, closed_at, "
                       "outcome_value, outcome_source, test_before, created_at, updated_at) VALUES (:a, :a, 'vision.ask', "
                       "'ask', 'act', 'juniper_answered', :o, :cl, :v, 'resolver', :tb, :o, :cl)"),
                  {"a": f"act-{k}", "o": NOW - timedelta(days=5), "cl": NOW - timedelta(days=4), "v": v,
                   "tb": f"camera_ask:{rule}"})
    c.commit()


class Market:
    """Funds an ask whose expected answer is at least ``line`` (the market's rank, simplified), holds the rest."""

    def __init__(self, line=0.5):
        self.line, self.wants = line, []

    def __call__(self, want):
        self.wants.append(want)
        ok = want.outcome_value is not None and want.outcome_value >= self.line
        return WantDecisionV1.model_validate({
            "want_id": want.want_id, "family": want.family, "producer": want.producer, "tier": 2, "verb": "ask",
            "decision_id": f"{want.want_id}@{len(self.wants)}", "decided_at": NOW.isoformat(), "shadow": True,
            "want": want.model_dump(mode="json"), "outcome": "funded" if ok else "deferred",
            "reason": "net_positive" if ok else "sated", "net": 0.9 if ok else 0.0, "value_rank": 0.9 if ok else 0.0,
            "charge": 0.0})


@pytest.fixture(autouse=True)
def _state(monkeypatch):
    ab.reset_state()
    monkeypatch.setattr(settings, "orion_physiology_bid_camera_asks", "on")
    monkeypatch.setattr(settings, "sql_writer_act_camera_asks_enabled", True)
    yield
    ab.reset_state()


def _open(c, t, market, monkeypatch):
    import orion.physiology.client as client

    monkeypatch.setattr(client, "bid", market)
    out = _open_sighting_asks(c, t, TZ, CFG)
    c.commit()
    return out


def _cards(c):
    return c.execute(text("SELECT ask_id, source_ref, evidence_refs FROM orion_ask WHERE status = 'open' "
                          "ORDER BY created_at")).mappings().all()


def test_held_vehicle_gets_no_card_funded_person_opens_then_a_later_rebid_opens_the_vehicle(conn, monkeypatch):
    _seed(conn)
    market = Market(line=0.5)
    opened = _open(conn, NOW, market, monkeypatch)
    # Candidates in the opener's order (moving sightings desc): VEHICLE (30), OTHER_VEHICLE (20), PERSON (12).
    assert [w.object_ref for w in market.wants] == [f"vision_individual:{x}" for x in (VEHICLE, OTHER_VEHICLE, PERSON)]
    m = 9 / 50                                   # the seeded family: vehicles 5 of 45 + people 4 of 5 answered
    assert [w.outcome_value for w in market.wants] == pytest.approx([(5 + 2 * m) / 47] * 2 + [(4 + 2 * m) / 7])
    assert [o["source_ref"] for o in opened] == [PERSON]
    cards = _cards(conn)
    assert [r["source_ref"] for r in cards] == [PERSON]
    assert bid_want_ref(cards[0]["evidence_refs"]) == market.wants[2].want_id
    # Juniper's story card is in no history: the salience history is the 25 Orion-made asks only.
    rows = conn.execute(text(ASKS_SQL), {"since": NOW - timedelta(days=30), "until": NOW}).mappings().all()
    assert "ask-juniper" not in {r["ask_id"] for r in rows} and len(rows) == 25   # the 25 past Orion-made asks

    # Next opener tick, 60 s later: both vehicles are in their backoff -> no bid, no card.
    _open(conn, NOW + timedelta(seconds=60), market, monkeypatch)
    assert len(market.wants) == 3 and len(_cards(conn)) == 1

    # The shadow loop reads the stamp: one want per ask, so it posts none for the person card.
    plan = plan_asks(conn.execute(text(ASKS_SQL), {"since": NOW - timedelta(days=31), "until": NOW + timedelta(minutes=5)})
                     .mappings().all(), now=NOW + timedelta(minutes=5), lookback=timedelta(hours=24))
    assert [(p["ask_id"] == cards[0]["ask_id"], p["bid_want"]) for p in plan] == [(True, market.wants[2].want_id)]

    # After the backoff the market has changed its mind (the line falls under the vehicles' 0.114): the vehicle re-bids
    # as the SAME want and
    # opens. The daily cap (2) is then full, so the other vehicle is not even bid.
    market.line = 0.1
    t = NOW + timedelta(seconds=601)
    _open(conn, t, market, monkeypatch)
    assert [r["source_ref"] for r in _cards(conn)] == [PERSON, VEHICLE]
    assert market.wants[3].want_id == market.wants[0].want_id and market.wants[3].opened_at == NOW
    assert len(market.wants) == 4
    assert _open(conn, t + timedelta(seconds=601), market, monkeypatch) == []


def test_shadow_mode_opens_first_with_no_bid(conn, monkeypatch):
    _seed(conn)
    monkeypatch.setattr(settings, "orion_physiology_bid_camera_asks", "shadow")
    market = Market(line=0.5)
    opened = _open(conn, NOW, market, monkeypatch)
    assert market.wants == []
    assert [o["source_ref"] for o in opened] == [VEHICLE, OTHER_VEHICLE]        # today's order, today's cap
    assert all(bid_want_ref(r["evidence_refs"]) is None for r in _cards(conn))


def test_market_down_opens_unstamped_so_the_shadow_loop_still_records_one_want(conn, monkeypatch):
    _seed(conn)
    opened = _open(conn, NOW, lambda want: None, monkeypatch)
    assert [o["source_ref"] for o in opened] == [VEHICLE, OTHER_VEHICLE]
    plan = plan_asks(conn.execute(text(ASKS_SQL), {"since": NOW - timedelta(days=31), "until": NOW + timedelta(minutes=5)})
                     .mappings().all(), now=NOW + timedelta(minutes=5), lookback=timedelta(hours=24))
    assert len(plan) == 2 and all(p["bid_want"] is None for p in plan)


def test_shadow_loop_sql_one_want_per_card_stamped_late_decided_or_neither(conn, monkeypatch):
    """The shadow loop's reads against real Postgres: a stamped card posts nothing; an unstamped card whose
    bid-at-open the market recorded anyway (client timeout) posts nothing; an unstamped card with no row posts one."""
    _seed(conn)
    _open(conn, NOW, lambda want: None, monkeypatch)                    # market down: both vehicles open unstamped
    cards = _cards(conn)
    late = cards[0]["ask_id"]
    conn.execute(text("INSERT INTO want_ledger VALUES ('d1', 'w1', 'vision.ask', :p, 'funded', 'net_positive', :t, "
                      "CAST(:d AS jsonb))"),
                 {"p": ab.PRODUCER, "t": NOW, "d": json.dumps({"want": {"event_ref": f"orion_ask:{late}"}})})
    conn.commit()
    market = []
    summary = run_one_cycle(postgres_uri="unused", lookback=timedelta(hours=24), posted=set(),
                            bid=lambda w: market.append(w) or None, now=NOW + timedelta(minutes=5), engine=conn.engine)
    assert summary["bid_at_open"] == 1
    assert [w.event_ref for w in market] == [f"orion_ask:{cards[1]['ask_id']}"]


def test_the_writers_family_share_is_the_markets_family_mean(conn):
    """Review of #3350: GRADES_SQL must select the acts the market's build_family_appetite reads (opened in the last
    30 days, producer or resolver outcome), or a rule with no closes would read a share the market does not use.
    Seeded: an act opened 31 days ago and closed 29 days ago (counted by neither), a withdrawn ``expired`` close (not
    a grade), a producer-graded close, and the 50 resolver closes of _seed."""
    from app.ask_event_wants import GRADES_SQL
    from orion.grammar.camera_ask_acts import rule_from_test_before, rule_outcome_values
    from orion.physiology.appetite import ActRecord, build_family_appetite
    from orion.physiology.families import parse_families

    _seed(conn)
    extra = [("old", NOW - timedelta(days=31), NOW - timedelta(days=29), 0.0, "resolver", "camera_ask:vehicle"),
             ("wd", NOW - timedelta(days=3), NOW - timedelta(days=2), 0.0, "expired", "camera_ask:confirm"),
             ("prod", NOW - timedelta(days=2), NOW - timedelta(days=1), 1.0, "producer", "camera_ask:confirm")]
    for a, o, cl, v, src, tb in extra:
        conn.execute(text("INSERT INTO grammar_acts (act_id, trace_id, actor, verb, level, signal_id, opened_at, "
                          "closed_at, outcome_value, outcome_source, test_before, created_at, updated_at) VALUES (:a, :a, "
                          "'vision.ask', 'ask', 'act', 'juniper_answered', :o, :cl, :v, :src, :tb, :o, :cl)"),
                     {"a": a, "o": o, "cl": cl, "v": v, "src": src, "tb": tb})
    conn.commit()
    rows = conn.execute(text(GRADES_SQL), {"actor": "vision.ask", "since": NOW - timedelta(days=30), "now": NOW}).fetchall()
    office = rule_outcome_values((rule_from_test_before(r[0]) or "", float(r[1])) for r in rows)["office"]  # no closes
    acts = [ActRecord(act_id=r["act_id"], family="vision.ask", opened_at=r["opened_at"], closed_at=r["closed_at"],
                      outcome_value=r["outcome_value"], outcome_source=r["outcome_source"])
            for r in conn.execute(text("SELECT act_id, opened_at, closed_at, outcome_value, outcome_source "
                                       "FROM grammar_acts WHERE actor = 'vision.ask'")).mappings()]
    cfg = parse_families({"version": 1, "families": {"vision.ask": {"tier": 2, "deferrable_window_s": 86400,
                                                                    "lanes": []}}}).get("vision.ask")
    app = build_family_appetite(cfg, acts, NOW)
    assert len(rows) == app.n_graded == 51
    assert office == pytest.approx(app.mean_outcome) == pytest.approx(10 / 51)

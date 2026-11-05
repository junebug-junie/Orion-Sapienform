"""Office sessions against a REAL Postgres: the session row, the timeline, the "Who is this?" ask.

Runs only when ``ORION_TEST_PG_URI`` points at a throwaway database -- it TRUNCATEs the walkway
tables -- with the walkway migrations plus vision_presence_transition_v1, vision_scene_inventory and
vision_office_session_v1 applied, e.g.:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55499:5432 postgres:16-alpine
    for f in walkway_camera_v1 walkway_privacy_v2 vision_confirm_asks_v1 vision_ask_names_v1 vision_split_v1 \\
             vision_osnet_shadow_v1 vision_person_album_v1 vision_person_basis_v1 vision_scene_inventory \\
             vision_presence_transition_v1 vision_office_session_v1; do
      docker exec -i pg psql -U postgres < services/orion-sql-db/manual_migration_$f.sql; done
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55499/postgres pytest tests/test_vision_office_sessions_pg.py

vision_events is created by sql-writer's ORM at boot; the fixture creates it the same way.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(REPO), str(SERVICE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import vision_individuals as vi  # noqa: E402
from app import vision_office_sessions as vos  # noqa: E402

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")

T0 = datetime(2026, 11, 5, 1, 24, 7, tzinfo=timezone.utc)
PRIOR = {"cam0": {"Juniper": 0.97, "Sibling": 0.01}}
CFG = vos.OfficeSessionsConfig(streams=("cam0",), settle_sec=660.0, ask_daily_cap=4)
ICFG = vi.IndividualsConfig()


def at(sec: float) -> datetime:
    return T0 + timedelta(seconds=sec)


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    from app.models.vision_event import VisionEventSQL

    engine = create_engine(PG)
    VisionEventSQL.__table__.create(engine, checkfirst=True)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE vision_events ADD COLUMN IF NOT EXISTS stream_id text"))
        conn.execute(text("TRUNCATE orion_ask, vision_individual_sighting, vision_individual, vision_crop_observation, "
                          "vision_person_exemplar, vision_person_attribution, vision_presence_transition, "
                          "vision_scene_inventory, vision_events, vision_office_session CASCADE"))
    yield engine
    engine.dispose()


def presence(conn, rows) -> None:
    from sqlalchemy import text

    for n, (to, frm, sec) in enumerate(rows):
        conn.execute(text("INSERT INTO vision_presence_transition (transition_id, presence_id, from_state, to_state, "
                          "occurred_at) VALUES (:i, 'cam0', :f, :t, :o)"),
                     {"i": f"tr{int(sec)}-{n}", "f": frm, "t": to, "o": at(sec)})


def record(conn, iid: str, secs, *, guess="Juniper", basis="prior", score=0.7, label=None, thumb=True) -> None:
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, first_seen_at, last_seen_at, "
        "sighting_count, distinct_days, label) VALUES (:i, 'cam0', 'person', '{1,0}', :a, :b, :n, 1, :l)"),
        {"i": iid, "a": at(secs[0][0]), "b": at(secs[-1][1]), "n": len(secs), "l": label})
    for j, (a, b) in enumerate(secs):
        conn.execute(text(
            "INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, started_at, ended_at, "
            "thumb_ref) VALUES (:s, :i, 'cam0', :a, :b, :t)"),
            {"s": f"{iid}-s{j}", "i": iid, "a": at(a), "b": at(b), "t": f"thumb:{iid}{j}" if thumb else None})
    if guess:
        conn.execute(text(
            "INSERT INTO vision_person_attribution (individual_id, person_label, channel, score, band, model_version, "
            "decided_at, basis) VALUES (:i, :p, 'body', :s, 'unsure', 'm', :t, :b)"),
            {"i": iid, "p": guess, "s": score, "t": at(secs[-1][1] + 5), "b": basis})


def caption(conn, eid: str, sec: float, text_: str, *, kind: str = "camera_caption") -> None:
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_events (event_id, event_type, narrative, entities, evidence_refs, created_at, stream_id) "
        "VALUES (:e, :k, :n, '[]', CAST(:r AS jsonb), :c, 'cam0')"),
        {"e": eid, "k": kind, "n": text_, "r": json.dumps([f"frame_ts:{at(sec).isoformat()}"]), "c": at(sec + 8)})


def counts(conn, sec: float, people: int) -> None:
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_scene_inventory (window_id, stream_id, observed_at, window_start_ts, window_end_ts, counts) "
        "VALUES (:w, 'cam0', :o, :a, :b, CAST(:c AS jsonb))"),
        {"w": f"w{sec}", "o": at(sec), "a": at(sec - 5).timestamp(), "b": at(sec).timestamp(),
         "c": json.dumps({"person": people, "desk": 2})})


def run(pg, now, **kw):
    with pg.begin() as conn:
        return vos.run_office_sessions(conn, now, kw.pop("cfg", CFG), ICFG, PRIOR)


def sessions(pg):
    from sqlalchemy import text

    with pg.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text("SELECT * FROM vision_office_session ORDER BY started_at"))]


def asks(pg):
    from sqlalchemy import text

    with pg.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text("SELECT * FROM orion_ask ORDER BY created_at"))]


def juniper_visit(conn, *, peak: int = 1) -> None:
    presence(conn, [("absent", None, -600), ("present", "absent", 0), ("recent", "present", 441),
                    ("absent", "recent", 551)])
    record(conn, "r1", [(-20, 5), (170, 200)], score=0.71)
    record(conn, "r2", [(20, 90)], score=0.68)
    record(conn, "r3", [(240, 300)], score=0.55)
    caption(conn, "c1", -19, "The person in the image is standing in a room, looking at a box on the floor.")
    caption(conn, "c2", 23, "The person in the image is sitting at a desk, opening a box.")
    caption(conn, "c3", 60, "The person in the image is sitting at a desk, opening a box.")
    caption(conn, "c4", 167, "The person in the image is reaching into a cabinet.")
    caption(conn, "v1", 100, "A table, two doors and a person are detected.", kind="visual_observation")
    for sec in range(0, 440, 30):
        counts(conn, sec, 1)
    if peak > 1:
        counts(conn, 260, peak)


@needs_pg
def test_one_visit_is_one_final_session_with_who_and_timeline(pg) -> None:
    with pg.begin() as conn:
        juniper_visit(conn)
    s, opened = run(pg, at(441 + 661))
    assert s["office_sessions_written"] == 1 and s["office_sessions_closed"] == 1 and opened == []
    (row,) = sessions(pg)
    assert row["status"] == "closed" and row["ended_at"] == at(441) and row["end_reason"] == "left"
    assert row["who_text"] == "Juniper (probably)" and row["unknown_people"] == 0
    assert [e["text"] for e in row["timeline"]] == [
        "standing in a room, looking at a box on the floor", "sitting at a desk, opening a box",
        "reaching into a cabinet"]
    assert row["timeline"][1]["repeats"] == 2
    assert row["label"] == ("Juniper (probably) in the office: standing in a room, looking at a box on the floor; "
                            "sitting at a desk, opening a box; reaching into a cabinet")
    assert sorted(row["individual_ids"]) == ["r1", "r2", "r3"] and row["image_refs"]
    assert asks(pg) == []
    # A final row is never rewritten.
    from sqlalchemy import text

    with pg.begin() as conn:
        conn.execute(text("UPDATE vision_person_attribution SET person_label = 'Neighbor', basis = 'evidence'"))
    run(pg, at(441 + 2000))
    assert sessions(pg)[0]["who_text"] == "Juniper (probably)"


@needs_pg
def test_open_then_settling_then_final(pg) -> None:
    with pg.begin() as conn:
        juniper_visit(conn)
    run(pg, at(200))     # mid-visit: only the rows before 'recent' exist in reality, all here is fine
    (row,) = sessions(pg)
    run(pg, at(441 + 100))   # ended, not settled yet
    assert sessions(pg)[0]["status"] == "open" and sessions(pg)[0]["closed_at"] is None
    run(pg, at(441 + 700))
    assert sessions(pg)[0]["status"] == "closed" and sessions(pg)[0]["closed_at"] == at(441 + 700)


@needs_pg
def test_two_people_in_view_open_one_who_is_this_card_right_away(pg) -> None:
    with pg.begin() as conn:
        juniper_visit(conn, peak=2)
    s, opened = run(pg, at(300))   # still in the office: the card opens now, not after the visit
    assert s["office_asks_opened"] == 1
    (ask,) = asks(pg)
    # The record seen while two people were in view (r3 at 240-300 s), least like Juniper.
    assert ask["source_kind"] == "vision_individual" and ask["source_ref"] == "r3" and ask["status"] == "open"
    assert vos.OFFICE_RULE_REF in ask["evidence_refs"]
    assert ask["question"].startswith("I think someone I don't recognize is in the office since 18:24.")
    assert "Also there: Juniper (probably)." in ask["question"]
    assert ask["image_refs"] == ["thumb:r30"]
    (row,) = sessions(pg)
    assert row["ask_id"] == ask["ask_id"] and row["unknown_people"] == 1
    assert row["who_text"] == "Juniper (probably) and someone I don't know"
    # One card per session: later passes and the final pass open nothing more.
    run(pg, at(441 + 700))
    assert len(asks(pg)) == 1 and sessions(pg)[0]["status"] == "closed"
    # The street rule's daily cap does not count it.
    from sqlalchemy import text

    with pg.connect() as conn:
        n = conn.execute(text(vi.ASKED_TODAY_SQL), {"d": at(-3600), "face": json.dumps([vi.FACE_RULE_REF])}).scalar()
    assert n == 0


@needs_pg
def test_daily_cap_cooldown_and_no_picture_no_ask(pg) -> None:
    with pg.begin() as conn:
        juniper_visit(conn, peak=2)
    run(pg, at(300), cfg=vos.OfficeSessionsConfig(streams=("cam0",), ask_daily_cap=0))
    assert asks(pg) == []   # cap 0 = none
    from sqlalchemy import text

    with pg.begin() as conn:
        conn.execute(text("UPDATE vision_individual_sighting SET thumb_ref = NULL"))
    run(pg, at(310))
    assert asks(pg) == []   # no picture, no ask; it stays a candidate
    with pg.begin() as conn:
        conn.execute(text("UPDATE vision_individual_sighting SET thumb_ref = 'thumb:' || sighting_id"))
        # Juniper already answered a card about r3: never asked again; the next candidate is.
        conn.execute(text("INSERT INTO orion_ask (ask_id, asked_of, question, evidence_refs, status, created_at, "
                          "source_kind, source_ref) VALUES ('old', 'juniper', 'q', '[]', 'answered', :t, "
                          "'vision_individual', 'r3')"), {"t": at(-86400 * 2)})
    run(pg, at(320))
    new = [a for a in asks(pg) if a["ask_id"] != "old"]
    assert [a["source_ref"] for a in new] == ["r2"]   # r1/r2 were not in the crowded window; r2 scores lower


@needs_pg
def test_missing_table_raises_migration_missing(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        conn.execute(text("ALTER TABLE vision_office_session RENAME TO vision_office_session_away"))
    try:
        with pytest.raises(vos.MigrationMissing):
            run(pg, at(0))
    finally:
        with pg.begin() as conn:
            conn.execute(text("ALTER TABLE vision_office_session_away RENAME TO vision_office_session"))

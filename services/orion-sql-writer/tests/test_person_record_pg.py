"""One record per person against a REAL Postgres (app/person_record.py).

Runs only when ``ORION_TEST_PG_URI`` points at a throwaway database with the
walkway migrations plus manual_migration_person_record_v1.sql applied -- it
TRUNCATEs the walkway and person tables:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55499:5432 postgres:16-alpine
    for f in walkway_camera_v1 walkway_privacy_v2 vision_confirm_asks_v1 vision_ask_names_v1 \\
             vision_split_v1 vision_osnet_shadow_v1 vision_person_album_v1 person_record_v1; do
      docker exec -i pg psql -U postgres < services/orion-sql-db/manual_migration_$f.sql; done
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55499/postgres pytest tests/test_person_record_pg.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from orion.schemas.person_model import PersonRefV1  # noqa: E402

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")
NOW = datetime(2026, 10, 22, 18, 0, tzinfo=timezone.utc)
PEOPLE = [PersonRefV1(node_id="referent-you", display_name="Alex", role="interlocutor", pronouns="she/her"),
          PersonRefV1(node_id="referent-self", display_name="Orion", role="self", pronouns="they/them")]


@pytest.fixture()
def pg(monkeypatch):
    from sqlalchemy import create_engine, text

    from app import person_record

    monkeypatch.setattr(person_record, "_people", lambda: list(PEOPLE))
    engine = create_engine(PG)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE orion_ask, vision_individual_pair_verdict, vision_individual_sighting, "
                          "vision_individual, person_fact, person_link, person CASCADE"))
    yield engine
    engine.dispose()


def _cluster(conn, iid, label, *, kind="person", days=1, ask=None, thumb=False):
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, centroid_n, first_seen_at, "
        "last_seen_at, sighting_count, distinct_days, label, label_ask_id) "
        "VALUES (:i, 'walkway', :k, '{1,0}', 1, :t, :t, :d, :d, :l, :a)"),
        {"i": iid, "k": kind, "t": NOW - timedelta(hours=1), "d": days, "l": label, "a": ask})
    if thumb:
        conn.execute(text(
            "INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, started_at, ended_at, "
            "dwell_sec, observation_count, thumb_ref) VALUES (:s, :i, 'walkway', :t, :t, 0, 1, :th)"),
            {"s": f"s-{iid}", "i": iid, "t": NOW - timedelta(hours=1), "th": "thumb:" + "ab" * 32})


def _run(engine, now=NOW, cap=3):
    from app import person_record as rec
    from app import vision_individuals as vi

    with engine.begin() as conn:
        return rec.run_person_record(conn, now, rec.PersonRecordConfig(ask_daily_cap=cap),
                                     vi.IndividualsConfig())


def _rows(engine, sql, **params):
    from sqlalchemy import text

    with engine.begin() as conn:
        return [tuple(r) for r in conn.execute(text(sql), params).fetchall()]


@needs_pg
def test_names_become_one_person_each_with_links_and_questions(pg) -> None:
    with pg.begin() as conn:
        _cluster(conn, "sam-1", "Sam", days=5, ask="ask-a", thumb=True)
        _cluster(conn, "sam-2", "sam", days=2)
        _cluster(conn, "bike", "Sam", kind="bicycle")
        _cluster(conn, "car", "no", kind="vehicle")
        _cluster(conn, "you-1", "alex", days=1)
    summary, opened = _run(pg)
    assert summary["link_conflicts"] == 0
    persons = dict(_rows(pg, "SELECT display_name, person_id FROM person"))
    assert set(persons) == {"Alex", "Orion", "Sam"}            # no person called "no"
    links = _rows(pg, "SELECT source_ref, source_kind, person_id FROM person_link WHERE revoked_at IS NULL "
                      "ORDER BY source_ref")
    assert links == [("bike", "vision_belonging", persons["Sam"]), ("sam-1", "vision_individual", persons["Sam"]),
                     ("sam-2", "vision_individual", persons["Sam"]), ("you-1", "vision_individual", "referent-you")]
    assert _rows(pg, "SELECT person_id, category, value_text, voice FROM person_fact") == [
        ("referent-you", "pronouns", "she/her", "authored_profile")]
    # One question per person, most-seen first, never about Orion; Sam's shows Sam's picture.
    assert [(a["source_ref"], a["question"]) for a in opened] == [
        (f"{persons['Sam']}|relationship", "Who is Sam to you?"),
        ("referent-you|gender", "What is your gender, in your own words?")]
    assert opened[0]["image_refs"] == ["thumb:" + "ab" * 32] and opened[1]["image_refs"] == []
    # Re-running changes nothing and opens nothing (one open question per person).
    summary2, opened2 = _run(pg, NOW + timedelta(minutes=1))
    assert (summary2["persons_added"], summary2["links_added"], summary2["links_revoked"], opened2) == (0, 0, 0, [])


@needs_pg
def test_an_answer_becomes_a_fact_and_the_next_question_follows(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _cluster(conn, "sam-1", "Sam")
    (ask,) = _run(pg, cap=1)[1]
    with pg.begin() as conn:
        conn.execute(text("UPDATE orion_ask SET status='answered', answer='  my neighbor ', answered_at=:t "
                          "WHERE ask_id=:a"), {"t": NOW, "a": ask["ask_id"]})
    summary, opened = _run(pg, NOW + timedelta(days=1), cap=2)
    sam = ask["source_ref"].split("|")[0]
    assert summary["facts_applied"] == 1
    assert _rows(pg, "SELECT fact_id, category, value_text, voice FROM person_fact WHERE person_id=:p", p=sam) == [
        (ask["ask_id"], "relationship", "my neighbor", "ask_answer")]
    # The least-asked person goes first (Alex, never asked), then Sam's next category.
    assert [a["source_ref"] for a in opened] == ["referent-you|gender", f"{sam}|pronouns"]
    assert _rows(pg, "SELECT applied_at IS NOT NULL FROM orion_ask WHERE ask_id=:a", a=ask["ask_id"]) == [(True,)]


@needs_pg
def test_a_dismissed_question_is_not_asked_again(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _cluster(conn, "sam-1", "Sam")
    (ask,) = _run(pg, cap=1)[1]
    with pg.begin() as conn:
        conn.execute(text("UPDATE orion_ask SET status='dismissed' WHERE ask_id=:a"), {"a": ask["ask_id"]})
    refs = [a["source_ref"] for a in _run(pg, NOW + timedelta(days=1), cap=2)[1]]
    assert ask["source_ref"] not in refs
    assert any(r.endswith("|pronouns") and not r.startswith("referent-") for r in refs)


@needs_pg
def test_the_daily_cap_holds_and_resets_tomorrow(pg) -> None:
    with pg.begin() as conn:
        for i in range(3):
            _cluster(conn, f"p{i}", f"Person{i}")
    assert len(_run(pg, cap=2)[1]) == 2
    assert _run(pg, NOW + timedelta(minutes=5), cap=2)[1] == []
    assert len(_run(pg, NOW + timedelta(days=1), cap=2)[1]) == 2


@needs_pg
def test_a_rename_moves_the_link_and_a_different_verdict_blocks_a_merge(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _cluster(conn, "a", "Sam")
        _cluster(conn, "b", "Sam")
    _run(pg, cap=0)
    with pg.begin() as conn:
        conn.execute(text("UPDATE vision_individual SET label='Alex' WHERE individual_id='a'"))
        conn.execute(text("INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, ask_id, decided_at) "
                          "VALUES ('b', 'c', 'different', 'ask-d', :t)"), {"t": NOW})
        _cluster(conn, "c", "Sam")
    summary, _ = _run(pg, cap=0)
    active = dict(_rows(pg, "SELECT source_ref, person_id FROM person_link WHERE revoked_at IS NULL"))
    assert active["a"] == "referent-you"
    assert "c" not in active and summary["link_conflicts"] == 1
    assert _rows(pg, "SELECT count(*) FROM person_link WHERE source_ref='a' AND revoked_at IS NOT NULL") == [(1,)]


@needs_pg
def test_migration_missing_is_a_quiet_skip(pg, monkeypatch) -> None:
    from app import person_record as rec

    monkeypatch.setattr(rec, "tables_present", lambda conn: False)
    assert _run(pg) == ({"person_record": "migration_missing"}, [])


@needs_pg
def test_a_corrected_typo_leaves_no_person_to_ask_about(pg) -> None:
    """Review 2026-10-22: after a rename the old name kept being asked about."""
    from sqlalchemy import text

    with pg.begin() as conn:
        _cluster(conn, "a", "Jhon")
    _run(pg, cap=0)
    with pg.begin() as conn:
        conn.execute(text("UPDATE vision_individual SET label='John' WHERE individual_id='a'"))
    _, opened = _run(pg, NOW + timedelta(days=1), cap=5)
    assert sorted(a["question"] for a in opened) == ["What is your gender, in your own words?", "Who is John to you?"]


@needs_pg
def test_a_removed_fact_is_not_asked_about_again(pg) -> None:
    """Hub review 2026-10-22: removing a fact that had no question behind it (people.yaml, Hub edit)
    made the category askable again."""
    from sqlalchemy import text

    with pg.begin() as conn:
        _cluster(conn, "sam-1", "Sam")
    _run(pg, cap=0)
    with pg.begin() as conn:
        pid = conn.execute(text("SELECT person_id FROM person WHERE display_name='Sam'")).scalar()
        conn.execute(text("INSERT INTO person_fact (fact_id, person_id, category, value_text, voice, removed_at) "
                          "VALUES ('f-x', :p, 'relationship', 'wrong', 'hub_edit', now())"), {"p": pid})
    refs = [a["source_ref"] for a in _run(pg, cap=5)[1]]
    assert f"{pid}|relationship" not in refs and f"{pid}|pronouns" in refs

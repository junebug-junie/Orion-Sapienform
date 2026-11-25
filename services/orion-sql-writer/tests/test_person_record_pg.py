"""One record per person against a REAL Postgres (app/person_record.py).

Runs only when ``ORION_TEST_PG_URI`` points at a throwaway database with the
walkway migrations plus manual_migration_person_record_v1.sql applied -- it
TRUNCATEs the walkway and person tables:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55499:5432 postgres:16-alpine
    for f in walkway_camera_v1 walkway_privacy_v2 vision_confirm_asks_v1 vision_ask_names_v1 \\
             vision_split_v1 vision_osnet_shadow_v1 vision_person_album_v1 person_record_v1 person_merge_v1; do
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
                          "vision_individual, vision_ask_picture_verdict, vision_person_exemplar, vision_crop_observation, "
                          "person_fact, person_link, person CASCADE"))
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


THUMB_NAMED, THUMB_OTHER, THUMB_PICKED = ("thumb:" + c * 64 for c in "cde")


def _who_is_this(conn, iid, answer, refs, *, ask_id=None, at=NOW - timedelta(hours=2)):
    """An answered "who is this?" card: its first picture is the one she named."""
    import json

    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answer, answered_at, created_at, "
        "image_ref, image_refs) VALUES (:a, 'Who is this?', 'vision_individual', :i, 'answered', :ans, :t, :t, "
        ":img, CAST(:imgs AS jsonb))"),
        {"a": ask_id or f"who-{iid}", "i": iid, "ans": answer, "t": at, "img": refs[0], "imgs": json.dumps(refs)})


def _run(engine, now=NOW, cap=3):
    from app import person_record as rec

    with engine.begin() as conn:
        return rec.run_person_record(conn, now, rec.PersonRecordConfig(ask_daily_cap=cap))


def _rows(engine, sql, **params):
    from sqlalchemy import text

    with engine.begin() as conn:
        return [tuple(r) for r in conn.execute(text(sql), params).fetchall()]


@needs_pg
def test_names_become_one_person_each_with_links_and_questions(pg) -> None:
    with pg.begin() as conn:
        _cluster(conn, "sam-1", "Sam", days=5, ask="ask-a", thumb=True)
        _who_is_this(conn, "sam-1", "Sam", [THUMB_NAMED, THUMB_OTHER])
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
    # One question per person, most-seen first, never about Orion. Sam's shows only the picture
    # she named Sam, not the cluster's newest sighting nor the card's other picture.
    assert [(a["source_ref"], a["question"]) for a in opened] == [
        (f"{persons['Sam']}|relationship", "How do you know Sam? For example: a family friend, a neighbor, a classmate of your kid."),
        ("referent-you|gender", "What is your gender, in your own words?")]
    assert opened[0]["image_refs"] == [THUMB_NAMED] and opened[1]["image_refs"] == []
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
    assert sorted(a["question"] for a in opened) == ["How do you know John? For example: a family friend, a neighbor, a classmate of your kid.",
                                                       "What is your gender, in your own words?"]


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


@needs_pg
def test_a_person_question_shows_only_pictures_she_tied_to_that_name(pg) -> None:
    """2026-10-23, live: "Who is Maeve to you?" showed three different people, the
    newest pictures of a mixed cluster. An open question opened before the fix is
    corrected too, and a picture picked on an "Is this <name>?" card joins."""
    from sqlalchemy import text

    with pg.begin() as conn:
        _cluster(conn, "sam-1", "Sam", days=5, thumb=True)
        _who_is_this(conn, "sam-1", "Sam", [THUMB_NAMED, THUMB_OTHER])
        # She also marked the second picture "someone else": it never shows.
        conn.execute(text(
            "INSERT INTO vision_ask_picture_verdict (ask_id, image_ref, picture_index, individual_id, verdict, name, "
            "answered_at) VALUES ('who-sam-1', :r, 1, 'sam-1', 'different', NULL, :t)"),
            {"r": THUMB_OTHER, "t": NOW - timedelta(hours=2)})
    _summary, opened = _run(pg)
    sam = opened[0]
    assert sam["question"] == "How do you know Sam? For example: a family friend, a neighbor, a classmate of your kid." and sam["image_refs"] == [THUMB_NAMED]
    # An old question still showing the mixed pictures is corrected on the next pass.
    with pg.begin() as conn:
        conn.execute(text("UPDATE orion_ask SET image_refs = CAST(:imgs AS jsonb), image_ref = :o "
                          "WHERE ask_id = :a"),
                     {"imgs": f'["{THUMB_OTHER}", "{THUMB_NAMED}"]', "o": THUMB_OTHER, "a": sam["ask_id"]})
        # A picture picked as Sam on an "Is this Sam?" card of another, unnamed cluster.
        conn.execute(text(
            "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, camera_id, artifact_id, "
            "observed_at, label, score, box_xyxy, thumb_ref) VALUES ('crop-p', 'o', 'walkway', 'c', 'a', :t, "
            "'person', 0.9, '{0,0,1,1}', :r)"), {"t": NOW - timedelta(minutes=30), "r": THUMB_PICKED})
        conn.execute(text(
            "INSERT INTO vision_person_exemplar (exemplar_id, person_label, crop_id, polarity, source, ask_id, added_at) "
            "VALUES ('e1', 'sam', 'crop-p', 1, 'confirm_pick_yes', 'ask-c', :t)"), {"t": NOW - timedelta(minutes=30)})
    summary, _ = _run(pg, NOW + timedelta(minutes=1))
    assert summary["person_ask_pictures_refreshed"] == 1
    assert _rows(pg, "SELECT image_refs, image_ref FROM orion_ask WHERE ask_id = :a", a=sam["ask_id"]) == [
        ([THUMB_PICKED, THUMB_NAMED], THUMB_PICKED)]
    summary, _ = _run(pg, NOW + timedelta(minutes=2))
    assert summary["person_ask_pictures_refreshed"] == 0
    # She later says the named picture is "not Sam" on another card: it stops showing.
    with pg.begin() as conn:
        conn.execute(text(
            "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, camera_id, artifact_id, "
            "observed_at, label, score, box_xyxy, thumb_ref) VALUES ('crop-n', 'o', 'walkway', 'c', 'a', :t, "
            "'person', 0.9, '{0,0,1,1}', :r)"), {"t": NOW - timedelta(hours=2), "r": THUMB_NAMED})
        conn.execute(text(
            "INSERT INTO vision_person_exemplar (exemplar_id, person_label, crop_id, polarity, source, ask_id, added_at) "
            "VALUES ('e2', 'Sam', 'crop-n', -1, 'confirm_pick_no', 'ask-d', :t)"), {"t": NOW})
    _run(pg, NOW + timedelta(minutes=3))
    assert _rows(pg, "SELECT image_refs FROM orion_ask WHERE ask_id = :a", a=sam["ask_id"]) == [([THUMB_PICKED],)]


@needs_pg
def test_a_name_joined_as_the_same_person_moves_its_clusters_and_is_not_asked_about(pg) -> None:
    """2026-10-29: "Same person as..." in the Hub sets person.merged_into; the next pass
    moves the joined name's clusters, and no question is opened about that name."""
    from sqlalchemy import text

    with pg.begin() as conn:
        _cluster(conn, "sam-1", "Sam", days=5)
        _cluster(conn, "full-1", "Sam Lee", days=1)
    _run(pg, cap=0)
    persons = dict(_rows(pg, "SELECT display_name, person_id FROM person"))
    with pg.begin() as conn:
        conn.execute(text("UPDATE person SET merged_into = :t WHERE person_id = :p"),
                     {"t": persons["Sam Lee"], "p": persons["Sam"]})
        _cluster(conn, "sam-2", "sam", days=2)
    summary, opened = _run(pg, NOW + timedelta(minutes=1))
    links = _rows(pg, "SELECT source_ref, person_id FROM person_link WHERE revoked_at IS NULL ORDER BY source_ref")
    assert links == [("full-1", persons["Sam Lee"]), ("sam-1", persons["Sam Lee"]), ("sam-2", persons["Sam Lee"])]
    assert summary["link_conflicts"] == 0
    assert all(not a["source_ref"].startswith(persons["Sam"]) for a in opened)
    assert [a["source_ref"] for a in opened if a["source_ref"].startswith(persons["Sam Lee"])] == [
        f"{persons['Sam Lee']}|relationship"]
    # An answer to a question about the joined name that was already in when she
    # joined (review 2026-10-29) lands on the person kept, not on the hidden row.
    with pg.begin() as conn:
        conn.execute(text(
            "INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answer, answered_at, created_at) "
            "VALUES ('ask-late', 'How do you know Sam?', 'person_fact', :r, 'answered', 'my neighbor', :t, :t)"),
            {"r": f"{persons['Sam']}|relationship", "t": NOW})
    _run(pg, NOW + timedelta(minutes=2))
    assert _rows(pg, "SELECT person_id, value_text FROM person_fact WHERE fact_id = 'ask-late'") == [
        (persons["Sam Lee"], "my neighbor")]


def _story_answer(conn, ask_id, kind, iid, detail, at=NOW - timedelta(minutes=30)):
    import json

    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answered_at, created_at, "
        "answer_detail) VALUES (:a, 'q', :k, :i, 'answered', :t, :t, CAST(:d AS jsonb))"),
        {"a": ask_id, "k": kind, "i": iid, "t": at, "d": json.dumps(detail)})


@needs_pg
def test_story_reasons_become_how_you_could_tell_facts_on_the_linked_person(pg) -> None:
    """Design 2026-11-24 patch 1: a walkway story's "How could you tell?" reason lands on
    the person that cluster is linked to -- for "someone else", the person Orion guessed."""
    from sqlalchemy import text

    with pg.begin() as conn:
        _cluster(conn, "sam-1", "Sam")
        _cluster(conn, "robin-1", "Robin")
        _cluster(conn, "anon-1", None)
        _cluster(conn, "dog-1", "Biscuit", kind="dog")
        story = {"story_id": "s" * 32, "via": "motion_story"}
        _story_answer(conn, "ask-else", "vision_story_person", "sam-1",
                      dict(story, action="someone_else", why="Sam rarely pushes a stroller.\nNot his build."))
        _story_answer(conn, "ask-name", "vision_individual", "robin-1",
                      dict(story, action="name", name="Robin", why="White dog, green leash."))
        _story_answer(conn, "ask-anon", "vision_story_person", "anon-1",
                      dict(story, action="name", name="Kit", why="Kit's red coat."))
        _story_answer(conn, "ask-unsure", "vision_story_person", "sam-1",
                      dict(story, action="unsure", why="Too dark to tell."))     # says nothing about Sam
        _story_answer(conn, "ask-dog", "vision_individual", "dog-1",
                      dict(story, action="name", name="Biscuit", why="Fluffy, green leash."))
        _story_answer(conn, "ask-nowhy", "vision_story_person", "sam-1", dict(story, action="unsure"))
        # A card answer with no answer text is not a card reason either (apply_card_reasons needs one).
        _story_answer(conn, "ask-card", "vision_individual", "sam-1", {"why": "not a story answer"})
    summary, _ = _run(pg)
    assert (summary["story_reasons_stored"], summary["story_reasons_waiting"],
            summary["story_reasons_skipped"]) == (2, 1, 1)   # a dog gets no person record
    persons = dict(_rows(pg, "SELECT display_name, person_id FROM person"))
    assert _rows(pg, "SELECT fact_id, person_id, category, value_text, voice, source_ref FROM person_fact "
                     "WHERE voice = 'story_reason' ORDER BY fact_id") == [
        ("why-ask-else", persons["Sam"], "how_you_could_tell",
         "A sighting that was NOT them: Sam rarely pushes a stroller.\nNot his build.",
         "story_reason", "ask:ask-else"),
        ("why-ask-name", persons["Robin"], "how_you_could_tell", "White dog, green leash.",
         "story_reason", "ask:ask-name")]
    # Never asked about as a question.
    assert not _rows(pg, "SELECT 1 FROM orion_ask WHERE source_ref LIKE '%|how_you_could_tell'")
    # Re-running stores nothing new; a reason she removed on the People page does not come back.
    with pg.begin() as conn:
        conn.execute(text("UPDATE person_fact SET removed_at = :t WHERE fact_id = 'why-ask-else'"), {"t": NOW})
    summary2, _ = _run(pg, NOW + timedelta(minutes=1))
    assert (summary2["story_reasons_stored"], summary2["story_reasons_waiting"]) == (0, 1)
    # The unnamed cluster gets a name: its reason lands then.
    with pg.begin() as conn:
        conn.execute(text("UPDATE vision_individual SET label = 'Kit' WHERE individual_id = 'anon-1'"))
    summary3, _ = _run(pg, NOW + timedelta(minutes=2))
    assert (summary3["story_reasons_stored"], summary3["story_reasons_waiting"]) == (1, 0)


def _card_answer(conn, ask_id, kind, ref, answer, detail, at=NOW - timedelta(minutes=30)):
    import json

    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answer, answered_at, created_at, "
        "answer_detail) VALUES (:a, 'q', :k, :r, 'answered', :ans, :t, :t, CAST(:d AS jsonb))"),
        {"a": ask_id, "k": kind, "r": ref, "ans": answer, "t": at, "d": json.dumps(detail)})


@needs_pg
def test_card_reasons_land_on_the_named_person_and_a_no_says_not_them(pg) -> None:
    """2026-11-25: "How could you tell?" on Hub "Who is this?" / "Is this <name>?" cards."""
    with pg.begin() as conn:
        _cluster(conn, "sam-1", "Sam")
        _cluster(conn, "robin-1", "Robin")
        _cluster(conn, "anon-1", None)
        _card_answer(conn, "c-who", "vision_individual", "robin-1", "Robin", {"why": "Green leash."})
        _card_answer(conn, "c-yes", "vision_person_confirm", "anon-1|Sam", "yes", {"why": "His red cap."})
        _card_answer(conn, "c-no", "vision_person_confirm", "anon-1|Robin", "no", {"why": "Robin has no stroller."})
        _card_answer(conn, "c-mixed", "vision_person_confirm", "anon-1|Sam", "yes",
                     {"why": "one of them", "pictures": [{"verdict": "same"}, {"verdict": "different"}]})
        _card_answer(conn, "c-nobody", "vision_person_confirm", "anon-1|Nobody", "no", {"why": "no such person"})
    summary, _ = _run(pg)
    assert (summary["card_reasons_stored"], summary["card_reasons_waiting"],
            summary["card_reasons_skipped"]) == (3, 1, 1)
    persons = dict(_rows(pg, "SELECT display_name, person_id FROM person"))
    assert _rows(pg, "SELECT fact_id, person_id, category, value_text, voice, source_ref FROM person_fact "
                     "WHERE voice = 'card_reason' ORDER BY fact_id") == [
        ("why-c-no", persons["Robin"], "how_you_could_tell", "A sighting that was NOT them: Robin has no stroller.",
         "card_reason", "ask:c-no"),
        ("why-c-who", persons["Robin"], "how_you_could_tell", "Green leash.", "card_reason", "ask:c-who"),
        ("why-c-yes", persons["Sam"], "how_you_could_tell", "His red cap.", "card_reason", "ask:c-yes")]
    summary2, _ = _run(pg, NOW + timedelta(minutes=1))
    assert (summary2["card_reasons_stored"], summary2["card_reasons_waiting"]) == (0, 1)


def test_card_reason_plan_is_pure_and_skips_what_it_cannot_place() -> None:
    from app.person_record import card_reason_plan

    assert card_reason_plan("vision_individual", "i1", "Sam", None, " x ") == ("cluster", "i1", "x")
    assert card_reason_plan("vision_person_confirm", "i1|Sam", "no", None, "y") == (
        "name", "Sam", "A sighting that was NOT them: y")
    assert card_reason_plan("vision_person_confirm", "i1|Sam", "yes", [{"verdict": "same"}], "y") == ("name", "Sam", "y")
    assert card_reason_plan("vision_person_confirm", "i1|Sam", "yes",
                            [{"verdict": "same"}, {"verdict": "different"}], "y") is None
    assert card_reason_plan("vision_person_confirm", "i1", "yes", None, "y") is None
    assert card_reason_plan("vision_individual", "i1", "Sam", None, "  ") is None


@needs_pg
def test_a_failing_story_reason_never_rolls_back_the_links(pg, monkeypatch) -> None:
    from app import person_record as rec

    def boom(conn):
        from sqlalchemy import text
        conn.execute(text("SELECT no_such_column FROM person"))

    monkeypatch.setattr(rec, "apply_story_reasons", boom)
    with pg.begin() as conn:
        _cluster(conn, "sam-1", "Sam")
    summary, _ = _run(pg)
    assert "story_reasons_error" in summary
    assert _rows(pg, "SELECT count(*) FROM person_link WHERE source_ref = 'sam-1' AND revoked_at IS NULL") == [(1,)]

"""Visual confirmation asks (2026-10-15): "who is this?" from faces, "are
these the same person?", several pictures per ask, their own daily cap.

Pure rules first. Then the whole answer path against a REAL Postgres with the
three walkway migrations applied (opener -> orion_ask row -> Hub-style answer
-> label / pair verdict / name carried across). That part runs only when
``ORION_TEST_PG_URI`` points at a throwaway database -- it TRUNCATEs the
walkway tables -- e.g.:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55499:5432 postgres:16-alpine
    for f in walkway_camera_v1 walkway_privacy_v2 vision_confirm_asks_v1; do
      docker exec -i pg psql -U postgres < services/orion-sql-db/manual_migration_$f.sql; done
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55499/postgres pytest tests/test_vision_confirm_asks.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import vision_individuals as vi  # noqa: E402

TZ = ZoneInfo("America/Denver")
NOW = datetime(2026, 10, 15, 21, 45, tzinfo=timezone.utc)
FACE = lambda c: "thumb:" + c * 64  # noqa: E731


def _vec(*xs: float, dim: int = 8) -> tuple:
    v = list(xs) + [0.0] * (dim - len(xs))
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


# --- pure rules ------------------------------------------------------------------------------------


def test_face_rule_asks_same_day_for_a_person_with_enough_faces() -> None:
    ok = dict(label=None, kind="person", face_n=5, stationary=False, min_faces=5)
    assert vi.should_ask_by_face(**ok)
    assert not vi.should_ask_by_face(**{**ok, "face_n": 4})
    assert not vi.should_ask_by_face(**{**ok, "label": "Juniper"})
    assert not vi.should_ask_by_face(**{**ok, "kind": "dog"})
    assert not vi.should_ask_by_face(**{**ok, "stationary": True})
    assert not vi.should_ask_by_face(**{**ok, "min_faces": 0})  # 0 disables


def test_old_rule_is_unchanged_for_non_face_individuals() -> None:
    assert vi.should_ask(label=None, sighting_count=10, distinct_days=5, min_sightings=10, min_days=5)
    assert not vi.should_ask(label=None, sighting_count=10, distinct_days=4, min_sightings=10, min_days=5)


def test_pair_ref_is_sorted_and_round_trips() -> None:
    assert vi.pair_ref("b", "a") == "a|b"
    assert vi.parse_pair_ref("a|b") == ("a", "b")
    for bad in ("", "a", "b|a", "a|a", "a|b|c", "|b"):
        assert vi.parse_pair_ref(bad) is None


def test_same_person_pairs_band_order_and_skips() -> None:
    a = vi.FaceCandidate("a", "person", _vec(1, 0), 10)
    b = vi.FaceCandidate("b", "person", _vec(1, 0.3), 10)          # ~0.96 to a
    c = vi.FaceCandidate("c", "person", _vec(1, 1.2), 10)          # ~0.64 to a
    d = vi.FaceCandidate("d", "person", _vec(0, 1), 10)            # 0 to a
    pairs = vi.same_person_pairs([a, b, c, d], min_cosine=0.40)
    assert pairs[0][:2] == ("a", "b")
    assert all(sim >= 0.40 for _, _, sim in pairs)
    assert pairs == sorted(pairs, key=lambda t: -t[2])
    assert ("a", "d") not in {(x, y) for x, y, _ in pairs}
    # Already asked / answered pairs are skipped.
    assert ("a", "b") not in {(x, y) for x, y, _ in vi.same_person_pairs([a, b, c], min_cosine=0.4, skip_refs={"a|b"})}
    # Both already named: Juniper said who each one is.
    named = [vi.FaceCandidate("a", "person", _vec(1, 0), 10, "Juniper"),
             vi.FaceCandidate("b", "person", _vec(1, 0.3), 10, "Kid")]
    assert vi.same_person_pairs(named, min_cosine=0.4) == []
    # Above the match threshold but still apart: still asked (the reducer matches crop by crop).
    assert vi.same_person_pairs([a, b], min_cosine=0.40)[0][2] > 0.9
    # A dog is not a person; a face of another size is a different model.
    dog = vi.FaceCandidate("e", "dog", _vec(1, 0), 10)
    other = vi.FaceCandidate("f", "person", _vec(1, 0, dim=4), 10)
    assert vi.same_person_pairs([a, dog, other], min_cosine=0.0) == []


def test_people_already_confirmed_the_same_count_as_one() -> None:
    groups = vi.same_groups([("a", "b"), ("b", "c")])
    assert groups["a"] == groups["b"] == groups["c"]
    a = vi.FaceCandidate("a", "person", _vec(1, 0), 10)
    b = vi.FaceCandidate("b", "person", _vec(1, 0.1), 10)
    x = vi.FaceCandidate("x", "person", _vec(1, 0.4), 10)
    pairs = vi.same_person_pairs([a, b, x], min_cosine=0.4, groups=groups)
    # a~b is inside a group: never asked. x~a and x~b are one question: the most alike one.
    assert [(p, q) for p, q, _ in pairs] == [("b", "x")]
    # And none at all once x~(that group) has been asked under either id.
    assert vi.same_person_pairs([a, b, x], min_cosine=0.4, groups=groups, skip_refs={"a|x"}) == []


def test_parse_yes_no() -> None:
    for y in ("yes", "Yes", " YES. ", "same", "same person"):
        assert vi.parse_yes_no(y) is True
    for n in ("no", "No!", "different", "different people"):
        assert vi.parse_yes_no(n) is False
    for x in (None, "", "maybe", "Bob"):
        assert vi.parse_yes_no(x) is None


def test_ask_image_refs_faces_first_newest_first_no_repeats() -> None:
    rows = [
        (FACE("1"), NOW - timedelta(hours=3), False, FACE("a")),
        (FACE("2"), NOW - timedelta(hours=1), False, FACE("b")),
        (FACE("3"), NOW - timedelta(hours=2), False, None),
        (FACE("4"), NOW - timedelta(days=9), False, FACE("c")),   # too old to outlive the ask
        (FACE("5"), NOW, True, FACE("d")),                         # no-embed zone
    ]
    kw = dict(now=NOW, thumb_retention_days=10, ask_expiry_days=7)
    assert vi.ask_image_refs(rows, **kw) == [FACE("b"), FACE("a"), FACE("2")]
    assert vi.ask_image_refs(rows, limit=5, **kw) == [FACE("b"), FACE("a"), FACE("2"), FACE("3"), FACE("1")]
    assert vi.ask_image_refs(rows, faces_only=True, **kw) == [FACE("b"), FACE("a")]
    # image_ref (single) is still the best one.
    assert vi.ask_image_ref(rows, **kw) == FACE("b")
    assert vi.ask_image_refs(rows, now=NOW, thumb_retention_days=7, ask_expiry_days=7) == []


def test_questions_are_plain() -> None:
    q = vi.face_ask_question(face_n=47, sighting_count=1)
    assert "who this is" in q and "once" in q
    assert "same person" in vi.same_person_question(face_n_a=18, face_n_b=19)


def test_loop_config_carries_the_new_keys() -> None:
    from app.settings import Settings
    from app.vision_individuals_loop import individuals_config

    cfg = individuals_config(Settings())
    assert (cfg.ask_face_min_faces, cfg.same_ask_min_face_cosine, cfg.confirm_ask_daily_cap) == (5, 0.40, 6)
    assert cfg.ask_daily_cap == 2  # the slow rule keeps its own cap


def test_slow_rule_cap_does_not_count_face_asks() -> None:
    assert "NOT (evidence_refs @> CAST(:face AS jsonb))" in vi.ASKED_TODAY_SQL
    assert "source_kind = :same" in vi.CONFIRM_ASKED_TODAY_SQL


def test_migration_declares_its_consumers_and_is_additive() -> None:
    sql = (REPO / "services/orion-sql-db/manual_migration_vision_confirm_asks_v1.sql").read_text()
    assert sql.splitlines()[0] == "-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub"
    body = "\n".join(line for line in sql.splitlines() if not line.startswith("--")).upper()
    assert "DROP" not in body and "DELETE" not in body


def test_ask_contract_image_refs_is_additive() -> None:
    from orion.schemas.ask import OrionAskV1

    old = OrionAskV1(ask_id="x", question="q", source_kind="vision_individual", source_ref="i", image_ref="thumb:a")
    assert old.image_refs == []
    new = OrionAskV1(ask_id="y", question="q", source_kind="vision_same_individual", source_ref="a|b",
                     image_ref=FACE("a"), image_refs=[FACE("a"), FACE("b")])
    assert new.model_dump(mode="json")["image_refs"] == [FACE("a"), FACE("b")]


# --- the whole path against real Postgres ------------------------------------------------------------

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE orion_ask, vision_individual_pair_verdict, vision_individual_sighting, "
                          "vision_individual CASCADE"))
    yield engine
    engine.dispose()


def _add_person(conn, iid: str, face, face_n: int, face_thumb: str, *, label=None, kind="person") -> None:
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, centroid_n, first_seen_at, "
        "last_seen_at, sighting_count, distinct_days, label, face_centroid, face_n) "
        "VALUES (:i, 'walkway', :k, :c, :n, :t, :t, 1, 1, :l, :f, :fn)"),
        {"i": iid, "k": kind, "c": list(_vec(1)), "n": face_n, "t": NOW - timedelta(minutes=10), "l": label,
         "f": list(face), "fn": face_n})
    conn.execute(text(
        "INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, started_at, ended_at, "
        "dwell_sec, observation_count, thumb_ref, face_thumb_ref) VALUES (:s, :i, 'walkway', :t, :t, 0, 3, :th, :f)"),
        {"s": "s-" + iid, "i": iid, "t": NOW - timedelta(minutes=10), "th": "thumb:" + ("f" + face_thumb[-1]) * 32,
         "f": face_thumb})


def _answer(conn, ask_id: str, answer) -> None:
    """What the Hub's POST /api/asks/{id}/answer does to the row."""
    from sqlalchemy import text

    conn.execute(text("UPDATE orion_ask SET status='answered', answer=:a, answered_at=:t WHERE ask_id=:i"),
                 {"a": answer, "t": NOW, "i": ask_id})


@needs_pg
def test_end_to_end_pairs_faces_answers_and_learning(pg) -> None:
    from sqlalchemy import text

    cfg = vi.IndividualsConfig()
    with pg.begin() as conn:
        vi._check_tables(conn)  # all three migrations present
        _add_person(conn, "aaa", _vec(1, 0.0), 18, FACE("a"))
        _add_person(conn, "bbb", _vec(1, 0.5), 19, FACE("b"))           # ~0.89 to aaa
        _add_person(conn, "ccc", _vec(0, 0, 1.0), 47, FACE("c"))        # far from both
        _add_person(conn, "ddd", _vec(0, 0, 1.0, 0, 0.1), 3, FACE("d"))  # too few faces
        _add_person(conn, "eee", _vec(0, 0, 1.0, 0.2), 9, FACE("e"), label="Juniper")
        opened = vi._open_asks(conn, NOW, TZ, cfg)

    kinds = sorted((a["source_kind"], a["source_ref"]) for a in opened)
    # ccc~eee is ~0.98 alike but only one is named: asked. aaa~bbb asked.
    assert ("vision_same_individual", "aaa|bbb") in kinds
    assert ("vision_same_individual", "ccc|eee") in kinds
    # aaa, bbb, ccc are in open pair asks -> no "who is this?" yet; ddd has too few faces; eee is named.
    assert [k for k in kinds if k[0] == "vision_individual"] == []
    pair = next(a for a in opened if a["source_ref"] == "aaa|bbb")
    assert pair["image_refs"] == [FACE("a"), FACE("b")] and pair["image_ref"] == FACE("a")

    with pg.begin() as conn:
        row = conn.execute(text("SELECT image_refs, image_ref FROM orion_ask WHERE source_ref='aaa|bbb'")).one()
        assert row[0] == [FACE("a"), FACE("b")] and row[1] == FACE("a")
        # Re-running opens nothing new (one open ask per subject, pairs already asked).
        assert vi._open_asks(conn, NOW, TZ, cfg) == []

    by_ref = {a["source_ref"]: a["ask_id"] for a in opened}
    with pg.begin() as conn:
        _answer(conn, by_ref["aaa|bbb"], "no")
        _answer(conn, by_ref["ccc|eee"], "yes")
        assert vi._apply_answers(conn, NOW) == 2

    with pg.begin() as conn:
        verdicts = {(r[0], r[1]): (r[2], r[3]) for r in conn.execute(text(
            "SELECT individual_a, individual_b, verdict, face_cosine FROM vision_individual_pair_verdict")).fetchall()}
        assert verdicts[("aaa", "bbb")][0] == "different"
        assert verdicts[("ccc", "eee")][0] == "same"
        assert abs(verdicts[("aaa", "bbb")][1] - vi.cosine(_vec(1, 0.0), _vec(1, 0.5))) < 1e-3  # calibration point
        labels = dict(conn.execute(text("SELECT individual_id, label FROM vision_individual")).fetchall())
        assert labels["ccc"] == "Juniper"   # carried across a 'same' verdict
        assert labels["aaa"] is None and labels["bbb"] is None
        # Next tick: the answered pairs are never asked again; aaa and bbb now get "who is this?".
        opened2 = vi._open_asks(conn, NOW + timedelta(minutes=1), TZ, cfg)
    refs2 = sorted((a["source_kind"], a["source_ref"]) for a in opened2)
    assert refs2 == [("vision_individual", "aaa"), ("vision_individual", "bbb")]
    who = next(a for a in opened2 if a["source_ref"] == "aaa")
    assert who["image_refs"][0] == FACE("a") and vi.FACE_RULE_REF in who["evidence_refs"]

    # Answering "who is this?" for aaa names aaa only (the pair said different).
    with pg.begin() as conn:
        _answer(conn, who["ask_id"], "Kid next door")
        vi._apply_answers(conn, NOW)
        labels = dict(conn.execute(text("SELECT individual_id, label FROM vision_individual")).fetchall())
    assert labels["aaa"] == "Kid next door" and labels["bbb"] is None


@needs_pg
def test_face_cap_is_separate_from_the_slow_rule_cap(pg) -> None:
    from sqlalchemy import text

    cfg = vi.IndividualsConfig(confirm_ask_daily_cap=1, ask_daily_cap=2)
    with pg.begin() as conn:
        for i, axis in enumerate(range(3)):
            _add_person(conn, f"p{i}", _vec(*([0.0] * axis + [1.0])), 6, FACE(str(i)))
        first = vi._open_asks(conn, NOW, TZ, cfg)
        assert len(first) == 1
        assert vi._open_asks(conn, NOW + timedelta(minutes=1), TZ, cfg) == []   # face cap used up today
        # The slow rule's own count ignores face asks.
        n = conn.execute(text(vi.ASKED_TODAY_SQL), {"d": vi.local_day_start(NOW, TZ),
                                                    "face": json.dumps([vi.FACE_RULE_REF])}).scalar()
        assert n == 0
        # Tomorrow (local), the cap resets.
        assert len(vi._open_asks(conn, NOW + timedelta(days=1), TZ, cfg)) == 1


@needs_pg
def test_unparseable_pair_answer_is_applied_as_nothing(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _add_person(conn, "aaa", _vec(1, 0.0), 18, FACE("a"))
        _add_person(conn, "bbb", _vec(1, 0.5), 19, FACE("b"))
        (ask,) = vi._open_asks(conn, NOW, TZ, vi.IndividualsConfig())
        _answer(conn, ask["ask_id"], "maybe")
        assert vi._apply_answers(conn, NOW) == 1
        assert conn.execute(text("SELECT count(*) FROM vision_individual_pair_verdict")).scalar() == 0
        assert conn.execute(text("SELECT applied_at FROM orion_ask WHERE ask_id=:a"), {"a": ask["ask_id"]}).scalar()


@needs_pg
def test_pair_needs_both_faces_on_the_card(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _add_person(conn, "aaa", _vec(1, 0.0), 18, FACE("a"))
        _add_person(conn, "bbb", _vec(1, 0.5), 19, FACE("b"))
        conn.execute(text("UPDATE vision_individual_sighting SET face_thumb_ref=NULL WHERE individual_id='bbb'"))
        opened = vi._open_asks(conn, NOW, TZ, vi.IndividualsConfig())
    assert all(a["source_kind"] != "vision_same_individual" for a in opened)


@needs_pg
def test_prune_keeps_individuals_in_an_open_pair_ask(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _add_person(conn, "aaa", _vec(1, 0.0), 18, FACE("a"))
        _add_person(conn, "bbb", _vec(1, 0.5), 19, FACE("b"))
        vi._open_asks(conn, NOW, TZ, vi.IndividualsConfig())
        conn.execute(text("UPDATE vision_individual SET last_seen_at = :t"), {"t": NOW - timedelta(days=200)})
        summary: dict = {}
        vi._prune(conn, NOW, vi.IndividualsConfig(), summary)
        left = {r[0] for r in conn.execute(text("SELECT individual_id FROM vision_individual")).fetchall()}
    assert {"aaa", "bbb"} <= left


@needs_pg
def test_one_who_is_this_per_confirmed_person(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _add_person(conn, "aaa", _vec(1, 0.0), 18, FACE("a"))
        _add_person(conn, "bbb", _vec(1, 0.5), 19, FACE("b"))
        conn.execute(text("INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, "
                          "face_cosine, ask_id, decided_at) VALUES ('aaa','bbb','same',0.89,'ask-x',:t)"), {"t": NOW})
        opened = vi._open_asks(conn, NOW, TZ, vi.IndividualsConfig())
    # No pair ask (already answered), and only ONE "who is this?" for the two ids.
    assert [(a["source_kind"], a["source_ref"]) for a in opened] == [("vision_individual", "bbb")]
    with pg.begin() as conn:
        _answer(conn, opened[0]["ask_id"], "Juniper")
        vi._apply_answers(conn, NOW)
        labels = dict(conn.execute(text("SELECT individual_id, label FROM vision_individual")).fetchall())
    assert labels == {"aaa": "Juniper", "bbb": "Juniper"}

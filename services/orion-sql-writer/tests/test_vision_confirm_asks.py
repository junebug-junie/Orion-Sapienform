"""Visual confirmation asks (2026-10-15): "who is this?" from faces, "are
these the same person?", several pictures per ask, their own daily cap.

Pure rules first. Then the whole answer path against a REAL Postgres with the
three walkway migrations applied (opener -> orion_ask row -> Hub-style answer
-> label / pair verdict / name carried across). That part runs only when
``ORION_TEST_PG_URI`` points at a throwaway database -- it TRUNCATEs the
walkway tables -- e.g.:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55499:5432 postgres:16-alpine
    for f in walkway_camera_v1 walkway_privacy_v2 vision_confirm_asks_v1 vision_ask_names_v1; do
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
                          "vision_individual, vision_ask_picture_verdict, vision_crop_observation CASCADE"))
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


@needs_pg
def test_a_name_reaches_the_whole_confirmed_group(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        for iid, c in (("aaa", "a"), ("bbb", "b"), ("ccc", "c")):
            _add_person(conn, iid, _vec(1, 0.0), 6, FACE(c))
        # b=c first, then a=b: naming a must reach c (two hops).
        for a, b in (("bbb", "ccc"), ("aaa", "bbb")):
            conn.execute(text("INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, "
                              "ask_id, decided_at) VALUES (:a, :b, 'same', 'ask-x', :t)"), {"a": a, "b": b, "t": NOW})
        conn.execute(text("INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answer, "
                          "answered_at) VALUES ('ask-who', 'q', 'vision_individual', 'aaa', 'answered', 'Juniper', :t)"),
                     {"t": NOW})
        vi._apply_answers(conn, NOW)
        labels = dict(conn.execute(text("SELECT individual_id, label FROM vision_individual")).fetchall())
    assert labels == {"aaa": "Juniper", "bbb": "Juniper", "ccc": "Juniper"}


@needs_pg
def test_not_sure_pair_comes_back_after_the_cooldown(pg) -> None:
    from sqlalchemy import text

    cfg = vi.IndividualsConfig()
    with pg.begin() as conn:
        _add_person(conn, "aaa", _vec(1, 0.0), 18, FACE("a"))
        _add_person(conn, "bbb", _vec(1, 0.5), 19, FACE("b"))
        (ask,) = [a for a in vi._open_asks(conn, NOW, TZ, cfg) if a["source_kind"] == vi.SOURCE_KIND_SAME]
        conn.execute(text("UPDATE orion_ask SET status='dismissed' WHERE ask_id=:a"), {"a": ask["ask_id"]})
        later = NOW + timedelta(days=1)
        assert all(a["source_kind"] != vi.SOURCE_KIND_SAME for a in vi._open_asks(conn, later, TZ, cfg))
        # Thumbs must still be young enough: move the sightings forward with the clock.
        much_later = NOW + timedelta(days=cfg.ask_cooldown_days + 1)
        conn.execute(text("UPDATE vision_individual SET last_seen_at=:t"), {"t": much_later})
        conn.execute(text("UPDATE vision_individual_sighting SET ended_at=:t, started_at=:t"), {"t": much_later})
        again = [a for a in vi._open_asks(conn, much_later, TZ, cfg) if a["source_kind"] == vi.SOURCE_KIND_SAME]
    assert [a["source_ref"] for a in again] == ["aaa|bbb"]


# --- names typed on the card (2026-10-15) ---------------------------------------------------------


def test_names_migration_declares_its_consumers_and_is_additive() -> None:
    sql = (REPO / "services/orion-sql-db/manual_migration_vision_ask_names_v1.sql").read_text()
    assert sql.splitlines()[0] == "-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub"
    body = "\n".join(line for line in sql.splitlines() if not line.startswith("--")).upper()
    assert "DROP" not in body and "DELETE" not in body and "TRUNCATE" not in body


def test_pair_names_rules() -> None:
    pair = ("aaa", "bbb")
    assert vi.pair_names(pair, None, {"names": {"aaa": "X"}}) == {}            # not a yes/no: nothing
    assert vi.pair_names(pair, True, {}) == {}
    assert vi.pair_names(pair, True, {"names": {"aaa": " Kid  next door "}}) == {"aaa": "Kid next door",
                                                                               "bbb": "Kid next door"}
    assert vi.pair_names(pair, True, {"names": {"aaa": "X", "bbb": "Y"}}) == {}  # two names for one person
    assert vi.pair_names(pair, False, {"names": {"aaa": "X", "bbb": "", "zzz": "Q"}}) == {"aaa": "X"}


def test_picture_rows_rules() -> None:
    detail = {"pictures": [
        {"image_ref": "r0", "index": 0, "verdict": "same", "name": "Theo"},
        {"image_ref": "r1", "index": 1, "verdict": "unsure", "name": "ignored"},
        {"image_ref": "r2", "index": 2, "verdict": "different", "name": " Other  kid "},
        {"image_ref": "r2", "index": 2, "verdict": "same"},       # repeat: dropped
        {"image_ref": "r3", "index": 3, "verdict": "bogus"},      # unknown verdict: dropped
    ]}
    rows = vi.picture_rows("ask", "ind", detail, NOW)
    assert [(r["ref"], r["idx"], r["v"], r["name"]) for r in rows] == [
        ("r0", 0, "same", "Theo"), ("r1", 1, "unsure", None), ("r2", 2, "different", "Other kid")]
    assert vi.picture_rows("ask", "ind", {}, NOW) == []


def _ask(conn, ask_id: str, kind: str, ref: str, *, status="open", answer=None, detail=None, at=NOW) -> None:
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answer, answered_at, answer_detail) "
        "VALUES (:i, 'q', :k, :r, :s, :a, :t, CAST(:d AS jsonb))"),
        {"i": ask_id, "k": kind, "r": ref, "s": status, "a": answer, "t": at if status != "open" else None,
         "d": None if detail is None else json.dumps(detail)})


def _labels(conn) -> dict:
    from sqlalchemy import text

    return {r[0]: (r[1], r[2]) for r in conn.execute(text(
        "SELECT individual_id, label, label_copied FROM vision_individual")).fetchall()}


def _same(conn, a: str, b: str) -> None:
    from sqlalchemy import text

    conn.execute(text("INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, ask_id, "
                      "decided_at) VALUES (:a, :b, 'same', 'ask-old', :t)"), {"a": a, "b": b, "t": NOW})


@needs_pg
def test_same_with_a_name_labels_the_whole_group_and_closes_its_cards(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        for iid, c in (("aaa", "a"), ("bbb", "b"), ("ccc", "c"), ("ddd", "d")):
            _add_person(conn, iid, _vec(1, 0.0), 6, FACE(c))
        # ccc was already confirmed the same as bbb and carries a COPIED name; ddd (same as aaa)
        # has a name Juniper typed for it.
        _same(conn, "bbb", "ccc")
        _same(conn, "aaa", "ddd")
        conn.execute(text("UPDATE vision_individual SET label='Old copy', label_copied=true WHERE individual_id='ccc'"))
        conn.execute(text("UPDATE vision_individual SET label='Typed', label_copied=false WHERE individual_id='ddd'"))
        _ask(conn, "who-ccc", "vision_individual", "ccc")     # still open: the name reaches it
        _ask(conn, "pair", "vision_same_individual", "aaa|bbb", status="answered", answer="yes",
             detail={"names": {"aaa": "Kid next door", "bbb": "Kid next door"}})
        assert vi._apply_answers(conn, NOW) == 1
        labels = _labels(conn)
        closed = conn.execute(text("SELECT status, answer, applied_at IS NOT NULL, answer_detail FROM orion_ask "
                                   "WHERE ask_id='who-ccc'")).one()
        verdict = conn.execute(text("SELECT verdict FROM vision_individual_pair_verdict "
                                    "WHERE individual_a='aaa' AND individual_b='bbb'")).scalar()
    assert verdict == "same"
    assert labels["aaa"] == ("Kid next door", False) and labels["bbb"] == ("Kid next door", False)
    assert labels["ccc"] == ("Kid next door", True)   # a copied name is replaced by a typed one
    assert labels["ddd"] == ("Typed", False)          # a name she typed for ddd is never replaced by a copy
    assert closed == ("answered", "Kid next door", True, {"resolved_by": "pair"})


@needs_pg
def test_different_with_two_names_labels_each(pg) -> None:
    with pg.begin() as conn:
        _add_person(conn, "aaa", _vec(1, 0.0), 6, FACE("a"))
        _add_person(conn, "bbb", _vec(1, 0.5), 6, FACE("b"))
        _ask(conn, "pair", "vision_same_individual", "aaa|bbb", status="answered", answer="no",
             detail={"names": {"aaa": "Theo", "bbb": "Maya"}})
        vi._apply_answers(conn, NOW)
        assert _labels(conn) == {"aaa": ("Theo", False), "bbb": ("Maya", False)}


@needs_pg
def test_no_names_is_a_verdict_only(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _add_person(conn, "aaa", _vec(1, 0.0), 6, FACE("a"))
        _add_person(conn, "bbb", _vec(1, 0.5), 6, FACE("b"))
        _ask(conn, "who-aaa", "vision_individual", "aaa")
        _ask(conn, "pair", "vision_same_individual", "aaa|bbb", status="answered", answer="no")
        vi._apply_answers(conn, NOW)
        assert _labels(conn) == {"aaa": (None, False), "bbb": (None, False)}
        assert conn.execute(text("SELECT status FROM orion_ask WHERE ask_id='who-aaa'")).scalar() == "open"
        assert conn.execute(text("SELECT count(*) FROM vision_individual_pair_verdict")).scalar() == 1


@needs_pg
def test_who_is_this_closed_by_the_hub_applies_as_a_typed_name(pg) -> None:
    """The Hub closes an open "who is this?" card with the pair's name (resolved_by);
    sql-writer applies that card like any answer, in answer order."""
    from sqlalchemy import text

    with pg.begin() as conn:
        _add_person(conn, "aaa", _vec(1, 0.0), 6, FACE("a"))
        _add_person(conn, "bbb", _vec(1, 0.5), 6, FACE("b"))
        _ask(conn, "pair", "vision_same_individual", "aaa|bbb", status="answered", answer="no",
             detail={"names": {"aaa": "Theo"}})
        _ask(conn, "who-aaa", "vision_individual", "aaa", status="answered", answer="Theo",
             detail={"resolved_by": "pair"})
        assert vi._apply_answers(conn, NOW) == 2
        assert _labels(conn)["aaa"] == ("Theo", False)
        assert conn.execute(text("SELECT count(*) FROM orion_ask WHERE applied_at IS NULL")).scalar() == 0


@needs_pg
@pytest.mark.parametrize("first,second,winner", [("Alice", "Bob", "Bob"), ("Bob", "Alice", "Alice")])
def test_last_typed_name_wins(pg, first, second, winner) -> None:
    with pg.begin() as conn:
        _add_person(conn, "aaa", _vec(1, 0.0), 6, FACE("a"))
        _add_person(conn, "bbb", _vec(1, 0.5), 6, FACE("b"))
        # Inserted in the opposite order to their answer times: answer time decides.
        _ask(conn, "z-pair", "vision_same_individual", "aaa|bbb", status="answered", answer="no",
             detail={"names": {"aaa": second}}, at=NOW)
        _ask(conn, "a-who", "vision_individual", "aaa", status="answered", answer=first,
             at=NOW - timedelta(minutes=1))
        vi._apply_answers(conn, NOW)
        assert _labels(conn)["aaa"] == (winner, False)


@needs_pg
def test_who_is_this_pictures_are_recorded_and_the_face_name_labels(pg) -> None:
    from sqlalchemy import text

    refs = [FACE("a"), "thumb:" + "e" * 64, "thumb:" + "9" * 64]
    with pg.begin() as conn:
        _add_person(conn, "aaa", _vec(1, 0.0), 6, FACE("a"))
        conn.execute(text(
            "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, observed_at, label, score, "
            "box_xyxy, thumb_ref) VALUES ('crop-2', 'obs', 'walkway', :t, 'person', 0.9, '{0,0,1,1}', :r)"),
            {"t": NOW, "r": refs[2]})
        _ask(conn, "who", "vision_individual", "aaa", status="answered", answer="Theo", detail={"pictures": [
            {"image_ref": refs[0], "index": 0, "verdict": "same", "name": "Theo"},
            {"image_ref": refs[1], "index": 1, "verdict": "unsure", "name": None},
            {"image_ref": refs[2], "index": 2, "verdict": "different", "name": "Other kid"}]})
        vi._apply_answers(conn, NOW)
        vi._apply_answers(conn, NOW)  # applied once: a second tick records nothing twice
        rows = conn.execute(text("SELECT picture_index, verdict, name, crop_id, individual_id "
                                 "FROM vision_ask_picture_verdict ORDER BY picture_index")).fetchall()
        assert _labels(conn)["aaa"] == ("Theo", False)   # the individual: the face picture's name
        # Record only: nothing split, no new individual, no pair verdict invented for a crop.
        assert conn.execute(text("SELECT count(*) FROM vision_individual")).scalar() == 1
        assert conn.execute(text("SELECT count(*) FROM vision_individual_pair_verdict")).scalar() == 0
    assert [tuple(r) for r in rows] == [(0, "same", "Theo", None, "aaa"), (1, "unsure", None, None, "aaa"),
                                        (2, "different", "Other kid", "crop-2", "aaa")]


@needs_pg
def test_check_tables_names_the_names_migration(pg) -> None:
    from sqlalchemy import text

    with pg.connect() as conn:
        tx = conn.begin()
        conn.execute(text("ALTER TABLE orion_ask DROP COLUMN answer_detail"))
        with pytest.raises(vi.MigrationMissing, match="manual_migration_vision_ask_names_v1"):
            vi._check_tables(conn)
        tx.rollback()

"""Splitting walkway individuals on "someone else" picture answers (2026-10-16).

docs/superpowers/specs/2026-10-16-vision-individual-split-proposal.md.

Pure rules first (who moves, who follows, majority guard, cannot-link in the
matcher, the migration's auto-apply safety). Then the whole path against a
REAL Postgres (``ORION_TEST_PG_URI``, a throwaway database: it TRUNCATEs the
walkway tables), with the five walkway migrations applied:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55499:5432 postgres:16-alpine
    for f in walkway_camera_v1 walkway_privacy_v2 vision_confirm_asks_v1 vision_ask_names_v1 vision_split_v1; do
      docker exec -i pg psql -U postgres < services/orion-sql-db/manual_migration_$f.sql; done
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55499/postgres pytest tests/test_vision_split.py
"""

from __future__ import annotations

import math
import os
import subprocess
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
from app import vision_split as vs  # noqa: E402

TZ = ZoneInfo("America/Denver")
NOW = datetime(2026, 10, 16, 3, 0, tzinfo=timezone.utc)
MIGRATION = REPO / "services" / "orion-sql-db" / "manual_migration_vision_split_v1.sql"


def _vec(*xs: float, dim: int = 8) -> tuple:
    v = list(xs) + [0.0] * (dim - len(xs))
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def _sc(cid: str, minute: int, emb=None, face=None) -> vs.SplitCrop:
    return vs.SplitCrop(cid, NOW + timedelta(minutes=minute), emb or _vec(1), face)


# --- pure: who moves ---------------------------------------------------------------------------------

FK = _vec(1, 0, 0)        # the confirmed picture's face
FM = _vec(0, 1, 0)        # the "someone else" picture's face


def test_named_crops_move_and_close_faces_follow_body_only_never_does() -> None:
    plan = vs.plan_split(
        source_n=100, move_seed=[_sc("m0", 0, face=FM)], keep_seed=[_sc("k0", 1, face=FK)],
        members=[
            _sc("f1", 2, face=_vec(0.1, 1, 0)),     # ~0.99 to FM, ~0.10 to FK: follows
            _sc("f2", 3, face=_vec(1, 1.05, 0)),    # ~0.72 to FM, ~0.69 to FK: margin too small
            _sc("f3", 4, face=_vec(1, 0.3, 0)),     # nearer the confirmed face
            _sc("b1", 5, emb=_vec(0, 1), face=None),  # body-only: never follows
        ])
    assert plan.refused is None
    assert plan.moved == [("m0", vs.MOVED_BY_VERDICT), ("f1", vs.MOVED_BY_FACE)]
    assert (plan.by_verdict, plan.by_face) == (1, 1)


def test_no_face_on_the_moved_picture_means_no_followers() -> None:
    plan = vs.plan_split(source_n=100, move_seed=[_sc("m0", 0), _sc("m1", 1)], keep_seed=[_sc("k0", 2, face=FK)],
                         members=[_sc("f1", 3, face=FM), _sc("f2", 4, face=FK)])
    assert plan.moved == [("m0", "verdict"), ("m1", "verdict")]


def test_keep_side_falls_back_to_the_source_face_centroid() -> None:
    # No face on the confirmed pictures: the source's own face centroid is the keep side.
    member = _sc("f1", 3, face=_vec(0.5, 1, 0))
    near_keep = vs.plan_split(source_n=100, move_seed=[_sc("m0", 0, face=FM)], keep_seed=[],
                              members=[member], source_face_centroid=list(_vec(0.5, 1, 0)))
    assert near_keep.by_face == 0
    far_keep = vs.plan_split(source_n=100, move_seed=[_sc("m0", 0, face=FM)], keep_seed=[],
                             members=[member], source_face_centroid=list(FK))
    assert far_keep.by_face == 1


def test_a_majority_move_is_refused() -> None:
    plan = vs.plan_split(source_n=3, move_seed=[_sc("m0", 0, face=FM), _sc("m1", 1)], keep_seed=[],
                         members=[])
    assert plan.refused == "majority"
    assert vs.plan_split(source_n=4, move_seed=[_sc("m0", 0), _sc("m1", 1)], keep_seed=[], members=[]).refused is None


def test_followers_that_push_a_split_over_the_limit_are_dropped_not_the_answer() -> None:
    plan = vs.plan_split(source_n=4, move_seed=[_sc("m0", 0, face=FM), _sc("m1", 1)], keep_seed=[],
                         members=[_sc("f1", 2, face=FM)], source_face_centroid=list(FK))
    assert plan.refused is None and plan.followers_dropped
    assert plan.moved == [("m0", "verdict"), ("m1", "verdict")]


def test_verdict_groups_by_name_case_insensitive_and_unnamed_together() -> None:
    P = vs.PictureVerdict
    keep, groups = vs.verdict_groups([
        P(0, "same", None, "c0"), P(1, "different", "Cora", "c1", "cora"), P(2, "different", "cora", "c2", "cora"),
        P(3, "different", None, "c3", ""), P(4, "different", None, "c4", ""), P(5, "unsure", None, "c5"),
        P(6, "different", "Dee", None, "dee"),
    ])
    assert keep == ["c0"]
    assert groups["cora"] == ("Cora", ["c1", "c2"])
    assert groups[""] == (None, ["c3", "c4"])
    assert groups["dee"] == ("Dee", [])


def test_group_sightings_uses_the_merge_gap_and_newest_pictures() -> None:
    def c(cid, sec, thumb=None):
        return {"crop_id": cid, "stream_id": "walkway", "observed_at": NOW + timedelta(seconds=sec), "zone": "walk",
                "box_xyxy": [0, 0, 1, 1], "thumb_ref": thumb, "face_thumb_ref": None, "embedding_ref": None}
    ids = iter(["s1", "s2"])
    out = vs.group_sightings([c("a", 0, "thumb:a"), c("b", 30), c("z", 500, "thumb:z")], individual_id="new",
                             merge_gap_sec=60, new_id=lambda: next(ids))
    assert [(s["sighting_id"], s["observation_count"], s["thumb_ref"], s["evidence_ref"]) for s in out] == [
        ("s1", 2, "thumb:a", "crop:b"), ("s2", 1, "thumb:z", "crop:z")]
    assert out[0]["dwell_sec"] == 30 and out[0]["zone"] == "walk"


def test_remove_then_add_centroid_round_trips() -> None:
    vecs = [_vec(1, 0.1 * i, 0.05 * i) for i in range(6)]
    c = vecs[0]
    n = 1
    for v in vecs[1:]:
        c = vi.update_centroid(c, n, v)
        n += 1
    less = vs.remove_from_centroid(c, n, vecs[-2:])
    back = vs.add_to_centroid(less, n - 2, vecs[-2:])
    assert vi.cosine(back, c) > 0.9999
    assert vs.remove_from_centroid(c, 2, vecs[:2]) is None


def test_face_threshold_calibration_holds_until_enough_separable_pairs() -> None:
    # Today's labelled data (the proposal's table): 2 same, 7 different, overlapping.
    today = vs.face_threshold_calibration([0.533, 0.151], [0.758, 0.646, 0.582, 0.511, 0.405, 0.145, 0.125])
    assert today["verdict"] == "hold" and today["overlap"] is True
    same = [0.70 + 0.01 * i for i in range(20)]
    diff = [0.20 + 0.01 * i for i in range(20)]
    ok = vs.face_threshold_calibration(same, diff)
    assert ok["verdict"] == "propose" and ok["false_accept"] <= 0.05 and ok["recall"] >= 0.5
    assert abs(ok["threshold"] - 0.41) < 1e-9          # smallest t with <= 5% different (0.39) + 0.02
    mixed = vs.face_threshold_calibration([0.3] * 20, [0.9] * 20)
    assert mixed["verdict"] == "hold" and "separates" in mixed["reason"]


# --- pure: cannot-link in the matcher -------------------------------------------------------------


def _crop(cid: str, face, emb=_vec(0, 0, 0, 0, 1), obs="o1") -> vi.CropRow:
    return vi.CropRow(crop_id=cid, observation_id=obs, stream_id="walkway", observed_at=NOW, label="person",
                      box_xyxy=(0, 0, 10, 10), zone=None, embedding=emb, embedding_ref=None, artifact_id=None,
                      face_embedding=face)


def _ind(iid: str, face, body=_vec(1)) -> vi.Individual:
    return vi.Individual(individual_id=iid, stream_id="walkway", kind="person", centroid=list(body), centroid_n=5,
                         first_seen_at=NOW - timedelta(hours=1), last_seen_at=NOW - timedelta(hours=1),
                         face_centroid=list(face), face_n=5)


def test_cannot_link_blocks_an_ambiguous_face_match_and_falls_to_body() -> None:
    x = _ind("x", _vec(1, 0.20))                      # crop face ~0.98 to x
    y = _ind("y", _vec(1, 0.45))                      # ~0.93 to y: within 0.10 of x
    face = _vec(1, 0.0)
    inds = {"x": x, "y": y}
    free = vi.apply_batch([_crop("c", face)], individuals=inds, latest_sightings={}, no_embed_zones=(),
                          match_threshold=0.8, merge_gap_sec=60, tz=TZ, new_id=lambda: "new")
    assert free.crop_individual == {"c": "x"} and free.assigned_by_face == 1
    linked = vi.apply_batch([_crop("c", face)], individuals=inds, latest_sightings={}, no_embed_zones=(),
                            match_threshold=0.8, merge_gap_sec=60, tz=TZ, new_id=lambda: "new",
                            cannot_link={"x": {"y"}, "y": {"x"}}, cannot_link_margin=0.10)
    # Body is far from both: the crop opens a new individual instead of joining x by face.
    assert linked.crop_individual == {"c": "new"}
    assert linked.assigned_by_face == 0 and linked.face_blocked_by_cannot_link == 1


def test_cannot_link_counter_only_counts_crops_it_actually_rerouted() -> None:
    # Body also matches x: the face refusal falls through and the body rule picks x anyway.
    inds = {"x": _ind("x", _vec(1, 0.20), body=_vec(0, 0, 0, 0, 1)), "y": _ind("y", _vec(1, 0.45))}
    out = vi.apply_batch([_crop("c", _vec(1, 0.0))], individuals=inds, latest_sightings={}, no_embed_zones=(),
                         match_threshold=0.8, merge_gap_sec=60, tz=TZ, cannot_link={"x": {"y"}, "y": {"x"}})
    assert out.crop_individual == {"c": "x"}
    assert out.assigned_by_face == 0 and out.face_blocked_by_cannot_link == 0


def test_cannot_link_leaves_a_clear_face_match_alone() -> None:
    x = _ind("x", _vec(1, 0.0))
    y = _ind("y", _vec(0, 1.0))                       # far from the crop's face
    out = vi.apply_batch([_crop("c", _vec(1, 0.05))], individuals={"x": x, "y": y}, latest_sightings={},
                         no_embed_zones=(), match_threshold=0.8, merge_gap_sec=60, tz=TZ,
                         cannot_link={"x": {"y"}, "y": {"x"}})
    assert out.crop_individual == {"c": "x"} and out.face_blocked_by_cannot_link == 0


def test_a_partner_already_taken_in_this_frame_does_not_block() -> None:
    inds = {"x": _ind("x", _vec(1, 0.20)), "y": _ind("y", _vec(1, 0.45))}
    assert vi.cannot_link_ambiguous(_vec(1, 0.0), "x", 0.98, individuals=inds, cannot_link={"x": {"y"}},
                                    margin=0.10) == "y"
    assert vi.cannot_link_ambiguous(_vec(1, 0.0), "x", 0.98, individuals=inds, cannot_link={"x": {"y"}},
                                    margin=0.10, taken={"y"}) is None


def test_every_assigned_crop_is_in_the_membership_map() -> None:
    out = vi.apply_batch([_crop("a", None, emb=_vec(1), obs="o1"), _crop("b", None, emb=_vec(1), obs="o2")],
                         individuals={}, latest_sightings={}, no_embed_zones=(), match_threshold=0.8,
                         merge_gap_sec=60, tz=TZ, new_id=iter(["i1", "s1", "s2"]).__next__)
    assert out.crop_individual == {"a": "i1", "b": "i1"}


# --- pure: kill switch, settings, migration ---------------------------------------------------------


class _NoConn:
    def __getattr__(self, name):
        raise AssertionError("the kill switch must not touch the database")


def test_kill_switch_off_does_nothing() -> None:
    cfg = vi.IndividualsConfig(split_on_verdict_enabled=False)
    assert vs.apply_splits(_NoConn(), NOW, cfg) == {"splits": 0, "splits_refused": 0, "splits_skipped": 0}


def test_settings_ship_the_split_on_and_reach_the_cycle_config() -> None:
    from app.settings import Settings
    from app.vision_individuals_loop import individuals_config

    s = Settings()
    cfg = individuals_config(s)
    assert s.vision_split_on_verdict_enabled is True and cfg.split_on_verdict_enabled is True
    assert (cfg.split_follow_min_face_cosine, cfg.split_follow_margin, cfg.split_max_move_fraction,
            cfg.cannot_link_margin) == (0.60, 0.10, 0.5, 0.10)
    off = individuals_config(Settings(VISION_SPLIT_ON_VERDICT_ENABLED=False))
    assert off.split_on_verdict_enabled is False
    env = (SERVICE / ".env_example").read_text()
    assert "VISION_SPLIT_ON_VERDICT_ENABLED=true" in env
    assert "VISION_SPLIT_ON_VERDICT_ENABLED=${VISION_SPLIT_ON_VERDICT_ENABLED:-true}" in (
        SERVICE / "docker-compose.yml").read_text()


def test_migration_passes_the_post_pull_auto_apply_filter() -> None:
    from orion.sql_migration_drift import auto_apply_refusals, parse_required_by, transaction_incompatible

    text = MIGRATION.read_text()
    live = {"vision_crop_observation", "vision_individual", "vision_individual_sighting"}
    assert auto_apply_refusals(text, live) == []
    assert auto_apply_refusals(text) == []          # unknown catalog: every table counts as existing
    rb = parse_required_by(text)
    assert rb is not None and rb.services == ("orion-sql-writer",) and not rb.errors
    assert transaction_incompatible(text)            # CONCURRENTLY: applied outside a transaction


def test_post_pull_applier_would_apply_it_without_a_wrapper() -> None:
    """What scripts/apply_post_pull_migrations.py decides for this file while it is MISSING live."""
    import importlib.util

    from orion import sql_migration_drift as d

    spec = importlib.util.spec_from_file_location("apply_post_pull_migrations",
                                                  REPO / "scripts" / "apply_post_pull_migrations.py")
    ap = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("apply_post_pull_migrations", ap)
    spec.loader.exec_module(ap)
    result = d.FileResult(name=MIGRATION.name, in_window=True, changed_at=NOW, status="MISSING")
    item = ap.classify(f"services/orion-sql-db/{MIGRATION.name}", MIGRATION.read_text(), result,
                       {"vision_crop_observation", "vision_individual"})
    assert item.action == "apply", item.reason
    assert item.no_txn == ["CREATE INDEX CONCURRENTLY"]


# --- the whole path against real Postgres ------------------------------------------------------------

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")
CFG = vi.IndividualsConfig()
ASK = "ask-test-1"


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE orion_ask, vision_individual_pair_verdict, vision_individual_sighting, "
                          "vision_individual, vision_ask_picture_verdict, vision_crop_observation, "
                          "vision_individual_split, vision_individuals_cursor CASCADE"))
    yield engine
    engine.dispose()


def _add_individual(conn, iid: str, *, n: int, face_n: int, label=None, labeled_at=None, face=FK,
                    body=_vec(1)) -> None:
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, centroid_n, first_seen_at, "
        "last_seen_at, sighting_count, distinct_days, label, labeled_at, label_ask_id, face_centroid, face_n) "
        "VALUES (:i, 'walkway', 'person', :c, :n, :t, :t, 1, 1, :l, :la, :lask, :f, :fn)"),
        {"i": iid, "c": list(body), "n": n, "t": NOW - timedelta(hours=2), "l": label, "la": labeled_at,
         "lask": "ask-name" if label else None, "f": list(face) if face else None, "fn": face_n})


def _add_crop(conn, cid: str, minute: int, *, member: str | None, emb=_vec(1), face=None, thumb=None,
              face_thumb=None) -> None:
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, observed_at, label, score, "
        "box_xyxy, zone, embedding, thumb_ref, face_embedding, face_thumb_ref, individual_id, created_at) VALUES "
        "(:c, :o, 'walkway', :t, 'person', 0.9, :b, 'walk', :e, :th, :f, :fth, :m, :t - interval '1 day')"),
        {"c": cid, "o": "obs-" + cid, "t": NOW - timedelta(hours=1) + timedelta(minutes=minute),
         "b": [0.0, 0.0, 10.0, 10.0], "e": list(emb), "th": thumb or f"thumb:{cid}", "f": list(face) if face else None,
         "fth": face_thumb, "m": member})


def _add_sighting(conn, sid: str, iid: str, start_min: int, end_min: int, count: int, *, thumb: str,
                  face_thumb=None, evidence=None) -> None:
    from sqlalchemy import text

    t0 = NOW - timedelta(hours=1)
    conn.execute(text(
        "INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, zone, started_at, ended_at, "
        "dwell_sec, observation_count, thumb_ref, face_thumb_ref, evidence_ref, zone_counts) "
        "VALUES (:s, :i, 'walkway', 'walk', :a, :b, 0, :n, :th, :fth, :ev, CAST(:zc AS jsonb))"),
        {"s": sid, "i": iid, "a": t0 + timedelta(minutes=start_min), "b": t0 + timedelta(minutes=end_min),
         "n": count, "th": thumb, "fth": face_thumb, "ev": evidence, "zc": f'{{"walk": {count}}}'})


def _verdict(conn, idx: int, verdict: str, crop_id: str | None, name=None, ask=ASK, iid="src") -> None:
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_ask_picture_verdict (ask_id, image_ref, picture_index, individual_id, verdict, name, "
        "crop_id, answered_at) VALUES (:a, :r, :i, :ind, :v, :n, :c, :t)"),
        {"a": ask, "r": f"thumb:pic{idx}-{ask}", "i": idx, "ind": iid, "v": verdict, "n": name, "c": crop_id,
         "t": NOW - timedelta(minutes=5)})


def _world(conn, *, legacy: bool = False) -> None:
    """src "Alder": 6 crops. k0 = the confirmed picture (face FK); m0 = the
    "someone else" picture (face FM); f1 a face close to FM (follows); f2 a
    face close to FK (stays); b1, b2 body-only crops (never follow).
    ``legacy``: every crop has NULL membership (older than the migration)."""
    _add_individual(conn, "src", n=6, face_n=3, label="Alder", labeled_at=NOW - timedelta(hours=3),
                    face=_vec(1, 0.4, 0))
    m = None if legacy else "src"
    _add_crop(conn, "k0", 0, member=m, face=FK, face_thumb="thumb:fk0")
    _add_crop(conn, "b1", 1, member=m, emb=_vec(1, 0.1))
    _add_crop(conn, "m0", 10, member=m, emb=_vec(1, 0.2), face=FM, face_thumb="thumb:fm0")
    _add_crop(conn, "f1", 11, member=m, emb=_vec(1, 0.3), face=_vec(0.1, 1, 0), face_thumb="thumb:ff1")
    _add_crop(conn, "b2", 12, member=m, emb=_vec(1, 0.4))
    _add_crop(conn, "f2", 30, member=m, face=_vec(1, 0.1, 0), face_thumb="thumb:ff2")
    _add_sighting(conn, "s-a", "src", 0, 1, 2, thumb="thumb:b1", face_thumb="thumb:fk0", evidence="crop:b1")
    _add_sighting(conn, "s-b", "src", 10, 12, 3, thumb="thumb:b2", face_thumb="thumb:ff1", evidence="crop:b2")
    _add_sighting(conn, "s-c", "src", 30, 30, 1, thumb="thumb:f2", face_thumb="thumb:ff2", evidence="crop:f2")
    _verdict(conn, 0, "same", "k0")
    _verdict(conn, 1, "different", "m0", name="Cora")


def _snapshot(conn) -> dict:
    from sqlalchemy import text

    q = lambda sql: [tuple(r) for r in conn.execute(text(sql)).fetchall()]  # noqa: E731
    return {
        "individuals": q("SELECT individual_id, to_jsonb(i)::text FROM vision_individual i ORDER BY 1"),
        "sightings": q("SELECT sighting_id, to_jsonb(s)::text FROM vision_individual_sighting s ORDER BY 1"),
        "crops": q("SELECT crop_id, individual_id FROM vision_crop_observation ORDER BY 1"),
        "pairs": q("SELECT individual_a, individual_b, verdict FROM vision_individual_pair_verdict ORDER BY 1, 2"),
    }


def _split_row(conn):
    from sqlalchemy import text

    return conn.execute(text("SELECT * FROM vision_individual_split")).mappings().one()


@needs_pg
def test_split_moves_the_named_picture_and_its_close_face_and_keeps_identity(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        vi._check_tables(conn)                       # the split migration is present
        _world(conn)
        src_label_before = conn.execute(text(
            "SELECT label, labeled_at, label_ask_id, label_copied FROM vision_individual WHERE individual_id='src'")).one()
    with pg.begin() as conn:
        assert vs.apply_splits(conn, NOW, CFG) == {"splits": 1, "splits_refused": 0, "splits_skipped": 0}
    with pg.begin() as conn:
        row = _split_row(conn)
        assert row["status"] == "split" and row["name"] == "Cora" and row["group_key"] == "cora"
        new = row["new_individual_id"]
        assert [(m["crop_id"], m["moved_by"]) for m in row["moved"]] == [("m0", "verdict"), ("f1", "face_follow")]
        crops = dict(conn.execute(text("SELECT crop_id, individual_id FROM vision_crop_observation")).fetchall())
        assert crops == {"k0": "src", "b1": "src", "m0": new, "f1": new, "b2": "src", "f2": "src"}
        ind = {r.individual_id: r for r in conn.execute(text("SELECT * FROM vision_individual")).fetchall()}
        # The confirmed side keeps its id and every label field (its substrate entity node is keyed on both).
        assert tuple(conn.execute(text(
            "SELECT label, labeled_at, label_ask_id, label_copied FROM vision_individual "
            "WHERE individual_id='src'")).one()) == tuple(src_label_before)
        assert (ind["src"].centroid_n, ind["src"].face_n) == (4, 1)
        assert row["after"]["method"] == "exact"
        # Exact: the remaining members' unit-vector mean.
        exact = vs.mean_unit([_vec(1), _vec(1, 0.1), _vec(1, 0.4), _vec(1)])
        assert vi.cosine(list(ind["src"].centroid), exact) > 0.9999
        assert (ind[new].label, ind[new].label_copied, ind[new].centroid_n, ind[new].face_n) == ("Cora", False, 2, 2)
        assert ind[new].label_ask_id == ASK
        # Cannot-link back to the source; no 'same' link (nobody else is called Cora).
        pairs = conn.execute(text("SELECT individual_a, individual_b, verdict, ask_id FROM "
                                  "vision_individual_pair_verdict")).fetchall()
        assert [(tuple(sorted((p[0], p[1]))), p[2], p[3]) for p in pairs] == [
            (tuple(sorted(("src", new))), "different", ASK)]
        # Source sightings: s-b lost two crops; its face picture (f1, moved) is gone, its body picture stays.
        sb = conn.execute(text("SELECT * FROM vision_individual_sighting WHERE sighting_id='s-b'")).one()
        assert (sb.observation_count, sb.zone_counts, sb.thumb_ref, sb.face_thumb_ref) == (1, {"walk": 1},
                                                                                          "thumb:b2", None)
        # Its time span shrinks to the crop that stays (b2 at minute 12); the source's seen fields follow.
        b2_at = NOW - timedelta(hours=1) + timedelta(minutes=12)
        assert (sb.started_at, sb.ended_at, sb.dwell_sec) == (b2_at, b2_at, 0)
        assert ind["src"].first_seen_at == NOW - timedelta(hours=1)
        assert ind["src"].last_seen_at == NOW - timedelta(hours=1) + timedelta(minutes=30)
        assert conn.execute(text("SELECT count(*) FROM vision_individual_sighting WHERE individual_id='src'")).scalar() == 3
        ns = conn.execute(text("SELECT * FROM vision_individual_sighting WHERE individual_id=:n"), {"n": new}).one()
        assert (ns.observation_count, ns.face_thumb_ref, ns.evidence_ref) == (2, "thumb:ff1", "crop:f1")
        # No thumb of a moved crop is left on the source: the next ask card cannot show Cora as Alder.
        refs = {r for row_ in conn.execute(text("SELECT thumb_ref, face_thumb_ref FROM vision_individual_sighting "
                                                "WHERE individual_id='src'")).fetchall() for r in row_}
        assert not refs & {"thumb:m0", "thumb:f1", "thumb:fm0", "thumb:ff1"}


@needs_pg
def test_out_of_retention_counts_only_sightings_whose_crops_are_gone(pg) -> None:
    """Person crops outlive VISION_CROP_RETENTION_DAYS (keep list): an old
    sighting that still has its crops was planned over, so it is not counted."""
    from sqlalchemy import text

    old = NOW - timedelta(days=10)
    with pg.begin() as conn:
        _world(conn)
        for sid, minute in (("s-old-kept", 0), ("s-old-gone", 30)):
            conn.execute(text(
                "INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, zone, started_at, "
                "ended_at, dwell_sec, observation_count, thumb_ref) VALUES (:s, 'src', 'walkway', 'walk', :t, :t, 0, "
                "1, 'thumb:old')"), {"s": sid, "t": old + timedelta(minutes=minute)})
        conn.execute(text(
            "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, observed_at, label, score, "
            "box_xyxy, zone, embedding, individual_id, created_at) VALUES ('old-kept', 'obs-old', 'walkway', :t, "
            "'person', 0.9, '{0,0,10,10}', 'walk', :e, 'src', :t)"), {"t": old, "e": list(_vec(1, 0.05))})
    with pg.begin() as conn:
        vs.apply_splits(conn, NOW, CFG)
    with pg.begin() as conn:
        assert _split_row(conn)["sightings_out_of_retention"] == 1


@needs_pg
def test_split_is_idempotent(pg) -> None:
    with pg.begin() as conn:
        _world(conn)
    with pg.begin() as conn:
        vs.apply_splits(conn, NOW, CFG)
        snap = _snapshot(conn)
    with pg.begin() as conn:
        assert vs.apply_splits(conn, NOW + timedelta(minutes=1), CFG) == {
            "splits": 0, "splits_refused": 0, "splits_skipped": 0}
        assert vs.pending_split_groups(conn) == []
        assert _snapshot(conn) == snap


@needs_pg
def test_a_crop_with_no_stored_membership_still_splits_by_its_crop_id(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn, legacy=True)
    with pg.begin() as conn:
        assert vs.apply_splits(conn, NOW, CFG)["splits"] == 1
        row = _split_row(conn)
        # Only the named picture: a crop with no stored membership can never be a face follower.
        assert [(m["crop_id"], m["moved_by"], m["prior_individual_id"]) for m in row["moved"]] == [
            ("m0", "verdict", None)]
        assert row["after"]["method"] == "subtract"
        src = conn.execute(text("SELECT centroid_n, face_n FROM vision_individual WHERE individual_id='src'")).one()
        assert tuple(src) == (5, 2)


@needs_pg
def test_a_crop_already_moved_or_pruned_does_nothing(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        conn.execute(text("UPDATE vision_crop_observation SET individual_id='elsewhere' WHERE crop_id='m0'"))
        _verdict(conn, 0, "different", "gone-crop", ask="ask-test-2")
    with pg.begin() as conn:
        assert vs.apply_splits(conn, NOW, CFG) == {"splits": 0, "splits_refused": 0, "splits_skipped": 2}
        reasons = sorted(r[0] for r in conn.execute(text("SELECT reason FROM vision_individual_split")).fetchall())
        assert reasons == ["no_member_crop gone-crop:pruned", "no_member_crop m0:not_a_member"]
        assert conn.execute(text("SELECT count(*) FROM vision_individual")).scalar() == 1


@needs_pg
def test_a_majority_split_is_refused_and_moves_nothing(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        _verdict(conn, 2, "different", "b1", name="Cora")
        _verdict(conn, 3, "different", "b2", name="Cora")
        _verdict(conn, 4, "different", "f2", name="Cora")
        before = _snapshot(conn)
    with pg.begin() as conn:
        assert vs.apply_splits(conn, NOW, CFG)["splits_refused"] == 1
        row = _split_row(conn)
        assert (row["status"], row["reason"], row["new_individual_id"]) == ("refused", "majority", None)
        assert _snapshot(conn) == before


@needs_pg
def test_typed_name_links_same_to_an_existing_individual_without_merging(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        _add_individual(conn, "other", n=9, face_n=0, face=None, label="cora", labeled_at=NOW - timedelta(days=1))
    with pg.begin() as conn:
        vs.apply_splits(conn, NOW, CFG)
        row = _split_row(conn)
        new = row["new_individual_id"]
        pairs = {(tuple(sorted((r[0], r[1]))), r[2]) for r in conn.execute(text(
            "SELECT individual_a, individual_b, verdict FROM vision_individual_pair_verdict")).fetchall()}
        assert pairs == {(tuple(sorted(("src", new))), "different"), (tuple(sorted(("other", new))), "same")}
        assert conn.execute(text("SELECT centroid_n FROM vision_individual WHERE individual_id='other'")).scalar() == 9
        assert sorted(p["verdict"] for p in row["linked_pair"]) == ["different", "same"]


@needs_pg
def test_unnamed_someone_else_gets_no_label_and_no_same_link(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        conn.execute(text("UPDATE vision_ask_picture_verdict SET name=NULL WHERE verdict='different'"))
    with pg.begin() as conn:
        vs.apply_splits(conn, NOW, CFG)
        new = _split_row(conn)["new_individual_id"]
        assert conn.execute(text("SELECT label FROM vision_individual WHERE individual_id=:n"), {"n": new}).scalar() is None
        assert conn.execute(text("SELECT count(*) FROM vision_individual_pair_verdict")).scalar() == 1


@needs_pg
def test_undo_restores_before_exactly(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        before = _snapshot(conn)
    with pg.begin() as conn:
        vs.apply_splits(conn, NOW, CFG)
        sid = _split_row(conn)["split_id"]
    with pg.begin() as conn:
        report = vs.undo_split(conn, sid, now=NOW)
        assert report["ok"] and report["source"]["restore"] == "exact" and report["crops_restored"] == 2
    with pg.begin() as conn:
        assert _snapshot(conn) == before                       # byte for byte, updated_at included
        assert _split_row(conn)["undone_at"] is not None
        assert vs.undo_split(conn, sid)["ok"] is False          # already undone
        assert vs.pending_split_groups(conn) == []              # and never re-split for that answer
        assert conn.execute(text("SELECT count(*) FROM vision_individual_split")).scalar() == 1


@needs_pg
def test_undo_keeps_crops_that_joined_the_source_later(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
    with pg.begin() as conn:
        vs.apply_splits(conn, NOW, CFG)
        sid = _split_row(conn)["split_id"]
        # A new crop joins the source after the split.
        conn.execute(text("UPDATE vision_individual SET centroid_n = centroid_n + 1, updated_at = now() "
                          "WHERE individual_id='src'"))
    with pg.begin() as conn:
        report = vs.undo_split(conn, sid, now=NOW)
        assert report["source"]["restore"].startswith("delta")
        assert conn.execute(text("SELECT centroid_n, face_n FROM vision_individual WHERE individual_id='src'")).one() \
            == (7, 3)


@needs_pg
def test_undo_refuses_when_crops_joined_the_new_individual(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
    with pg.begin() as conn:
        vs.apply_splits(conn, NOW, CFG)
        row = _split_row(conn)
        _add_crop(conn, "late", 40, member=row["new_individual_id"])
    with pg.begin() as conn:
        report = vs.undo_split(conn, row["split_id"])
        assert report["ok"] is False and report["joined_crops"] == ["late"]


@needs_pg
def test_undo_refuses_when_juniper_named_or_judged_the_new_individual_later(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
    with pg.begin() as conn:
        vs.apply_splits(conn, NOW, CFG)
        row = _split_row(conn)
        new = row["new_individual_id"]
        conn.execute(text("UPDATE vision_individual SET label='Dee', label_ask_id='ask-later' WHERE individual_id=:n"),
                     {"n": new})
    with pg.begin() as conn:
        assert "named the new individual" in vs.undo_split(conn, row["split_id"])["reason"]
        conn.execute(text("UPDATE vision_individual SET label='Cora', label_ask_id=:a WHERE individual_id=:n"),
                     {"a": ASK, "n": new})
        a, b = sorted((new, "zzz-other"))
        conn.execute(text("INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, ask_id, "
                          "decided_at) VALUES (:a, :b, 'same', 'ask-later', :t)"), {"a": a, "b": b, "t": NOW})
    with pg.begin() as conn:
        assert vs.undo_split(conn, row["split_id"])["pair_verdicts"] == [f"{a}|{b}:same"]


@needs_pg
def test_a_picture_both_kept_and_moved_is_contradictory(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        _verdict(conn, 2, "different", "k0", name="Cora")
    with pg.begin() as conn:
        assert vs.apply_splits(conn, NOW, CFG)["splits_skipped"] == 1
        assert _split_row(conn)["reason"] == "contradictory k0"
        assert conn.execute(text("SELECT count(*) FROM vision_individual")).scalar() == 1


@needs_pg
def test_a_failing_split_leaves_one_error_row_not_a_retry_loop(pg, monkeypatch) -> None:
    with pg.begin() as conn:
        _world(conn)

    def boom(*a, **k):
        raise ValueError("synthetic")
    monkeypatch.setattr(vs, "split_one", boom)
    with pg.begin() as conn:
        assert vs.apply_splits(conn, NOW, CFG)["splits_skipped"] == 1
        assert _split_row(conn)["reason"].startswith("error: ValueError")
        assert vs.pending_split_groups(conn) == []


@needs_pg
def test_undo_script_is_a_dry_run_unless_apply(pg) -> None:
    with pg.begin() as conn:
        _world(conn)
    with pg.begin() as conn:
        vs.apply_splits(conn, NOW, CFG)
        sid = _split_row(conn)["split_id"]
        after_split = _snapshot(conn)
    script = str(REPO / "scripts" / "undo_vision_split.py")
    dry = subprocess.run([sys.executable, script, sid, "--dsn", PG], capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    assert "DRY RUN" in dry.stdout
    with pg.begin() as conn:
        assert _snapshot(conn) == after_split
    done = subprocess.run([sys.executable, script, sid, "--dsn", PG, "--apply"], capture_output=True, text=True)
    assert done.returncode == 0 and "APPLIED" in done.stdout, done.stderr
    with pg.begin() as conn:
        assert _snapshot(conn) != after_split
        assert _split_row(conn)["undone_at"] is not None


@needs_pg
def test_the_cycle_writes_membership_honours_cannot_link_and_splits(pg) -> None:
    """The deployed path: run_one_individuals_cycle writes crop membership,
    refuses an ambiguous face match against a cannot-link, and applies a
    pending picture verdict -- unless the kill switch is off."""
    from sqlalchemy import text

    t = NOW - timedelta(minutes=5)
    with pg.begin() as conn:
        _add_individual(conn, "x", n=5, face_n=5, face=_vec(1, 0.20), body=_vec(0, 0, 1))
        _add_individual(conn, "y", n=5, face_n=5, face=_vec(1, 0.45), body=_vec(0, 0, 0, 1))
        conn.execute(text("INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, ask_id, "
                          "decided_at) VALUES ('x', 'y', 'different', 'ask-p', :t)"), {"t": t})
        conn.execute(text("INSERT INTO vision_individuals_cursor (stream_id, last_created_at) VALUES ('walkway', :t)"),
                     {"t": t - timedelta(minutes=1)})
        conn.execute(text(
            "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, observed_at, label, score, "
            "box_xyxy, embedding, face_embedding, created_at) VALUES ('c1', 'o1', 'walkway', :t, 'person', 0.9, "
            "ARRAY[0,0,10,10]::real[], :e, :f, :t)"),
            {"t": t, "e": list(_vec(0, 0, 0, 0, 1)), "f": list(_vec(1, 0.0))})
    off = vi.IndividualsConfig(split_on_verdict_enabled=False, ask_face_min_faces=0)
    on = vi.IndividualsConfig(ask_face_min_faces=0)
    summary, _ = vi.run_one_individuals_cycle(postgres_uri=PG, cfg=off, zones_by_stream={}, now=NOW)
    with pg.begin() as conn:
        assert conn.execute(text("SELECT individual_id FROM vision_crop_observation WHERE crop_id='c1'")).scalar() == "x"
        # Reset: the same crop again, now with the split/cannot-link switch on.
        conn.execute(text("UPDATE vision_individual SET centroid_n=5, face_n=5, face_centroid=:f WHERE individual_id='x'"),
                     {"f": list(_vec(1, 0.20))})
        conn.execute(text("DELETE FROM vision_individual WHERE individual_id NOT IN ('x', 'y')"))
        conn.execute(text("UPDATE vision_crop_observation SET individual_id=NULL"))
        conn.execute(text("UPDATE vision_individuals_cursor SET last_created_at=:t"), {"t": t - timedelta(minutes=1)})
        _world(conn)
    summary, _ = vi.run_one_individuals_cycle(postgres_uri=PG, cfg=on, zones_by_stream={}, now=NOW)
    assert summary["face_blocked_by_cannot_link"] == 1 and summary["splits"] == 1
    with pg.begin() as conn:
        assigned = conn.execute(text("SELECT individual_id FROM vision_crop_observation WHERE crop_id='c1'")).scalar()
        assert assigned not in (None, "x", "y")


# --- answers given on a Hub walkway story card (2026-10-18) ------------------------------------------
#
# The Hub (services/orion-hub/scripts/motion_story_routes.py) writes these rows already answered;
# its own real-Postgres test pins the exact row it writes. Here: what sql-writer does with them.

STORY = "f" * 32


def _story_ask(conn, ask_id: str, kind: str, iid: str, *, answer=None, action="name", name=None,
               pictures=None, minute=0) -> None:
    """The row POST /api/vision/motion-stories/{story}/people/{iid} inserts."""
    import json

    from sqlalchemy import text

    detail = {"story_id": STORY, "action": action, "via": "motion_story"}
    if name:
        detail["name"] = name
    if pictures is not None:
        detail["pictures"] = pictures
    conn.execute(text(
        "INSERT INTO orion_ask (ask_id, asked_of, question, evidence_refs, image_ref, image_refs, status, answer, "
        "answered_at, created_at, source_kind, source_ref, answer_detail) VALUES (:a, 'juniper', 'q', "
        "CAST(:ev AS jsonb), NULL, '[]'::jsonb, 'answered', :ans, :t, :t, :k, :i, CAST(:d AS jsonb))"),
        {"a": ask_id, "ev": json.dumps([f"motion_story:{STORY}", f"individual:{iid}"]), "ans": answer,
         "t": NOW - timedelta(minutes=30) + timedelta(minutes=minute), "k": kind, "i": iid, "d": json.dumps(detail)})


def _label(conn, iid: str):
    from sqlalchemy import text

    return tuple(conn.execute(text(
        "SELECT label, label_copied, label_ask_id FROM vision_individual WHERE individual_id=:i"), {"i": iid}).one())


@needs_pg
def test_a_name_typed_on_a_story_is_a_typed_label_linked_to_the_same_name(pg, caplog) -> None:
    import logging

    from sqlalchemy import text

    with pg.begin() as conn:
        _add_individual(conn, "anon", n=3, face_n=0, face=None)
        _add_individual(conn, "maya-1", n=3, face_n=0, face=None, label="Maya", labeled_at=NOW - timedelta(days=2))
        _add_individual(conn, "theo", n=3, face_n=0, face=None, label="Theo", labeled_at=NOW - timedelta(days=1))
        # An open "who is this?" card for anon, from the opener.
        conn.execute(text("INSERT INTO orion_ask (ask_id, question, source_kind, source_ref) "
                          "VALUES ('who-anon', 'q', 'vision_individual', 'anon')"))
        _story_ask(conn, "ask-story-1", "vision_individual", "anon", answer="maya", name="maya")
        # Last typed name wins: theo was typed "Theo", a story names them "Maya".
        _story_ask(conn, "ask-story-2", "vision_individual", "theo", answer="Maya", name="Maya", minute=1)
    caplog.set_level(logging.INFO, logger="sql-writer.vision_individuals")
    with pg.begin() as conn:
        assert vi._apply_answers(conn, NOW) >= 2
    with pg.begin() as conn:
        assert _label(conn, "anon") == ("maya", False, "ask-story-1")
        assert _label(conn, "theo") == ("Maya", False, "ask-story-2")
        assert any("[VISION_LABEL_REPLACED] individual_id=theo old='Theo'" in r.message for r in caplog.records)
        pairs = {(r[0], r[1], r[2], r[3]) for r in conn.execute(text(
            "SELECT individual_a, individual_b, verdict, ask_id FROM vision_individual_pair_verdict")).fetchall()}
        # Each story name links to the person she already typed that name for (case-insensitive).
        assert ("anon", "maya-1", "same", "ask-story-1") in pairs
        assert ("maya-1", "theo", "same", "ask-story-2") in pairs   # the earliest typed "Maya" (labeled_at)
        assert conn.execute(text("SELECT status, answer FROM orion_ask WHERE ask_id='who-anon'")).one() == (
            "answered", "maya")
        assert conn.execute(text("SELECT count(*) FROM orion_ask WHERE applied_at IS NULL")).scalar() == 0
        assert any("[VISION_SAME_NAME_LINK] individual_id=anon other=maya-1" in r.message for r in caplog.records)


@needs_pg
def test_story_name_skips_a_different_verdict_and_plain_answers_link_too(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _add_individual(conn, "anon", n=3, face_n=0, face=None)
        _add_individual(conn, "plain", n=3, face_n=0, face=None)
        _add_individual(conn, "maya-1", n=3, face_n=0, face=None, label="Maya", labeled_at=NOW - timedelta(days=2))
        conn.execute(text("INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, ask_id, "
                          "decided_at) VALUES ('anon', 'maya-1', 'different', 'ask-pair', :t)"), {"t": NOW})
        _add_individual(conn, "maya-2", n=3, face_n=0, face=None, label="Maya", labeled_at=NOW - timedelta(days=1))
        _story_ask(conn, "ask-story-1", "vision_individual", "anon", answer="Maya", name="Maya")
        # A plain "who is this?" answer (no story_id) links the same way (2026-10-20).
        conn.execute(text("INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answer, "
                          "answered_at) VALUES ('who-plain', 'q', 'vision_individual', 'plain', 'answered', 'Maya', :t)"),
                     {"t": NOW - timedelta(minutes=5)})
    with pg.begin() as conn:
        vi._apply_answers(conn, NOW)
    with pg.begin() as conn:
        assert _label(conn, "anon")[0] == "Maya" and _label(conn, "plain")[0] == "Maya"
        # maya-1 has a 'different' with anon: skipped, the next "Maya" is linked instead.
        # plain (answered later) joins the earliest typed "Maya", maya-1.
        assert [tuple(r) for r in conn.execute(text(
            "SELECT individual_a, individual_b, verdict, ask_id FROM vision_individual_pair_verdict "
            "ORDER BY 1, 2")).fetchall()] == [("anon", "maya-1", "different", "ask-pair"),
                                              ("anon", "maya-2", "same", "ask-story-1"),
                                              ("maya-1", "plain", "same", "who-plain")]


# --- every typed name links to the same name (2026-10-20) --------------------------------------------


def _who_ask(conn, ask_id: str, iid: str, answer: str, *, minute=0) -> None:
    """A plain Hub "who is this?" answer (no story_id), as the Hub writes it."""
    from sqlalchemy import text

    conn.execute(text("INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answer, "
                      "answered_at) VALUES (:a, 'q', 'vision_individual', :i, 'answered', :ans, :t)"),
                 {"a": ask_id, "i": iid, "ans": answer, "t": NOW - timedelta(minutes=30) + timedelta(minutes=minute)})


def _pair_ask(conn, ask_id: str, a: str, b: str, answer: str, names: dict, *, minute=0) -> None:
    """A "same person?" answer with name boxes, as the Hub writes it."""
    import json

    from sqlalchemy import text

    conn.execute(text("INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answer, "
                      "answered_at, answer_detail) VALUES (:a, 'q', 'vision_same_individual', :r, 'answered', "
                      ":ans, :t, CAST(:d AS jsonb))"),
                 {"a": ask_id, "r": vi.pair_ref(a, b), "ans": answer, "d": json.dumps({"names": names}),
                  "t": NOW - timedelta(minutes=30) + timedelta(minutes=minute)})


def _pairs(conn) -> list:
    from sqlalchemy import text

    return [tuple(r) for r in conn.execute(text(
        "SELECT individual_a, individual_b, verdict, ask_id FROM vision_individual_pair_verdict "
        "ORDER BY 1, 2")).fetchall()]


@needs_pg
def test_a_who_is_this_name_links_to_the_same_typed_name_and_logs(pg, caplog) -> None:
    import logging

    with pg.begin() as conn:
        _add_individual(conn, "anon", n=3, face_n=0, face=None)
        _add_individual(conn, "maya-1", n=3, face_n=0, face=None, label="Maya", labeled_at=NOW - timedelta(days=2))
        _who_ask(conn, "who-1", "anon", " maya ")
    caplog.set_level(logging.INFO, logger="sql-writer.vision_individuals")
    with pg.begin() as conn:
        vi._apply_answers(conn, NOW)
    with pg.begin() as conn:
        assert _label(conn, "anon") == ("maya", False, "who-1")
        assert _pairs(conn) == [("anon", "maya-1", "same", "who-1")]
    assert any("[VISION_SAME_NAME_LINK] individual_id=anon other=maya-1 name='maya' ask_id=who-1" in r.message
               for r in caplog.records)
    # Idempotent: a second tick (and re-linking the same name) adds nothing.
    with pg.begin() as conn:
        vi._apply_answers(conn, NOW)
        assert vi._link_same_name(conn, "anon", "Maya", "who-again", NOW) is None
        assert _pairs(conn) == [("anon", "maya-1", "same", "who-1")]


@needs_pg
def test_a_who_is_this_name_skips_a_copied_name_and_a_different_through_the_group(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _add_individual(conn, "anon", n=3, face_n=0, face=None)
        # Only copied "Maya" (never typed for this one): not a target.
        _add_individual(conn, "copy", n=3, face_n=0, face=None, label="Maya", labeled_at=NOW - timedelta(days=3))
        conn.execute(text("UPDATE vision_individual SET label_copied=true WHERE individual_id='copy'"))
        # maya-1 is confirmed the same as "x", and she said x is not anon: linking
        # anon to maya-1 would make anon = x through the group.
        _add_individual(conn, "maya-1", n=3, face_n=0, face=None, label="Maya", labeled_at=NOW - timedelta(days=2))
        _add_individual(conn, "x", n=3, face_n=0, face=None)
        conn.execute(text("INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, ask_id, "
                          "decided_at) VALUES ('maya-1', 'x', 'same', 'p1', :t), ('anon', 'x', 'different', 'p2', :t)"),
                     {"t": NOW})
        _who_ask(conn, "who-1", "anon", "Maya")
    with pg.begin() as conn:
        vi._apply_answers(conn, NOW)
    with pg.begin() as conn:
        assert _label(conn, "anon") == ("Maya", False, "who-1")
        assert _pairs(conn) == [("anon", "x", "different", "p2"), ("maya-1", "x", "same", "p1")]


@needs_pg
def test_pair_card_names_link_once_per_group_and_never_across_a_different(pg) -> None:
    with pg.begin() as conn:
        _add_individual(conn, "a1", n=3, face_n=0, face=None)
        _add_individual(conn, "b1", n=3, face_n=0, face=None)
        _add_individual(conn, "c1", n=3, face_n=0, face=None)
        _add_individual(conn, "d1", n=3, face_n=0, face=None)
        _add_individual(conn, "bo-1", n=3, face_n=0, face=None, label="Bo", labeled_at=NOW - timedelta(days=2))
        _add_individual(conn, "cy-1", n=3, face_n=0, face=None, label="Cy", labeled_at=NOW - timedelta(days=2))
        # "same" with one name: both faces get it, the pair joins "Bo" once (a1's link
        # puts b1 in bo-1's group already).
        _pair_ask(conn, "pair-1", "a1", "b1", "yes", {"a1": "Bo"})
        # "different" naming both "Cy": c1 links to cy-1; d1 must not (cy-1's group
        # now holds c1, whom she just said d1 is not).
        _pair_ask(conn, "pair-2", "c1", "d1", "no", {"c1": "Cy", "d1": "Cy"}, minute=1)
    with pg.begin() as conn:
        vi._apply_answers(conn, NOW)
    with pg.begin() as conn:
        assert _label(conn, "a1")[0] == "Bo" and _label(conn, "b1")[0] == "Bo"
        assert _label(conn, "c1")[0] == "Cy" and _label(conn, "d1")[0] == "Cy"
        assert _pairs(conn) == [("a1", "b1", "same", "pair-1"), ("a1", "bo-1", "same", "pair-1"),
                                ("c1", "cy-1", "same", "pair-2"), ("c1", "d1", "different", "pair-2")]


@needs_pg
def test_a_corrected_name_drops_the_old_names_links_but_not_her_pair_answers(pg, caplog) -> None:
    import logging

    from sqlalchemy import text

    with pg.begin() as conn:
        _add_individual(conn, "anon", n=3, face_n=0, face=None)
        _add_individual(conn, "maya-1", n=3, face_n=0, face=None, label="Maya", labeled_at=NOW - timedelta(days=2))
        _add_individual(conn, "kid", n=3, face_n=0, face=None, label="Maya", labeled_at=NOW - timedelta(days=1))
        conn.execute(text("UPDATE vision_individual SET label_copied=true WHERE individual_id='kid'"))
        # Her own "same person?" answer about anon and pal: never a name link, it stays.
        _add_individual(conn, "pal", n=3, face_n=0, face=None)
        conn.execute(text("INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answer, "
                          "answered_at, applied_at) VALUES ('pair-own', 'q', 'vision_same_individual', :r, "
                          "'answered', 'yes', :t, :t)"), {"r": vi.pair_ref("anon", "pal"), "t": NOW})
        conn.execute(text("INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, ask_id, "
                          "decided_at) VALUES ('anon', 'pal', 'same', 'pair-own', :t), "
                          "('kid', 'maya-1', 'same', 'p0', :t)"), {"t": NOW})
        _who_ask(conn, "who-1", "anon", "Maya")
        _who_ask(conn, "who-2", "anon", "Mya", minute=1)   # a typo fix
    caplog.set_level(logging.INFO, logger="sql-writer.vision_individuals")
    with pg.begin() as conn:
        vi._apply_answers(conn, NOW)
    with pg.begin() as conn:
        assert _label(conn, "anon") == ("Mya", False, "who-2")
        assert _pairs(conn) == [("anon", "pal", "same", "pair-own"), ("kid", "maya-1", "same", "p0")]
        # maya-1's group keeps its copied "Maya".
        assert _label(conn, "kid")[0] == "Maya"
    assert any("[VISION_SAME_NAME_UNLINK] individual_id=anon old='Maya' new='Mya' ask_id=who-2 others=maya-1"
               in r.message for r in caplog.records)


@needs_pg
def test_a_case_only_difference_is_one_name_not_a_conflict(pg, caplog) -> None:
    import logging

    from sqlalchemy import text

    with pg.begin() as conn:
        _add_individual(conn, "anon", n=3, face_n=0, face=None)
        _add_individual(conn, "maya-1", n=3, face_n=0, face=None, label="Maya", labeled_at=NOW - timedelta(days=2))
        _who_ask(conn, "who-1", "anon", "maya")
        _who_ask(conn, "who-2", "anon", "MAYA", minute=1)   # case only: the link stays
    caplog.set_level(logging.INFO, logger="sql-writer.vision_individuals")
    with pg.begin() as conn:
        vi._apply_answers(conn, NOW)
    with pg.begin() as conn:
        assert _pairs(conn) == [("anon", "maya-1", "same", "who-1")]
        assert conn.execute(text("SELECT label FROM vision_individual WHERE individual_id='maya-1'")).scalar() == "Maya"
    assert not any("VISION_GROUP_NAME_CONFLICT" in r.message or "VISION_SAME_NAME_UNLINK" in r.message
                   for r in caplog.records)


@needs_pg
def test_healed_pair_names_link_once(pg) -> None:
    import json

    from sqlalchemy import text

    with pg.begin() as conn:
        _add_individual(conn, "a1", n=3, face_n=0, face=None)
        _add_individual(conn, "b1", n=3, face_n=0, face=None)
        _add_individual(conn, "bo-1", n=3, face_n=0, face=None, label="Bo", labeled_at=NOW - timedelta(days=2))
        # An older writer applied the verdict but not the name.
        conn.execute(text("INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answer, "
                          "answered_at, applied_at, answer_detail) VALUES ('pair-old', 'q', 'vision_same_individual', "
                          ":r, 'answered', 'no', :t, :t, CAST(:d AS jsonb))"),
                     {"r": vi.pair_ref("a1", "b1"), "t": NOW - timedelta(hours=1), "d": json.dumps({"names": {"a1": "Bo"}})})
        conn.execute(text("INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, ask_id, "
                          "decided_at) VALUES ('a1', 'b1', 'different', 'pair-old', :t)"), {"t": NOW})
    for _ in range(2):
        with pg.begin() as conn:
            vi._apply_answers(conn, NOW)
    with pg.begin() as conn:
        assert _label(conn, "a1")[0] == "Bo"
        assert _pairs(conn) == [("a1", "b1", "different", "pair-old"), ("a1", "bo-1", "same", "pair-old")]


@needs_pg
def test_a_name_never_links_across_a_split(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        # A split moved "Cora" pictures off src into new-1 ('different') and linked new-1 to cora-1.
        _add_individual(conn, "src", n=3, face_n=0, face=None)
        _add_individual(conn, "new-1", n=3, face_n=0, face=None, label="Cora", labeled_at=NOW - timedelta(days=1))
        _add_individual(conn, "cora-1", n=3, face_n=0, face=None, label="Cora", labeled_at=NOW - timedelta(days=2))
        conn.execute(text("INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, ask_id, "
                          "decided_at) VALUES ('new-1', 'src', 'different', 'sp', :t), "
                          "('cora-1', 'new-1', 'same', 'sp', :t)"), {"t": NOW})
        _who_ask(conn, "who-1", "src", "Cora")
    with pg.begin() as conn:
        vi._apply_answers(conn, NOW)
    with pg.begin() as conn:
        assert _label(conn, "src")[0] == "Cora"
        assert _pairs(conn) == [("cora-1", "new-1", "same", "sp"), ("new-1", "src", "different", "sp")]


@needs_pg
def test_someone_else_on_a_story_records_crop_verdicts_and_the_split_moves_them(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _add_individual(conn, "src", n=6, face_n=3, label="Alder", labeled_at=NOW - timedelta(hours=3),
                        face=_vec(1, 0.4, 0))
        _add_crop(conn, "k0", 0, member="src", face=FK, face_thumb="thumb:fk0")
        _add_crop(conn, "b1", 1, member="src", emb=_vec(1, 0.1))
        _add_crop(conn, "m0", 10, member="src", emb=_vec(1, 0.2))
        _add_crop(conn, "m1", 11, member="src", emb=_vec(1, 0.3))
        _add_crop(conn, "b2", 12, member="src", emb=_vec(1, 0.4))
        _add_crop(conn, "f2", 30, member="src", face=_vec(1, 0.1, 0), face_thumb="thumb:ff2")
        _add_sighting(conn, "s-a", "src", 0, 1, 2, thumb="thumb:b1", face_thumb="thumb:fk0", evidence="crop:b1")
        _add_sighting(conn, "s-b", "src", 10, 12, 3, thumb="thumb:b2", evidence="crop:b2")
        _add_sighting(conn, "s-c", "src", 30, 30, 1, thumb="thumb:f2", face_thumb="thumb:ff2", evidence="crop:f2")
        pics = [{"image_ref": f"crop:{c}", "index": i, "verdict": "different", "crop_id": c, "name": "Cora"}
                for i, c in enumerate(("m0", "m1"))]
        _story_ask(conn, "ask-story-se", "vision_story_person", "src", action="someone_else", name="Cora",
                   pictures=pics)
        _story_ask(conn, "ask-story-un", "vision_story_person", "src", action="unsure", minute=1,
                   pictures=[{"image_ref": "crop:b1", "index": 0, "verdict": "unsure", "crop_id": "b1", "name": None}])
    with pg.begin() as conn:
        vi._apply_answers(conn, NOW)
    with pg.begin() as conn:
        rows = [tuple(r) for r in conn.execute(text(
            "SELECT ask_id, image_ref, individual_id, verdict, name, crop_id FROM vision_ask_picture_verdict "
            "ORDER BY ask_id, picture_index")).fetchall()]
        assert rows == [("ask-story-se", "crop:m0", "src", "different", "Cora", "m0"),
                        ("ask-story-se", "crop:m1", "src", "different", "Cora", "m1"),
                        ("ask-story-un", "crop:b1", "src", "unsure", None, "b1")]
        # "Someone else" is not a name for src: its typed label is untouched.
        assert _label(conn, "src") == ("Alder", False, "ask-name")
    with pg.begin() as conn:
        assert vs.apply_splits(conn, NOW, CFG) == {"splits": 1, "splits_refused": 0, "splits_skipped": 0}
    with pg.begin() as conn:
        row = _split_row(conn)
        assert (row["status"], row["name"], row["ask_id"]) == ("split", "Cora", "ask-story-se")
        new = row["new_individual_id"]
        crops = dict(conn.execute(text("SELECT crop_id, individual_id FROM vision_crop_observation")).fetchall())
        assert crops["m0"] == new and crops["m1"] == new and crops["k0"] == "src"
        assert _label(conn, new) == ("Cora", False, "ask-story-se")


@needs_pg
def test_story_names_do_not_use_up_the_who_is_this_daily_cap(pg) -> None:
    import json

    from sqlalchemy import text

    day = vi.local_day_start(NOW, ZoneInfo("America/Denver"))
    face = json.dumps([vi.FACE_RULE_REF])
    with pg.begin() as conn:
        for i in range(3):
            _add_individual(conn, f"p{i}", n=3, face_n=0, face=None)
            _story_ask(conn, f"ask-story-{i}", "vision_individual", f"p{i}", answer=f"N{i}", name=f"N{i}")
        conn.execute(text("UPDATE orion_ask SET created_at = :t"), {"t": NOW})
        assert conn.execute(text(vi.ASKED_TODAY_SQL), {"d": day, "face": face}).scalar() == 0
        conn.execute(text("INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, created_at) "
                          "VALUES ('who-real', 'q', 'vision_individual', 'p0', :t)"), {"t": NOW})
        assert conn.execute(text(vi.ASKED_TODAY_SQL), {"d": day, "face": face}).scalar() == 1

"""Street continuity loop (app/vision_handoff_loop.py) against a REAL Postgres.

``ORION_TEST_PG_URI`` must be a throwaway database: these tests TRUNCATE the
walkway tables.

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55511:5432 postgres:16-alpine
    for f in walkway_camera_v1 walkway_privacy_v2 vision_confirm_asks_v1 vision_ask_names_v1 vision_osnet_shadow_v1 \\
             vision_split_v1 vision_reanchor_v1 vision_motion_story_v1 vision_motion_story_strip_v2 \\
             vision_motion_story_edges_v1 vision_person_album_v1 vision_scene_inventory vision_handoff_v1; do
      docker exec -i pg psql -U postgres < services/orion-sql-db/manual_migration_$f.sql; done
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55511/postgres pytest tests/test_vision_handoff_pg.py
"""

from __future__ import annotations

import asyncio
import json
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

from app import vision_handoff as vh  # noqa: E402
from app import vision_handoff_loop as hl  # noqa: E402
from orion.vision.topology import load_topology  # noqa: E402

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")
pytestmark = needs_pg

T0 = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)
TOPO = load_topology(REPO / "config" / "vision_topology.yaml")
CFG = hl.HandoffLoopConfig()
W, H = 2560.0, 1920.0
VEC_A = [1.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1]
VEC_B = [0.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.5, 0.0]
TABLES = ("vision_handoff", "vision_handoff_hypothesis", "vision_handoff_cursor", "vision_handoff_street_outbox", "vision_motion_story",
          "vision_motion_story_cursor", "vision_crop_observation", "vision_scene_inventory", "vision_individual",
          "vision_individual_sighting", "vision_individual_pair_verdict", "orion_ask", "vision_unresolved",
          "vision_person_attribution", "vision_events")


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS vision_events (event_id VARCHAR PRIMARY KEY, event_type VARCHAR, narrative TEXT, "
            "entities JSONB, tags JSONB, confidence DOUBLE PRECISION, salience DOUBLE PRECISION, evidence_refs JSONB, "
            "correlation_id VARCHAR, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), stream_id TEXT)"))
        conn.execute(text("TRUNCATE " + ", ".join(TABLES) + " CASCADE"))
        for s in ("walkway", "doorbell"):
            conn.execute(text("INSERT INTO vision_motion_story_cursor (stream_id, processed_until) VALUES (:s, :p)"),
                         {"s": s, "p": T0 + timedelta(days=1)})
    yield engine
    engine.dispose()


def _crop(conn, crop_id, stream, t, *, vec, individual, label="person", box=(1000, 600, 1100, 1000)):
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, observed_at, label, score, box_xyxy, "
        "embedding, individual_id, created_at) VALUES (:c, :o, :s, :t, :l, 0.6, :b, :e, :i, :t)"),
        {"c": crop_id, "o": "obs-" + crop_id, "s": stream, "t": t, "l": label, "b": list(box), "e": vec, "i": individual})


def _individual(conn, iid, stream, t):
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, first_seen_at, last_seen_at) "
        "VALUES (:i, :s, 'person', :c, :t, :t)"), {"i": iid, "s": stream, "c": VEC_A, "t": t})
    conn.execute(text(
        "INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, started_at, ended_at, thumb_ref) "
        "VALUES (:sid, :i, :s, :t, :t, :th)"), {"sid": "sg-" + iid, "i": iid, "s": stream, "t": t, "th": f"thumb:{iid}"})


def _story(conn, story_id, stream, start, end, *, dir_lateral=0, dir_depth=0, exit_edge="inside", entry_edge="inside",
           exit_foot_x=0.5, exit_speed=0.05, individual=None, vec=VEC_A, kind="person", landed=None,
           first_x=None, zone="walkway"):
    """A story with three crops; the first box's foot sits at ``first_x`` of the frame (entry)."""
    from sqlalchemy import text

    fx = 0.5 if first_x is None else first_x
    crops = []
    for i, t in enumerate((start, start + (end - start) / 2, end)):
        cid = f"{story_id}-c{i}"
        x = fx * W if i == 0 else exit_foot_x * W
        box = (x - 50, 600, x + 50, 1000)
        _crop(conn, cid, stream, t, vec=vec, individual=individual, label=kind, box=box)
        crops.append({"crop_id": cid, "t": t.isoformat(), "box": list(box), "label": kind, "individual_id": individual,
                      "zone": zone})
    conn.execute(text(
        "INSERT INTO vision_motion_story (story_id, stream_id, kind, started_at, ended_at, detection_count, "
        "dir_lateral, dir_depth, zone_path, individual_id, published, crop_ids, evidence, rules_version, created_at, "
        "entry_edge, exit_edge, exit_foot_x, exit_speed_fw_per_sec, frame_w, frame_h) VALUES (:id, :s, :k, :a, :b, 3, "
        ":dl, :dd, :zp, :i, true, :cids, CAST(:ev AS jsonb), 1, :landed, :ee, :xe, :fx, :xs, 2560, 1920)"),
        {"id": story_id, "s": stream, "k": kind, "a": start, "b": end, "dl": dir_lateral, "dd": dir_depth,
         "zp": [zone], "i": None, "cids": [c["crop_id"] for c in crops], "ev": json.dumps(crops),
         "landed": landed or end + timedelta(seconds=60), "ee": entry_edge, "xe": exit_edge, "fx": exit_foot_x,
         "xs": exit_speed})


def _census(conn, stream, a, b, step=7.0, win=5.0):
    from sqlalchemy import text

    t = a
    n = 0
    while t < b:
        conn.execute(text(
            "INSERT INTO vision_scene_inventory (window_id, stream_id, observed_at, window_start_ts, window_end_ts, "
            "frame_count) VALUES (:w, :s, :t, :a, :b, 2)"),
            {"w": f"{stream}-{n}-{t.timestamp()}", "s": stream, "t": t + timedelta(seconds=win),
             "a": t.timestamp(), "b": t.timestamp() + win})
        t += timedelta(seconds=step)
        n += 1


def _cycle(engine, now, **kw):
    return hl.run_one_handoff_cycle(postgres_uri=PG, cfg=kw.pop("cfg", CFG), topology=TOPO, now=now,
                                    redis_url=None, engine=engine)


def _rows(engine, sql, **params):
    from sqlalchemy import text

    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(sql), params).mappings().fetchall()]


def _walker_left(conn, *, individual="ind-walk"):
    # Leaves the walkway toward the door: right-to-left, last box 30% from the left edge, 0.05 fw/s.
    _story(conn, "s_walk", "walkway", T0 - timedelta(seconds=20), T0, dir_lateral=-1, exit_foot_x=0.3,
           exit_speed=0.05, individual=individual, first_x=0.8)


def test_gap_handoff_met_trains_the_gap_and_opens_a_capped_ask_without_writing_a_label(pg) -> None:
    with pg.begin() as conn:
        _individual(conn, "ind-walk", "walkway", T0)
        _individual(conn, "ind-door", "doorbell", T0 + timedelta(seconds=40))
        _walker_left(conn)
        # Enters the doorbell on its right edge 40 s later, moving right-to-left; same appearance.
        _story(conn, "s_door", "doorbell", T0 + timedelta(seconds=40), T0 + timedelta(seconds=50), dir_lateral=-1,
               entry_edge="right", individual="ind-door", first_x=0.95)
    out = _cycle(pg, T0 + timedelta(seconds=400))
    assert out["opened_gap"] == 1 and out["met"] == 1, out
    (h,) = _rows(pg, "SELECT * FROM vision_handoff WHERE portal_id='walkway_to_door'")
    assert h["status"] == "met" and h["target_story_id"] == "s_door" and h["transit_sec"] == pytest.approx(40)
    assert h["g"] == pytest.approx(40 * 0.05 - 0.3) and 0 < h["p_value"] <= 1
    assert h["appearance_channel"] == "body" and h["appearance_score"] == pytest.approx(1.0, abs=1e-5)
    assert h["band_source"] == "prior" and h["steered"] is None
    ev = _rows(pg, "SELECT event_type, stream_id FROM vision_events")
    assert ev == [{"event_type": "handoff_met", "stream_id": "doorbell"}]
    (street,) = [e for _i, e in hl.fetch_outbox(pg)]
    assert street["kind"] == "handoff_met" and street["arrival_log_score"] == pytest.approx(h["arrival_log_score"], rel=1e-5)
    assert h["p_value"] >= vh.prior_p_floor(10) and h["attention_gate"] is False   # prior band: never surprising
    (ask,) = _rows(pg, "SELECT * FROM orion_ask")
    assert ask["source_kind"] == "vision_same_individual" and ask["source_ref"] == "ind-door|ind-walk"
    assert vh.HANDOFF_RULE_REF in ask["evidence_refs"] and len(ask["image_refs"]) == 2
    assert _rows(pg, "SELECT ask_id FROM vision_handoff WHERE portal_id='walkway_to_door'")[0]["ask_id"] == ask["ask_id"]
    # The doorbell story overlapped nothing on the walkway: recorded as unpaired, once.
    assert [r["status"] for r in _rows(pg, "SELECT status FROM vision_handoff WHERE portal_id='central_walk'")] == \
        ["unpaired"]
    # Never a label, never a merge, never a verdict.
    assert all(r["label"] is None for r in _rows(pg, "SELECT label FROM vision_individual"))
    assert _rows(pg, "SELECT * FROM vision_individual_pair_verdict") == []
    # Idempotent: a re-run neither duplicates nor asks again.
    _cycle(pg, T0 + timedelta(seconds=460))
    assert len(_rows(pg, "SELECT * FROM vision_handoff")) == 2 and len(_rows(pg, "SELECT * FROM orion_ask")) == 1


def test_ask_cap_zero_opens_none(pg) -> None:
    with pg.begin() as conn:
        _individual(conn, "ind-walk", "walkway", T0)
        _individual(conn, "ind-door", "doorbell", T0)
        _walker_left(conn)
        _story(conn, "s_door", "doorbell", T0 + timedelta(seconds=40), T0 + timedelta(seconds=50), dir_lateral=-1,
               entry_edge="right", individual="ind-door")
    cfg = hl.HandoffLoopConfig(rules=vh.HandoffConfig(same_ask_daily_cap=0), same_asks_enabled=False)
    _cycle(pg, T0 + timedelta(seconds=400), cfg=cfg)
    assert _rows(pg, "SELECT * FROM orion_ask") == []


def test_doorbell_not_looking_is_unscorable_not_missed(pg) -> None:
    with pg.begin() as conn:
        _walker_left(conn)
    out = _cycle(pg, T0 + timedelta(seconds=400))
    (h,) = _rows(pg, "SELECT status, coverage FROM vision_handoff")
    assert h == {"status": "unscorable", "coverage": 0.0} and out["unscorable"] == 1
    assert _rows(pg, "SELECT * FROM vision_handoff_hypothesis") == []
    assert _rows(pg, "SELECT * FROM vision_events") == []


def test_watched_miss_settles_unexplained_into_vision_unresolved_and_the_outbox(pg) -> None:
    with pg.begin() as conn:
        _walker_left(conn)
        _census(conn, "doorbell", T0, T0 + timedelta(seconds=300))   # the doorbell was looking
        _census(conn, "walkway", T0, T0 + timedelta(seconds=1200))
    out = _cycle(pg, T0 + timedelta(seconds=400))
    assert out["missed"] == 1, out
    (h,) = _rows(pg, "SELECT * FROM vision_handoff")
    assert h["status"] == "missed" and h["coverage"] >= 0.8 and h["p_value"] == pytest.approx(0.5)
    hyps = {r["kind"]: r for r in _rows(pg, "SELECT * FROM vision_handoff_hypothesis")}
    assert set(hyps) == set(vh.HYPOTHESES)
    assert hyps["detector_miss"]["status"] == "false" and hyps["late"]["status"] == "open"
    assert all(r["prior"] == pytest.approx(0.2) for r in hyps.values())
    assert [e["event_type"] for e in _rows(pg, "SELECT event_type FROM vision_events")] == ["handoff_missed"]
    # Past the deadline: nothing came true -> unexplained.
    _cycle(pg, h["late_until"] + timedelta(seconds=700))
    hyps = {r["kind"]: r["status"] for r in _rows(pg, "SELECT kind, status FROM vision_handoff_hypothesis")}
    assert hyps == {"detector_miss": "false", "late": "false", "turned_back": "false", "entered_place": "false",
                    "unexplained": "true"}
    (u,) = _rows(pg, "SELECT * FROM vision_unresolved")
    assert u["reason"] == "handoff_missed" and u["stream_id"] == "doorbell"
    out = hl.fetch_outbox(pg)
    assert len(out) == 1 and out[0][1]["kind"] == "handoff_missed" and "arrival_log_score" not in out[0][1]


def test_detector_miss_and_turned_back_settle_from_data(pg) -> None:
    with pg.begin() as conn:
        _walker_left(conn)
        _census(conn, "doorbell", T0, T0 + timedelta(seconds=300))
        # The doorbell detector saw a person in the window, but no story came of it.
        _crop(conn, "lonely", "doorbell", T0 + timedelta(seconds=50), vec=VEC_A, individual=None)
        # The walker comes back on the walkway, left-to-right.
        _story(conn, "s_back", "walkway", T0 + timedelta(seconds=200), T0 + timedelta(seconds=220), dir_lateral=1,
               exit_foot_x=0.9, landed=T0 + timedelta(seconds=280))
    _cycle(pg, T0 + timedelta(seconds=400))
    hyps = {r["kind"]: r["status"] for r in _rows(pg, "SELECT kind, status FROM vision_handoff_hypothesis")}
    assert hyps["detector_miss"] == "true" and hyps["turned_back"] == "true"
    h = _rows(pg, "SELECT late_until FROM vision_handoff WHERE source_story_id='s_walk'")[0]
    _cycle(pg, h["late_until"] + timedelta(seconds=700))
    hyps = {r["kind"]: r["status"] for r in _rows(pg, "SELECT kind, status FROM vision_handoff_hypothesis "
                                                      "WHERE handoff_id IN (SELECT handoff_id FROM vision_handoff "
                                                      "WHERE source_story_id='s_walk')")}
    assert hyps["unexplained"] == "false" and _rows(pg, "SELECT * FROM vision_unresolved") == []


def test_overlap_pair_on_the_central_walk(pg) -> None:
    with pg.begin() as conn:
        _story(conn, "d_porch", "doorbell", T0, T0 + timedelta(seconds=12), individual="ind-door")
        _story(conn, "w_patio", "walkway", T0 - timedelta(seconds=6), T0 + timedelta(seconds=2), dir_lateral=1,
               exit_foot_x=0.6, individual="ind-walk")
        _story(conn, "w_far", "walkway", T0 + timedelta(seconds=60), T0 + timedelta(seconds=70))
    out = _cycle(pg, T0 + timedelta(seconds=200))
    assert out["overlap_paired"] == 1, out
    (h,) = _rows(pg, "SELECT * FROM vision_handoff WHERE portal_id='central_walk'")
    assert h["source_story_id"] == "d_porch" and h["target_story_id"] == "w_patio" and h["status"] == "paired"
    (ev,) = [e for _i, e in hl.fetch_outbox(pg)]
    assert ev["kind"] == "overlap_pair" and ev["salience"] <= 0.1
    assert [e["event_type"] for e in _rows(pg, "SELECT event_type FROM vision_events")] == ["overlap_pair"]


def test_approach_door_met_by_ring_motion_and_an_unannounced_ring(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _story(conn, "w_toward", "walkway", T0 - timedelta(seconds=10), T0, dir_depth=1, individual="ind-walk")
        conn.execute(text("INSERT INTO vision_events (event_id, event_type, stream_id, created_at) VALUES "
                          "('ring-motion-1', 'doorbell_motion', 'doorbell', :t), "
                          "('ring-ding-2', 'doorbell_ring', 'doorbell', :t2)"),
                     {"t": T0 + timedelta(seconds=12), "t2": T0 + timedelta(seconds=3600)})
        _census(conn, "walkway", T0 + timedelta(seconds=3400), T0 + timedelta(seconds=3700))
    first = _cycle(pg, T0 + timedelta(seconds=100))
    assert first["opened_approach"] == 1, first
    out = _cycle(pg, T0 + timedelta(seconds=200))
    assert out["met"] == 1, out
    _cycle(pg, T0 + timedelta(seconds=3800))
    (a,) = _rows(pg, "SELECT * FROM vision_handoff WHERE source_story_id='w_toward'")
    assert a["status"] == "met" and a["target_event_id"] == "ring-motion-1" and a["transit_sec"] == pytest.approx(12)
    kinds = sorted(e["kind"] for _i, e in hl.fetch_outbox(pg))
    # One outbox row per event: the met grade does not overwrite the approach heads-up.
    assert kinds == ["approach_door", "handoff_met"]
    met = next(e for _i, e in hl.fetch_outbox(pg) if e["kind"] == "handoff_met")
    assert met["arrival_log_score"] == pytest.approx(a["arrival_log_score"], rel=1e-5)
    (r,) = _rows(pg, "SELECT * FROM vision_handoff WHERE status='unannounced'")
    assert r["target_event_id"] == "ring-ding-2" and r["coverage"] >= 0.8
    hyps = {h["kind"]: h["status"] for h in _rows(pg, "SELECT kind, status FROM vision_handoff_hypothesis")}
    assert hyps == {"detector_miss": "false", "out_of_view": "true"}


def test_live_window_sets_the_look_key(pg, monkeypatch) -> None:
    class FakeRedis:
        def __init__(self):
            self.kv, self.ttls = {}, {}

        def set(self, k, v, ex=None, nx=False):
            if nx and k in self.kv:
                return None
            self.kv[k], self.ttls[k] = v, ex
            return True

        def get(self, k):
            return self.kv.get(k)

        def ttl(self, k):
            return self.ttls.get(k, -2)

        def expire(self, k, t):
            self.ttls[k] = t

        def close(self):
            pass

    fake = FakeRedis()

    class FakeSteering(hl.Steering):
        def __init__(self, _url):
            self.client, self.failures = fake, 0

    monkeypatch.setattr(hl, "Steering", FakeSteering)
    with pg.begin() as conn:
        _walker_left(conn)
        conn.execute(__import__("sqlalchemy").text("UPDATE vision_motion_story SET created_at = :t"),
                     {"t": T0 + timedelta(seconds=5)})
    out = _cycle(pg, T0 + timedelta(seconds=10))
    (h,) = _rows(pg, "SELECT handoff_id, steered, window_end FROM vision_handoff")
    assert out["steered"] == 1 and h["steered"] == [vh.LOOK_KEY]
    assert fake.kv[vh.LOOK_KEY] == h["handoff_id"] and fake.ttls[vh.LOOK_KEY] == 86


def test_missing_migration_is_migration_missing(pg) -> None:
    from sqlalchemy import create_engine, text

    from app.vision_individuals import MigrationMissing

    with pg.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS handoff_empty CASCADE"))
        conn.execute(text("CREATE SCHEMA handoff_empty"))
    sep = "&" if "?" in PG else "?"
    empty = create_engine(f"{PG}{sep}options=-csearch_path%3Dhandoff_empty")
    try:
        with pytest.raises(MigrationMissing, match="manual_migration_vision_handoff_v1"):
            hl.run_one_handoff_cycle(postgres_uri=PG, cfg=CFG, topology=TOPO, now=T0, engine=empty)
    finally:
        empty.dispose()
        with pg.begin() as conn:
            conn.execute(text("DROP SCHEMA handoff_empty CASCADE"))


def test_outbox_publishes_and_marks_only_on_success(pg) -> None:
    with pg.begin() as conn:
        _story(conn, "d_porch", "doorbell", T0, T0 + timedelta(seconds=12))
        _story(conn, "w_patio", "walkway", T0, T0 + timedelta(seconds=10))
    _cycle(pg, T0 + timedelta(seconds=200))

    class Bus:
        def __init__(self, fail):
            self.fail, self.sent = fail, []

        async def publish(self, channel, env):
            if self.fail:
                raise RuntimeError("bus down")
            self.sent.append((channel, env))

    down = Bus(True)
    assert asyncio.run(hl.publish_outbox(down, pg, service_name="t", service_version="0")) == 0
    assert len(hl.fetch_outbox(pg)) == 1
    up = Bus(False)
    assert asyncio.run(hl.publish_outbox(up, pg, service_name="t", service_version="0")) == 1
    assert up.sent[0][0] == vh.STREET_CHANNEL and up.sent[0][1].kind == "vision.street.event.v1"
    assert hl.fetch_outbox(pg) == []


def test_stale_facts_are_recorded_but_not_published(pg) -> None:
    # A restart catching up on an hour-old approach must not bid for attention now.
    with pg.begin() as conn:
        _story(conn, "w_toward", "walkway", T0 - timedelta(seconds=10), T0, dir_depth=1, landed=T0 + timedelta(seconds=60))
    out = _cycle(pg, T0 + timedelta(seconds=1800))
    assert hl.fetch_outbox(pg) == [] and out["street_stale_skipped"] >= 1


def test_approach_miss_settles_unexplained_and_still_publishes(pg) -> None:
    # Review #1: the approach heads-up in the outbox must not hide the later miss.
    with pg.begin() as conn:
        _story(conn, "w_toward", "walkway", T0 - timedelta(seconds=10), T0, dir_depth=1)
        _census(conn, "doorbell", T0, T0 + timedelta(seconds=120))
        _census(conn, "walkway", T0, T0 + timedelta(seconds=1800))
    _cycle(pg, T0 + timedelta(seconds=100))
    _cycle(pg, T0 + timedelta(seconds=200))
    (h,) = _rows(pg, "SELECT status, late_until FROM vision_handoff WHERE source_story_id='w_toward'")
    assert h["status"] == "missed"
    _cycle(pg, h["late_until"] + timedelta(seconds=700))
    kinds = sorted(e["kind"] for _i, e in hl.fetch_outbox(pg))
    assert kinds == ["approach_door", "handoff_missed"]


def test_stuck_stories_settle_false_and_never_write_unresolved(pg) -> None:
    # Review #4: no data behind it means no "unexplained".
    from sqlalchemy import text

    with pg.begin() as conn:
        _walker_left(conn)
        _census(conn, "doorbell", T0, T0 + timedelta(seconds=300))
    _cycle(pg, T0 + timedelta(seconds=400))
    (h,) = _rows(pg, "SELECT late_until FROM vision_handoff WHERE portal_id='walkway_to_door'")
    with pg.begin() as conn:   # the target camera's stories stop catching up
        conn.execute(text("UPDATE vision_motion_story_cursor SET processed_until = :t WHERE stream_id='doorbell'"),
                     {"t": T0 + timedelta(seconds=300)})
    _cycle(pg, h["late_until"] + timedelta(seconds=700 + 3700))
    hyps = {r["kind"]: (r["status"], r["evidence"]) for r in _rows(pg, "SELECT kind, status, evidence FROM vision_handoff_hypothesis")}
    assert hyps["unexplained"][0] == "false" and hyps["unexplained"][1].get("data") == "never_caught_up"
    assert _rows(pg, "SELECT * FROM vision_unresolved") == [] and hl.fetch_outbox(pg) == []


def test_companion_story_opens_nothing(pg) -> None:
    # Review #5: a companion duplicates a walker already in its own story.
    from sqlalchemy import text

    with pg.begin() as conn:
        _walker_left(conn)
        conn.execute(text("UPDATE vision_motion_story SET drop_reason='companion'"))
    out = _cycle(pg, T0 + timedelta(seconds=400))
    assert "opened_gap" not in out and _rows(pg, "SELECT * FROM vision_handoff") == []


def test_announced_and_blind_rings_get_rows_so_they_are_examined_once(pg) -> None:
    # Review #6 / #8.
    from sqlalchemy import text

    with pg.begin() as conn:
        _story(conn, "w_still", "walkway", T0 - timedelta(seconds=30), T0 + timedelta(seconds=20))   # still there at ring
        conn.execute(text("INSERT INTO vision_events (event_id, event_type, stream_id, created_at) VALUES "
                          "('ding-a', 'doorbell_ring', 'doorbell', :t), ('ding-b', 'doorbell_ring', 'doorbell', :t2)"),
                     {"t": T0 + timedelta(seconds=5), "t2": T0 + timedelta(seconds=900)})
    out = _cycle(pg, T0 + timedelta(seconds=1000))
    rows = {r["target_event_id"]: r["status"] for r in _rows(pg, "SELECT target_event_id, status FROM vision_handoff "
                                                                  "WHERE target_event_id IS NOT NULL")}
    assert rows == {"ding-a": "announced", "ding-b": "unscorable"}, (rows, out)
    assert _rows(pg, "SELECT * FROM vision_handoff_hypothesis") == []


def test_a_poison_row_is_isolated_and_settled(pg, monkeypatch) -> None:
    # Review #3: one bad row must not stop the step (or the next tick) for the others.
    with pg.begin() as conn:
        _walker_left(conn)
        _story(conn, "s_walk2", "walkway", T0 + timedelta(seconds=100), T0 + timedelta(seconds=120), dir_lateral=-1,
               exit_foot_x=0.3, exit_speed=0.05, first_x=0.8)
    _cycle(pg, T0 + timedelta(seconds=200))
    real = hl._source

    def boom(ctx, story_id):
        if story_id == "s_walk":
            raise RuntimeError("poison")
        return real(ctx, story_id)

    monkeypatch.setattr(hl, "_source", boom)
    out = _cycle(pg, T0 + timedelta(seconds=600))
    st = {r["source_story_id"]: r["status"] for r in _rows(pg, "SELECT source_story_id, status FROM vision_handoff "
                                                                "WHERE portal_id='walkway_to_door'")}
    assert out["row_failed_grade"] == 1 and st == {"s_walk": "unscorable", "s_walk2": "unscorable"}

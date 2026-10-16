"""The walkway body-match line: continuation lane + re-identification at 0.90.

docs/superpowers/specs/2026-10-16-vision-body-match-proposal.md, acceptance
check 1. Pure ``apply_batch`` tests; the live replay is
evals/run_vision_body_match_replay_eval.py.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import vision_individuals as vi  # noqa: E402
import _body_match_fixture as fx  # noqa: E402

TZ = ZoneInfo("America/Denver")
T0 = datetime(2026, 10, 16, 1, 0, tzinfo=timezone.utc)
H = 200.0  # box height in px

# The pre-continuation apply_batch (main at f2bccf3dc) on fx.stream(), line 0.80.
GOLDEN_0_80 = "9fc55b3446db8a0966f899927076956db88c75bc652b3b1bcbfe40d7301ac8eb"


def _at(sim: float) -> tuple:
    """A unit vector with cosine ``sim`` to (1, 0)."""
    return (sim, math.sqrt(max(0.0, 1 - sim * sim)))


def _box(x: float, y: float = 500.0, h: float = H) -> tuple:
    return (x, y, x + h * 0.4, y + h)


def _crop(cid: str, obs: str, at: datetime, emb: tuple, box: tuple, face=None) -> vi.CropRow:
    return vi.CropRow(crop_id=cid, observation_id=obs, stream_id="walkway", observed_at=at, label="person",
                      box_xyxy=box, zone="walkway", embedding=emb, embedding_ref=None, artifact_id=None,
                      face_embedding=face)


def _ind(iid: str, centroid=(1.0, 0.0), seen: datetime = T0, n: int = 5) -> vi.Individual:
    return vi.Individual(iid, "walkway", "person", list(centroid), n, seen - timedelta(hours=1), seen, 1, 1)


def _sighting(iid: str, ended: datetime, box: tuple) -> vi.Sighting:
    return vi.Sighting(sighting_id=f"s-{iid}", individual_id=iid, stream_id="walkway",
                       started_at=ended - timedelta(seconds=20), ended_at=ended, observation_count=5,
                       zone_counts={"walkway": 5}, last_box_xyxy=list(box))


def _apply(crops, inds, sightings=(), **kw):
    args = dict(match_threshold=0.90, continuation_threshold=0.88, continuation_gap_sec=10.0,
                continuation_max_disp=1.5)
    args.update(kw)
    return vi.apply_batch(crops, individuals={i.individual_id: i for i in inds}, latest_sightings={},
                          nearby_sightings={s.individual_id: [s] for s in sightings}, no_embed_zones=(),
                          merge_gap_sec=60.0, tz=TZ, new_id=iter(f"new{k}" for k in range(99)).__next__, **args)


def test_a_close_recent_crop_joins_at_0_89() -> None:
    b = _apply([_crop("c", "o1", T0 + timedelta(seconds=5), _at(0.89), _box(100 + H))],
               [_ind("a")], [_sighting("a", T0, _box(100))])
    assert b.crop_individual["c"] == "a"
    assert (b.assigned_by_continuation, b.assigned_by_reid, b.new_individual_ids) == (1, 0, [])


def test_the_same_crop_20_s_later_opens_a_new_individual() -> None:
    b = _apply([_crop("c", "o1", T0 + timedelta(seconds=20), _at(0.89), _box(100 + H))],
               [_ind("a")], [_sighting("a", T0, _box(100))])
    assert b.crop_individual["c"] == "new0" and b.assigned_by_continuation == 0


def test_the_same_crop_3_box_heights_away_opens_a_new_individual() -> None:
    b = _apply([_crop("c", "o1", T0 + timedelta(seconds=5), _at(0.89), _box(100 + 3 * H))],
               [_ind("a")], [_sighting("a", T0, _box(100))])
    assert b.crop_individual["c"] == "new0"


def test_a_crop_0_91_from_someone_seen_an_hour_ago_joins_by_reid() -> None:
    b = _apply([_crop("c", "o1", T0 + timedelta(hours=1), _at(0.91), _box(1500))],
               [_ind("a")], [_sighting("a", T0, _box(100))])
    assert b.crop_individual["c"] == "a"
    assert (b.assigned_by_continuation, b.assigned_by_reid) == (0, 1)


def test_a_crop_0_85_from_someone_seen_an_hour_ago_no_longer_joins() -> None:
    """The old 0.80 line took it; the re-identification line at 0.90 does not."""
    crop = _crop("c", "o1", T0 + timedelta(hours=1), _at(0.85), _box(1500))
    assert _apply([crop], [_ind("a")]).crop_individual["c"] == "new0"
    old = _apply([crop], [_ind("a")], match_threshold=0.80, continuation_threshold=None)
    assert old.crop_individual["c"] == "a"


def test_a_close_recent_person_is_not_stolen_by_a_better_old_individual() -> None:
    """Continuation runs before re-identification: the crop stays with the
    person it is walking as, even if an old individual scores higher."""
    old = _ind("old", centroid=_at(0.99), seen=T0 - timedelta(hours=2))
    b = _apply([_crop("c", "o1", T0 + timedelta(seconds=3), _at(0.90), _box(120))],
               [_ind("a"), old], [_sighting("a", T0, _box(100))])
    assert b.crop_individual["c"] == "a" and b.assigned_by_continuation == 1


def test_same_frame_rule_holds_inside_the_continuation_lane() -> None:
    t = T0 + timedelta(seconds=3)
    crops = [_crop("c0", "o1", t, _at(0.95), _box(100)), _crop("c1", "o1", t, _at(0.94), _box(110))]
    b = _apply(crops, [_ind("a")], [_sighting("a", T0, _box(100))])
    assert b.crop_individual["c0"] == "a"
    assert b.crop_individual["c1"] != "a"


def test_continuation_follows_crops_assigned_in_the_same_batch() -> None:
    """A passing walks 1 box height a frame: each crop is close to the last
    one the batch assigned, never to where the individual started."""
    crops = [_crop(f"c{k}", f"o{k}", T0 + timedelta(seconds=2.6 * k), _at(0.885), _box(100 + k * H))
             for k in range(1, 6)]
    b = _apply(crops, [_ind("a")], [_sighting("a", T0, _box(100))])
    assert {b.crop_individual[c.crop_id] for c in crops} == {"a"}
    assert b.assigned_by_continuation == 5


def test_a_face_match_still_wins_first() -> None:
    other = _ind("f")
    other.face_centroid, other.face_n = [1.0, 0.0], 3
    other.centroid = list(_at(0.1))
    b = _apply([_crop("c", "o1", T0 + timedelta(seconds=3), _at(0.95), _box(100), face=(1.0, 0.0))],
               [_ind("a"), other], [_sighting("a", T0, _box(100))])
    assert b.crop_individual["c"] == "f" and b.assigned_by_face == 1 and b.assigned_by_continuation == 0


def test_lane_counts_add_up_to_assigned() -> None:
    crops = fx.build(vi, fx.stream())
    b = vi.apply_batch(crops[:300], individuals={}, latest_sightings={}, no_embed_zones=(),
                       match_threshold=0.90, merge_gap_sec=60.0, tz=TZ, continuation_threshold=0.88)
    assert b.assigned_by_face + b.assigned_by_continuation + b.assigned_by_reid + len(b.new_individual_ids) == b.assigned
    assert b.assigned_by_continuation > 0


def test_rollback_continuation_off_and_0_80_is_byte_identical_to_the_old_rule() -> None:
    """Kill switch: VISION_INDIVIDUALS_CONTINUATION_ENABLED=false + line 0.80
    reproduces the pre-patch apply_batch exactly (golden from main)."""
    rows = fx.stream()
    member, inds = fx.run_batches(vi, rows, TZ, match_threshold=0.80, face_match_threshold=0.60,
                                  continuation_threshold=None)
    assert fx.digest(member, inds) == GOLDEN_0_80
    # The new rule really does something different on the same stream.
    m2, i2 = fx.run_batches(vi, rows, TZ, match_threshold=0.90, face_match_threshold=0.60,
                            continuation_threshold=0.88)
    assert fx.digest(m2, i2) != GOLDEN_0_80


def test_eval_replay_baseline_equals_the_old_rule() -> None:
    """The eval's baseline row is the rollback config, driven through the same apply_batch."""
    sys.path.insert(0, str(SERVICE / "evals"))
    import run_vision_body_match_replay_eval as ev

    assert ev.BASELINE.match_threshold == 0.80 and ev.BASELINE.continuation_enabled is False
    rows = fx.stream()
    crops = fx.build(vi, rows)
    member, inds = ev.replay(crops, [r["landed"] for r in rows], ev.BASELINE, TZ)
    ref, ref_inds = fx.run_batches(vi, rows, TZ, match_threshold=0.80, face_match_threshold=0.60,
                                   continuation_threshold=None)
    canon = lambda m: sorted(sorted(c for c in m if m[c] == i) for i in set(m.values()))  # noqa: E731
    assert canon(member) == canon(ref)


def test_eval_scores_on_a_tiny_stream() -> None:
    sys.path.insert(0, str(SERVICE / "evals"))
    import run_vision_body_match_replay_eval as ev

    # Two people walking side by side for 3 frames.
    crops = []
    for k in range(3):
        t = T0 + timedelta(seconds=2.5 * k)
        crops.append(_crop(f"a{k}", f"o{k}", t, _at(0.99), _box(100 + 50 * k)))
        crops.append(_crop(f"b{k}", f"o{k}", t, _at(0.10), _box(900 + 50 * k)))
    tp = ev.track_pairs(crops)
    tl = ev.tracklets(len(crops), tp)
    conf = ev.conflicts(crops, tl)
    assert len(tp) == 4 and len(set(tl)) == 2 and len(conf) == 1
    good = {c.crop_id: c.crop_id[0] for c in crops}
    s = ev.score(crops, good, tp, tl, conf)
    assert (s["mixed"], s["split"], s["individuals"]) == (0.0, 0.0, 2)
    merged = {c.crop_id: "x" for c in crops}
    assert ev.score(crops, merged, tp, tl, conf)["mixed"] == 1.0


def test_box_distance_is_in_mean_box_heights() -> None:
    assert vi.box_centre_distance_in_heights((0, 0, 10, 100), (0, 150, 10, 250)) == 1.5
    assert vi.box_centre_distance_in_heights((0, 0, 10, 100), (0, 0, 10, 300)) == 100 / 200
    assert vi.box_centre_distance_in_heights((), (0, 0, 1, 1)) == math.inf


def test_settings_and_loop_wiring_ship_the_new_rule_on() -> None:
    from app.settings import Settings
    from app.vision_individuals_loop import individuals_config

    s = Settings(_env_file=None)
    cfg = individuals_config(s)
    assert (cfg.match_threshold, cfg.continuation_enabled, cfg.continuation_threshold,
            cfg.continuation_gap_sec, cfg.continuation_max_disp) == (0.90, True, 0.88, 10.0, 1.5)
    d = vi.IndividualsConfig()
    assert (d.match_threshold, d.continuation_enabled, d.continuation_threshold) == (0.90, True, 0.88)


def test_env_example_carries_the_new_keys_on() -> None:
    text = (SERVICE / ".env_example").read_text()
    for line in ("VISION_INDIVIDUALS_MATCH_THRESHOLD=0.90", "VISION_INDIVIDUALS_CONTINUATION_ENABLED=true",
                 "VISION_INDIVIDUALS_CONTINUATION_THRESHOLD=0.88", "VISION_INDIVIDUALS_CONTINUATION_GAP_SEC=10",
                 "VISION_INDIVIDUALS_CONTINUATION_MAX_DISP=1.5"):
        assert line in text.splitlines(), line


# --- the deployed cycle against REAL Postgres (ORION_TEST_PG_URI; see test_vision_reanchor.py) -------

import os  # noqa: E402

import pytest  # noqa: E402

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")


@needs_pg
@pytest.mark.parametrize("enabled,expected", [(True, "a"), (False, None)])
def test_the_cycle_continues_a_db_sighting_and_the_kill_switch_stops_it(enabled, expected) -> None:
    """run_one_individuals_cycle loads the individual's sighting (last box,
    ended_at) and joins a close crop 5 s later at 0.89 -- only with the
    continuation lane on; off, 0.89 < 0.90 opens a new individual."""
    from sqlalchemy import create_engine, text

    now = T0 + timedelta(minutes=5)
    seen = now - timedelta(minutes=2)
    engine = create_engine(PG)
    try:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE orion_ask, vision_individual_pair_verdict, vision_individual_sighting, "
                              "vision_individual, vision_crop_observation, vision_individuals_cursor CASCADE"))
            conn.execute(text(
                "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, centroid_n, first_seen_at, "
                "last_seen_at, sighting_count, distinct_days) VALUES ('a', 'walkway', 'person', :c, 5, :t, :t, 1, 1)"),
                {"c": [1.0, 0.0], "t": seen})
            conn.execute(text(
                "INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, started_at, ended_at, "
                "observation_count, zone_counts, last_box_xyxy) VALUES ('s-a', 'a', 'walkway', :s, :t, 5, "
                "'{\"walkway\": 5}'::jsonb, :b)"),
                {"s": seen - timedelta(seconds=20), "t": seen, "b": list(_box(100))})
            conn.execute(text("INSERT INTO vision_individuals_cursor (stream_id, last_created_at) VALUES ('walkway', :t)"),
                         {"t": seen})
            conn.execute(text(
                "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, observed_at, label, score, "
                "box_xyxy, zone, embedding, created_at) VALUES ('c1', 'o1', 'walkway', :t, 'person', 0.9, :b, "
                "'walkway', :e, :t)"),
                {"t": seen + timedelta(seconds=5), "b": list(_box(100 + H)), "e": list(_at(0.89))})
        cfg = vi.IndividualsConfig(continuation_enabled=enabled, ask_face_min_faces=0)
        summary, _ = vi.run_one_individuals_cycle(postgres_uri=PG, cfg=cfg, zones_by_stream={}, now=now)
        with engine.begin() as conn:
            got = conn.execute(text("SELECT individual_id FROM vision_crop_observation WHERE crop_id='c1'")).scalar()
        assert summary["assigned"] == 1
        if expected:
            assert got == expected and summary["assigned_by_continuation"] == 1
        else:
            assert got not in (None, "a") and summary["assigned_by_continuation"] == 0
    finally:
        engine.dispose()

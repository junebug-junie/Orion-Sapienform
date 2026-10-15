"""Walkway v2 (2026-10-15): face-preferred assignment and stationary fixtures.

Pure reducer functions (no DB) plus SQL-shape checks for the readers that
must leave a stationary fixture out.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "orion-sql-writer"))

from app import vision_individuals as vi  # noqa: E402
from app import vision_rhythm as vr  # noqa: E402

TZ = ZoneInfo("America/Denver")
T0 = datetime(2026, 10, 15, 4, 57, 24, tzinfo=timezone.utc)
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "walkway_patio_skeleton_boxes.csv"
THUMB = "thumb:" + "ab" * 32
FACE_THUMB = "thumb:" + "cd" * 32

APP_A = (1.0, 0.0, 0.0)
APP_B = (0.0, 1.0, 0.0)
FACE_A = (1.0, 0.0)
FACE_B = (0.0, 1.0)


def _ind(iid, centroid, face=None, last=T0):
    return vi.Individual(iid, "walkway", "person", list(centroid), 5, last - timedelta(days=3), last, 3, 2,
                         face_centroid=list(face) if face else None, face_n=3 if face else 0)


def _crop(cid, t, emb, face=None, box=(100.0, 100.0, 200.0, 400.0), zone="walkway", face_thumb=None):
    return vi.CropRow(crop_id=cid, observation_id=f"o-{cid}", stream_id="walkway", observed_at=t,
                      label="person", box_xyxy=tuple(box), zone=zone, embedding=tuple(emb),
                      embedding_ref="r", artifact_id="a", face_embedding=tuple(face) if face else None,
                      face_thumb_ref=face_thumb)


def _batch(crops, inds, **kw):
    kw.setdefault("match_threshold", 0.8)
    kw.setdefault("merge_gap_sec", 60)
    return vi.apply_batch(crops, individuals={i.individual_id: i for i in inds}, latest_sightings={},
                          no_embed_zones=set(), tz=TZ, **kw)


# --- faces --------------------------------------------------------------------


def test_confident_face_match_beats_appearance() -> None:
    # Appearance says A (same jacket); the face says B. The face wins.
    a, b = _ind("A", APP_A, FACE_A), _ind("B", APP_B, FACE_B)
    res = _batch([_crop("c1", T0 + timedelta(minutes=5), APP_A, face=FACE_B)], [a, b])
    (s,) = res.sightings.values()
    assert s.individual_id == "B"
    assert res.assigned_by_face == 1


def test_appearance_fallback_when_no_face_is_visible() -> None:
    a, b = _ind("A", APP_A, FACE_A), _ind("B", APP_B, FACE_B)
    res = _batch([_crop("c1", T0 + timedelta(minutes=5), APP_A)], [a, b])
    (s,) = res.sightings.values()
    assert s.individual_id == "A" and res.assigned_by_face == 0


def test_appearance_fallback_when_face_match_is_not_confident() -> None:
    a, b = _ind("A", APP_A, FACE_A), _ind("B", APP_B, FACE_B)
    weak = (0.7071, 0.7071)  # cosine 0.707 to both faces; below a 0.9 threshold
    res = _batch([_crop("c1", T0 + timedelta(minutes=5), APP_A, face=weak)], [a, b], face_match_threshold=0.9)
    (s,) = res.sightings.values()
    assert s.individual_id == "A" and res.assigned_by_face == 0


def test_face_seen_by_appearance_match_starts_a_face_centroid() -> None:
    a = _ind("A", APP_A)  # no face yet
    res = _batch([_crop("c1", T0 + timedelta(minutes=5), APP_A, face=FACE_B, face_thumb=FACE_THUMB)], [a])
    ind = res.individuals["A"]
    assert ind.face_centroid == [0.0, 1.0] and ind.face_n == 1
    (s,) = res.sightings.values()
    assert s.face_thumb_ref == FACE_THUMB


def test_new_individual_from_a_face_crop_carries_its_face() -> None:
    res = _batch([_crop("c1", T0, APP_A, face=FACE_A)], [], new_id=lambda: "N1")
    assert res.individuals["N1"].face_centroid == [1.0, 0.0]


def test_face_matching_can_be_disabled() -> None:
    a, b = _ind("A", APP_A, FACE_A), _ind("B", APP_B, FACE_B)
    res = _batch([_crop("c1", T0 + timedelta(minutes=5), APP_A, face=FACE_B)], [a, b], face_match_threshold=None)
    assert next(iter(res.sightings.values())).individual_id == "A"


def test_ask_image_prefers_the_face_thumbnail() -> None:
    now = T0
    rows = [(THUMB, now - timedelta(hours=1), False, None),
            (THUMB, now - timedelta(hours=3), False, FACE_THUMB)]
    assert vi.ask_image_ref(rows, now=now, thumb_retention_days=10, ask_expiry_days=7) == FACE_THUMB
    # Old 3-tuples still work (crop thumbnail only).
    assert vi.ask_image_ref([(THUMB, now, False)], now=now, thumb_retention_days=10, ask_expiry_days=7) == THUMB


# --- stationary fixtures, from the real patio skeleton boxes ---------------------


def _skeleton_rows(repeat: int = 2):
    """The real 25.7 min of boxes, replayed back-to-back ``repeat`` times (the
    prop sat there all night; the query window was 26 minutes)."""
    real = []
    for line in FIXTURE.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        t, *box = line.split(",")
        real.append((int(t), [float(v) for v in box]))
    span = real[-1][0] + 20
    return [(k * span + t, box) for k in range(repeat) for t, box in real]


def _skeleton_crops(repeat: int = 2):
    return [_crop(f"sk{i}", T0 + timedelta(seconds=t), APP_A, box=box, zone="patio")
            for i, (t, box) in enumerate(_skeleton_rows(repeat))]


def test_skeleton_box_pattern_becomes_stationary_after_thirty_minutes() -> None:
    crops = _skeleton_crops()
    res = _batch(crops, [], new_id=iter(f"id{k}" for k in range(1000)).__next__)
    assert len({s.individual_id for s in res.sightings.values()}) == 1  # one prop, one individual
    (ind,) = res.individuals.values()
    assert ind.stationary is True
    # Before 30 minutes it was not yet a fixture; after, every crop is.
    first_30 = [c for c in crops if (c.observed_at - T0).total_seconds() < 1800]
    after = [c for c in crops if (c.observed_at - T0).total_seconds() >= 1800]
    assert not any(res.crop_stationary[c.crop_id] for c in first_30)
    assert all(res.crop_stationary[c.crop_id] for c in after)
    # The sighting that holds the newest crop is flagged.
    newest = max(res.sightings.values(), key=lambda s: s.ended_at)
    assert newest.stationary is True


def test_real_gaps_up_to_105s_do_not_break_the_run() -> None:
    # Live max gap was 105 s, above the 60 s sighting merge gap: the run lives
    # on the individual, so split sightings still add up to one still run.
    res = _batch(_skeleton_crops(), [], new_id=iter(f"id{k}" for k in range(1000)).__next__)
    assert len(res.sightings) > 1
    (ind,) = res.individuals.values()
    assert ind.still_since == T0


def test_stationary_fixture_is_left_out_of_the_patio_head_count() -> None:
    crops = _skeleton_crops()
    res = _batch(crops, [], new_id=iter(f"id{k}" for k in range(1000)).__next__)
    late = [c for c in crops if (c.observed_at - T0).total_seconds() >= 1800]
    stationary_boxes = [(i.kind, i.still_box) for i in res.individuals.values() if i.stationary]
    assert vi.head_count_crops(late, "patio", crop_stationary=res.crop_stationary,
                               stationary_boxes=stationary_boxes, iou=0.5) == []
    # A real visitor elsewhere on the patio in the same tick still counts.
    visitor = _crop("v1", late[-1].observed_at, APP_B, box=(1500.0, 900.0, 1650.0, 1500.0), zone="patio")
    assert vi.head_count_crops(late + [visitor], "patio", crop_stationary={**res.crop_stationary, "v1": False},
                               stationary_boxes=stationary_boxes, iou=0.5) == [visitor]
    # An un-embedded frame of the prop (no individual) is matched by its box.
    bare = vi.CropRow("b1", "o-b1", "walkway", late[-1].observed_at, "person", (2061.0, 877.0, 2274.0, 1228.0),
                      "patio", None, None, None)
    assert vi.head_count_crops([bare], "patio", crop_stationary={}, stationary_boxes=stationary_boxes,
                               iou=0.5) == []


def test_moving_box_ends_stationary() -> None:
    crops = _skeleton_crops()
    res = _batch(crops, [], new_id=iter(f"id{k}" for k in range(1000)).__next__)
    (ind,) = res.individuals.values()
    assert ind.stationary
    # The "skeleton" was a person all along; they stand up and walk off.
    t = crops[-1].observed_at + timedelta(seconds=20)
    walk = _crop("w1", t, APP_A, box=(1700.0, 700.0, 1900.0, 1300.0), zone="patio")
    res2 = _batch([walk], [ind])
    moved = res2.individuals[ind.individual_id]
    assert moved.stationary is False and res2.crop_stationary["w1"] is False
    assert next(iter(res2.sightings.values())).stationary is False


def test_stationary_survives_a_long_gap_in_the_same_place() -> None:
    res = _batch(_skeleton_crops(), [], new_id=iter(f"id{k}" for k in range(1000)).__next__)
    (ind,) = res.individuals.values()
    back = _crop("n1", T0 + timedelta(hours=20), APP_A, box=(2062.0, 877.0, 2273.0, 1229.0), zone="patio")
    res2 = _batch([back], [ind])
    assert res2.individuals[ind.individual_id].stationary is True
    assert next(iter(res2.sightings.values())).stationary is True


def test_a_long_gap_restarts_a_run_that_was_not_yet_stationary() -> None:
    ind = _ind("A", APP_A)
    first = _crop("c1", T0, APP_A)
    gap = _crop("c2", T0 + timedelta(minutes=25), APP_A)  # 25 min > 300 s max gap
    later = _crop("c3", T0 + timedelta(minutes=40), APP_A)
    res = _batch([first, gap, later], [ind])
    assert res.individuals["A"].stationary is False
    assert res.individuals["A"].still_since == T0 + timedelta(minutes=40)


def test_a_walking_person_is_never_stationary() -> None:
    crops = [_crop(f"p{k}", T0 + timedelta(seconds=20 * k), APP_A, box=(100.0 + 120 * k, 500.0, 200.0 + 120 * k, 800.0))
             for k in range(200)]
    res = _batch(crops, [], new_id=iter(f"id{k}" for k in range(1000)).__next__)
    assert not any(res.crop_stationary.values())


def test_stationary_rule_can_be_disabled() -> None:
    res = _batch(_skeleton_crops(), [], stationary_min_sec=0,
                 new_id=iter(f"id{k}" for k in range(1000)).__next__)
    assert not any(res.crop_stationary.values())


# --- readers leave stationary out --------------------------------------------


class _Res:
    def __init__(self, rows=None):
        self._rows = rows or []

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._rows[0][0] if self._rows else None

    @property
    def rowcount(self):
        return 1


class _RecConn:
    def __init__(self, responses=None):
        self.sql = []
        self.params = []
        self.responses = responses or {}

    def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        self.sql.append(sql)
        self.params.append(params or {})
        for key, rows in self.responses.items():
            if key in sql:
                return _Res(rows)
        return _Res([])


def test_stationary_sighting_gets_no_attention_score_or_event() -> None:
    conn = _RecConn()
    s = vi.Sighting(sighting_id="s1", individual_id="A", stream_id="walkway", started_at=T0,
                    ended_at=T0 + timedelta(hours=1), stationary=True)
    assert vi._score_and_maybe_event(conn, s, _ind("A", APP_A), {}, TZ, vi.IndividualsConfig()) is False
    assert not any("INSERT INTO vision_events" in q for q in conn.sql)
    (upd,) = [p for q, p in zip(conn.sql, conn.params) if q.startswith("UPDATE vision_individual_sighting")]
    assert json.loads(upd["c"]) == {"_basis": "stationary"}


def test_ask_candidates_count_only_non_stationary_sightings() -> None:
    conn = _RecConn({"SELECT count(*) FROM orion_ask": [(0,)]})
    vi._open_asks(conn, T0, TZ, vi.IndividualsConfig())
    cand_sql = next(q for q in conn.sql if "FROM vision_individual i" in q and "JOIN LATERAL" in q)
    assert "NOT i.stationary" in cand_sql and "NOT s.stationary" in cand_sql
    # The face rule's candidates leave stationary fixtures out too.
    face_sql = next(q for q in conn.sql if "FROM vision_individual i" in q and "i.face_n >= :f" in q)
    assert "NOT i.stationary" in face_sql
    assert "v.n >= :n AND v.d >= :d" in cand_sql


def test_rhythm_subjects_skip_stationary_individuals_and_sightings() -> None:
    conn = _RecConn({"FROM vision_individual WHERE": [("A", "walkway", "person", None)]})
    cfg = vr.RhythmConfig(label_streams=())
    vr._load_subjects(conn, T0, cfg)
    assert any("FROM vision_individual WHERE" in q and "NOT stationary" in q for q in conn.sql)
    assert any("FROM vision_individual_sighting" in q and "NOT stationary" in q for q in conn.sql)
    vr._occurred(conn, "walkway", "individual:A", T0, T0 + timedelta(hours=1))
    assert "NOT stationary" in conn.sql[-1]


def test_head_count_snapshot_keeps_the_reader_flag() -> None:
    snap = vi.patio_snapshot(prev=None, batch_last_seen_at=T0, batch_count=1, now=T0, present_sec=120,
                             grace_sec=600, zone="patio")
    assert snap["head_count_zone"] is True and snap["no_embed_zone"] is True
    assert snap["subject"] == {"count": 1}


def test_cycle_refuses_to_run_without_the_v2_columns() -> None:
    class _C:
        def execute(self, stmt, params=None):
            sql = " ".join(str(stmt).split())
            if "to_regclass" in sql:
                return _Res([])
            return _Res([("vision_crop_observation", "face_embedding")])

    try:
        vi._check_tables(_C())
    except vi.MigrationMissing as exc:
        assert "manual_migration_walkway_privacy_v2.sql" in str(exc)
        assert "vision_individual.stationary" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected MigrationMissing")


def test_fixture_is_the_real_box() -> None:
    rows = list(csv.reader(line for line in FIXTURE.read_text().splitlines() if not line.startswith("#")))
    assert len(rows) == 79
    x1s = [float(r[1]) for r in rows]
    assert 2040 < min(x1s) and max(x1s) < 2090


def test_gapped_cadence_reflags_the_runs_earlier_sightings() -> None:
    # Review repro: 20 s steps with a ~100 s gap every few minutes (live max
    # gap 105 s) split the prop into a new sighting every ~3 min. Without the
    # re-flag, ~10 sightings from before the 30-minute mark stayed ordinary
    # and alone met ask_min_sightings=10 in one night.
    crops, t, k = [], 0, 0
    while t <= 2400:
        crops.append(_crop(f"g{k}", T0 + timedelta(seconds=t), APP_A, box=(2060.0, 877.0, 2274.0, 1227.0), zone="patio"))
        k += 1
        t += 100 if k % 8 == 0 else 20
    res = _batch(crops, [], new_id=iter(f"id{k}" for k in range(1000)).__next__)
    assert len(res.sightings) >= 8
    assert all(s.stationary for s in res.sightings.values())
    (iid,) = res.became_stationary
    assert res.became_stationary[iid] == T0


def test_already_stationary_individual_is_not_reported_as_flipping() -> None:
    res = _batch(_skeleton_crops(), [], new_id=iter(f"id{k}" for k in range(1000)).__next__)
    (ind,) = res.individuals.values()
    later = _crop("x1", T0 + timedelta(hours=2), APP_A, box=(2062.0, 877.0, 2273.0, 1229.0), zone="patio")
    assert _batch([later], [ind]).became_stationary == {}


def test_db_pass_reflags_older_sightings_of_the_run() -> None:
    import inspect

    src = inspect.getsource(vi._one_stream)
    assert "batch.became_stationary" in src
    assert "WHERE individual_id=:i AND ended_at >= :since AND NOT stationary" in " ".join(src.split())

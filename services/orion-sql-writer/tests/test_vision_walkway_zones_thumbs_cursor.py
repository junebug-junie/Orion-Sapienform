"""Walkway review fixes: no-embed zones by flag (not by name), thumbnails for
the ask card, and a landing-time cursor that never skips a late crop.

No Postgres: the pure functions directly, and ``_one_stream`` against a fake
connection that evaluates the crop query's time filter itself.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from app import vision_individuals as vi
from app.vision_crop_persist import build_crop_rows
from orion.schemas.vision import VisionCropObservationV1
from orion.vision.zones import Zone, load_zones

TZ = ZoneInfo("America/Denver")
T0 = datetime(2026, 9, 24, 14, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[3]
PATIO = Zone(name="patio", polygon=((0.0, 0.7), (0.35, 0.7), (0.35, 1.0), (0.0, 1.0)), embed=False)
GARDEN = Zone(name="garden", polygon=((0.65, 0.7), (1.0, 0.7), (1.0, 1.0), (0.65, 1.0)), embed=False)
WALK = Zone(name="walkway", polygon=((0.0, 0.35), (1.0, 0.35), (1.0, 1.0), (0.0, 1.0)), embed=True)
THUMB = "thumb:" + "ab" * 32


def _obs(crops):
    return VisionCropObservationV1(
        observation_id="obs-1", stream_id="walkway", observed_at=T0,
        frame_width=1000, frame_height=1000, crops=crops,
    )


def _crop_in(zone_center_x: float) -> dict:
    # bottom-center at (zone_center_x, 0.9) of a 1000x1000 frame
    x = zone_center_x * 1000
    return {"label": "person", "score": 0.9, "box_xyxy": [x - 20, 700, x + 20, 900],
            "embedding": [1.0, 0.0], "embedding_ref": "crop:e:1", "thumb_ref": THUMB}


# --- finding 6: the flag, not the name ---------------------------------------


def test_every_no_embed_zone_in_the_shipped_config_is_flagged_and_stripped() -> None:
    zones = load_zones(REPO / "config" / "vision_zones.yaml")
    no_embed = [(sid, z) for sid, zs in zones.items() for z in zs if not z.embed]
    assert no_embed, "shipped config has no embed:false zone -- the patio rule has nothing to guard"
    for stream_id, zone in no_embed:
        xs = [p[0] for p in zone.polygon]
        ys = [p[1] for p in zone.polygon]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        obs = VisionCropObservationV1(
            observation_id="o", stream_id=stream_id, observed_at=T0, frame_width=1000, frame_height=1000,
            crops=[{"label": "person", "score": 0.9,
                    "box_xyxy": [cx * 1000 - 5, cy * 1000 - 50, cx * 1000 + 5, cy * 1000],
                    "embedding": [1.0], "embedding_ref": "crop:x", "thumb_ref": THUMB}],
        )
        (row,) = build_crop_rows(obs, zones[stream_id])
        assert row["zone"] == zone.name
        assert row["zone_no_embed"] is True
        assert row["embedding"] is None and row["embedding_ref"] is None and row["thumb_ref"] is None


def test_a_second_no_embed_zone_is_covered_without_code_changes() -> None:
    zones = [PATIO, GARDEN, WALK]
    rows = build_crop_rows(_obs([_crop_in(0.8), _crop_in(0.5), _crop_in(0.15)]), zones)
    by_zone = {r["zone"]: r for r in rows}
    assert by_zone["garden"]["zone_no_embed"] is True and by_zone["garden"]["thumb_ref"] is None
    assert by_zone["patio"]["zone_no_embed"] is True and by_zone["patio"]["embedding"] is None
    assert by_zone["walkway"]["zone_no_embed"] is False
    assert by_zone["walkway"]["thumb_ref"] == THUMB and by_zone["walkway"]["embedding"] == [1.0, 0.0]


def test_reducer_treats_a_flagged_row_as_no_embed_whatever_its_name() -> None:
    crop = vi.CropRow(crop_id="c1", observation_id="o1", stream_id="walkway", observed_at=T0, label="person",
                      box_xyxy=(0, 0, 1, 1), zone="garden", embedding=(1.0, 0.0), embedding_ref="e",
                      artifact_id="a", zone_no_embed=True)
    r = vi.apply_batch([crop], individuals={}, latest_sightings={}, no_embed_zones=set(),
                       match_threshold=0.8, merge_gap_sec=60, tz=TZ)
    assert r.individuals == {} and [c.crop_id for c in r.patio_crops] == ["c1"]


def test_presence_row_id_comes_from_the_zone_name() -> None:
    assert vi.no_embed_presence_id("walkway", "patio") == "walkway:patio"
    assert vi.no_embed_presence_id("walkway", "garden") == "walkway:garden"
    snap = vi.patio_snapshot(prev=None, batch_last_seen_at=T0, batch_count=2, now=T0,
                             present_sec=120, grace_sec=600, zone="garden")
    assert snap["zone"] == "garden" and snap["no_embed_zone"] is True and snap["subject"] == {"count": 2}


def test_sighting_row_in_a_no_embed_zone_carries_no_refs() -> None:
    s = vi.Sighting(sighting_id="s", individual_id="i", stream_id="walkway", started_at=T0, ended_at=T0,
                    zone_counts={"garden": 3}, embedding_ref="e", thumb_ref=THUMB)
    row = vi.sighting_row(s, {"garden"})
    assert row["ne"] is True and row["er"] is None and row["th"] is None
    ok = vi.sighting_row(vi.Sighting(sighting_id="s", individual_id="i", stream_id="walkway", started_at=T0,
                                     ended_at=T0, zone_counts={"walkway": 1}, thumb_ref=THUMB), {"garden"})
    assert ok["ne"] is False and ok["th"] == THUMB


def test_no_patio_name_left_in_the_checks_or_readers() -> None:
    sql = (REPO / "services/orion-sql-db/manual_migration_walkway_camera_v1.sql").read_text()
    assert "'patio'" not in sql
    assert "NOT zone_no_embed OR (embedding IS NULL AND embedding_ref IS NULL AND thumb_ref IS NULL)" in sql
    assert "NOT zone_no_embed OR (embedding_ref IS NULL AND thumb_ref IS NULL)" in sql
    reader = (REPO / "orion/situational/perception_reader.py").read_text()
    assert "'patio'" not in reader and ":patio" not in reader


# --- finding 2: the ask's picture is a thumbnail -----------------------------


def test_ask_image_ref_is_the_most_recent_thumb() -> None:
    now = T0
    older = "thumb:" + "11" * 32
    ref = vi.ask_image_ref(
        [(older, now - timedelta(days=2), False), (THUMB, now - timedelta(hours=3), False),
         ("crop:cropobs:art:0", now, False), (None, now, False)],
        now=now, thumb_retention_days=14, ask_expiry_days=7)
    assert ref == THUMB


def test_ask_image_ref_skips_thumbs_that_would_be_pruned_before_the_ask_expires() -> None:
    now = T0
    assert vi.ask_image_ref([(THUMB, now - timedelta(days=8), False)], now=now,
                            thumb_retention_days=14, ask_expiry_days=7) is None
    assert vi.ask_image_ref([(THUMB, now - timedelta(days=6), False)], now=now,
                            thumb_retention_days=14, ask_expiry_days=7) == THUMB


def test_ask_image_ref_never_uses_a_no_embed_sighting() -> None:
    assert vi.ask_image_ref([(THUMB, T0, True)], now=T0, thumb_retention_days=14, ask_expiry_days=7) is None


def test_sighting_keeps_the_newest_crops_thumb() -> None:
    ids = itertools.count()
    mk = lambda i, t, th: vi.CropRow(  # noqa: E731
        crop_id=f"c{i}", observation_id=f"o{i}", stream_id="walkway", observed_at=t, label="person",
        box_xyxy=(0, 0, 1, 1), zone="walkway", embedding=(1.0, 0.0), embedding_ref=f"e{i}",
        artifact_id="a", thumb_ref=th)
    t1 = "thumb:" + "22" * 32
    r = vi.apply_batch([mk(1, T0, t1), mk(2, T0 + timedelta(seconds=5), THUMB),
                        mk(3, T0 + timedelta(seconds=9), None)],
                       individuals={}, latest_sightings={}, no_embed_zones={"patio"},
                       match_threshold=0.8, merge_gap_sec=60, tz=TZ, new_id=lambda: f"id{next(ids)}")
    (s,) = r.sightings.values()
    assert s.thumb_ref == THUMB


# --- finding 5: a late crop is not skipped -----------------------------------


def test_late_crop_older_than_latest_sighting_is_its_own_earlier_sighting() -> None:
    ind = vi.Individual(individual_id="i1", stream_id="walkway", kind="person", centroid=[1.0, 0.0],
                        centroid_n=5, first_seen_at=T0 - timedelta(days=3), last_seen_at=T0)
    latest = vi.Sighting(sighting_id="s-new", individual_id="i1", stream_id="walkway",
                         started_at=T0 - timedelta(minutes=1), ended_at=T0, zone_counts={"walkway": 4})
    late = vi.CropRow(crop_id="late", observation_id="ol", stream_id="walkway",
                      observed_at=T0 - timedelta(hours=2), label="person", box_xyxy=(0, 0, 1, 1),
                      zone="walkway", embedding=(1.0, 0.0), embedding_ref="e", artifact_id="a")
    r = vi.apply_batch([late], individuals={"i1": ind}, latest_sightings={"i1": latest},
                       no_embed_zones=set(), match_threshold=0.8, merge_gap_sec=60, tz=TZ,
                       new_id=lambda: "s-old")
    assert set(r.sightings) == {"s-old"}
    assert r.sightings["s-old"].started_at == T0 - timedelta(hours=2)
    # The newer sighting was not stretched back two hours.
    assert latest.started_at == T0 - timedelta(minutes=1)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    @property
    def rowcount(self):
        return len(self._rows)


class _Conn:
    """Answers _one_stream's reads; applies the crop query's time filter the
    way Postgres would, keyed on whichever column the SQL names."""

    def __init__(self, crops, cursor_at):
        self.crops = crops
        self.cursor_at = cursor_at
        self.cursor_writes = []

    def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        params = params or {}
        if sql.startswith("SELECT last_created_at FROM vision_individuals_cursor"):
            return _Result([(self.cursor_at,)])
        if "FROM vision_crop_observation" in sql:
            assert "created_at > :since AND created_at <= :upto" in sql, sql
            rows = [r for r in self.crops if params["since"] < r.created_at <= params["upto"]]
            return _Result(sorted(rows, key=lambda r: (r.created_at, r.crop_id))[: params["lim"]])
        if sql.startswith("INSERT INTO vision_individuals_cursor"):
            self.cursor_writes.append(params["t"])
            return _Result([])
        return _Result([])

    def begin_nested(self):
        return _Tx()


class _Tx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Engine:
    def __init__(self, conn):
        self.conn = conn

    def begin(self):
        conn = self.conn

        class _Ctx:
            def __enter__(self_inner):
                return conn

            def __exit__(self_inner, *a):
                return False

        return _Ctx()


def _row(crop_id, observed_at, created_at):
    return SimpleNamespace(
        crop_id=crop_id, observation_id=f"o-{crop_id}", stream_id="walkway", observed_at=observed_at,
        label="person", box_xyxy=[0, 0, 1, 1], zone="walkway", embedding=[1.0, 0.0], embedding_ref="e",
        artifact_id="a", zone_no_embed=False, thumb_ref=None, created_at=created_at)


def test_row_with_old_observed_at_but_new_created_at_is_processed(monkeypatch) -> None:
    now = T0
    cursor = now - timedelta(minutes=2)
    # Observed an hour ago (behind the cursor), landed a minute ago (after it).
    late = _row("late", observed_at=now - timedelta(hours=1), created_at=now - timedelta(minutes=1))
    already = _row("done", observed_at=now - timedelta(minutes=5), created_at=now - timedelta(minutes=3))
    conn = _Conn([late, already], cursor)
    seen = []
    monkeypatch.setattr(vi, "_upsert_individual", lambda c, ind: None)
    monkeypatch.setattr(vi, "_upsert_sighting", lambda c, s, z=(): seen.append(s))
    monkeypatch.setattr(vi, "_score_and_maybe_event", lambda *a, **k: False)
    summary = {k: 0 for k in ("crops_read", "assigned", "new_individuals", "sightings_touched",
                              "attention_events", "patio_writes")}
    vi._one_stream(_Engine(conn), "walkway", now, TZ, vi.IndividualsConfig(settle_sec=30), [WALK], summary)
    assert summary["crops_read"] == 1 and summary["assigned"] == 1
    (s,) = seen
    assert s.started_at == now - timedelta(hours=1)
    # Cursor advances on landing time: now - settle.
    assert conn.cursor_writes == [now - timedelta(seconds=30)]

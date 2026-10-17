"""Walkway story pictures of the moment (app/vision_story_strip.py, 2026-10-17).

Pure file/image steps on synthetic JPEGs first, then the whole loop against a
REAL Postgres (``ORION_TEST_PG_URI``, a throwaway database: it TRUNCATEs the
walkway tables), with the v1 + strip v2 migrations applied:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55499:5432 postgres:16-alpine
    for f in walkway_camera_v1 walkway_privacy_v2 vision_confirm_asks_v1 vision_ask_names_v1 \\
             vision_split_v1 vision_motion_story_v1 vision_motion_story_strip_v2; do
      docker exec -i pg psql -U postgres < services/orion-sql-db/manual_migration_$f.sql; done
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55499/postgres pytest tests/test_vision_story_strip.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from PIL import Image, ImageDraw  # noqa: E402

from app import vision_motion as vm  # noqa: E402
from app import vision_motion_loop as vl  # noqa: E402
from app import vision_story_strip as ss  # noqa: E402
from orion.vision.zones import load_zones  # noqa: E402

T0 = datetime(2026, 10, 16, 20, 0, tzinfo=timezone.utc)
ZONES = load_zones(REPO / "config" / "vision_zones.yaml")["walkway"]
MIGRATION_V2 = REPO / "services" / "orion-sql-db" / "manual_migration_vision_motion_story_strip_v2.sql"


def frame_name(t: datetime, stream: str = "walkway", cam: str = "walkway") -> str:
    return f"{cam}_{stream}_{int(t.timestamp() * 1000)}_deadbeef.jpg"


def write_frame(d: Path, t: datetime, *, level: int = 160, size=(640, 480), **kw) -> Path:
    """A small synthetic daylight frame (the loop scales boxes from 2560x1920 to the real size):
    vertical stripes level +- 70, i.e. detail a person can see (RMS contrast ~0.27)."""
    im = Image.new("RGB", size, (level, level, level))
    draw = ImageDraw.Draw(im)
    step = max(4, size[0] // 80)
    for x in range(0, size[0], 2 * step):
        draw.rectangle((x, 0, x + step - 1, size[1]), fill=(level - 70,) * 3)
        draw.rectangle((x + step, 0, x + 2 * step - 1, size[1]), fill=(level + 70,) * 3)
    draw.rectangle((0, 0, 40, 40), fill=(255, 255, 255))   # a bright "porch light"
    p = d / frame_name(t, **kw)
    im.save(p, "JPEG", quality=90)
    return p


def pc(cid: str, sec: float, x: float = 600.0, *, label: str = "person") -> ss.PendingCrop:
    return ss.PendingCrop(cid, T0 + timedelta(seconds=sec), label, (x - 40, 1150.0, x + 40, 1350.0), "walkway")


def cfg_for(tmp: Path, **kw) -> ss.StripConfig:
    (tmp / "frames").mkdir(exist_ok=True)
    return ss.StripConfig(frames_dir=str(tmp / "frames"), strip_dir=str(tmp / "strips"), **kw)


# --- pure ------------------------------------------------------------------------------------------------


def test_frame_is_found_by_name_within_two_ms_and_never_from_another_stream(tmp_path) -> None:
    cfg = cfg_for(tmp_path)
    t = T0 + timedelta(microseconds=123456)
    write_frame(Path(cfg.frames_dir), t)
    write_frame(Path(cfg.frames_dir), t, stream="cam0", cam="cam0")
    idx = ss.FrameIndex(cfg.frames_dir)
    assert idx.find("walkway", "walkway", t).endswith(frame_name(t))
    assert idx.find("walkway", "walkway", t + timedelta(milliseconds=2)) is not None
    assert idx.find("walkway", "walkway", t + timedelta(milliseconds=5)) is None
    assert idx.find(None, "walkway", t) is not None          # camera id falls back to the stream
    assert idx.find("walkway", "patio", t) is None


def test_capture_saves_a_256px_crop_and_a_640px_frame_or_says_the_frame_is_gone(tmp_path) -> None:
    cfg = cfg_for(tmp_path)
    write_frame(Path(cfg.frames_dir), T0, size=(2560, 1920))
    counts = ss.capture_pending([pc("a", 0), pc("b", 0, x=900), pc("gone", 5)], stream_id="walkway", cfg=cfg)
    assert counts == {"saved": 2, "frame_gone": 1, "already": 0, "failed": 0}
    pending = Path(cfg.strip_dir) / ss.PENDING
    crop = Image.open(pending / f"{ss.crop_key('a')}.jpg")
    assert crop.height == ss.CROP_H and crop.width < crop.height          # a person: taller than wide
    frame = Image.open(pending / f"{ss.frame_key('walkway', T0)}.jpg")
    assert frame.size == (640, 480)
    assert json.loads((pending / f"{ss.crop_key('gone')}.miss").read_text())["reason"] == "frame_gone"
    # A second pass does not redo anything.
    again = ss.capture_pending([pc("a", 0), pc("gone", 5)], stream_id="walkway", cfg=cfg)
    assert again["already"] == 2 and again["saved"] == 0


def test_spread_keeps_first_and_last() -> None:
    assert ss.spread(3) == [0, 1, 2]
    assert ss.spread(6) == [0, 2, 3, 5]
    assert ss.spread(10) == [0, 3, 6, 9]
    assert ss.spread(0) == []


def _evidence(crops):
    return [{"crop_id": c.crop_id, "t": c.t.isoformat(), "box": list(c.box)} for c in crops]


def test_assemble_writes_up_to_four_crops_one_boxed_frame_and_marks_what_is_missing(tmp_path) -> None:
    cfg = cfg_for(tmp_path)
    crops = [pc(f"w{i}", i * 2.5, x=600 + 160 * i) for i in range(6)]
    for c in crops[:4]:
        write_frame(Path(cfg.frames_dir), c.t)
    ss.capture_pending(crops[:5], stream_id="walkway", cfg=cfg)     # w4: frame gone; w5: never looked for
    sid = "a" * 32
    rec = ss.assemble_strip(sid, "walkway", _evidence(crops), cfg=cfg)
    out = Path(cfg.strip_dir) / sid
    assert [c["crop_id"] for c in rec["crops"]] == ["w0", "w2", "w3", "w5"]
    assert [c.get("file") or c["missing"] for c in rec["crops"]] == ["crop_0.jpg", "crop_1.jpg", "crop_2.jpg",
                                                                      "not_captured"]
    assert (out / "crop_0.jpg").is_file() and not (out / "crop_3.jpg").exists()
    assert rec["frame"]["file"] == "frame.jpg" and rec["frame"]["crop_id"] in {"w2", "w3"}  # nearest the middle
    assert Image.open(out / "frame.jpg").size == (640, 480)
    # Measured on the saved crops: striped daylight detail is high-contrast, so not dark.
    assert rec["scene_measured_on"] == "crops" and rec["scene_contrast"] > 0.2 and rec["dark"] is False
    assert 0.3 < rec["scene_luminance"] < 0.9 and 0.3 < rec["frame_luminance"] < 0.9
    # With a w4 marker present but w5 absent, the reasons differ.
    rec2 = ss.assemble_strip("b" * 32, "walkway", _evidence(crops[3:5]), cfg=cfg)
    assert [c.get("file") or c["missing"] for c in rec2["crops"]] == ["crop_0.jpg", "frame_gone"]


def test_a_night_street_is_dark_even_with_a_bright_porch_light(tmp_path) -> None:
    cfg = cfg_for(tmp_path, dark_contrast=0.10)
    c = pc("n0", 0)
    im = Image.new("RGB", (640, 480), (3, 3, 3))
    # Lit porch and pillars in the foreground; the passage (x 140-160, y 287-337 at 640 px) is the dark street.
    ImageDraw.Draw(im).rectangle((300, 0, 640, 480), fill=(240, 240, 240))
    ImageDraw.Draw(im).rectangle((0, 400, 640, 480), fill=(240, 240, 240))
    im.save(Path(cfg.frames_dir) / frame_name(c.t), "JPEG", quality=90)
    ss.capture_pending([c], stream_id="walkway", cfg=cfg)
    rec = ss.assemble_strip("c" * 32, "walkway", _evidence([c]), cfg=cfg)
    assert rec["frame_luminance"] > 0.3          # the whole-frame mean would call this "not dark"
    assert rec["scene_contrast"] < 0.10 and rec["dark"] is True


def test_a_lit_lawn_at_night_is_dark_although_it_is_as_bright_as_a_dark_coat_by_day(tmp_path) -> None:
    """Live 2026-10-17: a porch-lit lawn crop with nothing visible had median luma 0.11-0.125, the
    same as a dark-clothed person at 13:33 (0.114). Brightness cannot tell them apart; contrast can."""
    cfg = cfg_for(tmp_path, dark_contrast=0.10)
    c = pc("lawn", 0)
    Image.new("RGB", (640, 480), (24, 34, 24)).save(Path(cfg.frames_dir) / frame_name(c.t), "JPEG", quality=90)
    ss.capture_pending([c], stream_id="walkway", cfg=cfg)
    rec = ss.assemble_strip("9" * 32, "walkway", _evidence([c]), cfg=cfg)
    assert rec["scene_luminance"] > 0.1 and rec["dark"] is True


def test_with_no_saved_crop_the_box_areas_of_the_frame_are_measured(tmp_path) -> None:
    cfg = cfg_for(tmp_path)
    c = pc("f0", 0)
    write_frame(Path(cfg.frames_dir), c.t)
    ss.capture_pending([c], stream_id="walkway", cfg=cfg)
    (Path(cfg.strip_dir) / ss.PENDING / f"{ss.crop_key('f0')}.jpg").unlink()
    rec = ss.assemble_strip("8" * 32, "walkway", _evidence([c]), cfg=cfg)
    assert rec["frame"] is not None and rec["crops"][0]["missing"] == "not_captured"
    assert rec["scene_measured_on"] == "frame" and rec["dark"] is False


def test_no_frame_means_no_luminance_and_dark_unknown(tmp_path) -> None:
    cfg = cfg_for(tmp_path)
    rec = ss.assemble_strip("d" * 32, "walkway", _evidence([pc("x", 0)]), cfg=cfg)
    assert rec["frame"] is None and rec["dark"] is None and rec["scene_contrast"] is None
    assert rec["crops"] == [{"crop_id": "x", "t": pc("x", 0).t.isoformat(), "missing": "not_captured"}]


def test_bad_story_ids_and_links_are_never_written_or_deleted(tmp_path) -> None:
    cfg = cfg_for(tmp_path)
    with pytest.raises(ValueError):
        ss.assemble_strip("../etc", "walkway", [], cfg=cfg)
    root = Path(cfg.strip_dir)
    root.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep").write_text("x")
    (root / ("e" * 32)).symlink_to(victim, target_is_directory=True)
    assert ss.remove_strip_dir(cfg, "e" * 32) is False and (victim / "keep").exists()
    assert ss.remove_strip_dir(cfg, "..") is False
    (root / ("f" * 32)).mkdir()
    assert ss.strip_dirs(cfg) == ["f" * 32]
    assert ss.remove_strip_dir(cfg, "f" * 32) is True and not (root / ("f" * 32)).exists()


def test_pending_files_older_than_the_window_are_pruned(tmp_path) -> None:
    cfg = cfg_for(tmp_path, pending_sec=60)
    pending = Path(cfg.strip_dir) / ss.PENDING
    pending.mkdir(parents=True)
    old, new = pending / "c_old.jpg", pending / "c_new.jpg"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    os.utime(old, (time.time() - 120, time.time() - 120))
    assert ss.prune_pending(cfg) == 1 and new.exists() and not old.exists()


def test_config_parity_strip_keys_on_by_default() -> None:
    from app.settings import Settings

    env = (SERVICE / ".env_example").read_text()
    compose = (SERVICE / "docker-compose.yml").read_text()
    for key, field in [("VISION_STORY_STRIPS_ENABLED", "vision_story_strips_enabled"),
                       ("VISION_STORY_FRAMES_DIR", "vision_story_frames_dir"),
                       ("VISION_STORY_STRIP_DIR", "vision_story_strip_dir"),
                       ("VISION_STORY_STRIP_RETENTION_DAYS", "vision_story_strip_retention_days"),
                       ("VISION_STORY_STRIP_PENDING_SEC", "vision_story_strip_pending_sec"),
                       ("VISION_STORY_STRIP_DARK_CONTRAST", "vision_story_strip_dark_contrast")]:
        assert f"\n{key}=" in env, key
        assert Settings.model_fields[field].alias == key
    assert "VISION_STORY_STRIPS_ENABLED=true" in env
    assert "VISION_STORY_STRIPS_ENABLED=${VISION_STORY_STRIPS_ENABLED:-true}" in compose
    assert "/mnt/telemetry/vision/frames}:ro" in compose
    assert "Pillow==" in (SERVICE / "requirements.txt").read_text()


def test_strip_migration_is_auto_apply_safe() -> None:
    from orion.sql_migration_drift import auto_apply_refusals

    text = MIGRATION_V2.read_text()
    assert auto_apply_refusals(text) == []
    assert auto_apply_refusals(text, {"vision_motion_story", "vision_motion_story_verdict"}) == []


# --- the whole loop against real Postgres -----------------------------------------------------------------

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE vision_motion_story, vision_motion_story_verdict, vision_motion_story_cursor, "
                          "vision_crop_observation, vision_individual_sighting, vision_individual CASCADE"))
    vl._capture_cursor.clear()
    vl._last_orphan_sweep[0] = None
    yield engine
    engine.dispose()


def _insert(engine, crops) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        for c in crops:
            conn.execute(text(
                "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, camera_id, observed_at, "
                "label, score, box_xyxy, zone, created_at) VALUES (:c, :o, 'walkway', 'walkway', :t, :l, 0.5, :b, "
                "'walkway', :ct)"),
                {"c": c.crop_id, "o": "obs-" + c.crop_id, "t": c.t, "l": c.label, "b": list(c.box),
                 "ct": c.t + timedelta(seconds=2)})


def _loop_cfg(tmp: Path) -> vl.LoopConfig:
    return vl.LoopConfig(initial_lookback_sec=600, clean_since=T0 - timedelta(hours=1), strip=cfg_for(tmp))


def _cycle(cfg, now):
    return vl.run_one_motion_cycle(postgres_uri=PG, cfg=cfg, zones_by_stream={"walkway": ZONES}, now=now)


@needs_pg
def test_loop_captures_while_frames_exist_then_writes_the_strip_on_the_story(pg, tmp_path) -> None:
    from sqlalchemy import text

    cfg = _loop_cfg(tmp_path)
    frames = Path(cfg.strip.frames_dir)
    walk = [pc(f"w{i}", i * 2.5, x=600 + 160 * i) for i in range(5)]
    for c in walk:
        write_frame(frames, c.t)
    _insert(pg, walk)

    first = _cycle(cfg, T0 + timedelta(seconds=15))          # the walk is still open: no story yet
    assert first["capture_saved"] == 5 and first.get("published", 0) == 0
    for f in frames.iterdir():                               # the edge prunes the frames (~60 s)
        f.unlink()

    second = _cycle(cfg, T0 + timedelta(seconds=200))
    assert second["published"] == 1 and second["strips_written"] == 1
    with pg.connect() as conn:
        row = conn.execute(text(
            "SELECT story_id, evidence_strip, frame_luminance, scene_luminance, scene_contrast, dark "
            "FROM vision_motion_story "
            "WHERE published")).mappings().one()
    strip = row["evidence_strip"]
    assert [c.get("file") for c in strip["crops"]] == ["crop_0.jpg", "crop_1.jpg", "crop_2.jpg", "crop_3.jpg"]
    assert strip["frame"]["file"] == "frame.jpg" and row["dark"] is False
    assert row["scene_luminance"] == pytest.approx(strip["scene_luminance"], abs=1e-4)
    assert row["scene_contrast"] == pytest.approx(strip["scene_contrast"], abs=1e-4)
    assert (Path(cfg.strip.strip_dir) / row["story_id"] / "frame.jpg").is_file()
    # Nothing is redone on the next tick.
    assert _cycle(cfg, T0 + timedelta(seconds=215)).get("strips_written", 0) == 0


@needs_pg
def test_loop_records_a_story_whose_frames_were_already_gone(pg, tmp_path) -> None:
    from sqlalchemy import text

    cfg = _loop_cfg(tmp_path)
    walk = [pc(f"g{i}", i * 2.5, x=600 + 160 * i) for i in range(4)]
    _insert(pg, walk)                                          # no frames on disk at all
    _cycle(cfg, T0 + timedelta(seconds=15))
    _cycle(cfg, T0 + timedelta(seconds=200))
    with pg.connect() as conn:
        row = conn.execute(text("SELECT evidence_strip, dark FROM vision_motion_story WHERE published")).one()
    assert row[0]["frame"] is None and row[1] is None
    assert {c["missing"] for c in row[0]["crops"]} == {"frame_gone"}


@needs_pg
def test_strips_are_kept_until_a_verdict_and_thirty_days(pg, tmp_path) -> None:
    from sqlalchemy import text

    cfg = _loop_cfg(tmp_path)
    walk = [pc(f"r{i}", i * 2.5, x=600 + 160 * i) for i in range(4)]
    for c in walk:
        write_frame(Path(cfg.strip.frames_dir), c.t)
    _insert(pg, walk)
    _cycle(cfg, T0 + timedelta(seconds=15))
    _cycle(cfg, T0 + timedelta(seconds=200))
    with pg.connect() as conn:
        sid = conn.execute(text("SELECT story_id FROM vision_motion_story WHERE published")).scalar()
    d = Path(cfg.strip.strip_dir) / sid
    later = T0 + timedelta(days=31)
    _cycle(cfg, later)                                         # 31 days, no verdict: kept
    assert d.is_dir()
    with pg.begin() as conn:
        conn.execute(text("INSERT INTO vision_motion_story_verdict (story_id, verdict, rules_version) "
                          "VALUES (:s, 'right', 1)"), {"s": sid})
    s = _cycle(cfg, later + timedelta(seconds=15))
    assert s["strips_pruned"] == 1 and not d.exists()
    with pg.connect() as conn:
        strip = conn.execute(text("SELECT evidence_strip FROM vision_motion_story WHERE story_id = :s"),
                             {"s": sid}).scalar()
    assert "pruned_at" in strip and strip["frame"]["file"] == "frame.jpg"


@needs_pg
def test_orphan_strip_dirs_are_swept_but_young_ones_are_not(pg, tmp_path) -> None:
    cfg = _loop_cfg(tmp_path)
    root = Path(cfg.strip.strip_dir)
    old, young = root / ("1" * 32), root / ("2" * 32)
    old.mkdir(parents=True)
    young.mkdir()
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    s = _cycle(cfg, T0 + timedelta(seconds=15))
    assert s["strips_pruned"] == 1 and not old.exists() and young.exists()


@needs_pg
def test_the_first_orphan_sweep_runs_even_on_a_freshly_booted_host(pg, tmp_path, monkeypatch) -> None:
    """CI 2026-10-17: a runner up < 1 h has time.monotonic() < 3600, and a 0.0 'never swept'
    sentinel skipped the first sweep. Pin the clock low to reproduce that host."""
    cfg = _loop_cfg(tmp_path)
    old = Path(cfg.strip.strip_dir) / ("3" * 32)
    old.mkdir(parents=True)
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    monkeypatch.setattr(vl.time, "monotonic", lambda: 120.0)
    assert _cycle(cfg, T0 + timedelta(seconds=15))["strips_pruned"] == 1 and not old.exists()
    # ...and the next one waits the full interval.
    again = Path(cfg.strip.strip_dir) / ("4" * 32)
    again.mkdir()
    os.utime(again, (time.time() - 7200, time.time() - 7200))
    assert _cycle(cfg, T0 + timedelta(seconds=30))["strips_pruned"] == 0 and again.exists()


@needs_pg
def test_without_the_strip_migration_stories_still_run(pg, tmp_path) -> None:
    from sqlalchemy import text

    cfg = _loop_cfg(tmp_path)
    _insert(pg, [pc(f"m{i}", i * 2.5, x=600 + 160 * i) for i in range(4)])
    with pg.begin() as conn:
        conn.execute(text("ALTER TABLE vision_motion_story RENAME COLUMN evidence_strip TO evidence_strip_x"))
    try:
        s = _cycle(cfg, T0 + timedelta(seconds=200))
        assert s["published"] == 1 and s["strips"].startswith("migration missing: evidence_strip")
        assert "capture_saved" not in s
    finally:
        with pg.begin() as conn:
            conn.execute(text("ALTER TABLE vision_motion_story RENAME COLUMN evidence_strip_x TO evidence_strip"))


@needs_pg
def test_prop_boxes_are_not_captured_and_say_why(pg, tmp_path) -> None:
    """A fresh crop has no individual_id yet: the patio skeleton must be matched by its box."""
    from sqlalchemy import text

    cfg = _loop_cfg(tmp_path)
    with pg.begin() as conn:
        conn.execute(text(
            "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, first_seen_at, last_seen_at, "
            "stationary, still_box) VALUES ('skel', 'walkway', 'person', ARRAY[1.0]::real[], :t, :t, true, "
            "ARRAY[2136, 878, 2216, 1149]::real[])"), {"t": T0})
    skel = ss.PendingCrop("k0", T0, "person", (2136.0, 878.0, 2216.0, 1149.0), "walkway")
    walker = pc("w0", 0)
    for c in (skel, walker):
        write_frame(Path(cfg.strip.frames_dir), c.t)
    _insert(pg, [skel, walker])
    s = _cycle(cfg, T0 + timedelta(seconds=15))
    assert s["capture_prop"] == 1 and s["capture_saved"] == 1
    pending = Path(cfg.strip.strip_dir) / ss.PENDING
    assert json.loads((pending / f"{ss.crop_key('k0')}.miss").read_text())["reason"] == "prop_overlap"
    rec = ss.assemble_strip("7" * 32, "walkway", _evidence([skel]), cfg=cfg.strip)
    assert rec["crops"][0]["missing"] == "prop_overlap"


@needs_pg
def test_a_crop_committed_late_behind_the_cursor_is_still_captured(pg, tmp_path) -> None:
    cfg = _loop_cfg(tmp_path)
    a, late = pc("a0", 0), pc("late", 1, x=900)
    for c in (a, late):
        write_frame(Path(cfg.strip.frames_dir), c.t)
    _insert(pg, [a])
    assert _cycle(cfg, T0 + timedelta(seconds=15))["capture_saved"] == 1
    # created_at is the transaction START: this row sorts before the cursor but commits after the read.
    _insert(pg, [ss.PendingCrop("late", a.t - timedelta(seconds=1), "person", late.box, "walkway")])
    write_frame(Path(cfg.strip.frames_dir), a.t - timedelta(seconds=1))
    s = _cycle(cfg, T0 + timedelta(seconds=30))
    assert s["capture_saved"] == 1 and s["capture_already"] >= 1


@needs_pg
def test_a_strip_that_keeps_failing_gets_a_final_error_record(pg, tmp_path, monkeypatch) -> None:
    from sqlalchemy import text

    cfg = _loop_cfg(tmp_path)
    _insert(pg, [pc(f"x{i}", i * 2.5, x=600 + 160 * i) for i in range(4)])

    def boom(*a, **k):
        raise OSError("disk went away")

    monkeypatch.setattr(ss, "assemble_strip", boom)
    real_now = datetime.now(timezone.utc)
    _cycle(cfg, T0 + timedelta(seconds=200))
    with pg.connect() as conn:
        assert conn.execute(text("SELECT evidence_strip FROM vision_motion_story WHERE published")).scalar() is None
        # Pretend the story was written long ago (created_at is the DB clock).
        conn.execute(text("UPDATE vision_motion_story SET created_at = :t"),
                     {"t": real_now - timedelta(seconds=cfg.strip.pending_sec * 0.75)})
        conn.commit()
    vl.assemble_step(create_engine_for_test(), "walkway", real_now, cfg)
    with pg.connect() as conn:
        rec = conn.execute(text("SELECT evidence_strip FROM vision_motion_story WHERE published")).scalar()
    assert rec["error"] == "disk went away" and rec["frame"] is None and rec["dark"] is None


def create_engine_for_test():
    from sqlalchemy import create_engine

    return create_engine(PG)

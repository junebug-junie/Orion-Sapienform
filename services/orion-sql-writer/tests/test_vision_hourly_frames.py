"""Hourly frame samples (app/vision_hourly_frames.py): one frame per camera per
UTC hour, an index line per sample, and pruning of only its own day dirs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from PIL import Image

from app.settings import Settings
from app.vision_hourly_frames import (
    HourlyFramesConfig,
    config_from_settings,
    parse_streams,
    prune,
    run_one_tick,
    sample_path,
    take_samples,
)

NOW = datetime(2026, 10, 22, 18, 30, 0, tzinfo=timezone.utc)


def _frame(d, name, ms, shade=200):
    p = d / f"{name}_{ms}_abcdef12.jpg"
    Image.new("RGB", (64, 48), (shade, shade, shade)).save(p, "JPEG")
    return p


def _cfg(tmp_path, **kw):
    frames = tmp_path / "frames"
    frames.mkdir(exist_ok=True)
    base = dict(frames_dir=str(frames), out_dir=str(tmp_path / "hourly"),
                streams=parse_streams("walkway,cam0=frame,doorbell"))
    base.update(kw)
    return HourlyFramesConfig(**base), frames


def _ms(t):
    return int(t.timestamp() * 1000)


def test_parse_streams():
    assert parse_streams("walkway, cam0=frame ,doorbell,") == (
        ("walkway", "walkway_walkway"), ("cam0", "frame"), ("doorbell", "doorbell_doorbell"))


def test_saves_newest_frame_of_this_hour_once_with_index(tmp_path):
    cfg, frames = _cfg(tmp_path)
    _frame(frames, "walkway_walkway", _ms(NOW - timedelta(minutes=40)))   # last hour: never
    _frame(frames, "walkway_walkway", _ms(NOW - timedelta(minutes=5)), shade=10)
    newest = _frame(frames, "walkway_walkway", _ms(NOW - timedelta(seconds=30)), shade=240)
    _frame(frames, "walkway_walkway", _ms(NOW - timedelta(seconds=1)))    # may be mid-write: skipped
    _frame(frames, "frame", _ms(NOW - timedelta(minutes=1)))               # cam0's edge naming
    seen = []

    def summarize(stream, t):
        seen.append((stream, t))
        return {"windows": 3, "windows_with_detection": 1}

    out = take_samples(cfg, NOW, summarize)
    assert out == {"saved": 2, "already": 0, "no_frame": 1, "failed": 0}  # doorbell: no event this hour
    dest = sample_path(cfg.out_dir, "walkway", NOW)
    assert dest.name == "walkway_18Z.jpg" and dest.parent.name == "2026-10-22"
    assert dest.read_bytes() == newest.read_bytes()  # untouched full-resolution bytes
    lines = [json.loads(x) for x in (dest.parent / "index.jsonl").read_text().splitlines()]
    walk = next(x for x in lines if x["stream"] == "walkway")
    assert walk["source"] == newest.name
    assert walk["detections"] == {"windows": 3, "windows_with_detection": 1}
    assert walk["luma_mean"] > 0.8  # the bright frame, not the dark one
    assert (walk["width"], walk["height"]) == (64, 48)
    assert ("walkway", datetime.fromtimestamp(_ms(NOW - timedelta(seconds=30)) / 1000, tz=timezone.utc)) in seen

    again = take_samples(cfg, NOW + timedelta(minutes=10), summarize)
    assert again["saved"] == 0 and again["already"] == 2
    assert len((dest.parent / "index.jsonl").read_text().splitlines()) == 2


def test_new_hour_gets_a_new_sample(tmp_path):
    cfg, frames = _cfg(tmp_path, streams=parse_streams("walkway"))
    _frame(frames, "walkway_walkway", _ms(NOW - timedelta(minutes=1)))
    take_samples(cfg, NOW, None)
    later = NOW + timedelta(hours=1)
    _frame(frames, "walkway_walkway", _ms(later - timedelta(minutes=1)))
    assert take_samples(cfg, later, None)["saved"] == 1
    assert sample_path(cfg.out_dir, "walkway", later).exists()


def test_summary_failure_keeps_the_picture(tmp_path):
    cfg, frames = _cfg(tmp_path, streams=parse_streams("walkway"))
    _frame(frames, "walkway_walkway", _ms(NOW - timedelta(minutes=1)))

    def boom(stream, t):
        raise RuntimeError("db down")

    assert take_samples(cfg, NOW, boom)["saved"] == 1
    line = json.loads((sample_path(cfg.out_dir, "walkway", NOW).parent / "index.jsonl").read_text())
    assert line["detections"]["error"].startswith("RuntimeError")


def test_corrupt_frame_is_not_kept(tmp_path):
    cfg, frames = _cfg(tmp_path, streams=parse_streams("walkway"))
    (frames / f"walkway_walkway_{_ms(NOW - timedelta(minutes=1))}_abcdef12.jpg").write_bytes(b"not a jpeg")
    assert take_samples(cfg, NOW, None)["failed"] == 1
    dest = sample_path(cfg.out_dir, "walkway", NOW)
    assert not dest.exists() and not list(dest.parent.glob("*.tmp"))


def test_prune_removes_only_old_day_dirs(tmp_path):
    cfg, _ = _cfg(tmp_path, retention_days=14)
    root = tmp_path / "hourly"
    for name in ("2026-10-07", "2026-10-08", "2026-10-21", "notes", "2026-13-99"):
        (root / name).mkdir(parents=True)
    (root / "2026-10-01.txt").write_text("not a dir")
    assert prune(cfg, NOW) == 1
    assert sorted(os.listdir(root)) == ["2026-10-01.txt", "2026-10-08", "2026-10-21", "2026-13-99", "notes"]


def test_prune_zero_keeps_forever(tmp_path):
    cfg, _ = _cfg(tmp_path, retention_days=0)
    (tmp_path / "hourly" / "2020-01-01").mkdir(parents=True)
    assert prune(cfg, NOW) == 0


def test_run_one_tick_reports_prune(tmp_path):
    cfg, _ = _cfg(tmp_path)
    (tmp_path / "hourly" / "2026-09-01").mkdir(parents=True)
    out = run_one_tick(cfg, None, NOW)
    assert out["pruned_days"] == 1 and out["no_frame"] == 3


def test_settings_defaults_ship_on():
    s = Settings()
    assert s.vision_hourly_frames_enabled is True
    assert s.vision_hourly_frames_interval_sec > 0
    cfg = config_from_settings(s)
    assert cfg.retention_days == 14.0
    assert dict(cfg.streams) == {"walkway": "walkway_walkway", "cam0": "frame", "doorbell": "doorbell_doorbell"}

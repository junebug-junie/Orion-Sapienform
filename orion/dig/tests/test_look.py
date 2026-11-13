"""`dig look`: newest camera frame -> sandbox copy Orion can Read. No camera, no network."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.dig import __main__ as dig
from orion.dig import look as lk

NOW = datetime(2026, 11, 13, 21, 0, 0, tzinfo=timezone.utc)
NOW_MS = int(NOW.timestamp() * 1000)


def _frame(d: Path, camera: str, stream: str, ms: int, body: bytes = b"\xff\xd8jpeg") -> Path:
    # The name frame_store.save_frame writes: <camera>_<stream>_<epoch_ms>_<sha8>.jpg
    p = d / f"{camera}_{stream}_{ms}_0a1b2c3d.jpg"
    p.write_bytes(body)
    return p


@pytest.fixture()
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    frames, home = tmp_path / "frames", tmp_path / "home"
    frames.mkdir()
    home.mkdir()
    return frames, home


def test_newest_frame_for_that_stream_only(dirs) -> None:
    frames, _ = dirs
    _frame(frames, "cam0", "cam0", NOW_MS - 900)
    newest = _frame(frames, "cam0", "cam0", NOW_MS - 400)
    _frame(frames, "walkway", "walkway", NOW_MS - 100)  # newer, other camera
    (frames / "cam0_cam0_notes.txt").write_text("x")  # not a frame
    got = lk.newest_frame(frames, "cam0")
    assert got is not None and got.path == newest and got.epoch_ms == NOW_MS - 400


def test_look_copies_logs_and_says_live(dirs) -> None:
    frames, home = dirs
    _frame(frames, "cam0", "cam0", NOW_MS - 500, b"\xff\xd8office")
    lines, rec = lk.look("office", home=home, directory=frames, now=NOW)
    dest = Path(rec["path"])
    assert dest.read_bytes() == b"\xff\xd8office"
    assert dest.parent.parent == home / "look"
    assert rec["camera"] == "cam0" and rec["live"] is True and rec["age_sec"] == 0.5
    text = "\n".join(lines)
    assert "(live)" in text and str(dest) in text and "Read tool" in text
    logged = [json.loads(x) for x in (home / "look" / lk.LOG_NAME).read_text().splitlines()]
    assert logged == [rec]


def test_stale_camera_is_never_called_live(dirs) -> None:
    frames, home = dirs
    _frame(frames, "walkway", "walkway", NOW_MS - 240_000)
    lines, rec = lk.look("walkway", home=home, directory=frames, now=NOW)
    assert rec["live"] is False
    assert "NOT LIVE" in lines[1] and "4 min old" in lines[1]


def test_doorbell_stale_frame_says_event_driven(dirs) -> None:
    frames, home = dirs
    _frame(frames, "doorbell", "doorbell", NOW_MS - 120_000)
    lines, _ = lk.look("door", home=home, directory=frames, now=NOW)
    assert "only records after motion or a ring" in lines[1]


def test_no_frame_is_an_error_not_an_empty_look(dirs) -> None:
    frames, home = dirs
    with pytest.raises(lk.LookError, match="may be down"):
        lk.look("office", home=home, directory=frames, now=NOW)
    assert not (home / "look" / lk.LOG_NAME).exists()


def test_unreadable_frames_folder_names_the_mount(tmp_path: Path) -> None:
    with pytest.raises(lk.LookError, match="mounted read-only"):
        lk.look("office", home=tmp_path, directory=tmp_path / "missing", now=NOW)


def test_unknown_camera_lists_words(dirs) -> None:
    frames, home = dirs
    with pytest.raises(lk.LookError, match="office"):
        lk.look("kitchen", home=home, directory=frames, now=NOW)


def test_hourly_cap(dirs) -> None:
    frames, home = dirs
    _frame(frames, "cam0", "cam0", NOW_MS - 300)
    log = home / "look" / lk.LOG_NAME
    log.parent.mkdir(parents=True)
    old = (NOW - timedelta(hours=2)).isoformat()
    recent = (NOW - timedelta(minutes=10)).isoformat()
    log.write_text("".join(json.dumps({"looked_at": t}) + "\n"
                           for t in [old] * 5 + [recent] * (lk.LOOKS_PER_HOUR - 1)))
    lines, _ = lk.look("office", home=home, directory=frames, now=NOW)  # the 20th this hour
    assert lines[-1] == f"looks in the last hour: {lk.LOOKS_PER_HOUR} of {lk.LOOKS_PER_HOUR}"
    with pytest.raises(lk.LookError, match="cap is"):
        lk.look("office", home=home, directory=frames, now=NOW)


def test_sweep_only_old_day_folders(tmp_path: Path) -> None:
    root = tmp_path / "look"
    (root / "2026-11-01").mkdir(parents=True)
    (root / "2026-11-10").mkdir()
    (root / "keep-me").mkdir()
    (root / lk.LOG_NAME).write_text("")
    assert lk.sweep(root, datetime(2026, 11, 13).date()) == 1
    assert sorted(p.name for p in root.iterdir()) == ["2026-11-10", "keep-me", lk.LOG_NAME]


def test_cameras_listing(dirs) -> None:
    frames, _ = dirs
    _frame(frames, "cam0", "cam0", NOW_MS - 400)
    lines, payload = lk.cameras(directory=frames, now=NOW)
    assert lines[0].startswith("office:") and "(live)" in lines[0]
    assert payload["walkway"] is None and "no frame" in lines[1]


def test_dig_cli_look(dirs, monkeypatch, capsys) -> None:
    frames, home = dirs
    _frame(frames, "walkway", "walkway", int(datetime.now(timezone.utc).timestamp() * 1000) - 200)
    monkeypatch.setenv("ORION_DIG_HOME", str(home))
    monkeypatch.setenv("ORION_DIG_FRAMES_DIR", str(frames))
    assert dig.main(["look", "walkway"]) == 0
    out = capsys.readouterr().out
    assert "look: your outdoor camera (walkway)" in out and "full result:" in out
    assert dig.main(["look", "kitchen"]) == 1
    assert "unknown camera" in capsys.readouterr().err

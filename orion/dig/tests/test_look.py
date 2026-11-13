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


def _frame(d: Path, camera: str, stream: str, ms: int, body: bytes = b"\xff\xd8jpeg\xff\xd9") -> Path:
    # The name frame_store.save_frame writes: <camera>_<stream>_<epoch_ms>_<sha8>.jpg
    p = d / f"{camera}_{stream}_{ms}_0a1b2c3d.jpg"
    p.write_bytes(body)
    return p


@pytest.fixture(autouse=True)
def published(monkeypatch) -> list:
    """Every test's look publishes into this list, never to a real bus."""
    sent: list = []
    monkeypatch.setattr(lk, "_publish_bus", lambda events: sent.extend(events))
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    return sent


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
    _frame(frames, "cam0", "cam0", NOW_MS - 500, b"\xff\xd8office\xff\xd9")
    lines, rec = lk.look("office", home=home, directory=frames, now=NOW)
    dest = Path(rec["path"])
    assert dest.read_bytes() == b"\xff\xd8office\xff\xd9"
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
    with pytest.raises(lk.LookError, match="may be down") as err:
        lk.look("office", home=home, directory=frames, now=NOW)
    assert "act: act_" in str(err.value)  # a look that found the camera down is still an act
    (row,) = [json.loads(x) for x in (home / "look" / lk.LOG_NAME).read_text().splitlines()]
    assert row["live"] is False and "path" not in row and row["act_id"].startswith("act_")


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
    assert f"looks in the last hour: {lk.LOOKS_PER_HOUR} of {lk.LOOKS_PER_HOUR}" in lines
    rows_before = log.read_text()
    with pytest.raises(lk.LookError, match="cap is"):
        lk.look("office", home=home, directory=frames, now=NOW)
    assert log.read_text() == rows_before  # a refused look is no look: no row, no act


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


def test_half_written_newest_frame_falls_back_to_the_one_before(dirs) -> None:
    frames, home = dirs
    whole = _frame(frames, "cam0", "cam0", NOW_MS - 900, b"\xff\xd8whole\xff\xd9")
    _frame(frames, "cam0", "cam0", NOW_MS - 300, b"\xff\xd8cut-off mid-wri")  # cv2 still writing
    _, rec = lk.look("office", home=home, directory=frames, now=NOW)
    assert rec["source"] == str(whole) and Path(rec["path"]).read_bytes() == b"\xff\xd8whole\xff\xd9"


def test_all_newest_frames_cut_off_is_an_error(dirs) -> None:
    frames, home = dirs
    for k in range(3):
        _frame(frames, "cam0", "cam0", NOW_MS - 100 * k, b"\xff\xd8partial")
    with pytest.raises(lk.LookError, match="incomplete"):
        lk.look("office", home=home, directory=frames, now=NOW)
    (row,) = [json.loads(x) for x in (home / "look" / lk.LOG_NAME).read_text().splitlines()]
    assert row["live"] is False and "incomplete" in row["reason"]


def test_malformed_log_rows_do_not_crash_the_cap(tmp_path: Path) -> None:
    log = tmp_path / lk.LOG_NAME
    log.write_text("not json\n" + json.dumps({"looked_at": "2026-11-13T20:59:00"}) + "\n"
                   + json.dumps({"looked_at": None}) + "\n" + json.dumps({"looked_at": NOW.isoformat()}) + "\n")
    assert lk.looks_in_last_hour(log, NOW) == 1


# --- the act ledger (#3320) ---------------------------------------------------------------------

def test_a_live_look_is_one_act_opened_then_closed_1(dirs, published, monkeypatch) -> None:
    frames, home = dirs
    _frame(frames, "walkway", "walkway", NOW_MS - 400)
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Orion-Lease: x\nX-Orion-Correlation-Id: turn-123")
    lines, rec = lk.look("walkway", home=home, directory=frames, now=NOW)
    opened, closed = published
    assert [opened.event_kind, closed.event_kind] == ["act_opened", "act_closed"]
    assert opened.trace_id == closed.trace_id == f"act:{rec['act_id']}"
    act = closed.act
    assert (act.actor, act.verb, act.object_ref) == ("orion.look", "look", "camera:walkway")
    assert act.expect.signal_id == "artifact_produced" and act.outcome.value == 1.0
    assert closed.session_id == "turn-123" and closed.correlation_id is None  # joinable, never billed the turn
    assert act.cost is None  # not measured, never zero
    assert opened.provenance.source_service == "orion-harness-governor"
    assert f"python -P -m orion.dig act {rec['act_id']}" in lines[-1]


def test_a_stale_look_closes_0(dirs, published) -> None:
    frames, home = dirs
    _frame(frames, "walkway", "walkway", NOW_MS - 240_000)
    lk.look("walkway", home=home, directory=frames, now=NOW)
    assert published[1].act.outcome.value == 0.0


def test_two_looks_are_two_acts(dirs, published) -> None:
    frames, home = dirs
    _frame(frames, "cam0", "cam0", NOW_MS - 300)
    _, a = lk.look("office", home=home, directory=frames, now=NOW)
    _, b = lk.look("office", home=home, directory=frames, now=NOW + timedelta(seconds=1))
    assert a["act_id"] != b["act_id"]


def test_a_bus_failure_never_fails_the_look(dirs, monkeypatch) -> None:
    frames, home = dirs
    _frame(frames, "cam0", "cam0", NOW_MS - 300)

    def down(events):
        raise ConnectionError("bus down")

    monkeypatch.setattr(lk, "_publish_bus", down)
    lines, rec = lk.look("office", home=home, directory=frames, now=NOW)
    assert Path(rec["path"]).exists() and "act_id" not in rec
    assert lines[-1] == "act: not recorded (ConnectionError: bus down)"
    (row,) = [json.loads(x) for x in (home / "look" / lk.LOG_NAME).read_text().splitlines()]
    assert row["act_error"] == "ConnectionError: bus down"


def test_turn_id_header_matches_the_motor() -> None:
    from orion.harness.fcc_warm_relay import CORRELATION_HEADER

    assert lk.TURN_ID_HEADER == CORRELATION_HEADER
    assert lk.turn_id({"ANTHROPIC_CUSTOM_HEADERS": "x-orion-correlation-id:  t-9 "}) == "t-9"
    assert lk.turn_id({}) is None


def test_orion_look_has_a_family() -> None:
    from orion.physiology.families import load_families

    cfg, err = load_families(Path(__file__).resolve().parents[3] / "config/physiology/families.yaml")
    assert err is None and cfg.families[lk.ACTOR].tier == 0  # perception: never bids

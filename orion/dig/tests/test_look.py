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
    monkeypatch.setattr(lk, "read_records", lambda stream, frame_at: [])  # never a real database
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


def test_doorbell_stale_frame_says_event_driven_and_is_not_graded(dirs, published) -> None:
    frames, home = dirs
    _frame(frames, "doorbell", "doorbell", NOW_MS - 120_000)
    lines, rec = lk.look("door", home=home, directory=frames, now=NOW)
    assert "only records after motion or a ring" in lines[1]
    assert published == [] and "act_id" not in rec and lines[-1].startswith("act: none (")


def test_doorbell_with_no_event_is_waiting_not_a_0(dirs, published) -> None:
    frames, home = dirs
    with pytest.raises(lk.LookError, match="only records after motion") as err:
        lk.look("doorbell", home=home, directory=frames, now=NOW)
    assert published == [] and "act: none" in str(err.value)
    (row,) = [json.loads(x) for x in (home / "look" / lk.LOG_NAME).read_text().splitlines()]
    assert row["cap_exempt"] is True and lk.looks_in_last_hour(home / "look" / lk.LOG_NAME, NOW) == 0


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
    assert row["cap_exempt"] is True and "act_id" not in row  # a read race: no 0, no cap slot


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
    lines, rec = lk.look("walkway", home=home, directory=frames, now=NOW,
                         clock=lambda: NOW + timedelta(milliseconds=40))
    opened, closed = published
    assert [opened.event_kind, closed.event_kind] == ["act_opened", "act_closed"]
    assert opened.trace_id == closed.trace_id == f"act:{rec['act_id']}"
    act = closed.act
    assert (act.actor, act.verb, act.object_ref) == ("orion.look", "look", "camera:walkway")
    # real wall time (40 ms here): the ledger fills `seconds` from these, never a fake 0
    assert (closed.observed_at - opened.observed_at).total_seconds() == 0.04
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
    assert lines[-1] == "act: not recorded (ConnectionError('bus down'))"
    (row,) = [json.loads(x) for x in (home / "look" / lk.LOG_NAME).read_text().splitlines()]
    assert row["act_error"] == "ConnectionError('bus down')"


def test_turn_id_header_matches_the_motor() -> None:
    from orion.harness.fcc_warm_relay import CORRELATION_HEADER

    assert lk.TURN_ID_HEADER == CORRELATION_HEADER
    assert lk.turn_id({"ANTHROPIC_CUSTOM_HEADERS": "x-orion-correlation-id:  t-9 "}) == "t-9"
    assert lk.turn_id({}) is None


def test_orion_look_has_a_family() -> None:
    from orion.physiology.families import load_families

    cfg, err = load_families(Path(__file__).resolve().parents[3] / "config/physiology/families.yaml")
    assert err is None and cfg.families[lk.ACTOR].tier == 0  # perception: never bids


def test_office_act_names_the_camera_by_word(dirs, published) -> None:
    from orion.self_monitor.adapters import act_faculty

    frames, home = dirs
    _frame(frames, "cam0", "cam0", NOW_MS - 300)
    lk.look("office", home=home, directory=frames, now=NOW)
    act = published[1].act
    assert act.object_ref == "camera:office"
    faculty = act_faculty(act.actor, act.verb, act.object_ref)
    assert faculty.endswith(".office"), faculty  # same naming as walkway, not the bare verb


def test_open_sent_close_lost_says_so(dirs, monkeypatch) -> None:
    frames, home = dirs
    _frame(frames, "cam0", "cam0", NOW_MS - 300)

    def half(events):
        raise lk.PartialPublish(1, TimeoutError())

    monkeypatch.setattr(lk, "_publish_bus", half)
    lines, rec = lk.look("office", home=home, directory=frames, now=NOW)
    assert rec["act_id"].startswith("act_") and "OPENED BUT NOT CLOSED" in lines[-1] and "TimeoutError()" in lines[-1]


def test_turn_id_env_wins_and_matches_the_motor() -> None:
    from orion.harness.fcc_motor import TURN_CORRELATION_ENV

    assert lk.TURN_ID_ENV == TURN_CORRELATION_ENV
    env = {"ORION_TURN_CORRELATION_ID": "warm-turn", "ANTHROPIC_CUSTOM_HEADERS": "X-Orion-Correlation-Id: old"}
    assert lk.turn_id(env) == "warm-turn"


# --- records near the frame + look check (#3347 part 2a) ----------------------------------------

REC = [
    {"event_id": "a1b2c3d4-0000-0000-0000-000000000001", "event_type": "visual_observation", "confidence": 0.8,
     "entities": '["vehicle"]', "narrative": "A vehicle is parked on the walkway.", "offset_sec": 12.0},
    {"event_id": "ffff0000-0000-0000-0000-000000000002", "event_type": "camera_caption", "confidence": 0.4,
     "entities": "[]", "narrative": "a porch at night", "offset_sec": -30.5},
]


def _look_with_records(frames: Path, home: Path, records=None):
    _frame(frames, "walkway", "walkway", NOW_MS - 400)
    return lk.look("walkway", home=home, directory=frames, now=NOW,
                   records=(lambda s, t: REC) if records is None else records)


def test_look_shows_the_records_near_the_frame_and_keeps_them(dirs) -> None:
    frames, home = dirs
    lines, rec = _look_with_records(frames, home)
    text = "\n".join(lines)
    assert "A vehicle is parked on the walkway." in text and 'entities ["vehicle"]' in text
    assert f"look check {rec['act_id']}" in text
    assert [r["id"] for r in rec["records"]] == [r["event_id"] for r in REC]
    assert lines[-1].startswith("act: act_")  # the act line stays last


def test_unread_records_never_fail_the_look(dirs) -> None:
    frames, home = dirs

    def boom(stream, at):
        raise RuntimeError("db down")

    lines, rec = _look_with_records(frames, home, records=boom)
    assert Path(rec["path"]).exists() and "records: unread" in "\n".join(lines)


def test_records_sql_is_read_only_and_backslash_free() -> None:
    sql = lk.records_sql("walkway", NOW)
    assert sql.startswith("SELECT ") and "\\" not in sql and "stream_id = 'walkway'" in sql
    with pytest.raises(lk.LookError):
        lk.records_sql("walkway'; drop table x; --", NOW)


def test_check_differs_is_a_graded_0_child_of_the_look(dirs, published) -> None:
    frames, home = dirs
    _, look_rec = _look_with_records(frames, home)
    published.clear()
    lines, entry = lk.check(look_rec["act_id"], "differs", "porch, chair and lawn; no vehicle on the walkway",
                            "a1b2c3d4", home=home, now=NOW, clock=lambda: NOW + timedelta(milliseconds=5))
    opened, closed = published
    act = closed.act
    assert (act.actor, act.verb, act.parent_act_id) == ("orion.look_check", "infer", look_rec["act_id"])
    assert act.object_ref == "vision_event:a1b2c3d4-0000-0000-0000-000000000001" and act.outcome.value == 0.0
    assert act.outcome.note.startswith("differs: porch")
    logged = [json.loads(x) for x in (home / "look" / lk.CHECK_LOG).read_text().splitlines()]
    assert logged == [entry] and entry["frame_sha256"] == look_rec["sha256"]


def test_check_matches_closes_1_and_cannot_tell_is_never_graded(dirs, published) -> None:
    frames, home = dirs
    _, look_rec = _look_with_records(frames, home)
    published.clear()
    lk.check(look_rec["act_id"], "matches", "a dark porch at night, as captioned", "ffff0000", home=home, now=NOW)
    assert published[1].act.outcome.value == 1.0
    published.clear()
    lines, entry = lk.check(look_rec["act_id"], "cannot-tell", "the far half of the frame is too dark",
                            "a1b2c3d4", home=home, now=NOW)
    assert published == [] and "act_id" not in entry and "never graded" in lines[-1]


@pytest.mark.parametrize("args, match", [
    (("act_nope", "matches", "something long enough", "a1b2c3d4"), "no look with act"),
    (("{act}", "maybe", "something long enough", "a1b2c3d4"), "verdict must be"),
    (("{act}", "matches", "short", "a1b2c3d4"), "characters"),
    (("{act}", "matches", "something long enough", "deadbeef"), "not one this look showed"),
    (("{act}", "matches", "something long enough", "a1b"), "not one this look showed"),  # too short a prefix
])
def test_check_refuses(dirs, args, match) -> None:
    frames, home = dirs
    _, look_rec = _look_with_records(frames, home)
    act, verdict, seen, rid = args
    with pytest.raises(lk.LookError, match=match):
        lk.check(act.format(act=look_rec["act_id"]), verdict, seen, rid, home=home, now=NOW)


def test_a_record_is_checked_once_per_look(dirs) -> None:
    frames, home = dirs
    _, look_rec = _look_with_records(frames, home)
    lk.check(look_rec["act_id"], "differs", "no vehicle anywhere in view", "a1b2c3d4", home=home, now=NOW)
    with pytest.raises(lk.LookError, match="already checked"):
        lk.check(look_rec["act_id"], "matches", "changed my mind entirely", "a1b2c3d4", home=home, now=NOW)


def test_dig_cli_look_check(dirs, monkeypatch, capsys, published) -> None:
    frames, home = dirs
    monkeypatch.setenv("ORION_DIG_HOME", str(home))
    _, look_rec = _look_with_records(frames, home)
    assert dig.main(["look", "check", look_rec["act_id"], "differs", "no", "vehicle", "because", "it", "is",
                     "dark", "because", "a1b2c3d4"]) == 0
    out = capsys.readouterr().out
    assert "check: differs" in out and "you saw: no vehicle because it is dark" in out
    assert dig.main(["look", "check", look_rec["act_id"], "differs", "seen", "words", "because", "a1", "b2"]) == 1


def test_orion_look_check_has_a_tier_0_family() -> None:
    from orion.physiology.families import load_families

    cfg, err = load_families(Path(__file__).resolve().parents[3] / "config/physiology/families.yaml")
    assert err is None and cfg.families[lk.CHECK_ACTOR].tier == 0

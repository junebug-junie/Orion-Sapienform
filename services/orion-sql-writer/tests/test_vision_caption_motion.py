"""Caption motion cross-check (app/vision_caption_motion.py, spec PR #2763 item 3 C)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import vision_caption_motion as cm
from app.vision_caption_motion import (
    CaptionBox,
    FrameCrop,
    decide_caption,
    match_caption_crops,
    motion_claims,
    parse_caption_refs,
)
from orion.situational.street_material import caption_motion_check_from_tags, label_camera_caption

FIXTURE = Path(__file__).parent / "fixtures" / "caption_motion_driving_captions.json"
T0 = datetime(2026, 10, 19, 17, 12, 0, tzinfo=timezone.utc)


def _verdict_for(row: dict) -> cm.CaptionVerdict:
    _ts, boxes = parse_caption_refs(row["evidence_refs"])
    crops = [FrameCrop(c["crop_id"], c["label"], tuple(c["box"])) for c in row["crops"]]
    matched = match_caption_crops(boxes, crops)
    return decide_caption(row["narrative"], matched, row["story_class"], settled=True)


# --- Replay: the 9 known "driving" captions (live, read-only, 2026-10-17..20) ---


def test_the_nine_known_driving_captions_are_contradicted() -> None:
    rows = json.loads(FIXTURE.read_text())
    assert len(rows) == 9
    for row in rows:
        v = _verdict_for(row)
        assert v.status == "contradicted", row["narrative"]
        assert v.tags[0] == "motion_contradicted"
        assert v.tags[1] == "motion_check:vehicle:driving:parked"
        check = caption_motion_check_from_tags(["camera_caption", "motion_unverified", *v.tags])
        assert "caption said driving; the vehicle did not move" in label_camera_caption(
            row["narrative"], ["person"], motion_check=check)


def test_replay_matches_only_the_captions_own_boxes() -> None:
    """Each fixture frame: every crop the verdict used is one of the caption's own boxes."""
    for row in json.loads(FIXTURE.read_text()):
        _ts, boxes = parse_caption_refs(row["evidence_refs"])
        crops = [FrameCrop(c["crop_id"], c["label"], tuple(c["box"])) for c in row["crops"]]
        for c in match_caption_crops(boxes, crops):
            assert any(cm.box_iou(b.box, c.box) >= 0.5 for b in boxes)


# --- Claims: derived from the story speed vocabulary ---


@pytest.mark.parametrize("text,words", [
    ("A car is driving down a residential street.", ["driving"]),
    ("A truck drives past.", ["drives"]),
    ("A car is parked in the driveway.", []),  # 'driveway' is not a claim; 'parked' claims no motion
    ("A delivery driver stands by a van.", []),
    ("A person is walking a dog while a child is running.", ["walking", "running"]),
    ("A person is pushing a stroller.", ["pushing"]),
    ("A person is riding a bicycle.", ["riding"]),
    ("A person is mowing the lawn, passing by.", []),  # outside the story vocabulary: unchecked
])
def test_motion_claims(text, words) -> None:
    assert [c.word for c in motion_claims(text)] == words


def test_walking_is_a_claim_about_people_and_dogs_not_bicycles() -> None:
    (claim,) = motion_claims("A person is walking.")
    assert set(claim.kinds) == {"person", "dog"}
    (drive,) = motion_claims("A car is driving.")
    assert drive.kinds == ("vehicle",)


# --- IoU: the caption's own box, never a neighbour ---


def test_iou_picks_the_captions_box_not_a_neighbour() -> None:
    parked = FrameCrop("c-parked", "vehicle", (1236.8, 468.1, 1339.9, 539.6))
    moving_neighbour = FrameCrop("c-moving", "vehicle", (1700.0, 360.0, 1780.0, 420.0))
    refs = ["frame_ts:2026-10-19T17:12:00+00:00", "box:person:980,713,1095,967", "crop:1237,468,1340,540"]
    _ts, boxes = parse_caption_refs(refs)
    matched = match_caption_crops(boxes, [parked, moving_neighbour])
    assert [c.crop_id for c in matched] == ["c-parked"]
    v = decide_caption("A car is driving down the street.", matched,
                       {"c-parked": "parked", "c-moving": "driving"}, settled=True)
    assert v.status == "contradicted"
    # Same frame, the caption's box is the moving one this time.
    refs2 = ["frame_ts:2026-10-19T17:12:00+00:00", "crop:1700,360,1780,420"]
    _ts, boxes2 = parse_caption_refs(refs2)
    matched2 = match_caption_crops(boxes2, [parked, moving_neighbour])
    v2 = decide_caption("A car is driving down the street.", matched2,
                        {"c-parked": "parked", "c-moving": "driving"}, settled=True)
    assert v2.status == "supported"
    assert v2.tags == ("motion_supported", "motion_check:vehicle:driving:driving")


def test_best_iou_wins_when_two_crops_overlap_the_box() -> None:
    box = [CaptionBox(None, (100.0, 100.0, 200.0, 200.0))]
    near = FrameCrop("near", "vehicle", (102.0, 101.0, 201.0, 199.0))
    looser = FrameCrop("looser", "vehicle", (120.0, 110.0, 210.0, 205.0))
    assert [c.crop_id for c in match_caption_crops(box, [looser, near])] == ["near"]


def test_a_labelled_box_only_matches_its_own_kind() -> None:
    box = [CaptionBox("person", (100.0, 100.0, 200.0, 300.0))]
    bike = FrameCrop("bike", "bicycle", (100.0, 100.0, 200.0, 300.0))
    assert match_caption_crops(box, [bike]) == []
    # A gate label that is not a story kind ("mail truck") matches by IoU alone.
    truck = [CaptionBox("mail truck", (100.0, 100.0, 200.0, 300.0))]
    van = FrameCrop("van", "vehicle", (100.0, 100.0, 200.0, 300.0))
    assert [c.crop_id for c in match_caption_crops(truck, [van])] == ["van"]


# --- Undecidable stays untagged ---


def test_no_crop_of_the_claimed_kind_is_undecided_and_untagged() -> None:
    person = FrameCrop("p", "person", (0.0, 0.0, 10.0, 30.0))
    v = decide_caption("A car is driving.", [person], {"p": "walking"}, settled=True)
    assert v.status == "undecided" and v.tags == ()


def test_storyless_crop_is_pending_then_undecided() -> None:
    car = FrameCrop("v", "vehicle", (0.0, 0.0, 10.0, 10.0))
    assert decide_caption("A car is driving.", [car], {}, settled=False).status == "pending"
    v = decide_caption("A car is driving.", [car], {}, settled=True)
    assert v.status == "undecided" and v.tags == ()
    # A story with no speed (a glimpse) decides nothing either.
    assert decide_caption("A car is driving.", [car], {"v": None}, settled=True).tags == ()


def test_no_motion_claim_gets_no_tag() -> None:
    car = FrameCrop("v", "vehicle", (0.0, 0.0, 10.0, 10.0))
    v = decide_caption("A car is parked on the street.", [car], {"v": "parked"}, settled=True)
    assert v.status == "no_claim" and v.tags == ()


def test_one_contradicted_claim_makes_the_row_contradicted_and_leads_the_tags() -> None:
    car = FrameCrop("v", "vehicle", (0.0, 0.0, 10.0, 10.0))
    walker = FrameCrop("p", "person", (20.0, 0.0, 30.0, 30.0))
    v = decide_caption("A person walking while a car is driving.", [car, walker],
                       {"v": "parked", "p": "walking"}, settled=True)
    assert v.tags == ("motion_contradicted", "motion_check:vehicle:driving:parked",
                      "motion_check:person:walking:walking")


def test_parse_refs() -> None:
    ts, boxes = parse_caption_refs([
        "artifact:x", "frame_ts:2026-10-17T19:36:54.511902+00:00", "box:mail truck:1,2,3,4", "crop:5,6,7,8",
        "crop:bad", "model:Qwen"])
    assert ts == datetime(2026, 10, 17, 19, 36, 54, 511902, tzinfo=timezone.utc)
    assert boxes == [CaptionBox("mail truck", (1.0, 2.0, 3.0, 4.0)), CaptionBox(None, (5.0, 6.0, 7.0, 8.0))]


# --- The clocked step: reads, tags, never re-reads a final row ---


class _Result:
    def __init__(self, rows=None, scalar=None, rowcount=0):
        self._rows, self._scalar, self.rowcount = rows or [], scalar, rowcount

    def fetchall(self):
        return self._rows

    def scalar(self):
        return self._scalar


class _FakeDB:
    def __init__(self, captions, crops, stories, cursor):
        self.captions, self.crops, self.stories, self.cursor = captions, crops, stories, cursor
        self.updates: list[dict] = []
        self.reads = 0

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "vision_motion_story_cursor" in sql:
            return _Result(scalar=self.cursor)
        if sql.startswith("SELECT event_id"):
            return _Result(rows=[c for c in self.captions if c[0] not in {u["id"] for u in self.updates}])
        if "FROM vision_crop_observation" in sql:
            self.reads += 1
            return _Result(rows=self.crops)
        if "FROM vision_motion_story" in sql:
            return _Result(rows=self.stories)
        if sql.startswith("UPDATE vision_events"):
            self.updates.append(params)
            return _Result(rowcount=1)
        raise AssertionError(sql)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def connect(self):
        return self

    def begin(self):
        return self


def test_caption_motion_step_tags_once_and_skips_final_rows() -> None:
    cm._final_rows.clear()
    frame = T0 - timedelta(hours=1)
    refs = [f"frame_ts:{frame.isoformat()}", "crop:0,0,10,10"]
    captions = [("e-drive", "A car is driving.", json.dumps(refs)),
                ("e-still", "A car is parked.", json.dumps(refs)),
                ("e-walk", "A person is walking.", json.dumps(refs))]
    db = _FakeDB(captions, crops=[("v", "vehicle", [0, 0, 10, 10])], stories=[(["v"], "parked")], cursor=T0)
    counts = cm.caption_motion_step(db, "walkway", T0, lookback_sec=21600, settle_sec=911, min_iou=0.5)
    assert counts["tagged"] == 1 and counts["contradicted"] == 1
    assert counts["no_claim"] == 1 and counts["undecided"] == 1
    assert db.updates[0]["id"] == "e-drive"
    assert json.loads(db.updates[0]["add"]) == ["motion_contradicted", "motion_check:vehicle:driving:parked"]
    reads = db.reads
    again = cm.caption_motion_step(db, "walkway", T0, lookback_sec=21600, settle_sec=911, min_iou=0.5)
    assert again["checked"] == 0 and db.reads == reads  # final rows are not re-read
    cm._final_rows.clear()


def test_caption_before_the_story_cursor_waits() -> None:
    cm._final_rows.clear()
    frame = T0 - timedelta(seconds=30)
    refs = [f"frame_ts:{frame.isoformat()}", "crop:0,0,10,10"]
    db = _FakeDB([("e", "A car is driving.", refs)], crops=[], stories=[], cursor=T0 - timedelta(minutes=1))
    counts = cm.caption_motion_step(db, "walkway", T0, lookback_sec=21600, settle_sec=911, min_iou=0.5)
    assert counts["pending"] == 1 and not db.updates and "e" not in cm._final_rows


def test_loop_config_reads_the_settings() -> None:
    from app.settings import Settings
    from app.vision_motion_loop import loop_config

    cfg = loop_config(Settings())
    assert cfg.caption_check is True
    assert cfg.caption_lookback_sec == 21600.0 and cfg.caption_min_iou == 0.5

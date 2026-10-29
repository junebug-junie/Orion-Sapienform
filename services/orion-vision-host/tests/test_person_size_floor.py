"""cam0 person over-count: a desk photo of two people read as two people.

Found 2026-10-29. A framed photo of a couple stands on the desk in the cam0
room. GroundingDINO labels both pictured people `person` at 0.35-0.41, with
boxes 11-20 px wide and 15-31 px tall on the 640x480 frame. With Juniper in
the room the window read 3 people. The box coordinates below are copied from
real cam0 detection artifacts (10-22 19:30 MDT and 10-28 19:00 MDT).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
from PIL import Image

from app import runner as runner_mod
from app.detections import drop_undersized
from app.runner import VisionRunner

W, H = 640, 480
CAM0_RULE = {"person": 0.1}

# Real cam0 boxes.
PHOTO_LEFT = [254.0, 138.0, 267.0, 161.0]    # 10-28, score 0.397
PHOTO_RIGHT = [262.0, 136.0, 273.0, 161.0]   # 10-28, score 0.366
JUNIPER = [98.0, 158.0, 274.0, 383.0]        # 10-22 19:30, score 0.76
HEAD_AT_BOTTOM_EDGE = [390.0, 449.0, 462.0, 479.0]  # 08-15, real person entering from below


def _p(box, score=0.5, label="person"):
    return {"label": label, "score": score, "box_xyxy": list(box)}


def test_photo_people_dropped_real_person_kept() -> None:
    objs = [_p(JUNIPER, 0.76), _p(PHOTO_LEFT, 0.397), _p(PHOTO_RIGHT, 0.366)]
    kept, dropped = drop_undersized(objs, CAM0_RULE, W, H)
    assert [o["box_xyxy"] for o in kept] == [JUNIPER]
    assert dropped == 2


def test_two_real_people_side_by_side_still_count_two() -> None:
    # Two people standing next to each other (09-04 cam0 boxes, both ~0.7).
    a = [208.0, 109.0, 320.0, 338.0]
    b = [311.0, 138.0, 427.0, 304.0]
    kept, dropped = drop_undersized([_p(a, 0.62), _p(b, 0.71)], CAM0_RULE, W, H)
    assert len(kept) == 2
    assert dropped == 0


def test_box_cut_off_by_frame_edge_is_kept_even_when_small() -> None:
    kept, dropped = drop_undersized([_p(HEAD_AT_BOTTOM_EDGE)], CAM0_RULE, W, H)
    assert len(kept) == 1 and dropped == 0


def test_person_lying_down_is_judged_by_long_side() -> None:
    # 120 px wide, 30 px tall, mid-frame: long side 120 >= 48.
    kept, _ = drop_undersized([_p([200.0, 300.0, 320.0, 330.0])], CAM0_RULE, W, H)
    assert len(kept) == 1


def test_other_labels_untouched() -> None:
    cup = _p(PHOTO_LEFT, label="cup")
    kept, dropped = drop_undersized([cup], CAM0_RULE, W, H)
    assert kept == [cup] and dropped == 0


def test_empty_rule_is_a_passthrough() -> None:
    objs = [_p(PHOTO_LEFT)]
    kept, dropped = drop_undersized(objs, {}, W, H)
    assert kept == objs and dropped == 0


def test_floor_scales_with_frame_size() -> None:
    # Same photo box on a 2560x1920 still (4x): 52-100 px, still under 0.1*1920.
    big = [v * 4 for v in PHOTO_LEFT]
    kept, dropped = drop_undersized([_p(big)], CAM0_RULE, 2560, 1920)
    assert kept == [] and dropped == 1


def test_shipped_config_turns_the_floor_on_for_cam0_only(vision_profiles) -> None:
    params = vision_profiles.get_profile("retina_detect_open_vocab").params
    rule = params["min_box_long_side_frac_by_stream"]
    assert rule["cam0"]["person"] == pytest.approx(0.1)
    # The walkway's far sidewalk people are ~50 px on a 1920 px short side;
    # a floor there would delete real people.
    assert "walkway" not in rule


# ------------------------------------------------------------ runner wiring


@pytest.fixture
def cam0_runner(vision_profiles, monkeypatch, tmp_path):
    frame = tmp_path / "cam0.png"
    Image.new("RGB", (W, H), (90, 90, 90)).save(frame)
    r = VisionRunner(profiles=vision_profiles, enabled_names=["retina_detect_open_vocab"], cache_dir=tempfile.mkdtemp())
    r.artifacts_dir = tmp_path

    def fake(model, processor, images, *a, **k):
        return [[_p(JUNIPER, 0.76), _p(PHOTO_LEFT, 0.397), _p(PHOTO_RIGHT, 0.366)] for _ in images]

    monkeypatch.setattr(runner_mod, "detect_images", fake)
    monkeypatch.setattr(r.models, "load_grounding_dino", lambda **k: (object(), object()))
    return r, str(frame)


def _detect(runner, request):
    p = runner.profiles.get_profile("retina_detect_open_vocab")
    return runner._run_detect_grounding_dino(p, request, "cpu")


def test_runner_cam0_counts_one_person_not_three(cam0_runner) -> None:
    r, frame = cam0_runner
    out = _detect(r, {"image_path": frame, "stream_id": "cam0"})
    persons = [o for o in out["objects"] if o["label"] == "person"]
    assert [o["box_xyxy"] for o in persons] == [JUNIPER]


def test_runner_other_stream_is_unchanged(cam0_runner) -> None:
    r, frame = cam0_runner
    out = _detect(r, {"image_path": frame, "stream_id": "walkway"})
    assert sum(1 for o in out["objects"] if o["label"] == "person") == 3

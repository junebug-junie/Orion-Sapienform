"""Per-camera, per-label box cut-offs (app/box_thresholds.py) and their runner
wiring: the detector runs at the lowest cut-off for the request's stream, and
each box must reach its own label's cut-off."""

from __future__ import annotations

import tempfile

import pytest
from PIL import Image

import app.runner as runner_mod
from app.box_thresholds import BoxThresholds, apply_box_thresholds, resolve_box_thresholds
from app.runner import VisionRunner


def test_no_table_is_the_single_threshold():
    th = resolve_box_thresholds({"score_threshold": 0.35}, "walkway")
    assert th == BoxThresholds(0.35)
    assert th.floor == 0.35 and th.for_label("vehicle") == 0.35


def test_box_threshold_wins_over_score_threshold_alias():
    assert resolve_box_thresholds({"box_threshold": 0.3, "score_threshold": 0.35}, None).default == 0.3


def test_stream_entry_sets_labels_and_default():
    params = {"score_threshold": 0.35,
              "box_threshold_by_stream": {"walkway": {"default": 0.4, "Vehicle ": 0.3}}}
    th = resolve_box_thresholds(params, "walkway")
    assert th.default == 0.4
    assert th.for_label("vehicle") == 0.3
    assert th.for_label(" VEHICLE") == 0.3
    assert th.for_label("person") == 0.4
    assert th.floor == 0.3
    # Another stream is untouched.
    assert resolve_box_thresholds(params, "cam0") == BoxThresholds(0.35)


@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5, "x", None])
def test_invalid_value_is_ignored_never_lowers_to_zero(bad):
    th = resolve_box_thresholds({"score_threshold": 0.35, "box_threshold_by_stream": {"walkway": {"vehicle": bad}}}, "walkway")
    assert th.for_label("vehicle") == 0.35
    assert th.floor == 0.35


def test_apply_filters_each_box_by_its_own_label():
    th = BoxThresholds(0.35, {"vehicle": 0.30})
    objs = [
        {"label": "vehicle", "score": 0.31},   # kept: vehicle cut-off 0.30
        {"label": "person", "score": 0.31},    # dropped: default 0.35
        {"label": "person dog", "score": 0.36},  # merged phrase -> default, kept
        {"label": "vehicle", "score": 0.29},   # dropped
    ]
    kept = apply_box_thresholds(objs, th)
    assert [(o["label"], o["score"]) for o in kept] == [("vehicle", 0.31), ("person dog", 0.36)]


def test_shipped_profile_keeps_todays_values(vision_profiles):
    """Shipped config changes nothing: every camera and label stays at 0.35."""
    params = vision_profiles.get_profile("retina_detect_open_vocab").params
    for stream in ("walkway", "cam0", "doorbell", None):
        th = resolve_box_thresholds(params, stream)
        assert th.default == 0.35 and th.floor == 0.35
        for label in ("person", "vehicle", "dog", "mail truck"):
            assert th.for_label(label) == 0.35
    assert "cam0" not in (params.get("box_threshold_by_stream") or {})


@pytest.fixture
def runner_with_spy(vision_profiles, monkeypatch, tmp_path):
    frame = tmp_path / "f.png"
    Image.new("RGB", (640, 480)).save(frame)
    r = VisionRunner(profiles=vision_profiles, enabled_names=["retina_detect_open_vocab"], cache_dir=tempfile.mkdtemp())
    seen = []

    def fake(model, processor, images, text, box_th, text_th, device, cast=None):
        seen.append(box_th)
        return [[{"label": "vehicle", "score": 0.31, "box_xyxy": [0.0, 0.0, 50.0, 50.0]},
                 {"label": "person", "score": 0.33, "box_xyxy": [100.0, 0.0, 150.0, 80.0]},
                 {"label": "person", "score": 0.5, "box_xyxy": [300.0, 0.0, 350.0, 80.0]}] for _ in images]

    monkeypatch.setattr(runner_mod, "detect_images", fake)
    monkeypatch.setattr(r.models, "load_grounding_dino", lambda **k: (object(), object()))
    return r, str(frame), seen


def test_runner_runs_at_floor_and_filters_per_label(runner_with_spy):
    r, frame, seen = runner_with_spy
    p = r.profiles.get_profile("retina_detect_open_vocab")
    p.params = {**p.params, "box_threshold_by_stream": {"walkway": {"vehicle": 0.30}}}
    out = r._run_detect_grounding_dino(p, {"image_path": frame, "stream_id": "walkway"}, "cpu")
    assert seen == [0.30]
    assert sorted((o["label"], o["score"]) for o in out["objects"]) == [("person", 0.5), ("vehicle", 0.31)]

    seen.clear()
    out = r._run_detect_grounding_dino(p, {"image_path": frame, "stream_id": "cam0"}, "cpu")
    assert seen == [0.35]
    assert [(o["label"], o["score"]) for o in out["objects"]] == [("person", 0.5)]


def test_runner_shipped_profile_behaves_like_single_threshold(runner_with_spy):
    r, frame, seen = runner_with_spy
    p = r.profiles.get_profile("retina_detect_open_vocab")
    out = r._run_detect_grounding_dino(p, {"image_path": frame, "stream_id": "walkway"}, "cpu")
    assert seen == [0.35]
    assert [(o["label"], o["score"]) for o in out["objects"]] == [("person", 0.5)]

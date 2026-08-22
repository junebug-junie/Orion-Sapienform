"""GPU/model-free: exercises the Haar-cascade accounting logic against a
synthetic clip with no detectable face, so the "no face ever" fallback path
is deterministic and doesn't depend on real face-detection succeeding.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.face_extract import extract_face_crops


@pytest.fixture
def blank_video(tmp_path):
    """A short clip of solid-color frames -- Haar cascade should never find
    a face in this, exercising the full-frame fallback path deterministically."""
    path = str(tmp_path / "blank.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(6):
        frame = np.full((240, 320, 3), 127, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_no_face_falls_back_to_full_frame(blank_video):
    result = extract_face_crops(blank_video)

    assert result.frames_total == 6
    assert result.frames_detected == 0
    assert result.frames_carried_forward == 0
    assert result.frames_no_face_fallback_full_frame == 6
    assert result.detection_rate == 0.0
    assert result.faces.shape == (6, 224, 224, 3)
    assert result.faces.dtype == np.uint8


def test_as_meta_matches_counters(blank_video):
    result = extract_face_crops(blank_video)
    meta = result.as_meta()

    assert meta["frames_total"] == 6
    assert meta["frames_detected"] == 0
    assert meta["detection_rate"] == 0.0
    assert set(meta.keys()) == {
        "frames_total",
        "frames_detected",
        "frames_carried_forward",
        "frames_no_face_fallback_full_frame",
        "detection_rate",
    }


def test_missing_video_raises(tmp_path):
    with pytest.raises(RuntimeError):
        extract_face_crops(str(tmp_path / "does_not_exist.mp4"))


def test_empty_video_raises(tmp_path):
    path = str(tmp_path / "empty.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    writer.release()  # zero frames written
    with pytest.raises(RuntimeError):
        extract_face_crops(path)

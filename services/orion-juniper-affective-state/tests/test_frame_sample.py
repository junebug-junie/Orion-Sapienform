"""Frame selection and the quality signal derived from it.

Synthetic clips only -- no face is ever detected in generated noise, which is
exactly the branch worth pinning: a `detection_rate` of 0.0 must still return
usable frames AND report the zero honestly, because "send nothing" and "send
frames but flag them as untrustworthy" are different behaviours and the caller
decides between them. The live capture that motivated this whole patch scored
0.052, not 0.0, and the old pipeline treated it as a perfectly good read.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.frame_sample import FrameSampleError, _evenly_spaced, sample_frames

cv2 = pytest.importorskip("cv2")


def _write_clip(path, *, frames: int, width: int = 64, height: int = 48) -> str:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height)
    )
    assert writer.isOpened(), "could not open VideoWriter for the fixture clip"
    rng = np.random.default_rng(1234)
    try:
        for _ in range(frames):
            writer.write(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    finally:
        writer.release()
    return str(path)


def test_evenly_spaced_hits_both_ends_and_the_exact_midpoints():
    # Hand-computed: 9 candidates, want 5 -> step = (9-1)/(5-1) = 2.0 exactly,
    # so the picks are indices 0, 2, 4, 6, 8. Chosen with an integral step on
    # purpose: a fractional step lands on .5 for the middle pick, where
    # Python's banker's rounding makes the expected value non-obvious and the
    # test would be pinning round() rather than this function.
    assert _evenly_spaced(list(range(9)), 5) == [0, 2, 4, 6, 8]


def test_evenly_spaced_covers_the_whole_span_for_awkward_sizes():
    # The property that actually matters: the sample spans the clip. If it
    # collapsed to the front, the "you can see change across the frames"
    # premise of sending more than one frame would be false.
    picked = _evenly_spaced(list(range(231)), 5)
    assert picked[0] == 0
    assert picked[-1] == 230
    assert picked == sorted(picked)
    assert len(picked) == len(set(picked)) == 5


def test_evenly_spaced_returns_everything_when_asked_for_more_than_exists():
    assert _evenly_spaced([3, 9, 14], 5) == [3, 9, 14]


def test_sample_frames_reports_zero_detection_but_still_returns_frames(tmp_path):
    """Noise has no faces. The read is untrustworthy; the frames still exist."""
    clip = _write_clip(tmp_path / "noise.mp4", frames=40)
    result = sample_frames(clip, max_frames=5)

    assert result.frames_total == 40
    assert result.frames_detected == 0
    assert result.detection_rate == 0.0
    # The honest-but-useless case still yields input, so the CALLER's gate
    # decides, not this function silently refusing.
    assert len(result.frames) == 5
    assert all(f.jpeg.startswith(b"\xff\xd8") for f in result.frames)
    assert all(f.face_detected is False for f in result.frames)


def test_sample_frames_meta_matches_the_affectgpt_key_names(tmp_path):
    """detection_rate must mean the same thing across the backend cutover --
    a stored row from either backend is compared on these keys."""
    clip = _write_clip(tmp_path / "noise.mp4", frames=12)
    meta = sample_frames(clip, max_frames=3).as_meta()
    assert set(meta) == {
        "frames_total",
        "frames_detected",
        "detection_rate",
        "frames_sampled",
    }
    assert meta["frames_total"] == 12
    assert meta["frames_sampled"] == 3


def test_sample_frames_caps_at_the_clip_length(tmp_path):
    clip = _write_clip(tmp_path / "short.mp4", frames=2)
    result = sample_frames(clip, max_frames=5)
    assert len(result.frames) == 2
    assert result.frames_total == 2


def test_sample_frames_records_real_pixel_dimensions(tmp_path):
    clip = _write_clip(tmp_path / "dims.mp4", frames=6, width=64, height=48)
    frame = sample_frames(clip, max_frames=1).frames[0]
    # Not hardcoded 224 the way the cropping backend's output always was --
    # full frames go to the model now, so the ref's declared width/height must
    # be the clip's real size or the gateway's size cross-check would be wrong.
    assert (frame.width, frame.height) == (64, 48)


def test_sample_frames_raises_on_an_unreadable_clip(tmp_path):
    missing = tmp_path / "nope.mp4"
    with pytest.raises(FrameSampleError):
        sample_frames(str(missing), max_frames=3)


def test_sample_frames_raises_on_a_zero_frame_clip(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(FrameSampleError):
        sample_frames(str(empty), max_frames=3)


def test_sample_frames_rejects_a_nonsense_frame_budget(tmp_path):
    clip = _write_clip(tmp_path / "noise.mp4", frames=5)
    with pytest.raises(FrameSampleError):
        sample_frames(clip, max_frames=0)

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.runner import _load_image_from_request, _resolve_latest_frame_path
from app import runner as runner_module


def test_resolve_latest_frame_path_missing_dir_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner_module.settings, "VISION_FRAMES_DIR", str(tmp_path / "does-not-exist"))
    with pytest.raises(FileNotFoundError, match="frames directory not found"):
        _resolve_latest_frame_path()


def test_resolve_latest_frame_path_empty_dir_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner_module.settings, "VISION_FRAMES_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no frames found for stream 'cam0'"):
        _resolve_latest_frame_path()


def test_resolve_latest_frame_path_picks_the_newest_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner_module.settings, "VISION_FRAMES_DIR", str(tmp_path))
    older = tmp_path / "cam0_cam0_older.jpg"
    newer = tmp_path / "cam0_cam0_newer.jpg"
    older.write_bytes(b"old")
    time.sleep(0.01)
    newer.write_bytes(b"new")

    result = _resolve_latest_frame_path()
    assert result == newer


def test_resolve_latest_frame_path_ignores_non_jpg_files(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner_module.settings, "VISION_FRAMES_DIR", str(tmp_path))
    (tmp_path / "readme.txt").write_text("not a frame")
    jpg = tmp_path / "cam0_cam0_1.jpg"
    jpg.write_bytes(b"real frame")

    result = _resolve_latest_frame_path()
    assert result == jpg


def test_load_image_without_image_path_or_use_latest_frame_still_raises_as_before() -> None:
    """Backward compat: a request with neither image_path/frame_path NOR
    use_latest_frame must keep the exact prior behavior (a clear error),
    not silently fall back to the latest frame -- opt-in, not implicit.

    Assertion text updated (pre-existing bug, unrelated to this patch --
    found while running this file's suite to verify the vision_profiles.yaml
    cleanup): the percept_sha256 pointer path (PR #1867) changed the real
    message from "request.image_path is required" to the one below; this
    test never ran against that real message before now because torch isn't
    importable in this repo's shared dev venv, so pytest -q errors at
    collection instead of reaching this assertion."""
    with pytest.raises(
        ValueError,
        match=r"request needs image_path or percept_sha256",
    ):
        _load_image_from_request({})


def test_load_image_use_latest_frame_true_resolves_without_image_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner_module.settings, "VISION_FRAMES_DIR", str(tmp_path))
    # A minimal real JPEG so PIL.Image.open succeeds.
    from PIL import Image

    img_path = tmp_path / "cam0_cam0_1.jpg"
    Image.new("RGB", (4, 4), color="red").save(img_path)

    result = _load_image_from_request({"use_latest_frame": True})
    assert result.size == (4, 4)


def test_load_image_explicit_image_path_wins_over_use_latest_frame(monkeypatch, tmp_path) -> None:
    """An explicit image_path always takes priority -- use_latest_frame is
    only consulted when no explicit pointer is given."""
    from PIL import Image

    explicit_path = tmp_path / "explicit.jpg"
    Image.new("RGB", (2, 2), color="blue").save(explicit_path)

    # Frames dir intentionally left unset/nonexistent -- if the code ever
    # tried to resolve the latest frame here, this would raise instead of
    # returning the explicit image.
    monkeypatch.setattr(runner_module.settings, "VISION_FRAMES_DIR", str(tmp_path / "unused"))

    result = _load_image_from_request({"image_path": str(explicit_path), "use_latest_frame": True})
    assert result.size == (2, 2)


def test_latest_frame_is_the_room_cameras_even_when_another_camera_wrote_last(monkeypatch, tmp_path) -> None:
    """2026-11-04 regression: every athena retina writes into the same frames
    directory. "Look now" must answer from the office camera, not from
    whichever camera wrote last (the walkway writes 2 frames/s)."""
    monkeypatch.setattr(runner_module.settings, "VISION_FRAMES_DIR", str(tmp_path))
    room = tmp_path / "cam0_cam0_1793836160000_aa.jpg"
    room.write_bytes(b"room")
    time.sleep(0.01)
    (tmp_path / "walkway_walkway_1793836160100_bb.jpg").write_bytes(b"walkway")
    (tmp_path / "frame_1793836160200_cc.jpg").write_bytes(b"edge naming")

    assert _resolve_latest_frame_path() == room


def test_latest_frame_for_a_named_stream(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner_module.settings, "VISION_FRAMES_DIR", str(tmp_path))
    (tmp_path / "cam0_cam0_1.jpg").write_bytes(b"room")
    time.sleep(0.01)
    walk = tmp_path / "walkway_walkway_2.jpg"
    walk.write_bytes(b"walkway")
    time.sleep(0.01)
    (tmp_path / "cam0_cam0_3.jpg").write_bytes(b"room newer")

    assert _resolve_latest_frame_path("walkway") == walk
    with pytest.raises(FileNotFoundError, match="no frames found for stream 'doorbell'"):
        _resolve_latest_frame_path("doorbell")

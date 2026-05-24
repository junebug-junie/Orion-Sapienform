from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "orion-vision-retina"))
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from app.frame_store import SavedFrame, cleanup_old_frames, save_frame


def test_save_frame_writes_jpg(tmp_path: Path) -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    saved = save_frame(
        frame,
        directory=str(tmp_path),
        camera_id="cam/01",
        stream_id="stream:01",
        ts=1710000000.0,
        quality=90,
    )
    assert isinstance(saved, SavedFrame)
    assert saved.format == "jpg"
    assert saved.width == 64
    assert saved.height == 48
    assert Path(saved.image_path).is_file()
    assert saved.image_path.endswith(".jpg")
    assert "cam_01" in Path(saved.image_path).name
    assert "stream_01" in Path(saved.image_path).name


def test_cleanup_removes_old_keeps_fresh(tmp_path: Path) -> None:
    old = tmp_path / "old.jpg"
    fresh = tmp_path / "fresh.jpg"
    old.write_bytes(b"x")
    fresh.write_bytes(b"y")
    old_ts = time.time() - 600
    os.utime(old, (old_ts, old_ts))
    removed = cleanup_old_frames(str(tmp_path), max_age_seconds=300)
    assert removed >= 1
    assert not old.exists()
    assert fresh.exists()

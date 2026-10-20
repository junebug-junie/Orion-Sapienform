"""The edge prunes only its own frames in the shared dir, and survives files
vanishing mid-sweep (2026-10-17: it deleted walkway retina frames at 60 s)."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import frame_prune
from app.frame_prune import cleanup_old_frames


def _touch(d: Path, name: str, age: float) -> Path:
    p = d / name
    p.write_bytes(b"x")
    t = time.time() - age
    os.utime(p, (t, t))
    return p


def test_prunes_only_its_own_old_files(tmp_path):
    mine_old = _touch(tmp_path, "frame_1792261575263_44e535.jpg", 120)
    still_old = _touch(tmp_path, "still_1792261575263_44e535.jpg", 120)
    tmp_old = _touch(tmp_path, ".still_1792261575263_44e535.jpg.tmp", 120)
    mine_new = _touch(tmp_path, "frame_1792261575999_aaaaaa.jpg", 5)
    walkway = _touch(tmp_path, "walkway_walkway_1792261575263_0123abcd.jpg", 120)
    doorbell = _touch(tmp_path, "doorbell_front_1792261575263_0123abcd.jpg", 120)
    stray = _touch(tmp_path, "notes.txt", 120)

    assert cleanup_old_frames(str(tmp_path), 60) == 3

    assert not mine_old.exists() and not still_old.exists() and not tmp_old.exists()
    assert mine_new.exists()
    assert walkway.exists() and doorbell.exists() and stray.exists()


def test_file_vanishing_mid_sweep_is_skipped(tmp_path, monkeypatch):
    for i in range(3):
        _touch(tmp_path, f"frame_17922615752{i}0_44e53{i}.jpg", 120)
    real = os.scandir

    class _Racing:
        """Another pruner deletes each file right after it is listed."""

        def __init__(self, path):
            self._it = real(path)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self._it.close()

        def __iter__(self):
            for e in self._it:
                os.remove(e.path)
                yield e

    monkeypatch.setattr(frame_prune.os, "scandir", _Racing)
    assert cleanup_old_frames(str(tmp_path), 60) == 0
    assert list(tmp_path.iterdir()) == []


def test_missing_dir_is_a_no_op(tmp_path):
    assert cleanup_old_frames(str(tmp_path / "nope"), 60) == 0

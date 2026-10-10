"""Frames behind face checks are kept (2026-10-10): the camera buffer rolls over in ~65 s."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.face_frames import FaceFrameStore, best_candidate  # noqa: E402

CANDS = [{"subject": "juniper", "state": "unsure", "similarity": 0.209, "detect_confidence": 0.997},
         {"subject": "juniper", "state": "possible", "similarity": 0.41, "detect_confidence": 0.9}]


def _store(tmp_path, t, **kw):
    return FaceFrameStore(tmp_path, clock=lambda: t["now"], **kw)


def test_best_candidate_prefers_band_then_similarity():
    assert best_candidate(CANDS)["state"] == "possible"
    assert best_candidate([]) is None


def test_saves_frame_and_sidecar_named_by_best_match(tmp_path):
    t = {"now": 1_791_604_926.0}                       # 2026-10-10 04:02:06 UTC
    rel = _store(tmp_path, t).save("cam0", Image.new("RGB", (1920, 1080)), {"candidates": CANDS})
    assert rel == "cam0/2026-10-10/040206_000000_possible_0.410.jpg"
    img = Image.open(tmp_path / rel)
    assert max(img.size) == 1280                       # downscaled, aspect kept
    meta = json.loads((tmp_path / rel).with_suffix(".json").read_text())
    assert meta["stream_id"] == "cam0" and meta["candidates"] == CANDS


def test_rate_limited_per_stream(tmp_path):
    t = {"now": 1_000.0}
    store = _store(tmp_path, t, min_interval_sec=5.0)
    img = Image.new("RGB", (16, 16))
    assert store.save("cam0", img, {"candidates": CANDS})
    assert store.save("cam0", img, {"candidates": CANDS}) is None
    assert store.save("carbon", img, {"candidates": CANDS})


def test_prune_removes_only_old_files_and_empty_folders(tmp_path):
    t = {"now": 1_000_000.0}
    store = _store(tmp_path, t, retention_days=14.0, min_interval_sec=0.0)
    rel = store.save("cam0", Image.new("RGB", (16, 16)), {"candidates": CANDS})
    old = tmp_path / rel
    for f in (old, old.with_suffix(".json")):
        os.utime(f, (t["now"] - 15 * 86400, t["now"] - 15 * 86400))
    t["now"] += 1
    fresh = store.save("cam0", Image.new("RGB", (16, 16)), {"candidates": CANDS})
    assert store.prune() == 2
    assert not old.exists() and (tmp_path / fresh).exists()

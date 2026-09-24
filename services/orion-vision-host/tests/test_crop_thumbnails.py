"""Crop thumbnails for the ask card: only for crops the host embedded.

A thumbnail is made from the exact crop list the embedder is handed, so a
box in a no-embed zone (the patio) structurally cannot get one. Retention is
owned by the writer (ThumbStore prunes by file age).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from PIL import Image

from app.crop_embeddings import THUMB_MAX_SIDE, ThumbStore, attach_crop_embeddings
from orion.vision.zones import Zone

W, H = 1000, 1000
PATIO = Zone(name="patio", polygon=((0.0, 0.7), (0.35, 0.7), (0.35, 1.0), (0.0, 1.0)), embed=False)
WALKWAY = Zone(name="walkway", polygon=((0.0, 0.35), (1.0, 0.35), (1.0, 1.0), (0.0, 1.0)), embed=True)
ZONES = {"walkway": [PATIO, WALKWAY]}
PATIO_BOX = [100.0, 700.0, 200.0, 950.0]
WALKWAY_BOX = [600.0, 600.0, 700.0, 900.0]


def _embed(crops):
    return np.ones((len(crops), 4), dtype=np.float32)


def _run(objects, thumb_fn, *, zones=ZONES):
    return attach_crop_embeddings(
        objects, Image.new("RGB", (W, H), (40, 90, 160)),
        request={"want_crop_embeddings": True, "stream_id": "walkway"}, params={},
        embed_fn=_embed, zones_by_stream=zones, model_id="m", embed_profile="embed_image",
        min_score=0.35, thumb_fn=thumb_fn,
    )


class _SpyThumb:
    def __init__(self):
        self.sizes = []

    def __call__(self, crop):
        self.sizes.append(crop.size)
        return "thumb:" + "a" * 64


def test_host_never_writes_a_thumb_for_a_patio_box(tmp_path: Path) -> None:
    store = ThumbStore(tmp_path)
    objs = [{"label": "person", "score": 0.9, "box_xyxy": list(PATIO_BOX)},
            {"label": "person", "score": 0.9, "box_xyxy": list(WALKWAY_BOX)}]
    stats = _run(objs, store.put)
    patio, walk = objs
    assert patio["zone"] == "patio" and patio["thumb_ref"] is None and patio["embedding"] is None
    assert walk["thumb_ref"].startswith("thumb:")
    files = sorted(tmp_path.glob("*.jpg"))
    assert len(files) == 1 and stats["thumbs"] == 1
    digest = walk["thumb_ref"].split(":", 1)[1]
    assert files[0].name == f"{digest}.jpg"
    assert hashlib.sha256(files[0].read_bytes()).hexdigest() == digest  # content-addressed
    with Image.open(files[0]) as im:
        assert max(im.size) <= THUMB_MAX_SIDE and im.format == "JPEG"


def test_patio_only_frame_never_reaches_the_thumbnailer() -> None:
    spy = _SpyThumb()
    objs = [{"label": "person", "score": 0.9, "box_xyxy": list(PATIO_BOX)}]
    _run(objs, spy)
    assert spy.sizes == []
    assert objs[0]["thumb_ref"] is None


def test_thumbnailer_sees_exactly_the_embedded_crops() -> None:
    spy = _SpyThumb()
    objs = [{"label": "person", "score": 0.9, "box_xyxy": list(PATIO_BOX)},
            {"label": "dog", "score": 0.9, "box_xyxy": list(WALKWAY_BOX)}]
    _run(objs, spy)
    assert spy.sizes == [(100, 300)]  # the walkway crop, never the 100x250 patio one


def test_zones_unknown_means_no_thumbs() -> None:
    spy = _SpyThumb()
    objs = [{"label": "person", "score": 0.9, "box_xyxy": list(WALKWAY_BOX)}]
    _run(objs, spy, zones=None)
    assert spy.sizes == [] and objs[0]["thumb_ref"] is None


def test_thumb_failure_keeps_the_embedding() -> None:
    def boom(_crop):
        raise OSError("disk full")

    objs = [{"label": "person", "score": 0.9, "box_xyxy": list(WALKWAY_BOX)}]
    stats = _run(objs, boom)
    assert objs[0]["embedding"] is not None and objs[0]["thumb_ref"] is None
    assert stats["embedded"] == 1 and stats["thumbs"] == 0


def test_store_prunes_old_thumbs_and_rewrite_refreshes_age(tmp_path: Path) -> None:
    now = [1_000_000.0]
    store = ThumbStore(tmp_path, retention_days=14, prune_interval_sec=0, clock=lambda: now[0])
    img = Image.new("RGB", (50, 80), (1, 2, 3))
    ref = store.put(img)
    path = tmp_path / (ref.split(":", 1)[1] + ".jpg")
    old = tmp_path / ("b" * 64 + ".jpg")
    old.write_bytes(b"x")
    os.utime(old, (now[0] - 15 * 86400, now[0] - 15 * 86400))
    now[0] += 13 * 86400
    assert store.put(img) == ref  # same bytes, same ref; mtime refreshed
    assert not old.exists() and path.exists()
    now[0] += 13 * 86400
    store.prune()
    assert path.exists()  # refreshed 13 days ago, still inside 14
    now[0] += 2 * 86400
    store.prune()
    assert not path.exists()

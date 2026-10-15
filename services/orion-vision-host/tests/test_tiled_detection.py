"""Tiled full-resolution detection for the walkway (2026-10-15).

Pure tiling math (app/tiling.py) plus the runner wiring with the detector
mocked: the fake detector "sees" red rectangles in whatever image it is
handed, but only when the rectangle is at least MIN_SEEN_PX tall after the
GroundingDINO processor's own resize (800 short side / 1333 long side).
That reproduces the real failure: a small far object vanishes in the
downscaled whole frame and survives in a tile.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app import runner as runner_mod
from app.runner import VisionRunner
from app.tiling import (
    Tile,
    merge_tiled_detections,
    tile_layout,
    tiling_params,
    tiling_requested,
    to_frame_coords,
)

W, H = 2560, 1920


# ---------------------------------------------------------------- layout


def test_2x2_layout_covers_frame_with_overlap() -> None:
    tiles = tile_layout(W, H, (2, 2), 0.2)
    assert len(tiles) == 4
    assert min(t.x0 for t in tiles) == 0 and max(t.x1 for t in tiles) == W
    assert min(t.y0 for t in tiles) == 0 and max(t.y1 for t in tiles) == H
    left, right = tiles[0], tiles[1]
    # Neighbours share ~20% of a tile's width.
    shared = left.x1 - right.x0
    assert shared == pytest.approx(0.2 * left.width, abs=2)


def test_region_layout_stays_inside_region() -> None:
    tiles = tile_layout(W, H, (2, 1), 0.2, region=(0.0, 0.15, 1.0, 0.6))
    assert len(tiles) == 2
    assert all(t.y0 == 288 and t.y1 == 1152 for t in tiles)


def test_invalid_grid_or_region_raises() -> None:
    with pytest.raises(ValueError):
        tile_layout(W, H, (0, 2), 0.2)
    with pytest.raises(ValueError):
        tile_layout(W, H, (2, 2), 0.2, region=(0.5, 0.5, 0.5, 0.9))


# ---------------------------------------------------------------- mapping


def test_tile_box_maps_to_frame_coordinates() -> None:
    tile = Tile(1138, 853, 2560, 1920)
    obj = {"label": "person", "score": 0.6, "box_xyxy": [100.0, 200.0, 140.0, 300.0]}
    out = to_frame_coords(obj, tile, W, H)
    assert out["box_xyxy"] == [1238.0, 1053.0, 1278.0, 1153.0]
    assert out["_cut"] is False
    assert obj["box_xyxy"] == [100.0, 200.0, 140.0, 300.0]  # input untouched


def test_box_on_interior_edge_is_cut_but_frame_edge_is_not() -> None:
    left = Tile(0, 0, 1422, 1067)
    # Touches the tile's right edge (interior seam) -> cut.
    seam = to_frame_coords({"label": "person", "score": 0.5, "box_xyxy": [1380.0, 100.0, 1421.0, 300.0]}, left, W, H)
    assert seam["_cut"] is True
    # Touches the tile's left edge, which is also the frame edge -> not cut.
    edge = to_frame_coords({"label": "person", "score": 0.5, "box_xyxy": [0.0, 100.0, 40.0, 300.0]}, left, W, H)
    assert edge["_cut"] is False


# ---------------------------------------------------------------- merge


def test_person_split_by_a_seam_merges_into_one_box() -> None:
    """A tall person crosses the horizontal seam; neither tile sees all of
    them and the whole-frame pass missed them. The two cut halves (which
    overlap inside the shared band) become one box spanning both."""
    top, bottom = Tile(0, 0, 1422, 1067), Tile(0, 853, 1422, 1920)
    upper = to_frame_coords({"label": "person", "score": 0.55, "box_xyxy": [600.0, 700.0, 680.0, 1066.0]}, top, W, H)
    lower = to_frame_coords({"label": "person", "score": 0.48, "box_xyxy": [602.0, 0.0, 681.0, 400.0]}, bottom, W, H)
    assert upper["_cut"] and lower["_cut"]
    merged = merge_tiled_detections([], [upper, lower], nms_iou=0.6)
    assert len(merged) == 1
    box = merged[0]["box_xyxy"]
    assert box[1] == pytest.approx(700.0) and box[3] == pytest.approx(1253.0)
    assert merged[0]["score"] == pytest.approx(0.55)
    assert not any(k.startswith("_") for k in merged[0])


def test_cut_fragment_dropped_when_whole_box_exists() -> None:
    whole = {"label": "person", "score": 0.5, "box_xyxy": [1300.0, 400.0, 1500.0, 900.0]}
    left = Tile(0, 0, 1422, 1067)
    frag = to_frame_coords({"label": "person", "score": 0.7, "box_xyxy": [1300.0, 400.0, 1421.0, 900.0]}, left, W, H)
    assert frag["_cut"]
    merged = merge_tiled_detections([whole], [frag], nms_iou=0.6)
    assert merged == [whole]


def test_same_object_in_full_and_tile_passes_counted_once() -> None:
    whole = {"label": "dog", "score": 0.45, "box_xyxy": [200.0, 500.0, 260.0, 540.0]}
    tile = Tile(0, 0, 1422, 1067)
    seen = to_frame_coords({"label": "dog", "score": 0.6, "box_xyxy": [201.0, 501.0, 259.0, 541.0]}, tile, W, H)
    merged = merge_tiled_detections([whole], [seen], nms_iou=0.6)
    assert len(merged) == 1 and merged[0]["score"] == pytest.approx(0.6)


def test_different_labels_never_merge() -> None:
    top, bottom = Tile(0, 0, 1422, 1067), Tile(0, 853, 1422, 1920)
    a = to_frame_coords({"label": "person", "score": 0.5, "box_xyxy": [600.0, 700.0, 680.0, 1066.0]}, top, W, H)
    b = to_frame_coords({"label": "dog", "score": 0.5, "box_xyxy": [600.0, 0.0, 680.0, 400.0]}, bottom, W, H)
    assert len(merge_tiled_detections([], [a, b], nms_iou=0.6)) == 2


# ---------------------------------------------------------------- request knobs


def test_tiling_requested_only_when_asked() -> None:
    assert tiling_requested({"detect_tiles": True})
    assert tiling_requested({"detect_tiles": {"grid": [3, 2]}})
    assert not tiling_requested({})
    assert not tiling_requested({"detect_tiles": False})
    assert not tiling_requested({"detect_tiles": {"enabled": False}})


def test_tiling_params_profile_defaults_and_request_override(vision_profiles) -> None:
    params = vision_profiles.get_profile("retina_detect_open_vocab").params
    shipped = tiling_params(params, {"detect_tiles": True})
    assert shipped["grid"] == list(params["tile_grid"])
    assert shipped["include_full_frame"] is True
    over = tiling_params(params, {"detect_tiles": {"grid": [3, 1], "overlap": 0.1}})
    assert over["grid"] == [3, 1] and over["overlap"] == 0.1


# ---------------------------------------------------------------- runner wiring

MIN_SEEN_PX = 25  # fake detector's "can resolve it" height after resize
FAR_BOX = (1500, 600, 1530, 650)  # 30x50 px "person" on the far sidewalk


def _fake_detect_images(model, processor, images, text, box_th, text_th, device, cast=None):
    out = []
    for im in images:
        scale = min(800 / min(im.size), 1333 / max(im.size))
        arr = np.asarray(im)
        red = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 50) & (arr[:, :, 2] < 50)
        ys, xs = np.nonzero(red)
        objs = []
        if len(xs) and (ys.max() - ys.min() + 1) * scale >= MIN_SEEN_PX:
            objs.append({"label": "person", "score": 0.6,
                         "box_xyxy": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]})
        out.append(objs)
    return out


@pytest.fixture
def detect_runner(vision_profiles, monkeypatch, tmp_path):
    frame = tmp_path / "walkway.png"
    img = Image.new("RGB", (W, H), (40, 120, 40))
    ImageDraw.Draw(img).rectangle([FAR_BOX[0], FAR_BOX[1], FAR_BOX[2] - 1, FAR_BOX[3] - 1], fill=(255, 0, 0))
    img.save(frame)
    r = VisionRunner(profiles=vision_profiles, enabled_names=["retina_detect_open_vocab"], cache_dir=tempfile.mkdtemp())
    calls = []

    def spy(*a, **k):
        calls.append([im.size for im in a[2]])
        return _fake_detect_images(*a, **k)

    monkeypatch.setattr(runner_mod, "detect_images", spy)
    monkeypatch.setattr(r.models, "load_grounding_dino", lambda **k: (object(), object()))
    return r, str(frame), calls


def _detect(runner, request):
    p = runner.profiles.get_profile("retina_detect_open_vocab")
    return runner._run_detect_grounding_dino(p, request, "cpu")


def test_far_small_object_found_by_tiles_not_single_pass(detect_runner) -> None:
    r, frame, calls = detect_runner
    single = _detect(r, {"image_path": frame, "stream_id": "cam0"})
    assert single["objects"] == []
    assert "detect_tiles" not in single
    assert len(calls) == 1  # one whole-frame pass, no tiles

    calls.clear()
    tiled = _detect(r, {"image_path": frame, "stream_id": "walkway", "detect_tiles": True})
    assert len(tiled["objects"]) == 1
    assert tiled["objects"][0]["box_xyxy"] == [1500.0, 600.0, 1530.0, 650.0]  # full-frame pixels
    assert tiled["detect_tiles"]["tiles"] == 4
    assert len(calls) == 1 + 4
    # Boxes stay in the full frame's pixel space for zones/crops/faces.
    assert (tiled["frame_width"], tiled["frame_height"]) == (W, H)


def test_tiled_detect_respects_max_detections(detect_runner, monkeypatch) -> None:
    r, frame, _ = detect_runner

    def many(model, processor, images, *a, **k):
        return [[{"label": "vehicle", "score": 0.4 + i * 0.001, "box_xyxy": [i * 20.0, 10.0, i * 20.0 + 15, 30.0]}
                 for i in range(40)] for _ in images]

    monkeypatch.setattr(runner_mod, "detect_images", many)
    out = _detect(r, {"image_path": frame, "detect_tiles": True})
    cap = int(r.profiles.get_profile("retina_detect_open_vocab").params["max_detections"])
    assert len(out["objects"]) <= cap


def test_small_frame_is_not_tiled(detect_runner, tmp_path) -> None:
    r, _, calls = detect_runner
    small = tmp_path / "cam0.png"
    Image.new("RGB", (640, 480)).save(small)
    out = _detect(r, {"image_path": str(small), "detect_tiles": True})
    assert out["detect_tiles"]["skipped"].startswith("short_side<")
    assert len(calls) == 1

"""The walkway cut-off calibration eval replays the live detect pipeline on
stored low-floor boxes. These tests pin that replay to the runner's order
(per-label cut-off -> tile merge -> size floor -> NMS -> cap) and pin the
recall / false-positive bookkeeping, so the committed table can be trusted.
The last test re-runs the committed labelled set end to end."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[1] / "evals" / "walkway_cutoff_calibration_eval.py"
_spec = importlib.util.spec_from_file_location("walkway_cutoff_calibration_eval", EVAL)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

PARAMS = {
    "score_threshold": 0.35,
    "box_threshold_by_stream": {"walkway": {"person": 0.35, "vehicle": 0.30}},
    "nms_iou": 0.6,
    "max_detections": 60,
    "tile_grid": [2, 1],
    "tile_overlap": 0.2,
    "tile_region": [0.0, 0.15, 1.0, 0.6],
    "tile_include_full_frame": True,
    "tile_contain_frac": 0.7,
    "tile_seam_merge_frac": 0.25,
}


def _box(label, score, x1, y1, x2, y2):
    return {"label": label, "score": score, "box_xyxy": [x1, y1, x2, y2]}


def _rec(full, tiles=None, frame="2026-11-01/walkway_23Z.jpg"):
    return {"frame": frame, "width": 2560, "height": 1920, "floor": 0.20,
            "full_raw": full, "tiles_raw": tiles or []}


def test_configured_cutoffs_are_per_label():
    th = ev.thresholds_for(PARAMS, None)
    assert th.for_label("vehicle") == 0.30 and th.for_label("person") == 0.35
    assert th.for_label("mail truck") == 0.35  # unlisted phrase: stream default
    flat = ev.thresholds_for(PARAMS, 0.25)
    assert flat.for_label("person") == 0.25 and flat.floor == 0.25
    mixed = ev.thresholds_for(PARAMS, {"vehicle": 0.28})
    assert mixed.for_label("vehicle") == 0.28 and mixed.for_label("person") == 0.35


def test_replay_applies_each_label_cutoff():
    rec = _rec([_box("vehicle", 0.31, 0, 0, 100, 100), _box("person", 0.33, 500, 500, 560, 700),
                _box("person", 0.50, 900, 500, 960, 700)])
    kept = ev.replay(rec, PARAMS, ev.thresholds_for(PARAMS, None), tiled=False)
    assert sorted((o["label"], o["score"]) for o in kept) == [("person", 0.5), ("vehicle", 0.31)]


def test_replay_refuses_cutoff_below_scored_floor():
    with pytest.raises(SystemExit):
        ev.replay(_rec([]), PARAMS, ev.thresholds_for(PARAMS, 0.15), tiled=False)


def test_tiled_replay_adds_tile_only_box_and_whole_ignores_it():
    tile_only = _box("person", 0.40, 2000, 400, 2040, 500)
    rec = _rec([_box("vehicle", 0.40, 0, 300, 300, 500)], [tile_only])
    th = ev.thresholds_for(PARAMS, None)
    assert [o["label"] for o in ev.replay(rec, PARAMS, th, tiled=False)] == ["vehicle"]
    assert sorted(o["label"] for o in ev.replay(rec, PARAMS, th, tiled=True)) == ["person", "vehicle"]


def test_match_uses_family_and_overlap():
    pool = [{"family": "vehicle", "box_xyxy": [0, 0, 100, 100], "box_id": 1},
            {"family": "person", "box_xyxy": [0, 0, 100, 100], "box_id": 2}]
    assert ev.match(_box("mail truck", 0.4, 5, 5, 95, 95), pool)["box_id"] == 1
    assert ev.match(_box("person", 0.4, 5, 5, 95, 95), pool)["box_id"] == 2
    assert ev.match(_box("vehicle", 0.4, 500, 500, 600, 600), pool) is None


def test_evaluate_counts_recall_false_and_missed():
    rec = _rec([_box("vehicle", 0.31, 0, 0, 100, 100),       # real car, under 0.35
                _box("vehicle", 0.40, 1000, 0, 1100, 100),   # real car
                _box("vehicle", 0.29, 2000, 0, 2100, 100)])  # phantom
    labels = {rec["frame"]: [
        {"box_id": 1, "family": "vehicle", "verdict": "real", "object": "car_a", "box_xyxy": [0, 0, 100, 100]},
        {"box_id": 2, "family": "vehicle", "verdict": "real", "object": "car_b", "box_xyxy": [1000, 0, 1100, 100]},
        {"box_id": 3, "family": "vehicle", "verdict": "false", "object": "", "box_xyxy": [2000, 0, 2100, 100]},
    ]}
    frames = {rec["frame"]: {"condition": "rain", "missed": {"vehicle": 1}}}
    at35 = ev.evaluate([rec], labels, frames, PARAMS, 0.35, tiled=False)[("rain", "vehicle")]
    assert (at35["objects"], at35["recalled"], at35["false"]) == (3, 1, 0)
    at28 = ev.evaluate([rec], labels, frames, PARAMS, 0.28, tiled=False)[("all", "vehicle")]
    assert (at28["objects"], at28["recalled"], at28["false"], at28["frames_seen"]) == (3, 2, 1, 1)


def test_committed_labelled_set_reports():
    """The committed eval data loads and the report runs end to end."""
    if not (ev.SCORED.exists() and ev.LABELS.exists() and ev.FRAMES.exists()):
        pytest.skip("labelled set not committed")
    scored = ev.load_scored(ev.SCORED)
    labels = ev.load_labels(ev.LABELS)
    frames = ev.load_frames(ev.FRAMES)
    assert set(frames) <= {r["frame"] for r in scored}
    assert {v["verdict"] for rows in labels.values() for v in rows} <= {"real", "false", "dup"}
    params = ev.detect_params()
    acc = ev.evaluate(scored, labels, frames, params, None, tiled=False)
    json.dumps({str(k): v for k, v in acc.items()})
    # The shipped walkway cut-offs must keep what the calibration found:
    # rain-day vehicles seen (11-01 read 0 of 90 windows at 0.35) and no
    # false vehicle or person box on the whole-frame pass.
    rain = acc[("rain", "vehicle")]
    assert rain["objects"] >= 20 and rain["recalled"] / rain["objects"] >= 0.95
    old = ev.evaluate(scored, labels, frames, params, 0.35, tiled=False)[("rain", "vehicle")]
    assert old["recalled"] / old["objects"] < 0.5  # the blindness the change fixes
    assert acc[("all", "vehicle")]["false"] == 0
    assert acc.get(("all", "person"), {}).get("false", 0) == 0


def test_replay_matches_real_runner(vision_profiles, monkeypatch, tmp_path):
    """Parity: the same raw boxes through the REAL runner (whole frame + tiles)
    and through replay() give the same kept boxes. If the runner's order ever
    changes (NMS before merge, a new floor), this fails instead of the eval's
    numbers quietly going stale."""
    import tempfile

    from PIL import Image

    import app.runner as runner_mod
    from app.runner import VisionRunner
    from app.tiling import tile_layout, tiling_params, to_frame_coords

    W, H = 2560, 1920
    frame = tmp_path / "walkway.jpg"
    Image.new("RGB", (W, H)).save(frame)
    r = VisionRunner(profiles=vision_profiles, enabled_names=["retina_detect_open_vocab"], cache_dir=tempfile.mkdtemp())
    p = r.profiles.get_profile("retina_detect_open_vocab")
    # Exercise the size floor too (walkway ships none): a 30 px person is dropped.
    p.params = {**p.params, "min_box_long_side_frac_by_stream": {"walkway": {"person": 0.1}}}
    params = p.params

    whole = [_box("vehicle", 0.27, 1690, 360, 1780, 425),     # kept: walkway vehicle 0.25
             _box("person", 0.30, 600, 700, 660, 900),         # dropped: person 0.35
             _box("person", 0.50, 1300, 600, 1380, 820),       # kept
             _box("person", 0.45, 100, 1500, 115, 1530),       # dropped: under the size floor
             _box("mail truck", 0.30, 200, 750, 830, 1150)]    # dropped: mail truck stays 0.35
    tiles = tile_layout(W, H, *[tiling_params(params, {"detect_tiles": True})[k] for k in ("grid", "overlap", "region")])
    assert len(tiles) == 2
    seam0 = tiles[0].x1 - tiles[0].x0  # right edge of tile 0, in tile 0 coords
    seam1 = tiles[0].x1 - tiles[1].x0  # same frame x, in tile 1 coords
    tile_boxes = [
        [_box("person", 0.42, seam0 - 40, 300, seam0, 520),            # seam piece, left
         _box("vehicle", 0.31, 1690 - tiles[0].x0, 72, 1780 - tiles[0].x0, 137)],  # fragment of the whole-frame car
        [_box("person", 0.40, seam1 - 60, 300, seam1 + 30, 520),       # seam piece, right
         _box("vehicle", 0.26, 900, 100, 960, 140)],                   # tile-only far car
    ]
    calls = []

    def fake(model, processor, images, text, box_th, text_th, device, cast=None):
        out = []
        for im in images:
            if im.size == (W, H):
                raw = whole
            else:
                raw = tile_boxes[len(calls)]
                calls.append(1)
            out.append([dict(o) for o in raw if o["score"] >= box_th])
        return out

    monkeypatch.setattr(runner_mod, "detect_images", fake)
    monkeypatch.setattr(r.models, "load_grounding_dino", lambda **k: (object(), object()))
    live = r._run_detect_grounding_dino(p, {"image_path": str(frame), "stream_id": "walkway", "detect_tiles": True}, "cpu")
    assert len(calls) == 2

    rec = {"frame": "x", "width": W, "height": H, "floor": 0.20, "text_threshold": 0.25,
           "full_raw": whole,
           "tiles_raw": [ev._detuple(json.loads(json.dumps(to_frame_coords(o, t, W, H))))
                         for t, objs in zip(tiles, tile_boxes) for o in objs]}
    replayed = ev.replay(rec, params, ev.thresholds_for(params, None), tiled=True)

    def key(objs):
        return sorted((o["label"], round(o["score"], 4), tuple(round(v, 1) for v in o["box_xyxy"])) for o in objs)

    assert key(live["objects"]) == key(replayed)
    labels = [o["label"] for o in replayed]
    assert "vehicle" in labels and "mail truck" not in labels
    assert not any(o["label"] == "person" and o["score"] in (0.30, 0.45) for o in replayed)


def test_report_refuses_cutoff_below_text_threshold():
    """Boxes under text_threshold have no label, so 0.20 must not be reported
    as if it were measured."""
    rec = _rec([])
    rec["text_threshold"] = 0.25
    with pytest.raises(SystemExit):
        ev.replay(rec, PARAMS, ev.thresholds_for(PARAMS, 0.22), tiled=False)
    assert min(ev.CUTOFFS) >= 0.25

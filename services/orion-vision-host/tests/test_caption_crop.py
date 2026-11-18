"""Walkway captions crop to the detector's boxes (2026-10-16).

Regression: the VLM read the 2560x1920 walkway frame shrunk to ~1 MP and
answered "nothing notable" on 24 of 24 caption tasks the detector had found a
person or car in. These tests pin that the model is shown the boxes' region,
not the whole frame, and that a missing or useless crop falls back to the
whole frame instead of failing the caption.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.caption_crop import MIN_SIDE_PX, crop_box_for, crop_for_caption

FRAME = (2560, 1920)
CAR = [1280, 454, 1408, 518]  # a parked car down the street, live 10-16 frame


def test_small_far_box_gets_a_padded_crop_at_least_min_side():
    box = crop_box_for(FRAME, [CAR], pad=0.5)
    x1, y1, x2, y2 = box
    assert x1 <= CAR[0] and y1 <= CAR[1] and x2 >= CAR[2] and y2 >= CAR[3]
    assert x2 - x1 >= MIN_SIDE_PX and y2 - y1 >= MIN_SIDE_PX
    # The crop is a small part of the frame, which is the point.
    assert (x2 - x1) * (y2 - y1) < 0.1 * FRAME[0] * FRAME[1]


def test_nearby_boxes_are_cropped_together():
    # A person and the dog beside them: one crop holds both.
    box = crop_box_for(FRAME, [[100, 200, 180, 420], [200, 350, 260, 420]], pad=0.5)
    assert box[0] <= 100 and box[2] >= 260


def test_far_apart_boxes_crop_to_the_top_box_only():
    # Person (first = highest score) at the left edge, the patio prop at the
    # right: the crop stays on the person, not a frame-wide union.
    person, prop = [100, 600, 180, 820], [2060, 877, 2274, 1227]
    x1, y1, x2, y2 = crop_box_for(FRAME, [person, prop], pad=0.5)
    assert x1 <= person[0] and x2 >= person[2]
    assert x2 < prop[0]


def test_crop_clamped_to_frame_edges():
    x1, y1, x2, y2 = crop_box_for(FRAME, [[0, 0, 40, 40]], pad=0.5)
    assert (x1, y1) == (0, 0)
    assert x2 - x1 >= MIN_SIDE_PX and y2 - y1 >= MIN_SIDE_PX
    x1, y1, x2, y2 = crop_box_for(FRAME, [[2540, 1900, 2560, 1920]], pad=0.5)
    assert (x2, y2) == FRAME


def test_crop_covering_most_of_the_frame_is_skipped():
    assert crop_box_for(FRAME, [[10, 10, 2550, 1910]], pad=0.0) is None


def test_invalid_boxes_mean_whole_frame():
    assert crop_box_for(FRAME, [], pad=0.5) is None
    assert crop_box_for(FRAME, [[5, 5, 5, 9], "x", [1, 2, 3], [None, 1, 2, 3]], pad=0.5) is None
    assert crop_box_for(FRAME, [[3000, 3000, 3100, 3100]], pad=0.0) is None


def test_crop_for_caption_returns_the_cropped_image():
    img = Image.new("RGB", FRAME)
    out, box = crop_for_caption(img, {"caption_crop_boxes": [CAR], "caption_crop_pad": 0.5})
    assert box is not None and out.size == (box[2] - box[0], box[3] - box[1])


def test_no_boxes_or_bad_pad_keeps_the_frame_or_default():
    img = Image.new("RGB", FRAME)
    out, box = crop_for_caption(img, {})
    assert out is img and box is None
    out, box = crop_for_caption(img, {"caption_crop_boxes": [CAR], "caption_crop_pad": "bad"})
    assert box is not None

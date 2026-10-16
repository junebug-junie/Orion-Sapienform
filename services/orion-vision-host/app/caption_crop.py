"""Crop a caption frame to the detector's boxes before the VLM sees it.

Why (live 2026-10-16): the walkway frame is 2560x1920. Qwen2-VL-2B reads it
at VISION_VLM_QWEN_MAX_PIXELS (~1 MP), so the frame shrinks ~2.2x per side and
a person or car down the street is a few dozen pixels. On those frames the
model answered "nothing notable" for every walkway caption task (10-15: 44 of
50; 10-16: 24 of 24), even when the detector had found a person or a car. A
crop around the detector's boxes keeps the subject at full resolution.

The router sends the boxes in ``request.caption_crop_boxes`` (frame pixel
coordinates, ``[x1, y1, x2, y2]``) and the padding in
``request.caption_crop_pad`` (a fraction of the boxes' union size).
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

DEFAULT_PAD = 0.5
# Qwen2-VL works in 28 px patches; below ~448 px a side it has little to read.
MIN_SIDE_PX = 448
# A crop this close to the full frame gains nothing; send the frame as is.
MAX_AREA_FRACTION = 0.8

Box = Tuple[float, float, float, float]


def _valid_boxes(boxes: Iterable[Any]) -> List[Box]:
    out: List[Box] = []
    for b in boxes or []:
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def _grow(lo: float, hi: float, need: float, limit: float) -> Tuple[float, float]:
    """Widen [lo, hi] around its centre to at least ``need``, inside [0, limit]."""
    if hi - lo >= need:
        return lo, hi
    need = min(need, limit)
    centre = (lo + hi) / 2.0
    lo, hi = centre - need / 2.0, centre + need / 2.0
    if lo < 0:
        lo, hi = 0.0, need
    if hi > limit:
        lo, hi = limit - need, limit
    return lo, hi


def crop_box_for(
    size: Tuple[int, int],
    boxes: Sequence[Any],
    *,
    pad: float = DEFAULT_PAD,
    min_side: int = MIN_SIDE_PX,
    max_area_fraction: float = MAX_AREA_FRACTION,
) -> Optional[Tuple[int, int, int, int]]:
    """The region to caption, or None to caption the whole frame.

    Union of the valid boxes, padded by ``pad`` x the union's larger side,
    grown to at least ``min_side`` per side, clamped to the frame. None when
    there is no valid box or the crop would cover most of the frame anyway.
    """
    width, height = size
    valid = _valid_boxes(boxes)
    if not valid or width <= 0 or height <= 0:
        return None
    x1 = min(b[0] for b in valid)
    y1 = min(b[1] for b in valid)
    x2 = max(b[2] for b in valid)
    y2 = max(b[3] for b in valid)
    margin = max(0.0, float(pad)) * max(x2 - x1, y2 - y1)
    x1, y1, x2, y2 = x1 - margin, y1 - margin, x2 + margin, y2 + margin
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        return None  # boxes entirely outside the frame
    x1, x2 = _grow(x1, x2, float(min_side), float(width))
    y1, y2 = _grow(y1, y2, float(min_side), float(height))
    box = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
    if (box[2] - box[0]) * (box[3] - box[1]) >= max_area_fraction * width * height:
        return None
    return box


def crop_for_caption(img: Image.Image, request: dict) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
    """``(image to caption, crop box used or None)`` for a caption request."""
    boxes = (request or {}).get("caption_crop_boxes")
    if not boxes:
        return img, None
    try:
        pad = float((request or {}).get("caption_crop_pad", DEFAULT_PAD))
    except (TypeError, ValueError):
        pad = DEFAULT_PAD
    box = crop_box_for(img.size, boxes, pad=pad)
    if box is None:
        return img, None
    return img.crop(box), box

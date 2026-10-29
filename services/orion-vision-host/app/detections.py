"""
Detection post-processing: per-camera size floor, non-maximum suppression and
score-ordered capping.

Pure Python and dependency-free so it is testable without torch/numpy, which
are container-only. Detection counts here are in the tens, so the O(n^2)
pairwise comparison is not worth vectorising.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)

    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0

    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(objects: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    """
    Per-label non-maximum suppression, highest score first.

    Open-vocabulary detectors emit many overlapping boxes for one physical
    object. Measured live on 2026-08-12 against a room holding exactly one
    desk and one table, GroundingDINO returned `desk` x6, `box` x5, `table` x4
    for a single frame. Nothing suppressed them, so every downstream consumer
    counted one desk as six and the generated narrative said "multiple desks".

    Suppression is per-label deliberately: a person standing in front of a door
    should not suppress the door. Cross-label duplication (`screen` vs
    `monitor` on one physical monitor) is a vocabulary problem, not an NMS one,
    and is handled by not shipping both terms.

    Returns a new list, ordered by descending score.
    """
    if iou_threshold <= 0:
        return sorted(objects, key=lambda o: o.get("score", 0.0), reverse=True)

    ordered = sorted(objects, key=lambda o: o.get("score", 0.0), reverse=True)
    kept: List[Dict[str, Any]] = []

    for candidate in ordered:
        box = candidate.get("box_xyxy") or []
        label = candidate.get("label")
        if len(box) != 4:
            kept.append(candidate)
            continue

        suppressed = any(
            k.get("label") == label
            and len(k.get("box_xyxy") or []) == 4
            and _iou(box, k["box_xyxy"]) >= iou_threshold
            for k in kept
        )
        if not suppressed:
            kept.append(candidate)

    return kept


def drop_undersized(
    objects: List[Dict[str, Any]],
    min_long_side_frac: Mapping[str, float],
    frame_width: int,
    frame_height: int,
    edge_px: float = 2.0,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Drop boxes too small to be a real object of their label on this camera.

    ``min_long_side_frac`` maps a label to the smallest allowed long side of
    its box, as a fraction of the frame's short side. Labels not in the map
    are untouched. A box touching the frame border is always kept: it is cut
    off by the edge, so its size says nothing (a head entering from the
    bottom of the room camera is a short, wide box).

    Why (2026-10-29): a framed photo of two people stands on the desk in the
    cam0 room. GroundingDINO labels both pictured people `person` (scores
    0.35-0.41, boxes 11-20 px wide x 15-31 px tall on the 640x480 frame).
    Live cam0 detections 2026-08-12..10-28 hold ~250k person boxes under
    32 px tall (the photo, at each spot it stood on the desk), 73 between 32
    and 68 px (mostly people cut off at the bottom edge), then the real
    people. Replayed, this rule takes the 2+ person windows of 10-17..10-28
    from 14,364 to ~61.

    Returns ``(kept, dropped_count)``; order is preserved.
    """
    if not min_long_side_frac:
        return list(objects), 0
    short_side = float(min(frame_width, frame_height))
    if short_side <= 0:
        return list(objects), 0

    kept: List[Dict[str, Any]] = []
    dropped = 0
    for o in objects:
        frac = min_long_side_frac.get(str(o.get("label")))
        box = o.get("box_xyxy") or []
        if frac is None or len(box) != 4:
            kept.append(o)
            continue
        x0, y0, x1, y1 = (float(v) for v in box)
        touches_edge = (
            x0 <= edge_px
            or y0 <= edge_px
            or x1 >= frame_width - edge_px
            or y1 >= frame_height - edge_px
        )
        if touches_edge or max(x1 - x0, y1 - y0) >= float(frac) * short_side:
            kept.append(o)
        else:
            dropped += 1
    return kept, dropped


def cap_by_score(objects: List[Dict[str, Any]], max_detections: int) -> List[Dict[str, Any]]:
    """
    Keep the highest-scoring `max_detections`.

    The previous truncation walked the post-processor's output in query order
    and broke at the cap, so which detections survived depended on prompt
    ordering rather than confidence.
    """
    if max_detections is None or max_detections <= 0:
        return list(objects)
    return sorted(objects, key=lambda o: o.get("score", 0.0), reverse=True)[:max_detections]

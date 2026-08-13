"""
Detection post-processing: non-maximum suppression and score-ordered capping.

Pure Python and dependency-free so it is testable without torch/numpy, which
are container-only. Detection counts here are in the tens, so the O(n^2)
pairwise comparison is not worth vectorising.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


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

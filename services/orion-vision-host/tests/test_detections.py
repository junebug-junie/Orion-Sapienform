from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.detections import _iou, cap_by_score, nms


def _obj(label: str, score: float, box: list[float]) -> dict:
    return {"label": label, "score": score, "box_xyxy": box}


def test_iou_hand_computed() -> None:
    # Two 10x10 boxes offset by 5 in x: intersection 5x10=50,
    # union 100+100-50=150, IoU = 1/3.
    assert abs(_iou([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-9


def test_iou_disjoint_is_zero() -> None:
    assert _iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_identical_is_one() -> None:
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_duplicate_boxes_for_one_object_collapse_to_one() -> None:
    """
    The live defect: one physical desk produced six overlapping `desk` boxes,
    so the generated narrative said "multiple desks".
    """
    objs = [
        _obj("desk", 0.70, [100, 100, 200, 200]),
        _obj("desk", 0.65, [102, 101, 201, 199]),
        _obj("desk", 0.60, [ 98,  99, 199, 202]),
    ]
    kept = nms(objs, 0.6)
    assert len(kept) == 1
    assert kept[0]["score"] == 0.70, "must keep the highest-scoring box"


def test_distinct_objects_of_same_label_are_both_kept() -> None:
    # Two genuinely separate chairs must survive; suppression is about overlap,
    # not about collapsing a label to one instance.
    objs = [
        _obj("chair", 0.8, [0, 0, 10, 10]),
        _obj("chair", 0.7, [100, 100, 110, 110]),
    ]
    assert len(nms(objs, 0.6)) == 2


def test_suppression_is_per_label() -> None:
    # A person in front of a door overlaps it heavily; the door must survive.
    objs = [
        _obj("person", 0.9, [10, 10, 50, 90]),
        _obj("door", 0.6, [8, 5, 55, 95]),
    ]
    assert len(nms(objs, 0.6)) == 2


def test_overlap_below_threshold_is_not_suppressed() -> None:
    # IoU = 1/3 from the hand-computed case above, under a 0.6 threshold.
    objs = [
        _obj("box", 0.9, [0, 0, 10, 10]),
        _obj("box", 0.8, [5, 0, 15, 10]),
    ]
    assert len(nms(objs, 0.6)) == 2


def test_zero_threshold_disables_suppression_but_still_sorts() -> None:
    objs = [
        _obj("desk", 0.5, [0, 0, 10, 10]),
        _obj("desk", 0.9, [0, 0, 10, 10]),
    ]
    kept = nms(objs, 0.0)
    assert len(kept) == 2
    assert [o["score"] for o in kept] == [0.9, 0.5]


def test_malformed_box_is_kept_rather_than_dropped() -> None:
    objs = [_obj("desk", 0.5, [0, 0]), _obj("desk", 0.9, [0, 0, 10, 10])]
    assert len(nms(objs, 0.6)) == 2


def test_cap_keeps_highest_scores_not_query_order() -> None:
    """
    The previous truncation broke at the cap while walking the post-processor's
    query order, so which detections survived depended on prompt ordering
    rather than confidence.
    """
    objs = [
        _obj("cable", 0.26, [0, 0, 1, 1]),
        _obj("book", 0.27, [2, 2, 3, 3]),
        _obj("person", 0.95, [4, 4, 5, 5]),
    ]
    kept = cap_by_score(objs, 2)
    assert [o["label"] for o in kept] == ["person", "book"]


def test_cap_of_zero_or_none_is_a_passthrough() -> None:
    objs = [_obj("desk", 0.5, [0, 0, 1, 1])]
    assert len(cap_by_score(objs, 0)) == 1
    assert len(cap_by_score(objs, None)) == 1


def test_nms_does_not_mutate_input() -> None:
    objs = [_obj("desk", 0.7, [0, 0, 10, 10]), _obj("desk", 0.6, [0, 0, 10, 10])]
    nms(objs, 0.6)
    assert len(objs) == 2

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.activity import ActivityRateLimiter, labels_from_detections


def test_labels_from_detections_person_and_motion() -> None:
    detections = [
        {"kind": "yolo", "label": "person", "score": 0.9},
        {"kind": "motion", "label": "motion", "score": 1.0},
    ]
    assert labels_from_detections(detections) == ["person", "motion"]


def test_rate_limiter_blocks_duplicate_within_one_second() -> None:
    limiter = ActivityRateLimiter(min_interval_s=1.0)
    assert limiter.allow("cam0", "person", now=100.0) is True
    assert limiter.allow("cam0", "person", now=100.5) is False
    assert limiter.allow("cam0", "person", now=101.1) is True

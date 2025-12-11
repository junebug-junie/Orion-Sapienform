# app/utils.py
import time
from typing import List, Tuple

import cv2

from .settings import get_settings

settings = get_settings()

BBox = Tuple[int, int, int, int]


def draw_boxes(frame, boxes: List[BBox], label: str):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def encode_jpeg(frame):
    quality = settings.JPEG_QUALITY
    ok, buf = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        return None
    return buf.tobytes()


def now_ts():
    return int(time.time() * 1000)


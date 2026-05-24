from __future__ import annotations

import os
import re
import time
import uuid

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel


class SavedFrame(BaseModel):
    image_path: str
    width: int
    height: int
    format: str = "jpg"
    bytes_written: int | None = None


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def ensure_frame_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_frame(
    frame: np.ndarray,
    *,
    directory: str,
    camera_id: str,
    stream_id: str,
    ts: float,
    quality: int,
) -> SavedFrame:
    ensure_frame_dir(directory)
    cam = _sanitize_id(camera_id)
    stream = _sanitize_id(stream_id)
    epoch_ms = int(ts * 1000)
    short = uuid.uuid4().hex[:8]
    filename = f"{cam}_{stream}_{epoch_ms}_{short}.jpg"
    image_path = os.path.join(directory, filename)
    ok = cv2.imwrite(image_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise OSError(f"cv2.imwrite failed for {image_path}")
    h, w = frame.shape[:2]
    size = os.path.getsize(image_path)
    return SavedFrame(
        image_path=image_path,
        width=w,
        height=h,
        format="jpg",
        bytes_written=size,
    )


def cleanup_old_frames(directory: str, max_age_seconds: int) -> int:
    if not os.path.isdir(directory):
        return 0
    now = time.time()
    removed = 0
    for name in os.listdir(directory):
        fp = os.path.join(directory, name)
        if not os.path.isfile(fp):
            continue
        if now - os.path.getmtime(fp) > max_age_seconds:
            try:
                os.remove(fp)
                removed += 1
            except OSError as exc:
                logger.debug(f"[RETINA] cleanup skip {fp}: {exc}")
    return removed

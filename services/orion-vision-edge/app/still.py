"""One full-resolution still from the camera's main stream, on demand (2026-10-11).

The capture loop reads the camera's small substream (640x480 on cam0). At that size Juniper's
face at the desk is about 25 px wide, and the face model works from 160 px: five of her own
frames scored 0.09-0.18 against each other, while a back-of-the-head frame scored 0.34-0.38. So
the frame router asks for one still from the main stream (2560x1920 on cam0) right before a face
check, and only then. Opening the stream costs 2-5 s, which is why nothing reads it continuously.

The stream URL carries the camera password, so it stays in this service: callers get a file path
in the shared frame directory, which the capture loop's own retention already cleans up.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Callable, Dict, Optional

import cv2

OPEN_TIMEOUT_MS = 6000
READ_TIMEOUT_MS = 6000
MAX_READS = 5   # the first frames after opening can fail while the decoder waits for a keyframe


def _open(url: str):
    return cv2.VideoCapture(
        url,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS],
    )


def grab_still(
    url: str,
    out_dir: str,
    *,
    quality: int = 92,
    opener: Callable[[str], Any] = _open,
    clock: Callable[[], float] = time.time,
) -> Optional[Dict[str, Any]]:
    """Grab one frame from ``url`` and write it into ``out_dir``. None when no frame came back."""
    started = clock()
    cap = opener(url)
    try:
        if not cap.isOpened():
            return None
        frame = None
        for _ in range(MAX_READS):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            frame = None
        if frame is None:
            return None
    finally:
        cap.release()
    ts = clock()
    os.makedirs(out_dir, exist_ok=True)
    name = f"still_{int(ts * 1000)}_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(out_dir, name)
    tmp = os.path.join(out_dir, f".{name}.tmp")
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    # Write then rename: vision-host reads this path moments later and must never see half a JPEG.
    with open(tmp, "wb") as fh:
        fh.write(buf.tobytes())
    os.replace(tmp, path)
    return {
        "image_path": path,
        "frame_ts": ts,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "grab_ms": int((ts - started) * 1000),
    }

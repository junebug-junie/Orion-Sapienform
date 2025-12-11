# app/detector_worker.py
import asyncio
from datetime import datetime
from typing import List, Tuple

import numpy as np

from .context import settings, bus, camera, detectors
from .schemas import Detection, Event
from .state import state
from .utils import draw_boxes


def run_detectors_on_frame(frame) -> Tuple[np.ndarray, List[Detection]]:
    """
    Run all configured detectors on a single frame.

    Returns:
        annotated_frame (BGR np.ndarray),
        detections: List[Detection]
    """
    annotated = frame.copy()
    detections: List[Detection] = []

    # 1) Frame-based detectors: motion, face, yolo, etc.
    for name, mode, det in detectors:
        if mode != "frame":
            continue

        try:
            boxes = det.detect(frame)
        except Exception:
            boxes = []

        if boxes and settings.ANNOTATE:
            # YOLO returns 6-tuple; others 4-tuple
            if boxes and isinstance(boxes[0], tuple) and len(boxes[0]) == 6:
                draw_boxes(
                    annotated,
                    [(x, y, w, h) for (x, y, w, h, _, _) in boxes],
                    name,
                )
            else:
                draw_boxes(annotated, boxes, name)

        for b in boxes:
            if isinstance(b, tuple) and len(b) == 6:
                x, y, w, h, label, conf = b
                detections.append(
                    Detection(
                        kind=name,
                        bbox=(x, y, w, h),
                        score=float(conf),
                        label=label,
                    )
                )
            else:
                x, y, w, h = b
                detections.append(
                    Detection(
                        kind=name,
                        bbox=(x, y, w, h),
                        score=1.0,
                    )
                )

    # 2) Event-based detectors: presence, future identity/emotion, etc.
    for name, mode, det in detectors:
        if mode != "event":
            continue

        try:
            events = det.update([d.model_dump() for d in detections])
        except Exception:
            events = []

        for ev in events:
            detections.append(
                Detection(
                    kind=ev.get("kind", name),
                    bbox=(0, 0, 0, 0),
                    score=ev.get("score", 1.0),
                    label=ev.get("label"),
                    meta={
                        k: v
                        for k, v in ev.items()
                        if k not in ("kind", "label", "score")
                    },
                )
            )

    return annotated, detections


def build_event_from_detections(
    detections: List[Detection],
    frame_index: int,
) -> Event:
    return Event(
        ts=datetime.utcnow(),
        stream_id=settings.STREAM_ID,
        frame_index=frame_index,
        detections=detections,
        meta={"source": "edge", "camera": settings.SOURCE},
    )


async def run_detector_loop() -> None:
    """
    Background loop:

    - pulls frames from camera
    - runs detector pipeline
    - publishes raw events to Orion bus
    - updates latest annotated frame for UI
    """
    while True:
        frame = camera.get_frame()
        await asyncio.sleep(0.001)

        if frame is None:
            await asyncio.sleep(0.05)
            continue

        state.frame_counter += 1
        if state.frame_counter % max(settings.DETECT_EVERY_N_FRAMES, 1) != 0:
            continue

        annotated, detections = run_detectors_on_frame(frame)
        if not detections:
            continue

        ev = build_event_from_detections(detections, state.frame_counter)
        state.last_event = ev

        ev_dict = ev.model_dump()
        ev_dict["ts"] = ev.ts.isoformat()

        # Raw edge events channel
        bus.publish(settings.VISION_EVENTS_PUBLISH_RAW, ev_dict)

        # Update last annotated frame for MJPEG/snapshot
        camera.last_frame = annotated

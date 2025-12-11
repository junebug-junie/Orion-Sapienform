# app/detectors/__init__.py
from typing import Dict, List, Tuple

from ..settings import get_settings
from .motion import MotionDetector
from .face import FaceDetector
from .yolo import YoloDetector
from .presence import PresenceDetector

settings = get_settings()

# A detector always has:
#   - name (string)
#   - mode: "frame" or "event"
#   - callable:
#       frame-mode: detect(frame) -> list of detections/boxes
#       event-mode: update(detections) -> list of extra Detection-like dicts

DetectorSpec = Tuple[str, str, object]  # (name, mode, instance)


def build_detectors() -> List[DetectorSpec]:
    specs: List[DetectorSpec] = []
    names = set(settings.detector_names)

    if "motion" in names:
        specs.append(("motion", "frame", MotionDetector(min_area=settings.MOTION_MIN_AREA)))

    if "face" in names:
        w, h = settings.face_min_size_tuple
        specs.append(
            (
                "face",
                "frame",
                FaceDetector(
                    cascade_path=settings.FACE_CASCADE_PATH,
                    scale_factor=settings.FACE_SCALE_FACTOR,
                    min_neighbors=settings.FACE_MIN_NEIGHBORS,
                    min_size=min(w, h),
                ),
            )
        )

    if "yolo" in names and settings.ENABLE_YOLO:
        classes = settings.yolo_class_set or None
        # if YOLO_DEVICE is a digit, treat as cuda index, else use as-is
        device = f"cuda:{settings.YOLO_DEVICE}" if settings.YOLO_DEVICE.isdigit() else settings.YOLO_DEVICE
        specs.append(
            (
                "yolo",
                "frame",
                YoloDetector(
                    model_path=settings.YOLO_MODEL,
                    classes=classes,
                    conf=settings.YOLO_CONF,
                    device=device,
                ),
            )
        )

    if "presence" in names and settings.ENABLE_PRESENCE:
        specs.append(
            (
                "presence",
                "event",
                PresenceDetector(
                    timeout=settings.PRESENCE_TIMEOUT,
                    label=settings.PRESENCE_LABEL,
                ),
            )
        )

    return specs

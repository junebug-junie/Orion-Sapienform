from pydantic_settings import BaseSettings
from pydantic import AnyUrl
from typing import Optional

class Settings(BaseSettings):
    # Video source: e.g., "/dev/video0", "rtsp://user:pass@host:554/stream", "rtmp://..."
    SOURCE: str = "/dev/video0"
    WIDTH: int = 1280
    HEIGHT: int = 720
    FPS: int = 15

    # Comma-separated detector list: "motion,face"
    DETECTORS: str = "motion"

    # Minimum area (px) to report motion
    MOTION_MIN_AREA: int = 2000

    # Face detector scale/params
    FACE_SCALE_FACTOR: float = 1.1
    FACE_MIN_NEIGHBORS: int = 5
    FACE_MIN_SIZE: int = 30

    ENABLE_PRESENCE: bool = False
    PRESENCE_TIMEOUT: int = 60
    PRESENCE_LABEL: str = "Juniper"

    # Stream identifier (for events)
    STREAM_ID: str = "default"

    # Redis URL for publishing events (pub only; consumers can subscribe separately)
    REDIS_URL: Optional[str] = None  # e.g., "redis://redis:6379/0"

    # Optional HTTP webhook to receive events
    EVENT_WEBHOOK_URL: Optional[AnyUrl] = None

    # Throttle detection to every N frames (>=1)
    DETECT_EVERY_N_FRAMES: int = 2

    # Whether to draw annotations on frames
    ANNOTATE: bool = True

    # JPEG encoding quality (1-100)
    JPEG_QUALITY: int = 80

    # Expose basic web UI (index.html)
    ENABLE_UI: bool = True

settings = Settings()


# YOLO (GPU-friendly) toggles
ENABLE_YOLO: bool = False
YOLO_MODEL: str = "yolov8n.pt"   # file in /app/models or hub path
YOLO_CLASSES: str = "person"     # comma-separated names; leave empty for all
YOLO_CONF: float = 0.25
YOLO_DEVICE: str = "0"           # "0" (GPU0), "cuda:0", or "cpu"

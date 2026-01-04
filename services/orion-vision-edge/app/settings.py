# app/settings.py
from functools import lru_cache
from typing import List, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Vision edge service settings.

    - Reads from .env / environment
    - Strong types for everything critical
    - Drops old BRAIN_URL / LLM_MODEL coupling
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Service metadata ---
    SERVICE_NAME: str = "vision-edge"
    SERVICE_VERSION: str = "0.2.0"

    # --- Camera source ---
    SOURCE: str  # e.g. rtsp://USER:PASS@host:554/Preview_01_sub
    STREAM_ID: str = "cam0"

    WIDTH: int = 640
    HEIGHT: int = 360
    FPS: int = 15

    # Storage for "Pointer" architecture (shared frame paths)
    FRAME_STORAGE_DIR: str = "/mnt/frames"
    FRAME_RETENTION_SECONDS: int = 60

    # --- Detectors & sampling ---
    DETECTORS: str = "motion,face,yolo,presence"
    DETECT_EVERY_N_FRAMES: int = 10

    # Motion
    MOTION_MIN_AREA: int = 2000

    # Face
    FACE_CASCADE_PATH: str = "/app/haar/haarcascade_frontalface_default.xml"
    FACE_SCALE_FACTOR: float = 1.1
    FACE_MIN_NEIGHBORS: int = 5
    FACE_MIN_SIZE: str = "30,30"  # parsed into tuple below

    # Presence
    ENABLE_PRESENCE: bool = True
    PRESENCE_TIMEOUT: int = 60
    PRESENCE_LABEL: str = "Juniper"

    # YOLO
    ENABLE_YOLO: bool = True
    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_CLASSES: str = "person"  # comma-separated
    YOLO_CONF: float = 0.25
    YOLO_CONF_THRES: float = 0.25 # explicit new var
    YOLO_IOU_THRES: float = 0.45
    YOLO_IMG_SIZE: int = 640
    YOLO_DEVICE: str = "0"  # "0", "1", "cuda:0", "cpu", etc.
    YOLO_PERSON_RETRY_THRESHOLD: float = 0.15 # for "person-centric" retry

    # Debug / Telemetry
    EDGE_DEBUG_SAVE_FRAMES: bool = False
    EDGE_DEBUG_DIR: str = "/mnt/debug"
    EDGE_DEBUG_SAMPLE_RATE: int = 100 # save 1 in N

    # UI / JPEG
    ENABLE_UI: bool = True
    ANNOTATE: bool = True  # draw boxes / labels on frames
    JPEG_QUALITY: int = 90

    # --- Bus integration (bus-first, no brain HTTP) ---
    ORION_BUS_ENABLED: bool = True
    ORION_BUS_URL: str = "redis://100.92.216.81:6379/0"

    # --- Bus Channels
    VISION_EVENTS_PUBLISH_RAW: str = "orion:vision:edge:raw" # Deprecated but kept for legacy consumers
    CHANNEL_VISION_FRAMES: str = "vision.frames" # Pointer channel
    CHANNEL_VISION_ARTIFACTS: str = "vision.artifacts" # Edge Artifact channel
    VISION_EVENTS_PUBLISH_NOTABLE: str = "orion:vision:edge:event:notable"

    # ------ helpers / derived properties ------

    @property
    def detector_names(self) -> List[str]:
        return [d.strip() for d in self.DETECTORS.split(",") if d.strip()]

    @property
    def face_min_size_tuple(self) -> Tuple[int, int]:
        # we keep raw FACE_MIN_SIZE for env parity, and expose parsed tuple here
        x, y = self.FACE_MIN_SIZE.split(",")
        return int(x), int(y)

    @property
    def yolo_class_set(self):
        return {c.strip().lower() for c in self.YOLO_CLASSES.split(",") if c.strip()}

    @field_validator("JPEG_QUALITY")
    @classmethod
    def clamp_jpeg_quality(cls, v: int) -> int:
        return max(10, min(100, v))


@lru_cache
def get_settings() -> Settings:
    return Settings()

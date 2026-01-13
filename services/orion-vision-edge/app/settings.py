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
    SERVICE_NAME: str = Field("vision-edge", alias="SERVICE_NAME")
    SERVICE_VERSION: str = Field("0.2.0", alias="SERVICE_VERSION")

    # --- Camera source ---
    SOURCE: str = Field(..., alias="SOURCE") # e.g. rtsp://USER:PASS@host:554/Preview_01_sub
    STREAM_ID: str = Field("cam0", alias="STREAM_ID")

    WIDTH: int = Field(640, alias="WIDTH")
    HEIGHT: int = Field(360, alias="HEIGHT")
    FPS: int = Field(15, alias="FPS")

    # Storage for "Pointer" architecture (shared frame paths)
    FRAME_STORAGE_DIR: str = Field("/mnt/frames", alias="FRAME_STORAGE_DIR")
    FRAME_RETENTION_SECONDS: int = Field(60, alias="FRAME_RETENTION_SECONDS")

    # --- Detectors & sampling ---
    DETECTORS: str = Field("motion,face,yolo,presence", alias="DETECTORS")
    DETECT_EVERY_N_FRAMES: int = Field(10, alias="DETECT_EVERY_N_FRAMES")

    # Motion
    MOTION_MIN_AREA: int = Field(2000, alias="MOTION_MIN_AREA")

    # Face
    FACE_CASCADE_PATH: str = Field("/app/haar/haarcascade_frontalface_default.xml", alias="FACE_CASCADE_PATH")
    FACE_SCALE_FACTOR: float = Field(1.1, alias="FACE_SCALE_FACTOR")
    FACE_MIN_NEIGHBORS: int = Field(5, alias="FACE_MIN_NEIGHBORS")
    FACE_MIN_SIZE: str = Field("30,30", alias="FACE_MIN_SIZE")

    # Presence
    ENABLE_PRESENCE: bool = Field(True, alias="ENABLE_PRESENCE")
    PRESENCE_TIMEOUT: int = Field(60, alias="PRESENCE_TIMEOUT")
    PRESENCE_LABEL: str = Field("Juniper", alias="PRESENCE_LABEL")

    # YOLO
    ENABLE_YOLO: bool = Field(True, alias="ENABLE_YOLO")
    YOLO_MODEL: str = Field("yolov8n.pt", alias="YOLO_MODEL")
    YOLO_CLASSES: str = Field("person", alias="YOLO_CLASSES")
    YOLO_CONF: float = Field(0.25, alias="YOLO_CONF")
    YOLO_CONF_THRES: float = Field(0.25, alias="YOLO_CONF_THRES")
    YOLO_IOU_THRES: float = Field(0.45, alias="YOLO_IOU_THRES")
    YOLO_IMG_SIZE: int = Field(640, alias="YOLO_IMG_SIZE")
    YOLO_DEVICE: str = Field("0", alias="YOLO_DEVICE")
    YOLO_PERSON_RETRY_THRESHOLD: float = Field(0.15, alias="YOLO_PERSON_RETRY_THRESHOLD")

    # Debug / Telemetry
    EDGE_DEBUG_SAVE_FRAMES: bool = Field(False, alias="EDGE_DEBUG_SAVE_FRAMES")
    EDGE_DEBUG_DIR: str = Field("/mnt/debug", alias="EDGE_DEBUG_DIR")
    EDGE_DEBUG_SAMPLE_RATE: int = Field(100, alias="EDGE_DEBUG_SAMPLE_RATE")

    # UI / JPEG
    ENABLE_UI: bool = Field(True, alias="ENABLE_UI")
    ANNOTATE: bool = Field(True, alias="ANNOTATE")
    JPEG_QUALITY: int = Field(90, alias="JPEG_QUALITY")

    # --- Bus integration ---
    ORION_BUS_ENABLED: bool = Field(True, alias="ORION_BUS_ENABLED")
    ORION_BUS_ENFORCE_CATALOG: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    ORION_BUS_URL: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")

    # --- Bus Channels
    VISION_EVENTS_PUBLISH_RAW: str = Field("orion:vision:edge:raw", alias="VISION_EVENTS_PUBLISH_RAW")
    VISION_EVENTS_SUBSCRIBE_RAW: str = Field("orion:vision:edge:raw", alias="VISION_EVENTS_SUBSCRIBE_RAW")
    CHANNEL_VISION_FRAMES: str = Field("orion:vision:frames", alias="CHANNEL_VISION_FRAMES")
    CHANNEL_VISION_ARTIFACTS: str = Field("orion:vision:artifacts", alias="CHANNEL_VISION_ARTIFACTS")
    VISION_EVENTS_PUBLISH_NOTABLE: str = Field("orion:vision:edge:event:notable", alias="VISION_EVENTS_PUBLISH_NOTABLE")
    CHANNEL_VISION_EDGE_HEALTH: str = Field("orion:vision:edge:health", alias="CHANNEL_VISION_EDGE_HEALTH")
    CHANNEL_VISION_EDGE_ERROR: str = Field("orion:vision:edge:error", alias="CHANNEL_VISION_EDGE_ERROR")

    @property
    def detector_names(self) -> List[str]:
        return [d.strip() for d in self.DETECTORS.split(",") if d.strip()]

    @property
    def face_min_size_tuple(self) -> Tuple[int, int]:
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

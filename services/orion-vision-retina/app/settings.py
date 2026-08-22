from __future__ import annotations

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "vision-retina"
    SERVICE_VERSION: str = "0.2.0"
    LOG_LEVEL: str = "INFO"

    ORION_BUS_URL: str = "redis://localhost:6379/0"
    ORION_BUS_ENFORCE_CATALOG: bool = False

    CHANNEL_RETINA_PUB: str = "orion:vision:frames"
    CHANNEL_RETINA_ERROR: str = "orion:vision:retina:error"
    CHANNEL_SYSTEM_HEALTH: str = "orion:system:health"

    RETINA_SOURCE_TYPE: str = "folder"
    RETINA_SOURCE: str = "/mnt/telemetry/vision/intake"
    RETINA_SOURCE_PATH: str | None = None
    RETINA_CAMERA_ID: str = "retina-cam-01"
    RETINA_STREAM_ID: str = "retina-stream-01"

    RETINA_FPS: float = 1.0
    RETINA_WIDTH: int | None = None
    RETINA_HEIGHT: int | None = None

    FRAME_STORAGE_DIR: str = "/mnt/telemetry/vision/frames"

    # How this node addresses the frames it publishes.
    #   local         -- write to FRAME_STORAGE_DIR, publish image_path.
    #                    Correct ONLY when the vision host shares this
    #                    filesystem. athena. Default, so nothing changes.
    #   percept_store -- encode in memory, POST to orion-percept-store, publish
    #                    sha256. The only option for a node with no shared
    #                    disk. Nothing is written locally, on purpose: a
    #                    capture agent on a personal laptop must not accumulate
    #                    a spool of webcam frames.
    RETINA_FRAME_MODE: str = "local"
    RETINA_PERCEPT_STORE_URL: str = ""
    RETINA_PERCEPT_STORE_TOKEN: str = ""
    RETINA_PERCEPT_TIMEOUT_SEC: float = 10.0
    FRAME_RETENTION_SECONDS: int = 300
    JPEG_QUALITY: int = 90

    HEALTH_INTERVAL_SECONDS: float = 10.0
    SOURCE_RECONNECT_SECONDS: float = 5.0

    # On-demand video+audio clip capture for AffectGPT (app/clip_capture.py).
    # Separate from RETINA_SOURCE/RETINA_CAMERA_ID above -- those drive the
    # existing single-frame capture_loop, this is a distinct on-demand path
    # triggered via POST /capture/clip. UNVERIFIED against real hardware
    # (see clip_capture.py module docstring) -- defaults are a starting
    # point, not confirmed-correct for carbon's actual devices.
    RETINA_CLIP_ENABLED: bool = False
    # Shared-secret gate for POST /capture/clip, same convention as
    # orion-percept-store's PERCEPT_STORE_TOKEN. Added after review
    # (2026-08-22): docs/operations/carbon-webcam.md's whole documented
    # security posture is "nothing needs to reach carbon inbound" -- true
    # when this app's HTTP surface was health-only, no longer true once a
    # POST here triggers a live webcam+mic recording. Empty disables the
    # check (acceptable only on a closed tailnet, same caveat as percept
    # store) -- strongly set this one given what the endpoint actually does.
    RETINA_CLIP_TOKEN: str = ""
    RETINA_CLIP_FFMPEG_BIN: str = "ffmpeg"
    RETINA_CLIP_VIDEO_DEVICE: str = "/dev/video0"
    # PipeWire's pulse-compatible socket (the modern default on most desktop
    # Linux) accepts "default" as the default source name, same as plain
    # PulseAudio. Override if carbon's audio backend names sources differently.
    RETINA_CLIP_AUDIO_INPUT: str = "default"
    RETINA_CLIP_DURATION_SEC: float = 8.0
    RETINA_CLIP_FRAMERATE: int = 15
    RETINA_CLIP_WIDTH: int | None = None
    RETINA_CLIP_HEIGHT: int | None = None
    RETINA_CLIP_TIMEOUT_SEC: float = 30.0

    @field_validator("RETINA_CLIP_WIDTH", "RETINA_CLIP_HEIGHT", mode="before")
    @classmethod
    def _blank_clip_dims_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("JPEG_QUALITY", mode="before")
    @classmethod
    def _clamp_jpeg_quality(cls, v: object) -> int:
        try:
            n = int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 90
        return max(1, min(100, n))

    @field_validator("RETINA_WIDTH", "RETINA_HEIGHT", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @model_validator(mode="after")
    def _apply_source_path_alias(self) -> "Settings":
        if self.RETINA_SOURCE_PATH and "RETINA_SOURCE" not in self.model_fields_set:
            self.RETINA_SOURCE = self.RETINA_SOURCE_PATH
        return self


def get_settings() -> Settings:
    return Settings()

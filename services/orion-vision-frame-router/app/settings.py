from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "vision-frame-router"
    SERVICE_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    ORION_BUS_URL: str = "redis://localhost:6379/0"
    ORION_BUS_ENFORCE_CATALOG: bool = False

    CHANNEL_FRAMES_IN: str = "orion:vision:frames"
    CHANNEL_HOST_INTAKE: str = "orion:exec:request:VisionHostService"
    CHANNEL_HOST_ARTIFACTS: str = "orion:vision:artifacts"
    CHANNEL_REPLY_PREFIX: str = "orion:vision:reply"
    CHANNEL_SYSTEM_HEALTH: str = "orion:system:health"

    ROUTER_ENABLED: bool = True
    ROUTER_POLICY_PATH: str = "/app/config/vision_frame_router.yaml"

    DEFAULT_TASK_TYPE: str = "retina_fast"
    DEFAULT_EVERY_N_FRAMES: int = 10
    DEFAULT_MIN_SECONDS_PER_CAMERA: float = 5.0

    MAX_INFLIGHT_TOTAL: int = 2
    MAX_INFLIGHT_PER_CAMERA: int = 1
    TASK_TIMEOUT_SECONDS: float = 30.0

    REQUIRE_IMAGE_PATH_EXISTS: bool = True
    DRY_RUN: bool = False

    HEALTH_INTERVAL_SECONDS: float = 10.0

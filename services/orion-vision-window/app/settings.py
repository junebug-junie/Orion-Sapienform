from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "vision-window"
    SERVICE_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    # Bus
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    ORION_BUS_ENFORCE_CATALOG: bool = False

    # Channels
    CHANNEL_WINDOW_INTAKE: str = "orion:vision:artifacts"
    CHANNEL_WINDOW_PUB: str = "orion:vision:windows"
    CHANNEL_WINDOW_REQUEST: str = "orion:exec:request:VisionWindowService"

    # Rolling window (legacy WINDOW_SIZE_SEC kept as max wall clock span for a batch)
    WINDOW_SIZE_SEC: float = 30.0
    FLUSH_INTERVAL_MS: int = 5_000
    MAX_ARTIFACTS_PER_WINDOW: int = 64
    MAX_WINDOW_AGE_MS: int = 60_000
    STALE_AFTER_MS: int = 120_000

    # HTTP
    HTTP_HOST: str = "0.0.0.0"
    HTTP_PORT: int = 8000

    # Bounded recovery (§4.3) — same Redis URL as bus by default; dedicated URL optional
    VISION_WINDOW_RECOVERY_ENABLED: bool = True
    VISION_WINDOW_RECOVERY_REDIS_URL: str = ""
    VISION_WINDOW_RECOVERY_TTL_SEC: int = 3_600
    VISION_WINDOW_RECOVERY_MAX_N: int = 50
    VISION_WINDOW_HTTP_MAX_LIMIT: int = 50
    VISION_WINDOW_READY_REQUIRES_RECOVERY: bool = False

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    SERVICE_NAME: str = Field(default="orion-biometrics")
    SERVICE_VERSION: str = Field(default="0.1.0")
    NODE_NAME: str = Field(default="unknown")

    # Bus
    ORION_BUS_URL: str = Field(default="redis://orion-redis:6379/0")
    ORION_BUS_ENABLED: bool = Field(default=True)
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False)

    # Channels
    TELEMETRY_PUBLISH_CHANNEL: str = Field(default="orion:biometrics:telemetry")

    # Behavior
    TELEMETRY_INTERVAL: int = Field(default=30)
    LOG_LEVEL: str = Field(default="INFO")

    TABLE_NAME: str = Field(default="biometrics_raw")

    # Chassis Defaults
    HEARTBEAT_INTERVAL_SEC: float = 10.0
    ORION_HEALTH_CHANNEL: str = "orion:system:health"
    ERROR_CHANNEL: str = "system.error"
    SHUTDOWN_GRACE_SEC: float = 10.0

settings = Settings()

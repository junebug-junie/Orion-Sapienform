from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = Field(default="orion-office-env")
    SERVICE_VERSION: str = Field(default="0.1.0")
    INSTANCE_ID: str = Field(default="pan-1")
    ORION_BUS_URL: str = Field(default="redis://100.92.216.81:6379/0")
    ORION_BUS_ENABLED: bool = Field(default=True)
    ORION_HEALTH_CHANNEL: str = Field(default="orion:system:health")
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0)
    OFFICE_ENV_SAMPLE_CHANNEL: str = Field(default="orion:office:env:sample")
    # In-container path the compose file maps the host serial device to.
    OFFICE_SERIAL_PORT: str = Field(default="/dev/office-nano")
    OFFICE_SERIAL_BAUD: int = Field(default=115200)
    OFFICE_PUBLISH_INTERVAL_SEC: float = Field(default=30.0)
    OFFICE_STALE_AFTER_SEC: float = Field(default=10.0)
    OFFICE_SENSOR_ID: str = Field(default="sht40-office")


@lru_cache
def get_settings() -> Settings:
    return Settings()

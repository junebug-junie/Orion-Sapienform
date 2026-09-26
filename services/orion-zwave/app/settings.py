from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = Field(default="orion-zwave")
    SERVICE_VERSION: str = Field(default="0.1.0")
    INSTANCE_ID: str = Field(default="athena")
    ORION_BUS_URL: str = Field(default="redis://100.92.216.81:6379/0")
    ORION_BUS_ENABLED: bool = Field(default=True)
    ORION_ZWAVE_ENABLED: bool = Field(default=False)
    ZWAVE_JS_WS_URL: str = Field(default="ws://host.docker.internal:3000")
    ZWAVE_NODE_ID: int = Field(default=2)
    ZWAVE_DEVICE_ID: str = Field(default="shelly-wave-plug-ac")
    ZWAVE_DEVICE_NAME: str = Field(default="portable_ac")
    COOLING_SAMPLE_CHANNEL: str = Field(default="orion:home:cooling:sample")
    COOLING_POLL_INTERVAL_SEC: float = Field(default=5.0)
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0)
    ORION_HEALTH_CHANNEL: str = Field(default="orion:system:health")


@lru_cache
def get_settings() -> Settings:
    return Settings()

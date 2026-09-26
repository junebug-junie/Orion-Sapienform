from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = Field(default="orion-energy")
    SERVICE_VERSION: str = Field(default="0.1.0")
    INSTANCE_ID: str = Field(default="athena")
    ORION_BUS_URL: str = Field(default="redis://100.92.216.81:6379/0")
    ORION_BUS_ENABLED: bool = Field(default=True)

    ENERGY_INBOX_DIR: str = Field(default="/data/energy/inbox")
    ENERGY_PROCESSED_DIR: str = Field(default="/data/energy/processed")
    ENERGY_SCAN_INTERVAL_SEC: float = Field(default=60.0)
    ENERGY_TARIFF_PATH: str = Field(default="/app/config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml")
    ENERGY_TIMEZONE: str = Field(default="America/Denver")
    ENERGY_BILLING_CYCLE_START_DAY: int = Field(default=1, ge=1, le=31)
    # Empty = use the only usage point seen; required if the feed has several.
    ENERGY_USAGE_POINT_ID: str = Field(default="")
    ENERGY_RUN_COST_PENDING_HOURS: float = Field(default=96.0)
    # 0 disables pacing.
    ENERGY_PUBLISH_MAX_PER_SEC: float = Field(default=200.0, ge=0)

    ENERGY_USAGE_CHANNEL: str = Field(default="orion:energy:usage:observed")
    ENERGY_ACCRUED_CHANNEL: str = Field(default="orion:energy:cost:accrued")
    ENERGY_RUN_COST_CHANNEL: str = Field(default="orion:energy:run_cost:estimated")
    POWER_SETTLED_CHANNEL: str = Field(default="orion:power:intent:settled")

    HEARTBEAT_INTERVAL_SEC: float = Field(default=30.0)
    ORION_HEALTH_CHANNEL: str = Field(default="orion:system:health")


@lru_cache
def get_settings() -> Settings:
    return Settings()

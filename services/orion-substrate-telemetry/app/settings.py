from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = Field("substrate-telemetry", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    port: int = Field(8395, alias="PORT")

    orion_bus_url: str = Field("redis://127.0.0.1:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")

    channel_substrate_tier_outcomes: str = Field(
        "orion:substrate:tier_outcomes",
        alias="CHANNEL_SUBSTRATE_TIER_OUTCOMES",
    )

    postgres_uri: str = Field(
        "postgresql://postgres:postgres@localhost:5432/conjourney",
        alias="POSTGRES_URI",
    )

    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    retention_days: int = Field(7, alias="SUBSTRATE_TELEMETRY_RETENTION_DAYS")
    retention_scan_interval_sec: float = Field(3600.0, alias="SUBSTRATE_TELEMETRY_RETENTION_SCAN_SEC")
    per_correlation_row_cap: int = Field(100, alias="SUBSTRATE_TELEMETRY_PER_CORR_CAP")

    read_api_token: str | None = Field(None, alias="SUBSTRATE_TELEMETRY_READ_API_TOKEN")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

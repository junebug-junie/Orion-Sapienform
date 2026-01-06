from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = Field("state-journaler", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    port: int = Field(8380, alias="PORT")

    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")

    channel_spark_state_snapshot: str = Field("orion:spark:state:snapshot", alias="CHANNEL_SPARK_STATE_SNAPSHOT")
    channel_equilibrium_snapshot: str = Field("orion:equilibrium:snapshot", alias="CHANNEL_EQUILIBRIUM_SNAPSHOT")

    postgres_uri: str = Field("postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney", alias="POSTGRES_URI")
    rollup_table: str = Field("spark_state_rollups", alias="SPARK_ROLLUP_TABLE")

    windows_sec: List[int] = Field(default_factory=lambda: [60, 300, 3600], alias="ROLLUP_WINDOWS_SEC")
    rollup_interval_sec: float = Field(30.0, alias="ROLLUP_INTERVAL_SEC")
    retention_hours: int = Field(24, alias="ROLLUP_RETENTION_HOURS")

    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Identity
    service_name: str = Field("state-service", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    port: int = Field(8270, alias="PORT")

    # Bus / Redis (Titanium)
    orion_bus_url: str = Field("redis://orion-redis:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")

    # Input stream: Spark snapshots (real-time)
    channel_spark_state_snapshot: str = Field(
        "orion:spark:state:snapshot",
        validation_alias=AliasChoices("CHANNEL_SPARK_STATE_SNAPSHOT", "SPARK_STATE_SNAPSHOT_CHANNEL"),
    )

    # RPC intake channel
    state_request_channel: str = Field(
        "orion-state:request",
        validation_alias=AliasChoices("STATE_REQUEST_CHANNEL", "CHANNEL_STATE_REQUEST"),
    )

    # Cache
    cache_redis_url: str = Field(
        "redis://orion-redis:6379/0",
        validation_alias=AliasChoices("CACHE_REDIS_URL", "STATE_CACHE_REDIS_URL"),
    )
    state_cache_prefix: str = Field("orion:spark_state", alias="STATE_CACHE_PREFIX")
    primary_state_node: str = Field("athena", alias="PRIMARY_STATE_NODE")

    # Postgres hydration (durable recovery)
    postgres_uri: str = Field(
        "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        validation_alias=AliasChoices("POSTGRES_URI", "POSTGRESQL_URI", "DATABASE_URL"),
    )
    spark_telemetry_table: str = Field("spark_telemetry", alias="SPARK_TELEMETRY_TABLE")
    hydrate_limit: int = Field(500, alias="HYDRATE_LIMIT")

    # Chassis
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    health_channel: str = Field("system.health", alias="ORION_HEALTH_CHANNEL")
    error_channel: str = Field("system.error", alias="ORION_ERROR_CHANNEL")
    shutdown_grace_sec: float = Field(10.0, alias="ORION_SHUTDOWN_GRACE_SEC")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

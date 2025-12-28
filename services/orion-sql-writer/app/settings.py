from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Identity
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("sql-writer", alias="SERVICE_NAME")
    service_version: str = Field("0.4.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    port: int = Field(8220, alias="PORT")

    # Bus (legacy vars preserved)
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")

    # Chassis (new; must be wired in .env_example + docker-compose env)
    heartbeat_interval_sec: float = Field(10.0, alias="ORION_HEARTBEAT_INTERVAL_SEC")
    health_channel: str = Field("system.health", alias="ORION_HEALTH_CHANNEL")
    error_channel: str = Field("system.error", alias="ORION_ERROR_CHANNEL")
    shutdown_grace_sec: float = Field(10.0, alias="ORION_SHUTDOWN_GRACE_SEC")

    # --- Writer input channels ---
    channel_tags_raw: str = Field("orion:tags", alias="CHANNEL_TAGS_RAW")
    channel_tags_enriched: str = Field("orion:tags:enriched", alias="CHANNEL_TAGS_ENRICHED")

    channel_collapse_triage: str = Field("orion:collapse:triage", alias="CHANNEL_COLLAPSE_TRIAGE")
    channel_collapse_publish: str = Field("orion:collapse:sql-write", alias="CHANNEL_COLLAPSE_PUBLISH")

    channel_chat_log: str = Field("orion:chat:history:log", alias="CHANNEL_CHAT_LOG")

    channel_dream: str = Field("orion:dream:log", alias="CHANNEL_DREAM_LOG")
    channel_biometrics: str = Field("orion:biometrics:telemetry", alias="CHANNEL_BIOMETRICS_TELEMETRY")
    channel_spark_introspection: str = Field("orion:spark:introspection:log", alias="CHANNEL_SPARK_INTROSPECTION_LOG")

    # DB
    postgres_uri: str = Field("sqlite:///./orion.db", alias="POSTGRES_URI")
    database_url: str = Field("sqlite:///./orion.db", alias="DATABASE_URL")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

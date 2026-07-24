from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = Field(default="orion-graphiti-adapter", alias="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.1.0", alias="SERVICE_VERSION")
    NODE_NAME: str = Field(default="athena", alias="NODE_NAME")
    LOG_LEVEL: str = Field(default="INFO", alias="LOG_LEVEL")

    # Bus-native SystemHealthV1 heartbeat (orion:system:health). This service has no other
    # bus connection today (pure HTTP + Postgres + FalkorDB) -- these fields exist solely to
    # feed the HeartbeatOnly chassis. See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    ORION_BUS_URL: str = Field(default="redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, alias="ORION_BUS_ENABLED")
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0, alias="HEARTBEAT_INTERVAL_SEC")

    POSTGRES_URI: str = Field(default="", alias="POSTGRES_URI")
    GRAPHITI_AUTO_APPLY_SCHEMA: bool = Field(default=True, alias="GRAPHITI_AUTO_APPLY_SCHEMA")

    FALKORDB_URI: str = Field(default="", alias="FALKORDB_URI")
    FALKORDB_GRAPH: str = Field(default="graphiti_temporal", alias="FALKORDB_GRAPH")
    FALKORDB_ENABLED: bool = Field(default=True, alias="FALKORDB_ENABLED")

    GRAPHITI_BACKEND: Literal["orion_postgres", "graphiti_core"] = Field(
        default="graphiti_core", alias="GRAPHITI_BACKEND"
    )
    CRYSTALLIZER_EMBED_HOST_URL: str = Field(default="", alias="CRYSTALLIZER_EMBED_HOST_URL")
    # Bootstraps graphiti-core's RELATES_TO fulltext+range indices once at startup when the
    # graphiti_core backend + FalkorDB are both active. See
    # app/backends/graphiti_core.py::ensure_graphiti_indices for the idempotency handling
    # (FalkorDB's CREATE FULLTEXT INDEX has no IF NOT EXISTS guard).
    GRAPHITI_AUTO_BUILD_INDICES: bool = Field(default=True, alias="GRAPHITI_AUTO_BUILD_INDICES")

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


settings = Settings()

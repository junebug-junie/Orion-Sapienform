from __future__ import annotations

from functools import lru_cache
from typing import List, Dict
import json

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Identity
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("sql-writer", alias="SERVICE_NAME")
    service_version: str = Field("0.4.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    port: int = Field(8220, alias="PORT")

    # Bus
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")

    # Chassis
    heartbeat_interval_sec: float = Field(10.0, alias="ORION_HEARTBEAT_INTERVAL_SEC")
    health_channel: str = Field("orion:system:health", alias="ORION_HEALTH_CHANNEL")
    error_channel: str = Field("orion:system:error", alias="ORION_ERROR_CHANNEL")
    shutdown_grace_sec: float = Field(10.0, alias="ORION_SHUTDOWN_GRACE_SEC")

    # Routing
    # Comma-separated or JSON list of channels to subscribe to
    sql_writer_subscribe_channels: List[str] = Field(
        default=[
            "orion:tags:enriched",
            "orion:collapse:sql-write",
            "orion:chat:history:log",
            "orion:chat:history:turn",
            "orion:dream:log",
            "orion:telemetry:biometrics",
            "orion:biometrics:summary",
            "orion:biometrics:induction",
            "orion:spark:introspection:log", # legacy?
            "orion:spark:telemetry",
            "orion:cognition:trace",
            "orion:metacognition:tick",
            "orion:equilibrium:metacog:trigger"
        ],
        alias="SQL_WRITER_SUBSCRIBE_CHANNELS"
    )

    # JSON mapping from envelope.kind -> destination table (or internal model key)
    sql_writer_route_map_json: str = Field(
        default=json.dumps({
            "collapse.mirror": "CollapseMirror",
            "collapse.mirror.entry.v2": "CollapseMirror",
            "collapse.enrichment": "CollapseEnrichment",
            "tags.enriched": "CollapseEnrichment",
            "chat.history": "ChatHistoryLogSQL",
            "chat.log": "ChatHistoryLogSQL",
            "chat.history.message.v1": "ChatMessageSQL",
            "dream.log": "Dream",
            "biometrics.telemetry": "BiometricsTelemetry",
            "biometrics.summary.v1": "BiometricsSummarySQL",
            "biometrics.induction.v1": "BiometricsInductionSQL",
            "spark.introspection.log": "SparkIntrospectionLogSQL",
            "spark.introspection": "SparkIntrospectionLogSQL",
            "spark.telemetry": "SparkTelemetrySQL",
            "cognition.trace": "CognitionTraceSQL",
            "metacognition.tick.v1":"MetacognitionTickSQL",
            "orion.metacog.trigger.v1": "MetacogTriggerSQL"

        }),
        alias="SQL_WRITER_ROUTE_MAP_JSON"
    )

    @property
    def route_map(self) -> Dict[str, str]:
        try:
            return json.loads(self.sql_writer_route_map_json)
        except Exception:
            return {}

    @property
    def effective_subscribe_channels(self) -> List[str]:
        """Back-compat alias.

        Some refactor branches referenced `effective_subscribe_channels`.
        We keep env as source of truth and simply expose the configured list.
        """
        return list(self.sql_writer_subscribe_channels)


    # DB
    # Ensure default matches prod environment (Postgres), not SQLite.
    postgres_uri: str = Field("postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney", alias="POSTGRES_URI")
    database_url: str = Field("postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney", alias="DATABASE_URL")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

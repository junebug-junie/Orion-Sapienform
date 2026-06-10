from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service identity
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-memory-crystallizer", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("unknown", alias="NODE_NAME")
    port: int = Field(8634, alias="PORT")

    # Bus
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_url: str = Field("redis://127.0.0.1:6379/0", alias="ORION_BUS_URL")
    health_channel: str = Field("orion:system:health", alias="ORION_HEALTH_CHANNEL")
    error_channel: str = Field("orion:system:error", alias="ERROR_CHANNEL")
    heartbeat_interval_sec: float = Field(30.0, alias="HEARTBEAT_INTERVAL_SEC")

    # Crystallization lifecycle channels (must match orion/bus/channels.yaml)
    channel_proposed: str = Field(
        "orion:memory:crystallization:proposed", alias="CRYSTALLIZER_CHANNEL_PROPOSED"
    )
    channel_validated: str = Field(
        "orion:memory:crystallization:validated", alias="CRYSTALLIZER_CHANNEL_VALIDATED"
    )
    channel_approved: str = Field(
        "orion:memory:crystallization:approved", alias="CRYSTALLIZER_CHANNEL_APPROVED"
    )
    channel_rejected: str = Field(
        "orion:memory:crystallization:rejected", alias="CRYSTALLIZER_CHANNEL_REJECTED"
    )
    channel_quarantined: str = Field(
        "orion:memory:crystallization:quarantined", alias="CRYSTALLIZER_CHANNEL_QUARANTINED"
    )
    channel_project: str = Field(
        "orion:memory:crystallization:project", alias="CRYSTALLIZER_CHANNEL_PROJECT"
    )
    channel_retrieved: str = Field(
        "orion:memory:crystallization:retrieved", alias="CRYSTALLIZER_CHANNEL_RETRIEVED"
    )
    channel_vector_upsert: str = Field(
        "orion:memory:vector:upsert", alias="CRYSTALLIZER_CHANNEL_VECTOR_UPSERT"
    )

    # Storage
    postgres_uri: str = Field(
        "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        alias="POSTGRES_URI",
    )
    auto_apply_schema: bool = Field(True, alias="CRYSTALLIZER_AUTO_APPLY_SCHEMA")

    # Chroma projection (via orion-vector-writer; this service only emits
    # memory.vector.upsert.v1 payloads on the bus)
    vector_collection: str = Field(
        "orion_memory_crystallizations", alias="CRYSTALLIZER_VECTOR_COLLECTION"
    )
    embed_host_url: str = Field("", alias="CRYSTALLIZER_EMBED_HOST_URL")
    embed_timeout_ms: int = Field(8000, alias="CRYSTALLIZER_EMBED_TIMEOUT_MS")

    # Graphiti/FalkorDB temporal projection (additive; disabled until the
    # backend is deployed). This is NOT the RDF memory_graph path.
    graphiti_enabled: bool = Field(False, alias="GRAPHITI_ENABLED")
    graphiti_url: str = Field("", alias="GRAPHITI_URL")
    falkordb_uri: str = Field("", alias="FALKORDB_URI")

    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

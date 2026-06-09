from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service identity
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-graph-compression", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("unknown", alias="NODE_NAME")
    port: int = Field(8271, alias="PORT")

    # Bus
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_url: str = Field("redis://127.0.0.1:6379/0", alias="ORION_BUS_URL")
    health_channel: str = Field("orion:system:health", alias="ORION_HEALTH_CHANNEL")
    error_channel: str = Field("orion:system:error", alias="ERROR_CHANNEL")
    heartbeat_interval_sec: float = Field(30.0, alias="HEARTBEAT_INTERVAL_SEC")

    # Bus channels
    channel_rdf_enqueue: str = Field("orion:rdf:enqueue", alias="CHANNEL_RDF_ENQUEUE")
    channel_graph_compression_stale: str = Field(
        "orion:graph:compression:stale", alias="CHANNEL_GRAPH_COMPRESSION_STALE"
    )
    channel_graph_compression_events: str = Field(
        "orion:graph:compression:events", alias="CHANNEL_GRAPH_COMPRESSION_EVENTS"
    )
    channel_substrate_mutation_pressure: str = Field(
        "orion:substrate:mutation:pressure", alias="CHANNEL_SUBSTRATE_MUTATION_PRESSURE"
    )
    llm_gateway_bus_channel: str = Field(
        "orion:exec:request:LLMGatewayService", alias="LLM_GATEWAY_BUS_CHANNEL"
    )
    enable_llm_summaries: bool = Field(True, alias="ENABLE_LLM_SUMMARIES")

    # Postgres
    postgres_uri: str = Field(
        "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        alias="POSTGRES_URI",
    )

    # Fuseki
    rdf_store_query_url: str = Field(
        "http://orion-athena-fuseki:3030/orion/query", alias="RDF_STORE_QUERY_URL"
    )
    rdf_store_update_url: str = Field(
        "http://orion-athena-fuseki:3030/orion/update", alias="RDF_STORE_UPDATE_URL"
    )
    rdf_store_user: str = Field("admin", alias="RDF_STORE_USER")
    rdf_store_pass: str = Field("orion", alias="RDF_STORE_PASS")
    rdf_store_timeout_sec: float = Field(10.0, alias="RDF_STORE_TIMEOUT_SEC")

    # Worker tuning
    compression_poll_interval_sec: float = Field(300.0, alias="COMPRESSION_POLL_INTERVAL_SEC")
    compression_batch_size: int = Field(10, alias="COMPRESSION_BATCH_SIZE")
    compression_max_tokens_per_summary: int = Field(200, alias="COMPRESSION_MAX_TOKENS_PER_SUMMARY")
    compression_llm_budget_per_tick: int = Field(5000, alias="COMPRESSION_LLM_BUDGET_PER_TICK")
    compression_max_age_sec: int = Field(86400, alias="COMPRESSION_MAX_AGE_SEC")
    enable_compression_runtime: bool = Field(True, alias="ENABLE_COMPRESSION_RUNTIME")
    compression_policy_path: str = Field(
        "/app/config/compression_policy.v1.yaml", alias="COMPRESSION_POLICY_PATH"
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

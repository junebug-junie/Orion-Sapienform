from __future__ import annotations

from functools import lru_cache

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Identity
    service_name: str = Field("spark-introspector", alias="SERVICE_NAME")
    service_version: str = Field("0.2.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    # Bus
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")

    # Chassis
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    # Consume candidates (legacy dict payloads and/or envelopes)
    channel_spark_candidate: str = Field(
        "orion:spark:introspect:candidate:log",
        validation_alias=AliasChoices("CHANNEL_SPARK_INTROSPECT_CANDIDATE", "SPARK_CANDIDATE_CHANNEL"),
    )

    # Cognition Trace Intake
    channel_cognition_trace_pub: str = Field("orion:cognition:trace", alias="CHANNEL_COGNITION_TRACE_PUB")

    # Durable telemetry output (SQL writer ingests this)
    channel_spark_telemetry: str = Field("orion:spark:telemetry", alias="CHANNEL_SPARK_TELEMETRY")

    # Real-time snapshot stream (state-service + UI can subscribe)
    channel_spark_state_snapshot: str = Field(
        "orion:spark:state:snapshot",
        alias="CHANNEL_SPARK_STATE_SNAPSHOT",
    )

    # Freshness semantics (for read-model)
    spark_state_valid_for_ms: int = Field(15000, alias="SPARK_STATE_VALID_FOR_MS")

    # Tissue
    orion_tissue_snapshot_path: str = Field(
        "/mnt/storage-lukewarm/orion/spark/tissue-brain.npy",
        alias="ORION_TISSUE_SNAPSHOT_PATH",
    )

    # RPC to Cortex-Orch (Spark -> Cortex-Orch)
    channel_cortex_request: str = Field(
        "orion-cortex:request",
        validation_alias=AliasChoices("CORTEX_REQUEST_CHANNEL", "CORTEX_ORCH_REQUEST_CHANNEL", "ORCH_REQUEST_CHANNEL"),
    )

    # How long to wait for Cortex-Orch RPC reply
    cortex_timeout_sec: float = Field(15.0, alias="CORTEX_TIMEOUT_SEC")

    # Web UI
    port: int = Field(8444, alias="PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

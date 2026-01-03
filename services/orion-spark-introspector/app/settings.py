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
        "orion:spark:introspect:candidate",
        validation_alias=AliasChoices("CHANNEL_SPARK_INTROSPECT_CANDIDATE", "SPARK_CANDIDATE_CHANNEL"),
    )

    # Cognition Trace Intake
    channel_cognition_trace_pub: str = Field("orion:cognition:trace", alias="CHANNEL_COGNITION_TRACE_PUB")

    # Telemetry Output
    channel_spark_telemetry: str = Field("orion:spark:introspection:log", alias="CHANNEL_SPARK_TELEMETRY")

    # Tissue
    orion_tissue_snapshot_path: str = Field("/tmp/orion_tissue_snapshot.json", alias="ORION_TISSUE_SNAPSHOT_PATH")

    # RPC to Cortex-Orch (Spark -> Cortex-Orch)
    channel_cortex_request: str = Field(
        "orion-cortex:request",
        validation_alias=AliasChoices("CORTEX_REQUEST_CHANNEL", "CORTEX_ORCH_REQUEST_CHANNEL", "ORCH_REQUEST_CHANNEL"),
    )

    # How long to wait for Cortex-Orch RPC reply
    cortex_timeout_sec: float = Field(15.0, alias="CORTEX_TIMEOUT_SEC")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

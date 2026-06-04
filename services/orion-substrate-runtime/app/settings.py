from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-substrate-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    orion_bus_url: str = Field("redis://redis:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")

    enable_biometrics_node_reducer: bool = Field(True, alias="ENABLE_BIOMETRICS_NODE_REDUCER")
    enable_biometrics_pressure_organ: bool = Field(True, alias="ENABLE_BIOMETRICS_PRESSURE_ORGAN")
    enable_node_pressure_reducer: bool = Field(True, alias="ENABLE_NODE_PRESSURE_REDUCER")
    enable_execution_trajectory_reducer: bool = Field(
        False,
        alias="ENABLE_EXECUTION_TRAJECTORY_REDUCER",
    )
    biometrics_node_stale_after_sec: int = Field(180, alias="BIOMETRICS_NODE_STALE_AFTER_SEC")
    biometrics_pressure_min_confidence: float = Field(0.60, alias="BIOMETRICS_PRESSURE_MIN_CONFIDENCE")
    node_catalog_path: str = Field(
        "config/biometrics/node_catalog.yaml",
        alias="NODE_CATALOG_PATH",
    )
    grammar_poll_interval_sec: float = Field(2.0, alias="GRAMMAR_POLL_INTERVAL_SEC")
    grammar_event_channel: str = Field("orion:grammar:event", alias="GRAMMAR_EVENT_CHANNEL")
    publish_accepted_pressure_grammar: bool = Field(
        True,
        alias="PUBLISH_ACCEPTED_PRESSURE_GRAMMAR",
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    receipt_retention_success_minutes: int = Field(
        30, alias="ORION_RECEIPT_RETENTION_SUCCESS_MINUTES"
    )
    receipt_retention_error_hours: int = Field(6, alias="ORION_RECEIPT_RETENTION_ERROR_HOURS")
    receipt_full_payload_success: bool = Field(False, alias="ORION_RECEIPT_FULL_PAYLOAD_SUCCESS")
    receipt_full_payload_sample_rate: float = Field(
        0.0, alias="ORION_RECEIPT_FULL_PAYLOAD_SAMPLE_RATE"
    )
    receipt_max_table_gb: float = Field(25.0, alias="ORION_RECEIPT_MAX_TABLE_GB")
    receipt_warn_table_gb: float = Field(15.0, alias="ORION_RECEIPT_WARN_TABLE_GB")
    receipt_critical_table_gb: float = Field(20.0, alias="ORION_RECEIPT_CRITICAL_TABLE_GB")
    receipt_emergency_metadata_only: bool = Field(
        True, alias="ORION_RECEIPT_EMERGENCY_METADATA_ONLY"
    )
    receipt_prune_interval_sec: float = Field(300.0, alias="ORION_RECEIPT_PRUNE_INTERVAL_SEC")
    receipt_prune_batch_size: int = Field(10000, alias="ORION_RECEIPT_PRUNE_BATCH_SIZE")
    receipt_postgres_data_path: str = Field(
        "/mnt/postgres", alias="ORION_RECEIPT_POSTGRES_DATA_PATH"
    )
    receipt_disk_critical_pct: float = Field(85.0, alias="ORION_RECEIPT_DISK_CRITICAL_PCT")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-attention-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    # Bus-native SystemHealthV1 heartbeat (orion:system:health). This service has no other
    # bus connection today -- these fields exist solely to feed the HeartbeatOnly chassis. See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    attention_policy_path: str = Field(
        "config/attention/field_attention_policy.v1.yaml",
        alias="ATTENTION_POLICY_PATH",
    )
    attention_poll_interval_sec: float = Field(2.0, alias="ATTENTION_POLL_INTERVAL_SEC")
    enable_attention_runtime: bool = Field(True, alias="ENABLE_ATTENTION_RUNTIME")
    attention_frame_retention_hours: float = Field(72.0, alias="ATTENTION_FRAME_RETENTION_HOURS")
    attention_frame_prune_interval_sec: float = Field(3600.0, alias="ATTENTION_FRAME_PRUNE_INTERVAL_SEC")
    enable_transport_attention_visibility: bool = Field(
        False,
        alias="ENABLE_TRANSPORT_ATTENTION_VISIBILITY",
    )
    # Candidate A (precision-weighted salience, 2026-07-30): how many real
    # recent prediction-error rows to fetch per qualified node target, per
    # tick. Same order of magnitude as scripts/analysis/measure_precision_
    # weighted_salience_probe.py's own real replay windows (tens to ~100
    # rows within substrate_reduction_receipts' ~30-minute retention) --
    # not a calibration knob, a fetch-size bound.
    prediction_error_history_limit: int = Field(
        200, alias="ATTENTION_PREDICTION_ERROR_HISTORY_LIMIT"
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Health monitor -> orion-notify attention alerts. Edge-triggered (fires only
    # on healthy->unhealthy transitions), not polled-and-spammed.
    attention_frame_stall_multiplier: float = Field(1.5, alias="ATTENTION_FRAME_STALL_MULTIPLIER")
    health_check_interval_sec: float = Field(
        900.0, alias="ATTENTION_RUNTIME_HEALTH_CHECK_INTERVAL_SEC"
    )
    notify_base_url: str = Field("http://orion-athena-notify:7140", alias="NOTIFY_BASE_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

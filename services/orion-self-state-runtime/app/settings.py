from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-self-state-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    self_state_policy_path: str = Field(
        "config/self_state/self_state_policy.v1.yaml",
        alias="SELF_STATE_POLICY_PATH",
    )
    self_state_poll_interval_sec: float = Field(2.0, alias="SELF_STATE_POLL_INTERVAL_SEC")
    enable_self_state_runtime: bool = Field(True, alias="ENABLE_SELF_STATE_RUNTIME")
    enable_transport_self_state_influence: bool = Field(
        False,
        alias="ENABLE_TRANSPORT_SELF_STATE_INFLUENCE",
    )
    self_state_max_previous_age_sec: float = Field(
        300.0,
        alias="SELF_STATE_MAX_PREVIOUS_AGE_SEC",
    )
    self_state_retention_hours: float = Field(72.0, alias="SELF_STATE_RETENTION_HOURS")
    self_state_prune_interval_sec: float = Field(3600.0, alias="SELF_STATE_PRUNE_INTERVAL_SEC")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    channel_substrate_self_state: str = Field(
        "orion:substrate:self_state",
        alias="CHANNEL_SUBSTRATE_SELF_STATE",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

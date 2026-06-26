from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-feedback-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    feedback_policy_path: str = Field(
        "config/feedback/feedback_policy.v1.yaml",
        alias="FEEDBACK_POLICY_PATH",
    )
    feedback_poll_interval_sec: float = Field(2.0, alias="FEEDBACK_POLL_INTERVAL_SEC")
    enable_feedback_runtime: bool = Field(True, alias="ENABLE_FEEDBACK_RUNTIME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    feedback_bus_channel: str = Field("orion:feedback:frame", alias="FEEDBACK_BUS_CHANNEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

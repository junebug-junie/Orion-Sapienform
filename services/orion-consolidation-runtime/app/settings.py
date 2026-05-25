from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-consolidation-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    consolidation_policy_path: str = Field(
        "config/consolidation/consolidation_policy.v1.yaml",
        alias="CONSOLIDATION_POLICY_PATH",
    )
    consolidation_poll_interval_sec: float = Field(
        60.0,
        alias="CONSOLIDATION_POLL_INTERVAL_SEC",
    )
    enable_consolidation_runtime: bool = Field(
        True,
        alias="ENABLE_CONSOLIDATION_RUNTIME",
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-execution-dispatch-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    execution_dispatch_policy_path: str = Field(
        "config/execution_dispatch/execution_dispatch_policy.v1.yaml",
        alias="EXECUTION_DISPATCH_POLICY_PATH",
    )
    execution_dispatch_mode: Literal["dry_run", "prepare_only", "dispatch_read_only"] = Field(
        "dry_run",
        alias="EXECUTION_DISPATCH_MODE",
    )
    execution_dispatch_poll_interval_sec: float = Field(
        2.0,
        alias="EXECUTION_DISPATCH_POLL_INTERVAL_SEC",
    )
    enable_execution_dispatch_runtime: bool = Field(
        True,
        alias="ENABLE_EXECUTION_DISPATCH_RUNTIME",
    )
    cortex_exec_channel: str = Field("orion:cortex:request", alias="CORTEX_EXEC_CHANNEL")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

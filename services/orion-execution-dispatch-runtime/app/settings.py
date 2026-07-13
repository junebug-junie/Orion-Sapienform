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
    cortex_exec_channel: str = Field(
        "orion:cortex:exec:request:background", alias="CORTEX_EXEC_CHANNEL"
    )
    cortex_exec_result_prefix: str = Field(
        "orion:exec:result", alias="CORTEX_EXEC_RESULT_PREFIX"
    )
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    execution_dispatch_rpc_timeout_sec: float = Field(
        120.0, alias="EXECUTION_DISPATCH_RPC_TIMEOUT_SEC"
    )
    orion_dispatch_max_per_day: int = Field(24, alias="ORION_DISPATCH_MAX_PER_DAY")
    action_outcome_channel: str = Field(
        "orion:autonomy:action:outcome", alias="BUS_ACTION_OUTCOME_OUT"
    )
    notify_url: str = Field("http://orion-notify:7140", alias="NOTIFY_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

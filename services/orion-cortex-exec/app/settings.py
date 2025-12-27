"""orion-cortex-exec settings.

Goal: read shared mesh env (ORION_BUS_URL, etc.) by default, while allowing
service-specific overrides.

We also accept a few legacy env var names for backwards compatibility.
"""

from __future__ import annotations

import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for CortexExecService."""

    model_config = SettingsConfigDict(
        env_file=os.getenv("ORION_ENV_FILE", ".env"),
        extra="ignore",
        case_sensitive=True,
    )

    # Identity
    service_name: str = Field(
        default="CortexExecService",
        validation_alias=AliasChoices("SERVICE_NAME", "ORION_SERVICE_NAME"),
    )
    node_name: str = Field(
        default=os.getenv("ORION_NODE_NAME", "orion-node"),
        validation_alias=AliasChoices("NODE_NAME", "ORION_NODE_NAME"),
    )

    # Bus
    bus_url: str = Field(
        default="redis://100.92.216.81:6379/0",
        validation_alias=AliasChoices("ORION_BUS_URL", "BUS_URL", "REDIS_URL"),
    )
    bus_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("ORION_BUS_ENABLED", "BUS_ENABLED"),
    )

    # Orch → Exec RPC
    cortex_exec_request_channel: str = Field(
        default="orion-cortex-exec:request",
        validation_alias=AliasChoices(
            "CORTEX_EXEC_REQUEST_CHANNEL",
            "ORION_CORTEX_EXEC_REQUEST_CHANNEL",
            # Legacy prefix-style naming
            "ORION_CORTEX_EXEC_INTAKE_CHANNEL",
        ),
    )
    cortex_exec_result_prefix: str = Field(
        default="orion-cortex-exec:result",
        validation_alias=AliasChoices(
            "CORTEX_EXEC_RESULT_PREFIX",
            "ORION_CORTEX_EXEC_RESULT_PREFIX",
        ),
    )

    # Exec → Downstream fan-out
    exec_request_prefix: str = Field(
        default="orion-exec:request",
        validation_alias=AliasChoices(
            "EXEC_REQUEST_PREFIX",
            "ORION_EXEC_REQUEST_PREFIX",
            "ORION_CORTEX_EXEC_REQUEST_PREFIX",
        ),
    )
    exec_result_prefix: str = Field(
        default="orion-exec:result",
        validation_alias=AliasChoices(
            "EXEC_RESULT_PREFIX",
            "ORION_EXEC_RESULT_PREFIX",
            "ORION_CORTEX_EXEC_RESULT_PREFIX",
        ),
    )

    cortex_log_channel: str = Field(
        default="orion:cortex:telemetry",
        validation_alias=AliasChoices(
            "CORTEX_LOG_CHANNEL",
            "ORION_CORTEX_LOG_CHANNEL",
        ),
    )

    step_timeout_ms: int = Field(
        default=12000,
        validation_alias=AliasChoices(
            "STEP_TIMEOUT_MS",
            "ORION_STEP_TIMEOUT_MS",
            "ORION_EXEC_STEP_TIMEOUT_MS",
            "ORION_CORTEX_STEP_TIMEOUT_MS",
        ),
    )


settings = Settings()

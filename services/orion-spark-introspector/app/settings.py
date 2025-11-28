from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the Spark Introspector worker.

    - Reads from environment variables.
    - Provides sane defaults for local/dev.
    - Ignores extra env keys (prevents the usual Pydantic 'extra fields' drama).
    """

    # Redis / Orion Bus
    ORION_BUS_URL: str = "redis://orion-redis:6379/0"
    ORION_BUS_ENABLED: bool = True

    # Channel where brain publishes "spiky" Spark candidates
    CHANNEL_SPARK_INTROSPECT_CANDIDATE: str = "orion:spark:introspect:candidate"

    # Channel where SQL writer listens for inserts/updates
    SQL_WRITER_CHANNEL: str = "orion:sql:intake"

    # Cortex orchestrator HTTP endpoint
    CORTEX_ORCH_URL: str = "http://orion-cortex-orch:8072/orchestrate"

    # HTTP timeouts
    CONNECT_TIMEOUT: float = 5.0
    READ_TIMEOUT: float = 10.0

    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_prefix="",               # use variable names as-is
        env_file=".env",             # optional, only if present
        env_file_encoding="utf-8",
        extra="ignore",              # <<< avoids the classic 'extra fields' error
    )


# Instantiate once for the process
settings = Settings()

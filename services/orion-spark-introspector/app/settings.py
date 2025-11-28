from __future__ import annotations

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

    # Cortex orchestrator bus wiring
    # (these mirror ORCH_REQUEST_CHANNEL / ORCH_RESULT_PREFIX in cortex .env)
    CORTEX_ORCH_REQUEST_CHANNEL: str = "orion-cortex:request"
    CORTEX_ORCH_RESULT_PREFIX: str = "orion-cortex:result"

    # How long to wait for a cortex_orch result (seconds)
    CORTEX_ORCH_TIMEOUT_S: float = 10.0

    model_config = SettingsConfigDict(
        env_prefix="",               # use variable names as-is
        env_file=".env",             # optional, only if present
        env_file_encoding="utf-8",
        extra="ignore",              # don't explode on extra env vars
    )


settings = Settings()

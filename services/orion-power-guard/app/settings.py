from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Use pydantic-settings v2 config style
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # ─────────────────────────────────────────────
    # Service identity
    # ─────────────────────────────────────────────
    SERVICE_NAME: str = Field(default="power-guard")
    SERVICE_VERSION: str = Field(default="0.1.0")

    # ─────────────────────────────────────────────
    # Orion Bus config
    # ─────────────────────────────────────────────
    ORION_BUS_ENABLED: bool = Field(default=True)
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False)
    ORION_BUS_URL: str = Field(default="redis://100.92.216.81:6379/0")

    # ─────────────────────────────────────────────
    # Node / UPS identity
    # ─────────────────────────────────────────────
    POWER_GUARD_NODE_NAME: str = Field(
        default_factory=lambda: os.uname().nodename
    )
    POWER_GUARD_UPS_NAME: str = Field(default="apc_ups")

    # ─────────────────────────────────────────────
    # UPS via AP9640 SNMP
    # ─────────────────────────────────────────────
    POWER_GUARD_UPS_HOST: str = Field(
        default="192.168.0.50",
        description="IP of the AP9640 card",
    )
    POWER_GUARD_SNMP_PORT: int = Field(default=161)
    POWER_GUARD_SNMP_COMMUNITY: str = Field(default="public")

    # ─────────────────────────────────────────────
    # Polling + thresholds
    # ─────────────────────────────────────────────
    POWER_GUARD_POLL_INTERVAL_SEC: float = Field(default=5.0)
    POWER_GUARD_ONBATTERY_GRACE_SEC: float = Field(default=60.0)

    # ─────────────────────────────────────────────
    # Bus channels
    # ─────────────────────────────────────────────
    CHANNEL_POWER_EVENTS: str = Field(default="orion:power:events")

    # ─────────────────────────────────────────────
    # Shutdown behavior
    # ─────────────────────────────────────────────
    POWER_GUARD_ENABLE_SHUTDOWN: bool = Field(default=False)
    POWER_GUARD_SHUTDOWN_CMD: str = Field(
        default='/sbin/shutdown -h +1 "Orion PowerGuard: UPS on battery"'
    )

    @field_validator("POWER_GUARD_POLL_INTERVAL_SEC", "POWER_GUARD_ONBATTERY_GRACE_SEC")
    @classmethod
    def _ensure_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Interval/threshold must be positive")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()

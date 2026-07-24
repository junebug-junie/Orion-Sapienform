import json
import os
from typing import Dict

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    SERVICE_NAME: str = Field(default="orion-biometrics")
    SERVICE_VERSION: str = Field(default="0.1.0")
    NODE_NAME: str = Field(default="unknown")

    # Bus
    ORION_BUS_URL: str = Field(default="redis://orion-redis:6379/0")
    ORION_BUS_ENABLED: bool = Field(default=True)
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False)

    # Channels
    TELEMETRY_PUBLISH_CHANNEL: str = Field(default="orion:telemetry:biometrics")
    BIOMETRICS_SAMPLE_CHANNEL: str = Field(default="orion:biometrics:sample")
    BIOMETRICS_SUMMARY_CHANNEL: str = Field(default="orion:biometrics:summary")
    BIOMETRICS_INDUCTION_CHANNEL: str = Field(default="orion:biometrics:induction")
    BIOMETRICS_CLUSTER_CHANNEL: str = Field(default="orion:biometrics:cluster")
    SPARK_SIGNAL_CHANNEL: str = Field(default="orion:spark:signal")
    PUBLISH_BIOMETRICS_GRAMMAR: bool = Field(default=True)
    GRAMMAR_EVENT_CHANNEL: str = Field(default="orion:grammar:event")
    NODE_CATALOG_PATH: str = Field(default="/app/config/biometrics/node_catalog.yaml")

    # Behavior
    TELEMETRY_INTERVAL: int = Field(default=30)
    LOG_LEVEL: str = Field(default="INFO")
    BIOMETRICS_MODE: str = Field(default="agent")
    CLUSTER_PUBLISH_INTERVAL: int = Field(default=15)
    role_weights: Dict[str, float] = Field(
        default_factory=lambda: {"atlas": 0.7, "athena": 0.3, "other": 0.5},
        alias="CLUSTER_ROLE_WEIGHTS",
    )
    SPARK_SIGNAL_TTL_MS: int = Field(default=15000)

    THERMAL_MIN_C: float = Field(default=50.0)
    THERMAL_MAX_C: float = Field(default=85.0)
    DISK_BW_MBPS: float = Field(default=200.0)
    NET_BW_MBPS: float = Field(default=125.0)
    POWER_BAND_ALPHA: float = Field(default=0.1)

    TABLE_NAME: str = Field(default="biometrics_raw")

    # Chassis Defaults
    HEARTBEAT_INTERVAL_SEC: float = 10.0
    ORION_HEALTH_CHANNEL: str = "orion:system:health"
    ERROR_CHANNEL: str = "orion:system:error"
    SHUTDOWN_GRACE_SEC: float = 10.0

    # Disk-capacity telemetry (piggybacked onto the SystemHealthV1 heartbeat's
    # `details` dict -- see README.md "Disk capacity telemetry" section). Maps a
    # short mount name to the read-only in-container bind-mount path configured in
    # docker-compose.yml. Node-level, not per-service: every node running this
    # service reports its own 5 mounts independently once redeployed there.
    DISK_CAPACITY_MOUNTS: Dict[str, str] = Field(
        default_factory=lambda: {
            "docker": "/host_mnt/docker",
            "scripts": "/host_mnt/scripts",
            "postgres": "/host_mnt/postgres",
            "graphdb": "/host_mnt/graphdb",
            "telemetry": "/host_mnt/telemetry",
        }
    )

    @field_validator("role_weights", mode="before")
    @classmethod
    def _parse_role_weights(cls, value: object) -> Dict[str, float]:
        if isinstance(value, dict):
            return {str(k): float(v) for k, v in value.items()}
        if isinstance(value, str):
            try:
                data = json.loads(value)
            except json.JSONDecodeError:
                return {"atlas": 0.7, "athena": 0.3, "other": 0.5}
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
        return {"atlas": 0.7, "athena": 0.3, "other": 0.5}

    @field_validator("DISK_CAPACITY_MOUNTS", mode="before")
    @classmethod
    def _parse_disk_capacity_mounts(cls, value: object) -> Dict[str, str]:
        _default = {
            "docker": "/host_mnt/docker",
            "scripts": "/host_mnt/scripts",
            "postgres": "/host_mnt/postgres",
            "graphdb": "/host_mnt/graphdb",
            "telemetry": "/host_mnt/telemetry",
        }
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        if isinstance(value, str):
            if not value.strip():
                return _default
            try:
                data = json.loads(value)
            except json.JSONDecodeError:
                return _default
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        return _default

# pydantic-settings decodes Dict-typed env vars as JSON at its own source layer,
# before _parse_disk_capacity_mounts (which already handles "" gracefully) ever
# runs -- an empty string (e.g. docker-compose interpolating an unset host env
# var) throws SettingsError there instead of reaching our validator's fallback.
# Confirmed live 2026-07-24: orion-biometrics crash-looped on exactly this after
# .env_example gained DISK_CAPACITY_MOUNTS but a live .env hadn't been synced yet.
if not os.environ.get("DISK_CAPACITY_MOUNTS", "").strip():
    os.environ["DISK_CAPACITY_MOUNTS"] = "{}"

settings = Settings()

try:
    settings.role_weights = json.loads(settings.CLUSTER_ROLE_WEIGHTS)
except Exception:
    settings.role_weights = {"atlas": 0.7, "athena": 0.3, "other": 0.5}

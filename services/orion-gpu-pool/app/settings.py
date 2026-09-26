from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    service_name: str = Field("orion-gpu-pool", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    orion_bus_url: str = Field(..., alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    rpc_health_publish_enabled: bool = Field(True, alias="RPC_HEALTH_PUBLISH_ENABLED")
    rpc_health_publish_interval_sec: float = Field(30.0, alias="RPC_HEALTH_PUBLISH_INTERVAL_SEC")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")

    # Rules live in config/gpu_pool.yaml; models come from config/llm_profiles.yaml.
    config_path: str = Field("/app/config/gpu_pool.yaml", alias="GPU_POOL_CONFIG_PATH")
    profiles_path: str = Field("/app/config/llm_profiles.yaml", alias="GPU_POOL_PROFILES_PATH")

    # observe: the pool discovers, answers leases and publishes state, but nothing depends on it
    # and swap decisions are published as swap_requested events without actuation (stage 1).
    mode: str = Field("observe", alias="GPU_POOL_MODE")
    tick_sec: float = Field(1.0, alias="GPU_POOL_TICK_SEC")
    probe_interval_sec: float = Field(15.0, alias="GPU_POOL_PROBE_INTERVAL_SEC")
    probe_timeout_sec: float = Field(3.0, alias="GPU_POOL_PROBE_TIMEOUT_SEC")
    announce_stale_sec: float = Field(120.0, alias="GPU_POOL_ANNOUNCE_STALE_SEC")
    state_publish_sec: float = Field(5.0, alias="GPU_POOL_STATE_PUBLISH_SEC")
    replay_payload_max_bytes: int = Field(262144, alias="GPU_POOL_REPLAY_PAYLOAD_MAX_BYTES")
    lease_retention_hours: float = Field(168.0, gt=0, alias="GPU_POOL_LEASE_RETENTION_HOURS")

    # Stage 4.3 actuation. Comma-separated swap seats the pool may load/unload itself by sending
    # GpuActuateV1 to the seat's host actuator. Empty (default) = actuation OFF: every swap stays a
    # published swap_requested {actuated: false}, exactly as before. `agent-gpu2` at the 4.5 cutover.
    actuate_roles: str = Field("", alias="GPU_POOL_ACTUATE_ROLES")
    # Swap-load guards (read outside the runtime lock, every guard_refresh_sec).
    cabinet_url: str = Field("http://100.92.216.81:8080/api/cabinet/sensors/latest", alias="GPU_POOL_CABINET_URL")
    visual_activity_url: str = Field("http://orion-athena-thought:7155/visual-chain/activity",
                                     alias="GPU_POOL_VISUAL_ACTIVITY_URL")
    guard_refresh_sec: float = Field(30.0, gt=0, alias="GPU_POOL_GUARD_REFRESH_SEC")

    @property
    def actuate_role_list(self) -> list[str]:
        return [r.strip() for r in self.actuate_roles.split(",") if r.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

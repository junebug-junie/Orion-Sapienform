from __future__ import annotations

from functools import lru_cache
from typing import Literal

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

    # Stage 5.7 end state: enforce. Both modes actuate every swap seat that has a launch block in
    # config/gpu_pool.yaml. observe is the documented rollback and differs in exactly three ways:
    # a swap seat the pool has NEVER acted on is marked loaded when its worker answers (liveness
    # adoption; no such seat today), there is no boot/resume `status` reconcile, and operator holds
    # are refused.
    # A typo fails the boot. Emergency stop is the pause_actuation control verb, not a mode.
    mode: Literal["enforce", "observe"] = Field("enforce", alias="GPU_POOL_MODE")
    tick_sec: float = Field(1.0, alias="GPU_POOL_TICK_SEC")
    probe_interval_sec: float = Field(15.0, alias="GPU_POOL_PROBE_INTERVAL_SEC")
    probe_timeout_sec: float = Field(3.0, alias="GPU_POOL_PROBE_TIMEOUT_SEC")
    announce_stale_sec: float = Field(120.0, alias="GPU_POOL_ANNOUNCE_STALE_SEC")
    state_publish_sec: float = Field(5.0, alias="GPU_POOL_STATE_PUBLISH_SEC")
    replay_payload_max_bytes: int = Field(262144, alias="GPU_POOL_REPLAY_PAYLOAD_MAX_BYTES")
    lease_retention_hours: float = Field(168.0, gt=0, alias="GPU_POOL_LEASE_RETENTION_HOURS")

    # Swap-load guards (read outside the runtime lock, every guard_refresh_sec).
    cabinet_url: str = Field("http://100.92.216.81:8080/api/cabinet/sensors/latest", alias="GPU_POOL_CABINET_URL")
    guard_refresh_sec: float = Field(30.0, gt=0, alias="GPU_POOL_GUARD_REFRESH_SEC")
    # U4 shed lever kill switch (orion/gpu_pool/shed.py). OFF in code: the pool still receives and
    # shows shed signals (orion-hardware-watch cooling incidents) but blocks nothing. ON in .env_example.
    shed_enabled: bool = Field(False, alias="GPU_POOL_SHED_ENABLED")
    # Orion's learned shed (attend-to-act loop A1): the lower-precedence ``orion_self_shed`` reason.
    # OFF in code AND in .env_example -- Juniper flips it. false refuses every set and settles any
    # active Orion shed ``cancelled`` at boot; it never touches the reflex's cooling_incident.
    orion_shed_enabled: bool = Field(False, alias="GPU_POOL_ORION_SHED_ENABLED")
    # Caps, enforced in the pool (no caller can exceed them); orion_self_shed only, never the reflex.
    orion_shed_max_ttl_sec: float = Field(900.0, gt=0, le=3600, alias="GPU_POOL_ORION_SHED_MAX_TTL_SEC")
    orion_shed_max_sec_per_day: float = Field(3600.0, ge=0, alias="GPU_POOL_ORION_SHED_MAX_SEC_PER_DAY")
    orion_shed_min_gap_sec: float = Field(900.0, ge=0, alias="GPU_POOL_ORION_SHED_MIN_GAP_SEC")

    # Controller-health alert (orion/gpu_pool/controller_health.py): a Hub attention card when a seat's lane
    # controller keeps refusing with a reason a retry cannot fix (config_unloadable, digest mismatch,
    # ...). /health and the state payload show it regardless; this only gates the card.
    controller_alert_enabled: bool = Field(True, alias="GPU_POOL_CONTROLLER_ALERT_ENABLED")
    notify_base_url: str = Field("http://orion-athena-notify:7140", alias="NOTIFY_BASE_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")


@lru_cache
def get_settings() -> Settings:
    return Settings()

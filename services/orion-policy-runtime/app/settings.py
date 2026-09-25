from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-policy-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    # Bus-native SystemHealthV1 heartbeat (orion:system:health). Policy-runtime had no bus
    # connection at all before this -- these fields exist solely to carry an independent
    # heartbeat chassis. See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    substrate_policy_path: str = Field(
        "config/policy/substrate_policy.v1.yaml",
        alias="SUBSTRATE_POLICY_PATH",
    )
    policy_poll_interval_sec: float = Field(2.0, alias="POLICY_POLL_INTERVAL_SEC")
    # ROADMAP D2. How often to re-queue rows whose `*_pending` marker was cleared without the
    # downstream frame actually existing. The marker is cleared transactionally so this should
    # find nothing -- but the failure it guards is SILENT WORK LOSS, and it can only add work
    # back, never remove it. It runs the expensive anti-join the marker exists to avoid, hence
    # once every 15 min rather than on the 2s poll. 0 runs it every poll (do not).
    policy_reconcile_interval_sec: float = Field(
        900.0, alias="POLICY_RECONCILE_INTERVAL_SEC"
    )
    # 2026-09-25: the frequent sweep above only looks at rows generated inside this window
    # (seconds). Each probe is a random read, so keep it small; the full sweep covers the rest.
    policy_reconcile_window_sec: float = Field(
        7200.0, alias="POLICY_RECONCILE_WINDOW_SEC", gt=0.0
    )
    # How often the FULL-history sweep may run (seconds): one read-only hash anti-join, then
    # short batched UPDATEs. 0 disables it (the bounded sweep still runs).
    policy_reconcile_full_sweep_interval_sec: float = Field(
        86400.0, alias="POLICY_RECONCILE_FULL_SWEEP_INTERVAL_SEC", ge=0.0
    )
    # UTC hour the full sweep is allowed in (9 = 03:00 MDT / 02:00 MST). -1 = any hour, spaced by the
    # interval alone.
    policy_reconcile_full_sweep_hour_utc: int = Field(
        9, alias="POLICY_RECONCILE_FULL_SWEEP_HOUR_UTC", ge=-1, le=23
    )
    enable_policy_runtime: bool = Field(True, alias="ENABLE_POLICY_RUNTIME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

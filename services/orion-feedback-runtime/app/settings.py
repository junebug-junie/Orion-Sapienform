from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-feedback-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    feedback_policy_path: str = Field(
        "config/feedback/feedback_policy.v1.yaml",
        alias="FEEDBACK_POLICY_PATH",
    )
    feedback_poll_interval_sec: float = Field(2.0, alias="FEEDBACK_POLL_INTERVAL_SEC")
    # ROADMAP D2. How often to re-queue rows whose `*_pending` marker was cleared without the
    # downstream frame actually existing. The marker is cleared transactionally so this should
    # find nothing -- but the failure it guards is SILENT WORK LOSS, and it can only add work
    # back, never remove it. It runs the expensive anti-join the marker exists to avoid, hence
    # once every 15 min rather than on the 2s poll. 0 runs it every poll (do not).
    feedback_reconcile_interval_sec: float = Field(
        900.0, alias="FEEDBACK_RECONCILE_INTERVAL_SEC"
    )
    enable_feedback_runtime: bool = Field(True, alias="ENABLE_FEEDBACK_RUNTIME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Repo-wide bus convention (see root CLAUDE.md): always the real tailscale node address,
    # never bus-core/localhost. This default was stale (127.0.0.1) before this patch -- the
    # checked-in .env_example was also wrong (redis://bus-core:6379/0), fixed alongside this.
    bus_url: str = Field(default="redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    feedback_bus_channel: str = Field("orion:feedback:frame", alias="FEEDBACK_BUS_CHANNEL")
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health). See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

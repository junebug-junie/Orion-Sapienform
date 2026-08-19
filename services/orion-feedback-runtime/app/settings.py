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
    # ROADMAP D2 follow-through. The "oldest dispatch frame without feedback" lookup used to be
    # an unbounded anti-join over two ~420k-row tables -- 829 MB read + 465 MB temp spill per
    # execution, every 2s, and the single largest contributor to athena being fully I/O-stalled
    # ~20% of wall time. Bounded to a window it becomes two index scans (916 blocks, 116x less).
    #
    # 3600s is ~42x the measured maximum feedback lag (p99 77.1s, max 85.6s over 24h).
    # Set to 0 to disable the bound and restore the previous behaviour -- that is the rollback.
        # DISABLED BY DEFAULT, 2026-08-19, after code review found the bound UNSAFE as designed.
    # `fetch()` only reaches the backstop when the fast path is EMPTY -- so during a real
    # backlog, where fresh in-window work always exists, the backstop never fires and pre-window
    # rows are stranded permanently. Live evidence: on 2026-08-14 this stage produced 29,264
    # feedback frames for dispatch rows ~34 HOURS old, while 26,148 new rows arrived the same
    # day. 8 of the last 30 days were entirely in that regime. The measurement that justified a
    # 1h window (n=514, max 85.6s) was taken during an unrepresentative quiet spell; over 7 days
    # the lag p50 is 124,613s and the max is 975,770s.
    # Do NOT re-enable without redesigning -- see orion/db/pending_scan.py.
    feedback_scan_window_sec: float = Field(0.0, alias="FEEDBACK_SCAN_WINDOW_SEC")
    # How often the unbounded backstop may run when the bounded query finds nothing. This is
    # what guarantees a frame older than the window is never skipped forever -- it is picked up
    # within one interval instead of instantly, and logged as a tripwire when it happens.
    feedback_scan_backstop_interval_sec: float = Field(
        300.0, alias="FEEDBACK_SCAN_BACKSTOP_INTERVAL_SEC"
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

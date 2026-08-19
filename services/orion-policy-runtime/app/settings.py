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
    # ROADMAP D2 follow-through. The "next proposal without a policy frame" lookup was an
    # unbounded anti-join over two ~420k-row tables and, after the dispatch->feedback stage was
    # bounded, the largest remaining sequential-scan source in the database (340,766 tuples/sec).
    # 3600s is ~113x the measured maximum proposal->policy lag (p99 17.0s, max 31.8s over 24h).
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
    policy_scan_window_sec: float = Field(0.0, alias="POLICY_SCAN_WINDOW_SEC")
    policy_scan_backstop_interval_sec: float = Field(
        300.0, alias="POLICY_SCAN_BACKSTOP_INTERVAL_SEC"
    )
    enable_policy_runtime: bool = Field(True, alias="ENABLE_POLICY_RUNTIME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-execution-dispatch-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    execution_dispatch_policy_path: str = Field(
        "config/execution_dispatch/execution_dispatch_policy.v1.yaml",
        alias="EXECUTION_DISPATCH_POLICY_PATH",
    )
    execution_dispatch_mode: Literal["dry_run", "prepare_only", "dispatch_read_only"] = Field(
        "dry_run",
        alias="EXECUTION_DISPATCH_MODE",
    )
    execution_dispatch_poll_interval_sec: float = Field(
        2.0,
        alias="EXECUTION_DISPATCH_POLL_INTERVAL_SEC",
    )
    enable_execution_dispatch_runtime: bool = Field(
        True,
        alias="ENABLE_EXECUTION_DISPATCH_RUNTIME",
    )
    cortex_exec_channel: str = Field(
        "orion:cortex:exec:request:background", alias="CORTEX_EXEC_CHANNEL"
    )
    cortex_exec_result_prefix: str = Field(
        "orion:exec:result", alias="CORTEX_EXEC_RESULT_PREFIX"
    )
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health). See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    execution_dispatch_rpc_timeout_sec: float = Field(
        120.0, alias="EXECUTION_DISPATCH_RPC_TIMEOUT_SEC"
    )
    # 2026-07-29: no longer the primary mechanism -- see
    # orion_dispatch_risk_cap_advisory_only's comment below and
    # app/worker.py::ExecutionDispatchRuntimeWorker._derive_daily_risk_cap
    # for the real, self-calibrating EWMA ceiling that replaced this fixed
    # number as of this patch. This value is now only the last-resort
    # fallback used when the daily-risk EWMA baseline has never been seeded
    # AND no historical closed day with real candidate data exists at all
    # (i.e. a truly first-ever tick against an empty
    # substrate_execution_dispatch_frames table) -- in this repo's real
    # history that never actually triggers (2026-07-28 already has real
    # closed-day data: 817.65), but the fallback still has to be something.
    # Kept at its old value for continuity of what a fresh/empty deployment
    # gets before any real data exists, not because 10.0 means anything
    # about real risk demand -- it never did (see the old comment history in
    # this file's git blame and services/orion-execution-dispatch-runtime/
    # README.md for how this number's original "~2x one day's total"
    # justification was itself later found to be ~2x a *clamped* value, not
    # real demand).
    orion_dispatch_max_risk_per_day: float = Field(10.0, alias="ORION_DISPATCH_MAX_RISK_PER_DAY")
    # 2026-07-29: enforcement is back ON (default flipped True -> False).
    # Real sequence, not "we always knew this": ORION_DISPATCH_MAX_RISK_PER_DAY
    # was a fixed 10.0 constant, ENFORCED, from 2026-07-26 through 2026-07-27
    # -- and it worked exactly like a real ceiling, clamping dispatched-risk
    # totals at exactly 10.00/day both days (150 candidates on the 26th, 88
    # on the 27th), which looked like a healthy, working cap. It wasn't --
    # advisory_only=True shipped 2026-07-28 specifically to observe what real
    # *uncapped* demand looked like, and the answer was 817.65/day (15,099
    # candidates) and climbing, ~80x the old enforced number. That one day of
    # advisory-only data is exactly what
    # app/worker.py::ExecutionDispatchRuntimeWorker._derive_daily_risk_cap
    # now feeds into a real EWMA baseline (orion/bus/ewma.py::
    # compute_ewma_update, same mechanism as PR #1433's recent_perturbation
    # fix and PR #1434's execution_prediction_error fix) instead of a
    # hand-picked multiplier -- so re-enforcing now, on a derived ceiling
    # instead of a guessed one, closes the loop advisory-only was opened for.
    # Real per-tick spend still reads from sum_risk_dispatched_today()
    # (right-censored at whatever cap is in force, and correctly so -- that's
    # what "spend" means); the EWMA baseline itself is fed from
    # store.py::sum_uncapped_risk_for_day's *uncapped* demand instead, so the
    # new cap can't recreate the same clamped-value-masks-true-magnitude trap
    # one layer down. Explicit operator override: set this back to True to
    # return to log-only behavior without touching the derived-cap machinery
    # itself.
    orion_dispatch_risk_cap_advisory_only: bool = Field(
        False, alias="ORION_DISPATCH_RISK_CAP_ADVISORY_ONLY"
    )
    action_outcome_channel: str = Field(
        "orion:autonomy:action:outcome", alias="BUS_ACTION_OUTCOME_OUT"
    )
    notify_url: str = Field("http://notify:7140", alias="NOTIFY_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

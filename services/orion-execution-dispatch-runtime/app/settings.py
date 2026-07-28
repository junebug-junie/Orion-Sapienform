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
    # Replaces the old ORION_DISPATCH_MAX_PER_DAY (a blind action count, never
    # empirically derived -- confirmed 2026-07-25 the checked-in default (24)
    # and the live drifted value (88) both trace to a hand-picked round
    # number in docs/superpowers/specs/2026-07-13-endogenous-action-motor-
    # nerve-spec.md's risk table, never validated against real behavior).
    # This budget is spent against each dispatched candidate's own real,
    # already-computed `risk_score` (ExecutionDispatchCandidateV1.risk_score,
    # [0,1] per candidate) instead of counting actions -- five trivial
    # inspects (risk_score ~0.05 each) no longer cost the same as five
    # higher-risk candidates. Starting value anchored to real observed
    # data, not another guess: the first real day this pipeline dispatched
    # successfully (2026-07-26, post theater-tripwire fix) spent a real
    # cumulative risk of 4.4 across 88 dispatches (all risk_score=0.05 that
    # day). This default is ~2x that real observed total -- a margin over
    # actually-observed behavior, not a fresh round number. Needs real
    # re-derivation as more real history accumulates across a wider mix of
    # dispatch_kind/risk_score, per this program's own "measure before
    # minting" discipline -- treat this as a disclosed starting judgment
    # call, not a settled constant.
    orion_dispatch_max_risk_per_day: float = Field(10.0, alias="ORION_DISPATCH_MAX_RISK_PER_DAY")
    # 2026-07-28: the *shape* of this budget (spend real risk_score, not a blind
    # count) is real; the *number* is not -- "~2x one day's observed total" is
    # still a hand-picked multiplier, no different in kind from the old 24/88
    # this replaced, just wearing better math. Confirmed live the same day:
    # every dispatched candidate so far has had an identical risk_score=0.05,
    # so there is still no real distribution to derive a principled ceiling
    # from. Advisory-only by default: the cap is computed and logged every
    # tick (execution_dispatch_risk_budget_status) but does not block sends --
    # flip to false only once enough risk_score variance has accumulated to
    # derive max_risk_per_day from real data instead of a guessed multiplier.
    orion_dispatch_risk_cap_advisory_only: bool = Field(
        True, alias="ORION_DISPATCH_RISK_CAP_ADVISORY_ONLY"
    )
    action_outcome_channel: str = Field(
        "orion:autonomy:action:outcome", alias="BUS_ACTION_OUTCOME_OUT"
    )
    notify_url: str = Field("http://orion-notify:7140", alias="NOTIFY_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

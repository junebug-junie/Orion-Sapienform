from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from orion.schemas.durable_run import DURABLE_RUN_REQUEST_CHANNEL, DURABLE_RUN_STATE_CHANNEL


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-durable-runs", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    # RPC-health snapshot publish (orion:rpc_health:snapshot) of the long-lived rpc_bus:
    # runner rpc_request calls + GPU pool lease RPCs (no HTTP hops since 4.5). Mesh transport
    # coverage of docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-
    # baseline-design.md. Defaults match .env_example.
    rpc_health_publish_enabled: bool = Field(True, alias="RPC_HEALTH_PUBLISH_ENABLED")
    rpc_health_publish_interval_sec: float = Field(30.0, gt=0.0, alias="RPC_HEALTH_PUBLISH_INTERVAL_SEC")
    # Per-hop channel_latency in each snapshot. RpcHealthSnapshotV1 is extra="forbid":
    # orion-signal-gateway and orion-equilibrium-service must already run PR #2312's build.
    rpc_health_channel_latency_enabled: bool = Field(True, alias="RPC_HEALTH_CHANNEL_LATENCY_ENABLED")
    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Master switch. Off = the service heartbeats and answers /health but
    # consumes nothing; every kickoff sits unanswered on the request channel.
    enabled: bool = Field(True, alias="DURABLE_RUNS_ENABLED")
    # On boot, every checkpointed thread whose graph still has a next node is
    # picked back up. This is the whole point of the service.
    resume_on_boot: bool = Field(True, alias="DURABLE_RUNS_RESUME_ON_BOOT")
    # And again every N seconds, for runs whose last node attempt failed
    # (Hub was away mid-turn, the graph did not answer) -- a failed node
    # leaves the thread with the same next node, so the sweep re-invokes it.
    resume_sweep_sec: float = Field(120.0, gt=0.0, alias="DURABLE_RUNS_RESUME_SWEEP_SEC")
    # A checkpoint older than this is abandoned, not resumed: a day-old
    # curiosity run resuming into a different day's material is not
    # continuity, it is a ghost (design doc MQ2/MQ3).
    max_age_hours: float = Field(24.0, gt=0.0, alias="DURABLE_RUNS_MAX_AGE_HOURS")
    # The harness turn RPC to Hub. Hub's own investigation budget is 8840s
    # (HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC); this sits above it so the
    # runner never gives up on a turn Hub is still running.
    turn_rpc_timeout_sec: float = Field(8940.0, gt=0.0, alias="DURABLE_RUNS_TURN_RPC_TIMEOUT_SEC")
    # Orion's own graph, for read_turn_result. Same host/port/name Hub uses
    # (HUB_CURIOSITY_GRAPH_*); blank host = graph half off, same as Hub.
    graph_host: str = Field("", alias="DURABLE_RUNS_GRAPH_HOST")
    graph_port: int = Field(6379, alias="DURABLE_RUNS_GRAPH_PORT")
    graph_own: str = Field("orion_worldview", alias="DURABLE_RUNS_GRAPH_OWN")

    # Admitted runs (resource admission) are driven here on GPU pool holds (stage 4.5): the pool is
    # the only scheduler. The broker, widening and gpu2 elastic keys were deleted with the broker.
    admission_enabled: bool = Field(False, alias="DURABLE_RUNS_ADMISSION_ENABLED")
    # Gateway capacity permits for world-model and the visual chain (/capacity). NOT GPU pool
    # holds; stays until stage 5 moves those onto pool leases.
    capacity_enabled: bool = Field(False, alias="DURABLE_RUNS_CAPACITY_ENABLED")
    admission_tick_sec: float = Field(5.0, gt=0.0, alias="DURABLE_RUNS_ADMISSION_TICK_SEC")
    # Capacity permit TTL (/capacity). Admitted runs' holds use the pool's hold_lease_ttl_sec.
    lease_seconds: float = Field(90.0, ge=15.0, alias="DURABLE_RUNS_LEASE_SECONDS")
    # How often a working run heartbeats its pool hold (must be at most half the pool's
    # hold_lease_ttl_sec, checked at startup) and how often a Door-A hold is kept alive.
    lease_heartbeat_sec: float = Field(15.0, gt=0.0, alias="DURABLE_RUNS_LEASE_HEARTBEAT_SEC")
    # A run waiting in the pool's queue is woken by the pool's "granted" event for its holder.
    # This is only the missed-event fallback: one pool status read per waiting run per interval.
    hold_status_poll_sec: float = Field(60.0, gt=0.0, alias="DURABLE_RUNS_HOLD_STATUS_POLL_SEC")
    # Door-A: how long a completed run's hold is kept for Hub's outreach composition before
    # durable-runs releases it itself (Hub normally releases it within minutes).
    outreach_hold_max_sec: float = Field(1800.0, gt=0.0, alias="DURABLE_RUNS_OUTREACH_HOLD_MAX_SEC")
    # A hold request the pool could not take -- unreachable (RPC timeout), or a refusal that is
    # about the pool rather than the run (version skew "invalid:*", a config roll's
    # "unknown_class") -- keeps the run waiting and asks again after base * 2^(n-1) seconds,
    # capped at max. Each refusal is a visible run.waiting_resource event with the reason.
    pool_retry_base_sec: float = Field(15.0, gt=0.0, alias="DURABLE_RUNS_POOL_RETRY_BASE_SEC")
    pool_retry_max_sec: float = Field(300.0, gt=0.0, alias="DURABLE_RUNS_POOL_RETRY_MAX_SEC")
    retry_max_attempts: int = Field(3, ge=1, le=20, alias="DURABLE_RUNS_RETRY_MAX_ATTEMPTS")
    retry_base_sec: float = Field(30.0, gt=0.0, alias="DURABLE_RUNS_RETRY_BASE_SEC")
    retry_max_sec: float = Field(300.0, gt=0.0, alias="DURABLE_RUNS_RETRY_MAX_SEC")
    # Admitted runs: a run is failed terminally once it has at least
    # RESUME_MAX_FAILURES resume failures since its last real node progress
    # AND the first of them is RESUME_MIN_FAILURE_SPAN_SEC old. Both are
    # needed: the reconcile loop wakes early on other runs' activity, so a
    # count alone could fail healthy runs in a brief infrastructure outage.
    resume_max_failures: int = Field(10, ge=1, le=1000, alias="DURABLE_RUNS_RESUME_MAX_FAILURES")
    resume_min_failure_span_sec: float = Field(600.0, ge=0.0, alias="DURABLE_RUNS_RESUME_MIN_FAILURE_SPAN_SEC")

    # self_study.reflect's llm_call node only: the SAME channel cortex-exec's
    # own self_study.py reads via its CORTEX_REQUEST_CHANNEL env var
    # (default "orion:cortex:request") -- this service becomes a caller of
    # it for the first time with the reflect workflow (every other graph
    # here only talks to Hub and the state channel).
    cortex_request_channel: str = Field("orion:cortex:request", alias="CORTEX_REQUEST_CHANNEL")
    # RPC budget for that single LLM call. Mirrors cortex-exec's own
    # SELF_STUDY_REFLECT_TIMEOUT_SEC (480s -> 1400s, 2026-09-21) plus slack
    # for the round trip -- cortex-exec's synchronous wait for this run's
    # completion event uses its OWN timeout as the real deadline; this is
    # only how long the runner itself waits on the verb-dispatch RPC before
    # giving up and failing the node (resumable, same as harness_turn).
    reflect_llm_call_timeout_sec: float = Field(1500.0, gt=0.0, alias="DURABLE_RUNS_REFLECT_LLM_CALL_TIMEOUT_SEC")

    @model_validator(mode="after")
    def valid_lease_heartbeat(self):
        if self.lease_heartbeat_sec >= self.lease_seconds:
            raise ValueError("lease heartbeat interval must be shorter than lease duration")
        return self

    request_channel: str = DURABLE_RUN_REQUEST_CHANNEL
    state_channel: str = DURABLE_RUN_STATE_CHANNEL

    model_config = {"env_file": ".env", "extra": "ignore", "populate_by_name": True}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

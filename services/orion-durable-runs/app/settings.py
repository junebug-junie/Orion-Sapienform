from __future__ import annotations

from pydantic import Field
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
    # The harness turn RPC to Hub. Hub's own investigation budget is 3500s
    # (HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC); this sits above it so the
    # runner never gives up on a turn Hub is still running.
    turn_rpc_timeout_sec: float = Field(3600.0, gt=0.0, alias="DURABLE_RUNS_TURN_RPC_TIMEOUT_SEC")
    # Orion's own graph, for read_turn_result. Same host/port/name Hub uses
    # (HUB_CURIOSITY_GRAPH_*); blank host = graph half off, same as Hub.
    graph_host: str = Field("", alias="DURABLE_RUNS_GRAPH_HOST")
    graph_port: int = Field(6379, alias="DURABLE_RUNS_GRAPH_PORT")
    graph_own: str = Field("orion_worldview", alias="DURABLE_RUNS_GRAPH_OWN")

    request_channel: str = DURABLE_RUN_REQUEST_CHANNEL
    state_channel: str = DURABLE_RUN_STATE_CHANNEL

    model_config = {"env_file": ".env", "extra": "ignore", "populate_by_name": True}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

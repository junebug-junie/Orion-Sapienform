from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
    )

    service_name: str = Field("orion-heartbeat", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: Optional[str] = Field(None, alias="NODE_NAME")
    instance_id: Optional[str] = Field(None, alias="INSTANCE_ID")

    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    health_channel: str = Field("orion:system:health", alias="ORION_HEALTH_CHANNEL")

    # v0 is additive/read-only: subscribes to the existing, already-standardized
    # grammar stream (see design doc, "Current architecture" -- GrammarEventV1 +
    # orion-substrate-runtime already solved the per-organ ingestion problem the
    # 2026-05-01 heartbeat charter stalled on). No new channel is registered;
    # this is a second, additive consumer of a channel that already exists.
    channel_grammar_event: str = Field("orion:grammar:event", alias="CHANNEL_GRAMMAR_EVENT")

    # H1 (boundary/bulk entanglement) is computed periodically, not on every
    # absorbed atom -- entropy_profile() is cheap (~6ms) but there's no reason
    # to recompute it more often than an operator/debug surface would ever
    # read it.
    h1_interval_sec: float = Field(30.0, alias="HEARTBEAT_H1_INTERVAL_SEC")

    # Deterministic substrate seed (initial random MPS state before any
    # absorption) -- fixed by default so a restart without a snapshot
    # reproduces the same starting point; v0 has no crash-safe persistence
    # (charter's own snapshot/restore machinery is explicitly deferred, see
    # design doc "Explicit deferrals").
    substrate_seed: int = Field(42, alias="HEARTBEAT_SUBSTRATE_SEED")

    http_port: int = Field(7251, alias="HEARTBEAT_HTTP_PORT")


settings = Settings()

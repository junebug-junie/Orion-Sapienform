from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    SERVICE_NAME: str = Field(default="orion-bus")
    SERVICE_VERSION: str = Field(default="0.1.0")
    NODE_NAME: str = Field(default="athena", alias="NODE_NAME")
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health), published by
    # the bus-observer process on its own independent bus connection. See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0, alias="HEARTBEAT_INTERVAL_SEC")

    REDIS_URL: str = Field(default="redis://bus-core:6379/0")

    publish_orion_bus_grammar: bool = Field(False, alias="PUBLISH_ORION_BUS_GRAMMAR")
    grammar_event_channel: str = Field("orion:grammar:event", alias="GRAMMAR_EVENT_CHANNEL")

    bus_observer_enabled: bool = Field(True, alias="BUS_OBSERVER_ENABLED")
    bus_observer_poll_interval_sec: float = Field(10.0, alias="BUS_OBSERVER_POLL_INTERVAL_SEC")
    # 2026-07-18 fix: this field's default is the fallback used whenever
    # `env_file=".env"` (a path relative to process cwd, not to this file)
    # fails to resolve -- e.g. pytest/any invocation from the repo root
    # instead of services/orion-bus/. Confirmed this is a live, not
    # theoretical, path: an orchestrator verification run from repo root hit
    # exactly this fallback and got the OLD default because only
    # .env/.env_example had been updated, not this field -- silently
    # reintroducing the bug this whole fix exists to close. Do not let this
    # default drift from .env_example's BUS_OBSERVER_STREAMS value again.
    #
    # What these keys are used for (2026-09-25): catalog membership
    # (bus_configured_stream_uncataloged, the census-off fallback for
    # catalog_drift_pressure) and a bounded XREVRANGE schema sample
    # (contract_pressure). They are NOT sampled for depth any more: the XLEN
    # depth / backpressure family and BUS_STREAM_DEPTH_WARNING/CRITICAL were
    # retired (fix/bus-observer-scope). A live SCAN found 5 Redis Streams on
    # the whole bus; the only live consumer group sat at lag=0 pending=0, and
    # XLEN is retained length, not backlog. Mesh-wide transport health is the
    # census (BUS_OBSERVER_CENSUS_ENABLED), bus_synaptic and RPC health.
    # The DLQ key does not exist live (XLEN of a missing key reads 0).
    bus_observer_streams: str = Field(
        "orion:stream:world_pulse:run:result,"
        "orion:stream:world_pulse:run:result:dlq",
        alias="BUS_OBSERVER_STREAMS",
    )
    bus_observer_node_id: str = Field("athena", alias="BUS_OBSERVER_NODE_ID")
    # Bounded per-stream XREVRANGE sample size used to check recent entries on
    # each *cataloged* configured stream against that channel's registered
    # schema_id (orion/bus/channels.yaml). Kept small: cost is
    # len(observer_stream_list) * this value extra Redis reads per tick, on
    # top of the existing 1 PING.
    bus_observer_schema_sample_count: int = Field(
        5, alias="BUS_OBSERVER_SCHEMA_SAMPLE_COUNT"
    )

    channels_catalog_path: str = Field(
        "orion/bus/channels.yaml",
        alias="BUS_CHANNELS_CATALOG_PATH",
    )

    # Mesh-wide census diff (orion.bus.census.compute_census() over
    # orion.bus.velocity.scan_active_channels()'s SCAN of the full
    # orion:bus:velocity:* namespace), backing catalog_drift_pressure's fix
    # (2026-07-25). Gated: SCAN cost at real mesh scale was an explicitly
    # named, unmeasured question in the parent design doc -- default False
    # (code-level safe fallback) until live-verified cheap, matching this
    # repo's convention of the real intended default living in .env_example,
    # not here.
    bus_observer_census_enabled: bool = Field(False, alias="BUS_OBSERVER_CENSUS_ENABLED")
    bus_observer_census_window_minutes: int = Field(5, alias="BUS_OBSERVER_CENSUS_WINDOW_MINUTES")

    # Idea 6 (bus_activity_zscore, docs/superpowers/specs/2026-07-24-bus-
    # vitality-field-signal-brainstorm.md) -- EWMA smoothing for the rolling
    # total-mesh-publish-rate baseline. Rides on bus_observer_census_enabled
    # (reuses that same scan_active_channels() call, no separate gate) --
    # same alpha default as the bus-synaptic-graph arc's own EWMA convention.
    bus_activity_ewma_alpha: float = Field(0.2, alias="BUS_ACTIVITY_EWMA_ALPHA", ge=0.0, le=1.0)

    @property
    def observer_stream_list(self) -> list[str]:
        return [s.strip() for s in self.bus_observer_streams.split(",") if s.strip()]


settings = Settings()

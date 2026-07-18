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
    # orion:evt:gateway/orion:bus:out were placeholder names from the
    # original bus-observer commit (ee810551, 2026-05-25), never once
    # cataloged in orion/bus/channels.yaml at any point in git history, and
    # verified live TYPE=none (no key exists at all) -- structurally
    # incapable of ever producing a depth sample, confirmed by 180
    # substrate_reduction_receipts over 15 days reading transport_pressure/
    # stream_depth_pressure/backpressure at a flat 0.0 the entire time.
    # orion:grammar:event is cataloged but Pub/Sub-only (verified live
    # TYPE=none, delivered via OrionBusAsync.publish() -> redis.publish(),
    # never XADD'd) -- XLEN/XREVRANGE can never see it regardless of real
    # traffic. orion-bus today routes almost everything through pub/sub,
    # which has no persistent backlog to observe; depth/backpressure is only
    # meaningful for genuinely XADD'd channels. orion/bus/channels.yaml
    # catalogs exactly two kind="stream" channels, and both are now the
    # default: orion:stream:world_pulse:run:result (real XADD by
    # orion-world-pulse, verified live TYPE=stream XLEN=82) and its
    # dead-letter sibling orion:stream:world_pulse:run:result:dlq (verified
    # live TYPE=none/XLEN=0 right now -- expected-healthy for a DLQ, not
    # broken; a nonzero DLQ depth is itself a real failure signal worth
    # having wired).
    bus_observer_streams: str = Field(
        "orion:stream:world_pulse:run:result,"
        "orion:stream:world_pulse:run:result:dlq",
        alias="BUS_OBSERVER_STREAMS",
    )
    bus_stream_depth_warning: int = Field(25000, alias="BUS_STREAM_DEPTH_WARNING")
    bus_stream_depth_critical: int = Field(100000, alias="BUS_STREAM_DEPTH_CRITICAL")
    bus_observer_node_id: str = Field("athena", alias="BUS_OBSERVER_NODE_ID")
    # Bounded per-stream XREVRANGE sample size used to check recent entries on
    # each *cataloged* configured stream against that channel's registered
    # schema_id (orion/bus/channels.yaml). Kept small: cost is
    # len(observer_stream_list) * this value extra Redis reads per tick, on
    # top of the existing 1 PING + len(observer_stream_list) XLEN calls.
    bus_observer_schema_sample_count: int = Field(
        5, alias="BUS_OBSERVER_SCHEMA_SAMPLE_COUNT"
    )

    channels_catalog_path: str = Field(
        "orion/bus/channels.yaml",
        alias="BUS_CHANNELS_CATALOG_PATH",
    )

    @property
    def observer_stream_list(self) -> list[str]:
        return [s.strip() for s in self.bus_observer_streams.split(",") if s.strip()]


settings = Settings()

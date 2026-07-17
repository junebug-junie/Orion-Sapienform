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
    # orion:evt:gateway/orion:bus:out are depth/backpressure-monitoring
    # targets only. Verified live against bus-core Redis (TYPE + XLEN):
    # both come back TYPE=none -- neither is an existing key of any kind,
    # let alone a Stream. They are also not cataloged in
    # orion/bus/channels.yaml, so they never contribute a schema_id either
    # way; schema-mismatch sampling (contract_pressure) never fires for
    # them, and (separately, pre-existing, not touched here) their
    # depth/backpressure numbers have presumably always read 0 too.
    #
    # orion:grammar:event IS cataloged (schema_id=GrammarEventV1) but is
    # NOT a fix for schema-mismatch sampling being dead: verified live
    # (TYPE orion:grammar:event -> "none") that it is delivered purely via
    # Redis PUBLISH (OrionBusAsync.publish() -> redis.publish(), see
    # orion/grammar/publish.py / orion/core/bus/async_service.py) and never
    # XADD'd anywhere in this codebase -- Pub/Sub messages are never stored,
    # so XLEN/XREVRANGE against this key structurally can never see
    # anything, no matter how much real traffic passes through it. Kept in
    # the default anyway (harmless, correctly cataloged, and would start
    # working for free if a future change ever dual-writes it to a Stream)
    # but it is NOT what makes schema-mismatch sampling live.
    #
    # orion:stream:world_pulse:run:result is the channel that actually
    # fixes it: cataloged (schema_id=WorldPulseRunResultV1), kind="stream"
    # in channels.yaml, written via RedisStreamWorkQueue.enqueue() (real
    # XADD) by orion-world-pulse, gated by WP_RUN_RESULT_STREAM_ENABLED
    # which defaults to true in services/orion-world-pulse/.env_example.
    # Verified live against bus-core Redis: TYPE=stream, XLEN=82 real
    # entries at verification time, and a sampled XREVRANGE entry decodes
    # via the "envelope" field exactly as count_schema_mismatches() expects.
    # Low/bursty volume ("once per run" per the comment in channels.yaml
    # above this entry) -- sampled_count can legitimately be 0 between runs,
    # that's expected, not a bug.
    bus_observer_streams: str = Field(
        "orion:evt:gateway,orion:bus:out,orion:grammar:event,"
        "orion:stream:world_pulse:run:result",
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

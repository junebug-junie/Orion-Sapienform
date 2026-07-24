from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ORION_BUS_URL: str = Field(default="redis://localhost:6379/0")
    MIRROR_PATTERN: str = Field(default="orion:*")
    MIRROR_SQLITE_PATH: str = Field(default="/data/bus_mirror.sqlite")
    MIRROR_PARQUET_DIR: str = Field(default="/data/parquet")

    # Bus synaptic graph (Phase 1 of docs/superpowers/specs/2026-07-24-bus-vitality-field-signal-brainstorm.md's
    # "Big-swing direction"). Bounded, aggregated FalkorDB edges instead of the
    # raw-log SQLite path above -- additive, independently gated, off by default.
    MIRROR_GRAPH_ENABLED: bool = Field(default=False)
    # Hard-defaulted to the standard FalkorDB hostname (matching every other
    # FALKORDB_URI consumer in this repo -- orion-recall, orion-meta-tags,
    # orion-substrate-runtime, etc.) rather than falling back to ORION_BUS_URL
    # if unset. Falling back to the bus URL would silently point GRAPH.QUERY
    # calls at the pub/sub Redis instead of FalkorDB -- caught in review.
    FALKORDB_URI: str = Field(default="redis://orion-athena-falkordb:6379")
    FALKORDB_BUS_GRAPH: str = Field(default="orion_bus_synapse")
    # Bounded TTL for the in-memory correlation_id -> (organ, epoch) lookup used
    # to derive CAUSALLY_FOLLOWED_BY edges. Entries older than this are evicted
    # so the table can't grow unboundedly under full-tilt "orion:*" traffic --
    # the same O(N)-growth failure class this repo has hit before (see
    # feedback_substrate_performance / feedback_execution_merge_cap memories).
    MIRROR_GRAPH_CHAIN_TTL_SEC: float = Field(default=120.0)
    MIRROR_GRAPH_EWMA_ALPHA: float = Field(default=0.2, ge=0.0, le=1.0)

    # Phase 2: in-flight/long-running chain visibility. Threshold is
    # deliberately well below MIRROR_GRAPH_CHAIN_TTL_SEC's default (120s) so a
    # genuinely slow, still-active multi-hop chain gets flagged before it
    # would ever be evicted as stale.
    MIRROR_GRAPH_INFLIGHT_LOG_INTERVAL_SEC: float = Field(default=30.0)
    MIRROR_GRAPH_LONG_RUNNING_THRESHOLD_SEC: float = Field(default=30.0)
    # Caps payload.steps[] processing per envelope -- a buggy/adversarial
    # producer sending a huge steps array would otherwise stall the mirror's
    # message loop for the full duration of writing that many edges
    # sequentially (found in review). Real cognition traces have a handful
    # of steps; 200 is generous headroom, not a tuned real-world ceiling.
    MIRROR_GRAPH_MAX_VERB_STEPS: int = Field(default=200, ge=1)


settings = Settings()

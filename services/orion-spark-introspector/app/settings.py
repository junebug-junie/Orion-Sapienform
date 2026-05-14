from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Identity
    service_name: str = Field("spark-introspector", alias="SERVICE_NAME")
    service_version: str = Field("0.2.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    # Bus
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")

    # Chassis
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    # Consume candidates (legacy dict payloads and/or envelopes)
    channel_spark_candidate: str = Field(
        "orion:spark:introspect:candidate*",
        validation_alias=AliasChoices("CHANNEL_SPARK_INTROSPECT_CANDIDATE", "SPARK_CANDIDATE_CHANNEL"),
    )

    # Cognition Trace Intake
    channel_cognition_trace_pub: str = Field("orion:cognition:trace", alias="CHANNEL_COGNITION_TRACE_PUB")

    # Durable telemetry output (SQL writer ingests this)
    channel_spark_telemetry: str = Field("orion:spark:telemetry", alias="CHANNEL_SPARK_TELEMETRY")

    # Real-time snapshot stream (state-service + UI can subscribe)
    channel_spark_state_snapshot: str = Field(
        "orion:spark:state:snapshot",
        alias="CHANNEL_SPARK_STATE_SNAPSHOT",
    )

    # Spark signals (normalized distress/equilibrium)
    channel_spark_signal: str = Field("orion:spark:signal", alias="CHANNEL_SPARK_SIGNAL")

    # Core events (legacy bus events)
    channel_core_events: str = Field("orion:core:events", alias="CHANNEL_CORE_EVENTS")

    # Semantic vector upserts
    channel_vector_semantic_upsert: str = Field(
        "orion:vector:semantic:upsert",
        alias="CHANNEL_VECTOR_SEMANTIC_UPSERT",
    )

    # Embedding RPC (to orion-vector-host) for valence anchors
    channel_embedding_generate: str = Field(
        "orion:embedding:generate",
        alias="CHANNEL_EMBEDDING_GENERATE",
    )
    embedding_result_prefix: str = Field(
        "orion:embedding:result:",
        alias="EMBEDDING_RESULT_PREFIX",
    )

    spark_vector_collection: str = Field(
        "orion_spark_store",
        alias="SPARK_VECTOR_COLLECTION",
    )

    # Valence anchor texts (semantic axis), gain, and refresh cadence
    valence_anchor_pos_text: str = Field(
        "I feel hopeful and grateful.",
        alias="VALENCE_ANCHOR_POS_TEXT",
    )
    valence_anchor_neg_text: str = Field(
        "I feel hopeless and afraid.",
        alias="VALENCE_ANCHOR_NEG_TEXT",
    )
    valence_gain: float = Field(
        0.35,
        alias="VALENCE_GAIN",
    )
    valence_anchor_refresh_sec: int = Field(
        6 * 60 * 60,
        alias="VALENCE_ANCHOR_REFRESH_SEC",
    )
    valence_anchor_timeout_sec: int = Field(
        10,
        alias="VALENCE_ANCHOR_TIMEOUT_SEC",
    )

    # Freshness semantics (for read-model)
    spark_state_valid_for_ms: int = Field(15000, alias="SPARK_STATE_VALID_FOR_MS")

    # Tissue
    orion_tissue_snapshot_path: str = Field(
        "/mnt/storage-lukewarm/orion/spark/tissue-brain.npy",
        alias="ORION_TISSUE_SNAPSHOT_PATH",
    )

    # RPC to Cortex-Orch (Spark -> Cortex-Orch)
    channel_cortex_request: str = Field(
        "orion:cortex:request",
        validation_alias=AliasChoices("CORTEX_REQUEST_CHANNEL", "CORTEX_ORCH_REQUEST_CHANNEL", "ORCH_REQUEST_CHANNEL"),
    )

    # How long to wait for Cortex-Orch RPC reply
    cortex_timeout_sec: float = Field(120.0, alias="CORTEX_TIMEOUT_SEC")

    # Stamped into orch request metadata (orch resolver remains authoritative)
    spark_introspection_execution_lane: str = Field("spark", alias="SPARK_INTROSPECTION_EXECUTION_LANE")
    spark_introspection_llm_lane: str = Field("spark", alias="SPARK_INTROSPECTION_LLM_LANE")
    spark_introspection_allow_chat_fallback: bool = Field(False, alias="SPARK_INTROSPECTION_ALLOW_CHAT_FALLBACK")
    spark_introspection_max_tokens: int = Field(384, alias="SPARK_INTROSPECTION_MAX_TOKENS")

    # Phase 1 heavy-path controls (semaphore, Redis idempotency, staleness)
    spark_introspection_enable_heavy: bool = Field(True, alias="SPARK_INTROSPECTION_ENABLE_HEAVY")
    spark_introspection_max_inflight: int = Field(1, alias="SPARK_INTROSPECTION_MAX_INFLIGHT")
    spark_introspection_timeout_sec: float = Field(45.0, alias="SPARK_INTROSPECTION_TIMEOUT_SEC")
    spark_introspection_queue_max_age_sec: float = Field(180.0, alias="SPARK_INTROSPECTION_QUEUE_MAX_AGE_SEC")
    spark_introspection_drop_on_pressure: bool = Field(True, alias="SPARK_INTROSPECTION_DROP_ON_PRESSURE")
    spark_introspection_acquire_timeout_sec: float = Field(0.0, alias="SPARK_INTROSPECTION_ACQUIRE_TIMEOUT_SEC")
    spark_introspection_min_interval_sec: float = Field(0.0, alias="SPARK_INTROSPECTION_MIN_INTERVAL_SEC")
    spark_introspection_idempotency_enable: bool = Field(True, alias="SPARK_INTROSPECTION_IDEMPOTENCY_ENABLE")
    spark_introspection_redis_url: Optional[str] = Field(None, alias="SPARK_INTROSPECTION_REDIS_URL")
    spark_introspection_key_prefix: str = Field("spark:introspection", alias="SPARK_INTROSPECTION_KEY_PREFIX")
    spark_introspection_inflight_ttl_sec: int = Field(300, alias="SPARK_INTROSPECTION_INFLIGHT_TTL_SEC")
    spark_introspection_done_ttl_sec: int = Field(86400, alias="SPARK_INTROSPECTION_DONE_TTL_SEC")

    # Phase 4C queue-backed heavy introspection
    spark_introspection_queue_enabled: bool = Field(False, alias="SPARK_INTROSPECTION_QUEUE_ENABLED")
    spark_introspection_queue_stream: str = Field(
        "orion:queue:spark:introspection", alias="SPARK_INTROSPECTION_QUEUE_STREAM"
    )
    spark_introspection_queue_group: str = Field(
        "spark-introspector-workers", alias="SPARK_INTROSPECTION_QUEUE_GROUP"
    )
    spark_introspection_queue_consumer: str = Field("", alias="SPARK_INTROSPECTION_QUEUE_CONSUMER")
    spark_introspection_queue_dlq: str = Field(
        "orion:queue:spark:introspection:dlq", alias="SPARK_INTROSPECTION_QUEUE_DLQ"
    )
    spark_introspection_queue_maxlen: Optional[int] = Field(None, alias="SPARK_INTROSPECTION_QUEUE_MAXLEN")
    spark_introspection_queue_read_count: int = Field(1, alias="SPARK_INTROSPECTION_QUEUE_READ_COUNT")
    spark_introspection_queue_block_ms: int = Field(5000, alias="SPARK_INTROSPECTION_QUEUE_BLOCK_MS")
    spark_introspection_queue_max_inflight: int = Field(1, alias="SPARK_INTROSPECTION_QUEUE_MAX_INFLIGHT")
    spark_introspection_queue_max_attempts: int = Field(3, alias="SPARK_INTROSPECTION_QUEUE_MAX_ATTEMPTS")
    spark_introspection_queue_reclaim_pending: bool = Field(True, alias="SPARK_INTROSPECTION_QUEUE_RECLAIM_PENDING")
    spark_introspection_queue_reclaim_min_idle_ms: int = Field(
        120_000, alias="SPARK_INTROSPECTION_QUEUE_RECLAIM_MIN_IDLE_MS"
    )
    spark_introspection_queue_stale_policy: str = Field("drop", alias="SPARK_INTROSPECTION_QUEUE_STALE_POLICY")
    spark_introspection_inline_heavy_enabled: bool = Field(True, alias="SPARK_INTROSPECTION_INLINE_HEAVY_ENABLED")
    # GET /debug/spark/introspection-queue — off by default; when on, requires shared secret (header or query).
    spark_introspection_queue_debug_enabled: bool = Field(False, alias="SPARK_INTROSPECTION_QUEUE_DEBUG_ENABLE")
    spark_introspection_queue_debug_token: Optional[str] = Field(None, alias="SPARK_INTROSPECTION_QUEUE_DEBUG_TOKEN")

    # Web UI
    port: int = Field(8444, alias="PORT")

    # Turn effect alerting (opt-in)
    turn_effect_alerts_enable: bool = Field(False, alias="TURN_EFFECT_ALERTS_ENABLE")
    turn_effect_alerts_coherence_drop: float = Field(0.25, alias="TURN_EFFECT_ALERTS_COHERENCE_DROP")
    turn_effect_alerts_valence_drop: float = Field(0.25, alias="TURN_EFFECT_ALERTS_VALENCE_DROP")
    turn_effect_alerts_novelty_spike: float = Field(0.35, alias="TURN_EFFECT_ALERTS_NOVELTY_SPIKE")
    turn_effect_alerts_cooldown_sec: int = Field(120, alias="TURN_EFFECT_ALERTS_COOLDOWN_SEC")
    turn_effect_alerts_notify_enable: bool = Field(False, alias="TURN_EFFECT_ALERTS_NOTIFY_ENABLE")
    turn_effect_alerts_dedupe_enable: bool = Field(True, alias="TURN_EFFECT_ALERTS_DEDUPE_ENABLE")
    turn_effect_alerts_dedupe_window_sec: int = Field(600, alias="TURN_EFFECT_ALERTS_DEDUPE_WINDOW_SEC")
    turn_effect_alerts_dedupe_eps: float = Field(0.02, alias="TURN_EFFECT_ALERTS_DEDUPE_EPS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-field-digester", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    lattice_path: str = Field(
        "config/field/orion_field_topology.v1.yaml",
        alias="LATTICE_PATH",
    )
    receipt_poll_interval_sec: float = Field(2.0, alias="RECEIPT_POLL_INTERVAL_SEC")
    biometrics_field_decay_rate: float = Field(0.92, alias="BIOMETRICS_FIELD_DECAY_RATE")
    biometrics_field_diffusion_rate: float = Field(1.0, alias="BIOMETRICS_FIELD_DIFFUSION_RATE")
    enable_transport_field_digestion: bool = Field(
        False,
        alias="ENABLE_TRANSPORT_FIELD_DIGESTION",
    )
    enable_idle_tick: bool = Field(True, alias="FIELD_DIGESTER_IDLE_TICK_ENABLED")
    field_state_retention_hours: float = Field(72.0, alias="FIELD_STATE_RETENTION_HOURS")
    field_state_prune_interval_sec: float = Field(3600.0, alias="FIELD_STATE_PRUNE_INTERVAL_SEC")
    field_applied_deltas_prune_min_age_hours: float = Field(
        1.0, alias="FIELD_APPLIED_DELTAS_PRUNE_MIN_AGE_HOURS"
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Health monitor -> orion-notify attention alerts. Edge-triggered (fires only
    # on healthy->unhealthy transitions), not polled-and-spammed.
    health_check_interval_sec: float = Field(900.0, alias="FIELD_DIGESTER_HEALTH_CHECK_INTERVAL_SEC")
    field_state_stall_multiplier: float = Field(1.5, alias="FIELD_STATE_STALL_MULTIPLIER")
    applied_deltas_alert_row_count: int = Field(
        5_000_000, alias="FIELD_APPLIED_DELTAS_ALERT_ROW_COUNT"
    )
    # Default set with real headroom above the observed conjourney baseline
    # (~37.5GB as of 2026-07-12) -- not a round number picked in the abstract.
    db_size_alert_gb: float = Field(60.0, alias="FIELD_DIGESTER_DB_SIZE_ALERT_GB")
    notify_base_url: str = Field("http://orion-athena-notify:7140", alias="NOTIFY_BASE_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")

    # Field-channel raw-substrate corpus collector (Item 1 v2, roadmap item 1
    # correction, 2026-07-13) -- docs/superpowers/specs/2026-07-13-felt-
    # state-arc-roadmap-spec.md. Off by default: a real, live precedent of
    # unbounded corpus growth from a sibling sink (INNER_FEATURES_CORPUS_PATH,
    # ~104MB/36.8k rows over 5 days) means anything new stays inert until an
    # operator opts in.
    field_channel_corpus_path: str = Field("", alias="FIELD_CHANNEL_CORPUS_PATH")
    # Same rotation/retention policy shape as orion-spark-introspector's
    # corpus_sink_max_bytes/corpus_sink_rotated_keep (services/orion-spark-
    # introspector/app/settings.py) -- each service reads its own env file
    # independently, so there is no collision between the two services'
    # settings despite the identical field names/aliases/defaults.
    corpus_sink_max_bytes: int = Field(200_000_000, ge=1_000_000, alias="CORPUS_SINK_MAX_BYTES")
    corpus_sink_rotated_keep: int = Field(5, ge=0, alias="CORPUS_SINK_ROTATED_KEEP")

    # Causal Geometry v1 follow-up: the scheduled Phase A -> Phase B producer
    # (orion/substrate/causal_geometry_producer.py, run from
    # app/worker.py's _causal_geometry_producer_loop). Independent of
    # FIELD_PLASTICITY_ENABLED (which gates *consuming* the learned overlay in
    # diffusion) -- this gates *producing* proposals into the HITL queue.
    # Off by default: an operator must opt in explicitly, same posture as
    # every other plasticity flag in this file.
    field_plasticity_producer_enabled: bool = Field(False, alias="FIELD_PLASTICITY_PRODUCER_ENABLED")
    # 24h: matches the design spec's "deterministic nightly job" framing for
    # Phase A. This is a full Postgres pull + O(channels^2) lagged-correlation
    # + surrogate-significance computation, not something to run on the
    # digester's own ~2s tick cadence.
    field_plasticity_producer_interval_hours: float = Field(
        24.0, alias="FIELD_PLASTICITY_PRODUCER_INTERVAL_HOURS"
    )
    # Matches causal_geometry_engine.DEFAULT_WINDOW_HOURS (168h / 7 days).
    field_plasticity_producer_window_hours: float = Field(
        168.0, alias="FIELD_PLASTICITY_PRODUCER_WINDOW_HOURS"
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

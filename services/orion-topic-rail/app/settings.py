from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple
import json

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Identity
    service_name: str = Field("orion-topic-rail", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Database
    topic_rail_pg_dsn: str = Field(
        "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        validation_alias=AliasChoices("TOPIC_RAIL_PG_DSN", "POSTGRES_URI"),
    )

    # Embeddings
    topic_rail_embedding_url: str = Field(
        "http://orion-vector-host:8000/embedding",
        alias="TOPIC_RAIL_EMBEDDING_URL",
    )

    # Model artifacts
    topic_rail_model_dir: str = Field(
        "/mnt/telemetry/models/topic-rail",
        alias="TOPIC_RAIL_MODEL_DIR",
    )
    topic_rail_model_version: str = Field(
        ..., alias="TOPIC_RAIL_MODEL_VERSION"
    )

    # Loop behavior
    topic_rail_poll_seconds: int = Field(300, alias="TOPIC_RAIL_POLL_SECONDS")
    topic_rail_train_limit: int = Field(5000, alias="TOPIC_RAIL_TRAIN_LIMIT")
    topic_rail_assign_limit: int = Field(500, alias="TOPIC_RAIL_ASSIGN_LIMIT")
    topic_rail_train_time_window_days: Optional[int] = Field(
        None, alias="TOPIC_RAIL_TRAIN_TIME_WINDOW_DAYS"
    )
    topic_rail_run_once: bool = Field(False, alias="TOPIC_RAIL_RUN_ONCE")
    topic_rail_mode: str = Field("daemon", alias="TOPIC_RAIL_MODE")
    topic_rail_force_refit: bool = Field(False, alias="TOPIC_RAIL_FORCE_REFIT")

    # Document build
    topic_rail_doc_mode: str = Field("prompt+response", alias="TOPIC_RAIL_DOC_MODE")
    topic_rail_include_session: bool = Field(
        True, alias="TOPIC_RAIL_INCLUDE_SESSION"
    )

    # BERTopic
    topic_rail_calc_probs: bool = Field(True, alias="TOPIC_RAIL_CALC_PROBS")
    topic_rail_use_keybert: bool = Field(False, alias="TOPIC_RAIL_USE_KEYBERT")
    topic_rail_min_topic_size: int = Field(15, alias="TOPIC_RAIL_MIN_TOPIC_SIZE")
    topic_rail_ngram_range: str = Field("1,2", alias="TOPIC_RAIL_NGRAM_RANGE")
    topic_rail_stopwords: Optional[str] = Field(
        "english", alias="TOPIC_RAIL_STOPWORDS"
    )

    # Retry behavior
    topic_rail_embedding_retries: int = Field(
        2, alias="TOPIC_RAIL_EMBEDDING_RETRIES"
    )
    topic_rail_embedding_backoff_sec: float = Field(
        0.75, alias="TOPIC_RAIL_EMBEDDING_BACKOFF_SEC"
    )

    # Summary outputs
    topic_rail_summary_enabled: bool = Field(False, alias="TOPIC_RAIL_SUMMARY_ENABLED")
    topic_rail_summary_window_minutes: int = Field(
        1440, alias="TOPIC_RAIL_SUMMARY_WINDOW_MINUTES"
    )
    topic_rail_summary_min_docs: int = Field(
        10, alias="TOPIC_RAIL_SUMMARY_MIN_DOCS"
    )
    topic_rail_summary_max_topics: int = Field(
        20, alias="TOPIC_RAIL_SUMMARY_MAX_TOPICS"
    )

    # Drift outputs
    topic_rail_drift_enabled: bool = Field(False, alias="TOPIC_RAIL_DRIFT_ENABLED")
    topic_rail_drift_window_minutes: int = Field(
        1440, alias="TOPIC_RAIL_DRIFT_WINDOW_MINUTES"
    )
    topic_rail_drift_min_turns: int = Field(
        10, alias="TOPIC_RAIL_DRIFT_MIN_TURNS"
    )

    # Bus publishing (optional)
    topic_rail_bus_publish_enabled: bool = Field(
        False, alias="TOPIC_RAIL_BUS_PUBLISH_ENABLED"
    )
    topic_rail_bus_url: str = Field(
        "redis://100.92.216.81:6379/0", alias="TOPIC_RAIL_BUS_URL"
    )
    topic_rail_bus_topic_summary_channel: str = Field(
        "orion:topic:summary.v1", alias="TOPIC_RAIL_BUS_TOPIC_SUMMARY_CHANNEL"
    )
    topic_rail_bus_topic_shift_channel: str = Field(
        "orion:topic:shift.v1", alias="TOPIC_RAIL_BUS_TOPIC_SHIFT_CHANNEL"
    )
    topic_rail_bus_topic_assigned_channel: str = Field(
        "orion:topic:rail:assigned.v1", alias="TOPIC_RAIL_BUS_TOPIC_ASSIGNED_CHANNEL"
    )
    topic_rail_shift_switch_rate_threshold: float = Field(
        0.35, alias="TOPIC_RAIL_SHIFT_SWITCH_RATE_THRESHOLD"
    )
    topic_rail_allow_embed_model_mismatch: bool = Field(
        False, alias="TOPIC_RAIL_ALLOW_EMBED_MODEL_MISMATCH"
    )

    # Lifecycle policy
    topic_rail_refit_policy: str = Field(
        "never", alias="TOPIC_RAIL_REFIT_POLICY"
    )
    topic_rail_refit_ttl_hours: int = Field(
        168, alias="TOPIC_RAIL_REFIT_TTL_HOURS"
    )
    topic_rail_refit_doc_threshold: int = Field(
        20000, alias="TOPIC_RAIL_REFIT_DOC_THRESHOLD"
    )
    topic_rail_allow_refit_in_daemon: bool = Field(
        False, alias="TOPIC_RAIL_ALLOW_REFIT_IN_DAEMON"
    )

    # Outlier handling
    topic_rail_outlier_enabled: bool = Field(
        True, alias="TOPIC_RAIL_OUTLIER_ENABLED"
    )
    topic_rail_outlier_max_pct: float = Field(
        0.40, alias="TOPIC_RAIL_OUTLIER_MAX_PCT"
    )

    # HTTP health endpoint
    topic_rail_http_enabled: bool = Field(
        False, alias="TOPIC_RAIL_HTTP_ENABLED"
    )
    topic_rail_http_port: int = Field(
        8610, alias="TOPIC_RAIL_HTTP_PORT"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True

    @field_validator("topic_rail_model_version")
    @classmethod
    def _require_model_version(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("TOPIC_RAIL_MODEL_VERSION is required")
        return v

    @property
    def ngram_range(self) -> Tuple[int, int]:
        raw = str(self.topic_rail_ngram_range or "1,1")
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) == 1:
            val = int(parts[0])
            return (val, val)
        return (int(parts[0]), int(parts[1]))

    @property
    def stopwords(self) -> Optional[str]:
        raw = self.topic_rail_stopwords
        if raw is None:
            return None
        if isinstance(raw, str) and raw.strip().lower() in {"none", "null", ""}:
            return None
        return raw

    def settings_snapshot(self) -> dict:
        return {
            "model_version": self.topic_rail_model_version,
            "doc_mode": self.topic_rail_doc_mode,
            "min_topic_size": self.topic_rail_min_topic_size,
            "ngram_range": list(self.ngram_range),
            "stopwords": self.stopwords,
            "calc_probs": self.topic_rail_calc_probs,
            "use_keybert": self.topic_rail_use_keybert,
            "train_limit": self.topic_rail_train_limit,
            "assign_limit": self.topic_rail_assign_limit,
            "train_time_window_days": self.topic_rail_train_time_window_days,
            "run_once": self.topic_rail_run_once,
            "mode": self.topic_rail_mode,
            "force_refit": self.topic_rail_force_refit,
            "summary_enabled": self.topic_rail_summary_enabled,
            "summary_window_minutes": self.topic_rail_summary_window_minutes,
            "summary_min_docs": self.topic_rail_summary_min_docs,
            "summary_max_topics": self.topic_rail_summary_max_topics,
            "drift_enabled": self.topic_rail_drift_enabled,
            "drift_window_minutes": self.topic_rail_drift_window_minutes,
            "drift_min_turns": self.topic_rail_drift_min_turns,
            "bus_publish_enabled": self.topic_rail_bus_publish_enabled,
            "bus_topic_summary_channel": self.topic_rail_bus_topic_summary_channel,
            "bus_topic_shift_channel": self.topic_rail_bus_topic_shift_channel,
            "bus_topic_assigned_channel": self.topic_rail_bus_topic_assigned_channel,
            "shift_switch_rate_threshold": self.topic_rail_shift_switch_rate_threshold,
            "allow_embed_model_mismatch": self.topic_rail_allow_embed_model_mismatch,
            "refit_policy": self.topic_rail_refit_policy,
            "refit_ttl_hours": self.topic_rail_refit_ttl_hours,
            "refit_doc_threshold": self.topic_rail_refit_doc_threshold,
            "allow_refit_in_daemon": self.topic_rail_allow_refit_in_daemon,
            "outlier_enabled": self.topic_rail_outlier_enabled,
            "outlier_max_pct": self.topic_rail_outlier_max_pct,
            "http_enabled": self.topic_rail_http_enabled,
            "http_port": self.topic_rail_http_port,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

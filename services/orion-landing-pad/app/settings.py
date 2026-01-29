from __future__ import annotations

from uuid import uuid4
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = Field("orion-landing-pad", alias="APP_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("unknown", alias="NODE_NAME")
    instance_id: Optional[str] = Field(None, alias="INSTANCE_ID")
    boot_id: str = Field(default_factory=lambda: str(uuid4()), alias="BOOT_ID")
    port: int = Field(8370, alias="PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    orion_bus_url: str = Field("redis://orion-redis:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")

    pad_input_allowlist_patterns: List[str] | str = Field(
        default_factory=lambda: [
            "orion:telemetry:*",
            "orion:cortex:*",
            "orion:spark:*",
        ],
        alias="PAD_INPUT_ALLOWLIST_PATTERNS",
    )
    pad_input_denylist_patterns: List[str] | str = Field(
        default_factory=lambda: ["orion:pad:*"],
        alias="PAD_INPUT_DENYLIST_PATTERNS",
    )

    pad_output_event_channel: str = Field("orion:pad:event", alias="PAD_OUTPUT_EVENT_CHANNEL")
    pad_output_frame_channel: str = Field("orion:pad:frame", alias="PAD_OUTPUT_FRAME_CHANNEL")
    pad_output_signal_channel: str = Field("orion:pad:signal", alias="PAD_OUTPUT_SIGNAL_CHANNEL")
    pad_output_stats_channel: str = Field("orion:pad:stats", alias="PAD_OUTPUT_STATS_CHANNEL")

    pad_rpc_request_channel: str = Field("orion:pad:rpc:request", alias="PAD_RPC_REQUEST_CHANNEL")

    pad_frame_tick_ms: int = Field(5000, alias="PAD_FRAME_TICK_MS")
    pad_frame_window_ms: int = Field(15000, alias="PAD_FRAME_WINDOW_MS")
    pad_max_events_per_tick: int = Field(50, alias="PAD_MAX_EVENTS_PER_TICK")

    pad_min_salience: float = Field(0.05, alias="PAD_MIN_SALIENCE")
    pad_pulse_salience: float = Field(0.8, alias="PAD_PULSE_SALIENCE")

    pad_max_queue_size: int = Field(500, alias="PAD_MAX_QUEUE_SIZE")
    pad_queue_drop_policy: str = Field("drop_low_priority_first", alias="PAD_QUEUE_DROP_POLICY")

    pad_event_ttl_sec: int = Field(600, alias="PAD_EVENT_TTL_SEC")
    pad_frame_ttl_sec: int = Field(900, alias="PAD_FRAME_TTL_SEC")

    pad_events_stream_key: str = Field("orion:pad:events", alias="PAD_EVENTS_STREAM_KEY")
    pad_frames_stream_key: str = Field("orion:pad:frames", alias="PAD_FRAMES_STREAM_KEY")
    pad_stream_maxlen: int = Field(500, alias="PAD_STREAM_MAXLEN")

    pad_stats_tick_sec: int = Field(15, alias="PAD_STATS_TICK_SEC")
    pad_tensor_dim: int = Field(32, alias="PAD_TENSOR_DIM")

    redis_url: Optional[str] = Field(None, alias="REDIS_URL")

    public_base_path: str = Field("/landing-pad", alias="PUBLIC_BASE_PATH")
    ui_sample_limit: int = Field(500, alias="UI_SAMPLE_LIMIT")
    ui_query_limit: int = Field(2000, alias="UI_QUERY_LIMIT")
    ui_default_lookback_minutes: int = Field(60, alias="UI_DEFAULT_LOOKBACK_MINUTES")

    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    shutdown_grace_sec: float = Field(10.0, alias="SHUTDOWN_GRACE_SEC")
    health_channel: str = Field("orion:system:health", alias="ORION_HEALTH_CHANNEL")
    error_channel: str = Field("orion:system:error", alias="ERROR_CHANNEL")

    @field_validator("pad_input_allowlist_patterns", "pad_input_denylist_patterns", mode="before")
    @classmethod
    def _split_patterns(cls, v: List[str] | str | None) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        # Accept comma-separated env strings
        parts = [p.strip() for p in str(v).split(",") if p.strip()]
        return parts

    @field_validator("public_base_path")
    @classmethod
    def _normalize_base_path(cls, v: str) -> str:
        if not v:
            return "/"
        base = v.strip()
        if not base.startswith("/"):
            base = f"/{base}"
        if base != "/" and base.endswith("/"):
            base = base.rstrip("/")
        return base

    def merged_redis_url(self) -> str:
        return self.redis_url or self.orion_bus_url


settings = Settings()

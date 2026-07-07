from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-substrate-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")

    enable_biometrics_node_reducer: bool = Field(True, alias="ENABLE_BIOMETRICS_NODE_REDUCER")
    enable_biometrics_pressure_organ: bool = Field(True, alias="ENABLE_BIOMETRICS_PRESSURE_ORGAN")
    enable_node_pressure_reducer: bool = Field(True, alias="ENABLE_NODE_PRESSURE_REDUCER")
    enable_execution_trajectory_reducer: bool = Field(
        False,
        alias="ENABLE_EXECUTION_TRAJECTORY_REDUCER",
    )
    enable_transport_bus_reducer: bool = Field(
        False,
        alias="ENABLE_TRANSPORT_BUS_REDUCER",
    )
    enable_chat_grammar_reducer: bool = Field(False, alias="ENABLE_CHAT_GRAMMAR_REDUCER")
    chat_grammar_batch_limit: int = Field(100, alias="CHAT_GRAMMAR_BATCH_LIMIT")
    bus_stream_depth_critical: int = Field(100_000, alias="BUS_STREAM_DEPTH_CRITICAL")
    transport_substrate_maturity: str = Field(
        "trace_only",
        alias="TRANSPORT_SUBSTRATE_MATURITY",
    )
    biometrics_node_stale_after_sec: int = Field(180, alias="BIOMETRICS_NODE_STALE_AFTER_SEC")
    biometrics_pressure_min_confidence: float = Field(0.60, alias="BIOMETRICS_PRESSURE_MIN_CONFIDENCE")
    node_catalog_path: str = Field(
        "config/biometrics/node_catalog.yaml",
        alias="NODE_CATALOG_PATH",
    )
    grammar_poll_interval_sec: float = Field(5.0, alias="GRAMMAR_POLL_INTERVAL_SEC")
    enable_dynamics_tick: bool = Field(False, alias="SUBSTRATE_DYNAMICS_TICK_ENABLED")
    dynamics_tick_interval_sec: float = Field(30.0, alias="SUBSTRATE_DYNAMICS_TICK_INTERVAL_SEC")
    enable_episodic_tick: bool = Field(False, alias="SUBSTRATE_EPISODIC_TICK_ENABLED")
    episodic_tick_interval_sec: float = Field(300.0, alias="SUBSTRATE_EPISODIC_TICK_INTERVAL_SEC")
    # Window + tick interval must stay under receipt retention (default 30 min),
    # or completed windows will already be pruned when consolidated.
    episodic_window_seconds: int = Field(900, alias="SUBSTRATE_EPISODIC_WINDOW_SECONDS")
    episodic_max_receipts: int = Field(64, alias="SUBSTRATE_EPISODIC_MAX_RECEIPTS")
    episodic_retention_days: float = Field(14.0, alias="SUBSTRATE_EPISODIC_RETENTION_DAYS")
    enable_attention_broadcast: bool = Field(False, alias="ORION_ATTENTION_BROADCAST_ENABLED")
    attention_broadcast_interval_sec: float = Field(
        30.0, alias="ORION_ATTENTION_BROADCAST_INTERVAL_SEC"
    )
    attention_broadcast_min_salience: float = Field(
        0.2, alias="ORION_ATTENTION_BROADCAST_MIN_SALIENCE"
    )
    enable_endogenous_curiosity: bool = Field(
        False, alias="ORION_ENDOGENOUS_CURIOSITY_ENABLED"
    )
    endogenous_curiosity_kill_switch: bool = Field(
        False, alias="ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH"
    )
    endogenous_curiosity_budget: int = Field(3, alias="ORION_ENDOGENOUS_CURIOSITY_BUDGET")
    endogenous_curiosity_min_repair_level: float = Field(
        0.6, alias="ORION_ENDOGENOUS_CURIOSITY_MIN_REPAIR_LEVEL"
    )
    endogenous_curiosity_tick_interval_sec: float = Field(
        60.0, alias="ORION_ENDOGENOUS_CURIOSITY_TICK_INTERVAL_SEC"
    )
    biometrics_grammar_batch_limit: int = Field(50, alias="BIOMETRICS_GRAMMAR_BATCH_LIMIT")
    execution_grammar_batch_limit: int = Field(100, alias="EXECUTION_GRAMMAR_BATCH_LIMIT")
    transport_grammar_batch_limit: int = Field(500, alias="TRANSPORT_GRAMMAR_BATCH_LIMIT")
    reducer_heartbeat_stale_sec: float = Field(120.0, alias="REDUCER_HEARTBEAT_STALE_SEC")
    reducer_poison_max_retries: int = Field(3, alias="REDUCER_POISON_MAX_RETRIES")
    channel_finalize_appraisal_request: str = Field(
        "orion:substrate:finalize_appraisal:request",
        alias="CHANNEL_FINALIZE_APPRAISAL_REQUEST",
    )
    channel_finalize_appraisal_result_prefix: str = Field(
        "orion:substrate:finalize_appraisal:result:",
        alias="CHANNEL_FINALIZE_APPRAISAL_RESULT_PREFIX",
    )
    channel_post_turn_closure: str = Field(
        "orion:substrate:post_turn_closure",
        alias="CHANNEL_POST_TURN_CLOSURE",
    )
    enable_post_turn_closure_listener: bool = Field(
        True,
        alias="ENABLE_POST_TURN_CLOSURE_LISTENER",
    )
    grammar_event_channel: str = Field("orion:grammar:event", alias="GRAMMAR_EVENT_CHANNEL")
    accepted_pressure_grammar_channel: str = Field(
        "orion:grammar:accepted-pressure",
        alias="ACCEPTED_PRESSURE_GRAMMAR_CHANNEL",
    )
    publish_accepted_pressure_grammar: bool = Field(
        True,
        alias="PUBLISH_ACCEPTED_PRESSURE_GRAMMAR",
    )
    substrate_cursor_tail_seed_on_lag: bool = Field(
        False,
        alias="SUBSTRATE_CURSOR_TAIL_SEED_ON_LAG",
    )
    substrate_cursor_lag_resync_hours: float = Field(6.0, alias="SUBSTRATE_CURSOR_LAG_RESYNC_HOURS")
    substrate_cursor_reset_operator_token: str = Field(
        "",
        alias="SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN",
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    receipt_retention_success_minutes: int = Field(
        30, alias="ORION_RECEIPT_RETENTION_SUCCESS_MINUTES"
    )
    receipt_retention_error_hours: int = Field(6, alias="ORION_RECEIPT_RETENTION_ERROR_HOURS")
    receipt_full_payload_success: bool = Field(False, alias="ORION_RECEIPT_FULL_PAYLOAD_SUCCESS")
    receipt_full_payload_sample_rate: float = Field(
        0.0, alias="ORION_RECEIPT_FULL_PAYLOAD_SAMPLE_RATE"
    )
    receipt_max_table_gb: float = Field(25.0, alias="ORION_RECEIPT_MAX_TABLE_GB")
    receipt_warn_table_gb: float = Field(15.0, alias="ORION_RECEIPT_WARN_TABLE_GB")
    receipt_critical_table_gb: float = Field(20.0, alias="ORION_RECEIPT_CRITICAL_TABLE_GB")
    receipt_emergency_metadata_only: bool = Field(
        True, alias="ORION_RECEIPT_EMERGENCY_METADATA_ONLY"
    )
    receipt_prune_interval_sec: float = Field(300.0, alias="ORION_RECEIPT_PRUNE_INTERVAL_SEC")
    receipt_prune_batch_size: int = Field(10000, alias="ORION_RECEIPT_PRUNE_BATCH_SIZE")
    receipt_postgres_data_path: str = Field(
        "/mnt/postgres", alias="ORION_RECEIPT_POSTGRES_DATA_PATH"
    )
    receipt_disk_critical_pct: float = Field(85.0, alias="ORION_RECEIPT_DISK_CRITICAL_PCT")

    # Orion embodiment (mind-to-sprite) hooks — all default off / empty-safe.
    embodiment_c_tick_enabled: bool = Field(False, alias="EMBODIMENT_C_TICK_ENABLED")
    embodiment_perception_substrate_enabled: bool = Field(
        False, alias="EMBODIMENT_PERCEPTION_SUBSTRATE_ENABLED"
    )
    embodiment_channel_intent: str = Field(
        "orion:embodiment:intent", alias="EMBODIMENT_CHANNEL_INTENT"
    )
    embodiment_channel_perception: str = Field(
        "orion:embodiment:perception", alias="EMBODIMENT_CHANNEL_PERCEPTION"
    )
    drives_state_channel: str = Field(
        "orion:memory:drives:state", alias="EMBODIMENT_DRIVES_STATE_CHANNEL"
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

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
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health). See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    enable_biometrics_node_reducer: bool = Field(True, alias="ENABLE_BIOMETRICS_NODE_REDUCER")
    enable_biometrics_pressure_organ: bool = Field(True, alias="ENABLE_BIOMETRICS_PRESSURE_ORGAN")
    enable_node_pressure_reducer: bool = Field(True, alias="ENABLE_NODE_PRESSURE_REDUCER")
    enable_execution_trajectory_reducer: bool = Field(
        True,
        alias="ENABLE_EXECUTION_TRAJECTORY_REDUCER",
    )
    # Caps on active_execution_trajectory.runs (LRU by last_updated_at). Only the
    # freshest ~120s of runs are ever consumed downstream (orion-spark-introspector).
    execution_trajectory_max_runs: int = Field(2000, alias="EXECUTION_TRAJECTORY_MAX_RUNS")
    execution_trajectory_max_age_sec: int = Field(
        86400, alias="EXECUTION_TRAJECTORY_MAX_AGE_SEC"
    )
    enable_transport_bus_reducer: bool = Field(
        False,
        alias="ENABLE_TRANSPORT_BUS_REDUCER",
    )
    enable_chat_grammar_reducer: bool = Field(True, alias="ENABLE_CHAT_GRAMMAR_REDUCER")
    chat_grammar_batch_limit: int = Field(100, alias="CHAT_GRAMMAR_BATCH_LIMIT")
    enable_route_grammar_reducer: bool = Field(True, alias="ENABLE_ROUTE_GRAMMAR_REDUCER")
    route_grammar_batch_limit: int = Field(100, alias="ROUTE_GRAMMAR_BATCH_LIMIT")
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
    # bus_synaptic_prediction_error: own explicit flag, not piggybacked on
    # SUBSTRATE_WRITE_PREDICTION_ERROR_NODES -- new kind of signal (reads FalkorDB
    # directly, no grammar-event batch), not a sixth instance of the existing
    # grammar-event-driven prediction-error shape. See
    # docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md.
    enable_bus_synaptic_tick: bool = Field(False, alias="SUBSTRATE_BUS_SYNAPTIC_TICK_ENABLED")
    bus_synaptic_tick_interval_sec: float = Field(
        30.0, alias="SUBSTRATE_BUS_SYNAPTIC_TICK_INTERVAL_SEC"
    )
    # Cold-start reliability floor, mirrors Hub's own min_count=5
    # (services/orion-hub/scripts/bus_synaptic_graph_routes.py::anomalies()).
    bus_synaptic_min_edge_count: int = Field(5, alias="SUBSTRATE_BUS_SYNAPTIC_MIN_EDGE_COUNT")
    # Staleness floor (review finding, 2026-07-25): an edge from a
    # decommissioned organ/channel keeps its frozen z-score forever once
    # count clears the floor above -- exclude edges not updated in this
    # window so a long-dead edge ages out of the aggregate.
    bus_synaptic_max_edge_age_sec: float = Field(
        3600.0, alias="SUBSTRATE_BUS_SYNAPTIC_MAX_EDGE_AGE_SEC"
    )
    # Same env var name services/orion-bus-mirror and services/orion-recall
    # already use for the same graph -- not a new name.
    falkordb_bus_graph: str = Field("orion_bus_synapse", alias="FALKORDB_BUS_GRAPH")
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
    # Append-only companion log to the singleton substrate_attention_broadcast_
    # projection table (see manual_migration_attention_broadcast_log_v1.sql).
    # 168h (7 days) covers the Phase 1/2/3 replay scripts' default 48h analysis
    # window with margin.
    attention_broadcast_log_retention_hours: float = Field(
        168.0, alias="ORION_ATTENTION_BROADCAST_LOG_RETENTION_HOURS"
    )
    # AST/HOT self-model live tick (docs/superpowers/specs/2026-07-29-ast-hot-
    # reducer-live-ticking-design.md). Appended to the tail of
    # _attention_broadcast_tick() -- no separate timer, rides that tick's own
    # ORION_ATTENTION_BROADCAST_INTERVAL_SEC cadence (broadcast is the
    # slowest of the two real inputs, so there's no resolution to gain from a
    # faster independent timer). Default-off, matching every other tick in
    # this file (dynamics, bus_synaptic, episodic) -- flip only after a live-
    # data sanity check against the new substrate_attention_self_model table.
    enable_attention_self_model_tick: bool = Field(
        False, alias="SUBSTRATE_ATTENTION_SELF_MODEL_TICK_ENABLED"
    )
    # In-process rolling-window size for prediction_error_trend_by_domain
    # (orion/substrate/prediction_error_trend.py), appended once per
    # attention-broadcast tick (~30s cadence by default) rather than once per
    # ~2s field-lane tick like the offline replay script's own
    # PREDICTION_ERROR_TREND_WINDOW_TICKS=30 default -- 10 ticks here is a
    # real-world-time-comparable starting anchor (10 * 30s = 5min), not
    # independently calibrated. Needs its own live-data check before being
    # trusted, same as the offline constant's own documented status.
    attention_self_model_trend_window_ticks: int = Field(
        10, alias="SUBSTRATE_ATTENTION_SELF_MODEL_TREND_WINDOW_TICKS"
    )
    # Same 168h (7-day) default as attention_broadcast_log_retention_hours
    # above -- covers this repo's default 48h analysis-window scripts with
    # margin.
    attention_self_model_log_retention_hours: float = Field(
        168.0, alias="SUBSTRATE_ATTENTION_SELF_MODEL_LOG_RETENTION_HOURS"
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
    # Self-tab brain-EKG frame producer. Enabled by default (operator directive).
    brain_frame_enabled: bool = Field(True, alias="SUBSTRATE_BRAIN_FRAME_ENABLED")
    brain_frame_interval_sec: float = Field(5.0, alias="BRAIN_FRAME_INTERVAL_SEC")
    brain_frame_retention_hours: int = Field(24, alias="BRAIN_FRAME_RETENTION_HOURS")
    brain_frame_sample_nodes: int = Field(40, alias="BRAIN_FRAME_SAMPLE_NODES")
    brain_frame_sample_edges: int = Field(60, alias="BRAIN_FRAME_SAMPLE_EDGES")
    brain_frame_firing_threshold: float = Field(0.5, alias="BRAIN_FRAME_FIRING_THRESHOLD")
    brain_frame_starving_threshold: float = Field(0.1, alias="BRAIN_FRAME_STARVING_THRESHOLD")
    # A dimension renders stale when generated_at - as_of exceeds its cadence.
    brain_frame_self_state_cadence_sec: float = Field(
        30.0, alias="BRAIN_FRAME_SELF_STATE_CADENCE_SEC"
    )
    brain_frame_spotlight_cadence_sec: float = Field(
        30.0, alias="BRAIN_FRAME_SPOTLIGHT_CADENCE_SEC"
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
    # Voluntary attention (ORION_ATTENTION_TOPDOWN_ENABLED) goal-context feed:
    # populates the in-memory active-goal store from GoalProposalV1 events.
    channel_goal_proposal: str = Field(
        "orion:memory:goals:proposed",
        alias="CHANNEL_GOAL_PROPOSAL",
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
    drives_audit_channel: str = Field(
        "orion:memory:drives:audit", alias="DRIVES_AUDIT_CHANNEL"
    )
    # Consumed for the brain-frame honesty_metrics/field_anomaly dimensions.
    # Producer: orion-field-digester's mood-arc reconstruction-error scorer.
    field_channel_anomaly_score_channel: str = Field(
        "orion:field_channel:anomaly_score", alias="FIELD_CHANNEL_ANOMALY_SCORE_CHANNEL"
    )
    # Formerly materialized DriveEngine drive_state/drive_audit into the substrate
    # graph (snapshot_source="drive_state"). Chat stance / Mind measurement SoR is
    # Postgres drive_audits (bus → sql-writer). Default off. Enabling this only
    # restores graph writes — it does not restore stance reads.
    drive_state_substrate_materialization_enabled: bool = Field(
        False, alias="DRIVE_STATE_SUBSTRATE_MATERIALIZATION_ENABLED"
    )

    # Health monitor -> orion-notify attention alerts. Edge-triggered (fires only
    # on healthy->unhealthy transitions), not polled-and-spammed.
    health_check_interval_sec: float = Field(
        900.0, alias="SUBSTRATE_RUNTIME_HEALTH_CHECK_INTERVAL_SEC"
    )
    # Before paging on a fresh unhealthy transition, wait this long and recheck
    # once -- filters out single-tick reducer-health blips (e.g. one cursor
    # commit racing transient DB pressure, self-healing on the very next poll)
    # without delaying a genuinely sustained incident by more than this.
    health_recheck_delay_sec: float = Field(
        15.0, alias="SUBSTRATE_RUNTIME_HEALTH_RECHECK_DELAY_SEC"
    )
    notify_base_url: str = Field("http://orion-athena-notify:7140", alias="NOTIFY_BASE_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

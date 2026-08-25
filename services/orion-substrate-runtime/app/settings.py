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
    # Perceptual availability, feeding node:substrate.vision ->
    # capability:vision so that capability has a real edge instead of a
    # fabricated constant. A bus listener on the vision artifact channel feeds
    # a clock-driven tick -- NOT a bus-cadence statistic, which was tried and
    # deleted (it z-scored a fixed scheduler, and froze rather than rose when
    # the eye went silent). Own flag and interval, like bus_synaptic, because
    # it is a different question rather than another grammar-event domain.
    # Deliberately NOT added to ACTIVE_INFERENCE_DOMAINS or to worker.py's
    # _PREDICTION_ERROR_DOMAIN_NODE_IDS in this patch; see the perception
    # design doc's metric gate, item 6.
    enable_vision_channel_tick: bool = Field(
        False, alias="SUBSTRATE_VISION_CHANNEL_TICK_ENABLED"
    )
    vision_channel_tick_interval_sec: float = Field(
        30.0, alias="SUBSTRATE_VISION_CHANNEL_TICK_INTERVAL_SEC"
    )
    # The channel carrying the detector's real output. Chosen over
    # orion:vision:events (~11/hour -- far too sparse to read as liveness) and
    # over orion:vision:frames (0.1s, but pre-detector, so it stays healthy
    # while the eye is blind). Measured live 2026-08-13: one message every 5.0s.
    vision_artifacts_channel: str = Field(
        "orion:vision:artifacts", alias="SUBSTRATE_VISION_ARTIFACTS_CHANNEL"
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
    # PREDICTION_ERROR_TREND_WINDOW_TICKS=30 default.
    #
    # 2026-08-19: calibrated via real TRAIN/TEST validation, closing the gap
    # this constant's own comment previously flagged ("not independently
    # calibrated"). Swept {2, 10 (the old default), 30} against real
    # substrate_attention_self_model biometrics prediction_error history
    # (19,425 ticks, 7-day span, this exact ~30s cadence -- see
    # docs/superpowers/pr-reports/2026-08-19-l6-item34-hit-it-all-pr.md for the
    # full methodology, chronological 70/30 split, held-out TEST results):
    #   window= 2: TEST reversion accuracy 61.9% (n=4885, z=+16.6 vs 50% null)
    #   window=10: TEST reversion accuracy 56.4% (n=5300, z=+9.3)  <- old default
    #   window=30: TEST reversion accuracy 54.2% (n=5293, z=+6.1) <- offline default
    # window=2 wins on held-out TEST, not just TRAIN it was tuned on (TEST
    # accuracy is actually slightly *higher* than TRAIN's 59.9% -- not an
    # overfit result). This is item 4 sub-idea #3's original "hit it all"
    # deliverable, paused since 2026-07-23 by a Postgres data loss that also
    # wiped the table this validation used to depend on (substrate_field_state,
    # since repurposed for unrelated content -- see
    # project_item4_predicted_shift_reversion_paused_data_loss.md's 2026-08-19
    # correction). Real trade-off, not a free win: at window=2, mid=1, so the
    # trend is a single-prior-sample vs single-recent-sample comparison -- very
    # reactive, no smoothing. Validated on biometrics only (the only domain
    # with enough real variance, same caveat as the reversion-sign fix itself
    # -- docs/superpowers/specs/2026-07-23-predicted-shift-reversion-finding.md),
    # applied uniformly for the same reasoned-extrapolation reason that doc
    # already accepted for the sign fix.
    attention_self_model_trend_window_ticks: int = Field(
        2, alias="SUBSTRATE_ATTENTION_SELF_MODEL_TREND_WINDOW_TICKS"
    )
    # Same 168h (7-day) default as attention_broadcast_log_retention_hours
    # above -- covers this repo's default 48h analysis-window scripts with
    # margin.
    attention_self_model_log_retention_hours: float = Field(
        168.0, alias="SUBSTRATE_ATTENTION_SELF_MODEL_LOG_RETENTION_HOURS"
    )
    # orion-heartbeat's /h1 ensemble verdict, fetched once per
    # _attention_self_model_tick() (~30s cadence, same tick this rides) and
    # threaded into reduce_attention_self_model()'s heartbeat_h1 param.
    # Empty string (default) disables the fetch entirely -- this reducer
    # already treats a missing heartbeat_h1 input as honestly absent, so
    # there's no separate enable flag needed beyond the URL itself being
    # set. Named SUBSTRATE_HEARTBEAT_H1_* (not HEARTBEAT_*) to avoid
    # colliding with this file's own unrelated `heartbeat_interval_sec`
    # above, which is this service's own bus-native SystemHealthV1
    # heartbeat cadence -- a different concept entirely.
    heartbeat_h1_url: str = Field("", alias="SUBSTRATE_HEARTBEAT_H1_URL")
    heartbeat_h1_fetch_timeout_sec: float = Field(
        2.0, alias="SUBSTRATE_HEARTBEAT_H1_FETCH_TIMEOUT_SEC"
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
    # Still real: cached by the embodiment C producer (embodiment_c_tick_enabled)
    # below via _drive_state_listener_loop. The optional legacy substrate-graph
    # materialization that used to also listen on this + a drives_audit_channel
    # (DRIVE_STATE_SUBSTRATE_MATERIALIZATION_ENABLED) was removed 2026-07-30:
    # DriveStateV1/DriveAuditV1 are write-never now that DriveEngine (the only
    # producer) is deleted, so that path was permanently dead weight, not a live
    # toggle. See services/orion-substrate-runtime/app/worker.py.
    drives_state_channel: str = Field(
        "orion:memory:drives:state", alias="EMBODIMENT_DRIVES_STATE_CHANNEL"
    )
    # Consumed for the brain-frame honesty_metrics/field_anomaly dimensions.
    # Producer: orion-field-digester's mood-arc reconstruction-error scorer.
    field_channel_anomaly_score_channel: str = Field(
        "orion:field_channel:anomaly_score", alias="FIELD_CHANNEL_ANOMALY_SCORE_CHANNEL"
    )
    # codebase_prediction_error consumer (docs/superpowers/specs/2026-07-30-
    # codebase-mass-signal-design.md, "Producer + consumer patch design").
    # Own dedicated flag, not piggybacked on SUBSTRATE_WRITE_PREDICTION_ERROR_NODES
    # -- same reasoning as SUBSTRATE_BUS_SYNAPTIC_TICK_ENABLED above: this
    # domain has a materially different risk surface (external I/O lives in
    # a separate service, orion-cocreation-signals, but the consumer side
    # here adds a new Postgres table + a new bus subscription, not a
    # reducer-projection read like the other five domains). Default-off,
    # matching every other tick in this file -- flip only after a live-data
    # sanity check against the new substrate_codebase_mass_baseline table.
    enable_codebase_prediction_error_node: bool = Field(
        False, alias="SUBSTRATE_WRITE_CODEBASE_PREDICTION_ERROR_NODE"
    )
    codebase_delta_channel: str = Field(
        "orion:substrate:codebase_delta", alias="CHANNEL_CODEBASE_DELTA"
    )
    # Append-only substrate_codebase_mass_baseline retention -- this domain's
    # tick cadence is far coarser than the AST self-model's (real events at
    # minutes-to-hours granularity, not a 30s broadcast cadence), so 30 days
    # is a real-world-comparable window, not copied verbatim from
    # ORION_ATTENTION_BROADCAST_LOG_RETENTION_HOURS's 7-day default.
    codebase_mass_baseline_retention_days: float = Field(
        30.0, alias="SUBSTRATE_CODEBASE_MASS_BASELINE_RETENTION_DAYS"
    )
    # Append-only substrate_codebase_delta_log retention. Longer than the
    # baseline table's 30 days on purpose: this table holds the raw per-tick
    # payload (real commit counts, PR numbers, graph deltas) that a future
    # Hub "cocreation signals" analytics tab needs real history to plot --
    # the baseline table only needs its latest row functionally, this one's
    # whole point is accumulated history.
    codebase_delta_log_retention_days: float = Field(
        180.0, alias="SUBSTRATE_CODEBASE_DELTA_LOG_RETENTION_DAYS"
    )

    # Perceptual prediction error (P2, docs/superpowers/specs/2026-08-12-
    # perception-frontier-design.md): a two-stage 0-1 surprise score per
    # camera stream (updated 2026-08-19, review finding -- this comment
    # used to describe only stage 1 and was stale the moment stage 2
    # shipped the same day). Stage 1: raw magnitude
    # `1 - cos(frame_embedding, EWMA_embedding)`. Stage 2: that magnitude
    # z-scored against a second EWMA baseline of the magnitude itself,
    # saturating at 3-sigma -- see orion/substrate/prediction_error.py::
    # perception_prediction_error()'s own docstring for why stage 1 alone
    # (the value this flag published for its first day live) was found
    # numerically incomparable to every other prediction_error domain's
    # min_error threshold and migrated to include stage 2. Own explicit
    # flag, not piggybacked on SUBSTRATE_VISION_CHANNEL_TICK_ENABLED or
    # SUBSTRATE_WRITE_PREDICTION_ERROR_NODES -- same domain-independence
    # convention every tick in this file follows (bus_synaptic vs
    # vision_channel vs codebase all have their own flags despite
    # structural similarity). Distinct from node:substrate.vision's P3
    # channels: perception_staleness measures ARRIVAL TIMING,
    # perception_yield measures the detector's OBJECT COUNT; this measures
    # the embedding model's own CONTENT ENCODING of the frame (see
    # orion/substrate/prediction_error.py's P2 section for the full
    # independence check).
    #
    # This Field's own default stays False (an operator with no .env
    # override gets the conservative default); Juniper's explicit
    # go-ahead to flip the live .env to true landed as a separate commit
    # the same day as the original shadow-only PR (chore(substrate-
    # runtime): enable P2 shadow tick by default) -- SUBSTRATE_PERCEPTION_
    # PREDICTION_ERROR_TICK_ENABLED=true is live on this host now, so the
    # live-data sanity check documented in perception_prediction_error()'s
    # own docstring is not a future gate, it already ran once (against
    # stage 1) and needs re-running against stage 2's real output before
    # a min_error crossing here should be trusted for any consumer
    # decision.
    enable_perception_prediction_error_tick: bool = Field(
        False, alias="SUBSTRATE_PERCEPTION_PREDICTION_ERROR_TICK_ENABLED"
    )
    # Clock-driven companion tick to the event-driven listener above -- writes
    # node:substrate.perception (prediction_error + embedding_staleness) on a
    # fixed interval regardless of whether a new embedding arrived this tick.
    # Not optional: an event-only write would reproduce the node:substrate.
    # route decay-to-zero incident CLAUDE.md section 0A names -- silence would
    # mean this node is never rewritten again, and orion-field-digester's
    # generic per-tick staleness decay would multiply whatever was last
    # written toward 0.0 forever, indistinguishable from genuine calm. Same
    # 30s default as bus_synaptic/vision_channel above.
    perception_prediction_error_tick_interval_sec: float = Field(
        30.0, alias="SUBSTRATE_PERCEPTION_PREDICTION_ERROR_TICK_INTERVAL_SEC"
    )
    # Append-only substrate_perception_embedding_baseline retention, same
    # 30-day convention as codebase_mass_baseline_retention_days above (each
    # stream's own running EWMA state only needs its latest row functionally).
    perception_baseline_retention_days: float = Field(
        30.0, alias="SUBSTRATE_PERCEPTION_BASELINE_RETENTION_DAYS"
    )

    # Health monitor -> orion-notify attention alerts. Edge-triggered (fires only
    # on healthy->unhealthy transitions), not polled-and-spammed.
    health_check_interval_sec: float = Field(
        900.0, alias="SUBSTRATE_RUNTIME_HEALTH_CHECK_INTERVAL_SEC"
    )
    # Before paging on a fresh unhealthy transition, wait this long and recheck
    # once -- filters out single-tick reducer-health blips without delaying a
    # genuinely sustained incident by more than this.
    #
    # 2026-07-31: the example that used to be cited here ("one cursor commit
    # racing transient DB pressure, self-healing on the very next poll") was
    # wrong. That alert was a detector bug -- `classify()` treated the normal
    # in-flight ordering between `record_success()` and `record_cursor_advance()`
    # as a commit failure, ~20% of the time on a healthy system. Fixed in
    # `app/reducer_health.py`. This debounce is still useful for genuine
    # single-tick blips; it just was not filtering what it thought it was.
    health_recheck_delay_sec: float = Field(
        15.0, alias="SUBSTRATE_RUNTIME_HEALTH_RECHECK_DELAY_SEC"
    )
    notify_base_url: str = Field("http://orion-athena-notify:7140", alias="NOTIFY_BASE_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")

    # World-model publish tick: the first real producer for
    # orion:exec:request:WorldModelService (services/orion-world-model, PR
    # #1775/#1861). Default-off like every other tick in this file -- this is
    # a brand-new producer wiring an untrained-weights scaffold service, not a
    # decision to make this worker's output feed real cognition yet. See
    # services/orion-substrate-runtime/README.md "World-model publish tick"
    # for the full honesty split (which of the six feature groups are real
    # vs. explicitly zero-filled this patch).
    enable_world_model_publish_tick: bool = Field(
        False, alias="SUBSTRATE_WORLD_MODEL_PUBLISH_TICK_ENABLED"
    )
    world_model_publish_tick_interval_sec: float = Field(
        30.0, alias="SUBSTRATE_WORLD_MODEL_PUBLISH_TICK_INTERVAL_SEC"
    )
    world_model_request_channel: str = Field(
        "orion:exec:request:WorldModelService", alias="SUBSTRATE_WORLD_MODEL_REQUEST_CHANNEL"
    )
    # Feature-group dims. Deliberately this service's OWN env keys, not a
    # cross-service import of services/orion-world-model/app/settings.py
    # (CLAUDE.md section 5: do not reach into another service's internals).
    # Defaults mirror that service's own WM_DIM_* defaults exactly as of this
    # patch (services/orion-world-model/app/settings.py) so an out-of-the-box
    # deployment matches without operator action -- but this is a REAL
    # coupling, not a coincidence: orion-world-model's
    # trajectory_steps_to_tensors() (app/main.py) rejects any request where a
    # feature group's declared `dim` does not equal that service's configured
    # WM_DIM_<GROUP>. If an operator changes one side's dim, the other side's
    # env key must be changed too, in the same changeset -- nothing enforces
    # this automatically across the service boundary.
    world_model_dim_biometrics: int = Field(32, alias="SUBSTRATE_WORLD_MODEL_DIM_BIOMETRICS")
    world_model_dim_affect: int = Field(16, alias="SUBSTRATE_WORLD_MODEL_DIM_AFFECT")
    world_model_dim_execution_context: int = Field(
        16, alias="SUBSTRATE_WORLD_MODEL_DIM_EXECUTION_CONTEXT"
    )
    world_model_dim_memory_pointers: int = Field(
        32, alias="SUBSTRATE_WORLD_MODEL_DIM_MEMORY_POINTERS"
    )
    world_model_dim_temporal: int = Field(8, alias="SUBSTRATE_WORLD_MODEL_DIM_TEMPORAL")
    # See app/world_model_features.py's module docstring for why this is
    # NEVER hardcoded to a "corrected" value even though WM_DIM_VISION_
    # EMBEDDING=512 was never verified against a real deployed SigLIP2
    # profile as of this patch -- the assembly code compares the real
    # observed vector length to this configured value at publish time and
    # zero-fills + logs loudly on any mismatch, rather than guessing.
    world_model_dim_vision_embedding: int = Field(
        512, alias="SUBSTRATE_WORLD_MODEL_DIM_VISION_EMBEDDING"
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

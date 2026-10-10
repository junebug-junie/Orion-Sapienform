from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-attention-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    # Bus-native SystemHealthV1 heartbeat (orion:system:health). This service has no other
    # bus connection today -- these fields exist solely to feed the HeartbeatOnly chassis. See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    attention_policy_path: str = Field(
        "config/attention/field_attention_policy.v1.yaml",
        alias="ATTENTION_POLICY_PATH",
    )
    attention_poll_interval_sec: float = Field(2.0, alias="ATTENTION_POLL_INTERVAL_SEC")
    enable_attention_runtime: bool = Field(True, alias="ENABLE_ATTENTION_RUNTIME")
    attention_frame_retention_hours: float = Field(72.0, alias="ATTENTION_FRAME_RETENTION_HOURS")
    attention_frame_prune_interval_sec: float = Field(3600.0, alias="ATTENTION_FRAME_PRUNE_INTERVAL_SEC")
    # Candidate A (precision-weighted salience). 2026-07-30 EWMA-baseline fix
    # (see AttentionRuntimeStore.advance_node_prediction_error_baseline and
    # orion/sentience_striving_program/README.md §12): this now bounds how many
    # real NEW substrate_reduction_receipts rows (since the persisted baseline's
    # own cursor) are folded into a node target's baseline per tick, not the size
    # of a raw window recomputed from scratch every tick -- a fetch-size cap on a
    # real backlog (e.g. after a restart), not a calibration knob. Kept at the
    # same value used when this was still a per-tick rolling-window fetch size
    # (tens to ~100 real rows within substrate_reduction_receipts' ~30-minute
    # retention, per scripts/analysis/measure_precision_weighted_salience_probe.py's
    # own real replay windows) -- still comfortably above any real per-tick
    # receipt volume observed live.
    prediction_error_history_limit: int = Field(
        200, alias="ATTENTION_PREDICTION_ERROR_HISTORY_LIMIT"
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    # World-first attention (spec docs/superpowers/specs/2026-10-07-orion-self-
    # calibration-design.md, section A; Juniper 2026-10-10). On: the field
    # contest attends to the world (chat, camera surprise) by default and to
    # a body signal only when it is high/unusual against its own 7 days; a
    # calm tick has no winner. false = the previous five-node min-max ranking
    # exactly. Ships on (Juniper's rule).
    attention_world_first_enabled: bool = Field(True, alias="ATTENTION_WORLD_FIRST_ENABLED")
    # Event-written body nodes (glossary semantics: one reading per event,
    # carried forward) compete only for 300 s after the event that wrote
    # them, fading in rank (orion.attention.world_first.
    # EVENT_ORIENTING_WINDOW_SEC). World-first only. false = world-first
    # without the fade. Ships on (Juniper's rule).
    attention_event_decay_enabled: bool = Field(True, alias="ATTENTION_EVENT_DECAY_ENABLED")

    # Field-native goal-provenance producer (Sentience Striving Program sec6
    # Objective 3, 2026-07-30) -- see docs/superpowers/specs/2026-07-30-goal-
    # provenance-and-decision-lattice-observability-design.md. Publishes
    # FieldGoalProvenanceV1 to orion:memory:goals:proposed (repointed from the
    # deleted GoalProposalEngine's old contract) when the same real
    # node:substrate.* target sustains the node-target subset's top-1 rank for
    # goal_provenance_min_streak consecutive real field ticks. Uses this
    # service's existing orion_bus_url/orion_bus_enabled (previously only fed
    # the HeartbeatOnly chassis) for a second, independent bus connection.
    enable_goal_provenance_producer: bool = Field(
        True, alias="ORION_GOAL_PROVENANCE_PRODUCER_ENABLED"
    )
    # min_streak's real value is an unmeasured, disclosed placeholder debounce
    # (see design doc Part A, Missing Question 2) -- not a calibrated
    # threshold. Revisit once live trigger-rate data exists.
    goal_provenance_min_streak: int = Field(3, ge=1, alias="ORION_GOAL_PROVENANCE_MIN_STREAK")
    # THE ONE BRIDGE (2026-09-06): before choosing a goal target, read which
    # node ids the substrate's workspace competition currently holds as open
    # loops and prefer a qualified candidate the competition can see. Kill
    # switch for a cognition-loop change; off = pre-bridge behaviour exactly.
    enable_goal_reads_competition: bool = Field(
        True, alias="ORION_GOAL_PROVENANCE_READS_COMPETITION"
    )
    # A projection older than this reads as "competition unknown" (fall back),
    # never as "nothing is competing". 120s = four ~30s substrate broadcast
    # ticks -- one missed tick is normal, four is a stalled producer.
    goal_competition_max_age_sec: float = Field(
        120.0, gt=0.0, alias="ORION_GOAL_PROVENANCE_COMPETITION_MAX_AGE_SEC"
    )
    channel_goal_proposal: str = Field(
        "orion:memory:goals:proposed", alias="CHANNEL_GOAL_PROPOSAL"
    )
    # Health monitor -> orion-notify attention alerts. Edge-triggered (fires only
    # on healthy->unhealthy transitions), not polled-and-spammed.
    attention_frame_stall_multiplier: float = Field(1.5, alias="ATTENTION_FRAME_STALL_MULTIPLIER")
    health_check_interval_sec: float = Field(
        900.0, alias="ATTENTION_RUNTIME_HEALTH_CHECK_INTERVAL_SEC"
    )
    notify_base_url: str = Field("http://orion-athena-notify:7140", alias="NOTIFY_BASE_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

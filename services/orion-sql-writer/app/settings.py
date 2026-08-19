from __future__ import annotations

from functools import lru_cache
import json

from pydantic import Field
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger("sql-writer.settings")

DEFAULT_ROUTE_MAP: dict[str, str] = {
    "collapse.mirror": "CollapseMirror",
    "collapse.mirror.entry.v2": "CollapseMirror",
    "metacog.entry.v1": "MetacogEntry",
    "repair_pressure.appraisal.v1": "RepairPressureAppraisalLog",
    "collapse.enrichment": "CollapseEnrichment",
    "tags.enriched": "CollapseEnrichment",
    "chat.history": "ChatHistoryLogSQL",
    "chat.log": "ChatHistoryLogSQL",
    "chat.history.message.v1": "ChatMessageSQL",
    "chat.response.feedback.v1": "ChatResponseFeedbackSQL",
    "chat.gpt.log.v1": "ChatGptLogSQL",
    "chat.gpt.turn.v1": "ChatGptLogSQL",
    "chat.gpt.message.v1": "ChatGptMessageSQL",
    "chat.gpt.import.run.v1": "ChatGptImportRunSQL",
    "chat.gpt.conversation.v1": "ChatGptConversationSQL",
    "chat.gpt.example.v1": "ChatGptDerivedExampleSQL",
    "dream.log": "Dream",
    "dream.result.v1": "Dream",
    "biometrics.telemetry": "BiometricsTelemetry",
    "biometrics.summary.v1": "BiometricsSummarySQL",
    "biometrics.induction.v1": "BiometricsInductionSQL",
    "causal.geometry.snapshot.v1": "CausalGeometrySnapshotSQL",
    "spark.telemetry": "SparkTelemetrySQL",
    "spark.state.snapshot.v1": "SparkTelemetrySQL",
    "cognition.trace": "CognitionTraceSQL",
    "thought.event.v1": "ThoughtDecisionSQL",
    "harness.run.v1": "HarnessTurnTraceSQL",
    "harness.verdict.molecule.v1": "HarnessTurnTraceSQL",
    "harness.turn.outcome.v1": "HarnessTurnTraceSQL",
    "harness.post_turn.closure.v1": "HarnessTurnTraceSQL",
    "metacognition.tick.v1": "MetacognitionTickSQL",
    "orion.metacog.trigger.v1": "MetacogTriggerSQL",
    "metacognitive.trace.v1": "MetacognitiveTraceSQL",
    "notify.notification.request.v1": "NotificationRequestDB",
    "notify.notification.receipt.v1": "NotificationReceiptDB",
    "notify.recipient.update.v1": "RecipientProfileDB",
    "notify.preference.update.v1": "NotificationPreferenceDB",
    "journal.entry.write.v1": "JournalEntrySQL",
    "journal.entry.index.v1": "JournalEntryIndexSQL",
    "evidence.unit.v1": "EvidenceUnitSQL",
    "social.turn.v1": "SocialRoomTurnSQL",
    "external.room.message.v1": "ExternalRoomMessageSQL",
    "external.room.post.result.v1": "ExternalRoomMessageSQL",
    "external.room.turn.skipped.v1": "ExternalRoomMessageSQL",
    "external.room.participant.v1": "ExternalRoomParticipantSQL",
    "endogenous.runtime.record.v1": "EndogenousRuntimeRecordSQL",
    "endogenous.runtime.audit.v1": "EndogenousRuntimeAuditSQL",
    "calibration.profile.audit.v1": "CalibrationProfileAuditSQL",
    "chat.response.feedback.v1": "ChatResponseFeedbackSQL",
    "world.pulse.run.result.v1": "WorldPulseRunSQL",
    "world.pulse.digest.created.v1": "WorldPulseDigestSQL",
    "world.pulse.digest.item.v1": "WorldPulseDigestItemSQL",
    "world.pulse.article.emit.v1": "WorldPulseArticleSQL",
    "world.pulse.cluster.emit.v1": "WorldPulseArticleClusterSQL",
    "world.pulse.claim.emit.v1": "WorldPulseClaimSQL",
    "world.pulse.event.emit.v1": "WorldPulseEventSQL",
    "world.pulse.entity.emit.v1": "WorldPulseEntitySQL",
    "world.pulse.situation.brief.upsert.v1": "WorldPulseSituationBriefSQL",
    "world.pulse.situation.change.emit.v1": "WorldPulseSituationChangeSQL",
    "world.pulse.learning.emit.v1": "WorldPulseLearningDeltaSQL",
    "world.pulse.worth.reading.v1": "WorldPulseWorthReadingSQL",
    "world.pulse.worth.watching.v1": "WorldPulseWorthWatchingSQL",
    "world.context.daily.capsule.v1": "WorldPulseContextCapsuleSQL",
    "world.pulse.publish.status.v1": "WorldPulsePublishStatusSQL",
    "hub.messages.create.v1": "WorldPulseHubMessageSQL",
    "mind.run.artifact.v1": "MindRunSQL",
    "grammar.event.v1": "GrammarEventSQL",
    "chat.history.spark_meta.patch.v1": "__patch_chat_history__",
    "vision.event.v1": "VisionEventSQL",
    "action.outcome.emit.v1": "ActionOutcomeSQL",
    "debug.attention.streak_tick.v1": "DominanceStreakTickSQL",
    "substrate.dev_economics_ledger.v1": "DevEconomicsLedgerSQL",
    "substrate.doc_semantic_drift.v1": "DocSemanticDriftSQL",
    "juniper.affective_state.v1": "JuniperAffectiveStateSQL",
    "self.phi_reward.v1": "PhiRewardSQL",
    "equilibrium.service.transition.v1": "EquilibriumServiceTransitionSQL",
}


class Settings(BaseSettings):
    # Identity
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("sql-writer", alias="SERVICE_NAME")
    service_version: str = Field("0.4.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    port: int = Field(8220, alias="PORT")

    # Bus
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")

    # Chassis
    heartbeat_interval_sec: float = Field(10.0, alias="ORION_HEARTBEAT_INTERVAL_SEC")
    health_channel: str = Field("orion:system:health", alias="ORION_HEALTH_CHANNEL")
    error_channel: str = Field("orion:system:error", alias="ORION_ERROR_CHANNEL")
    shutdown_grace_sec: float = Field(10.0, alias="ORION_SHUTDOWN_GRACE_SEC")

    # Routing
    # Comma-separated or JSON list of channels to subscribe to
    sql_writer_subscribe_channels: list[str] = Field(
        default=[
            "orion:tags:enriched",
            "orion:collapse:sql-write",
            "orion:metacog:sql-write",
            "orion:repair_pressure:appraisal",
            "orion:vision:events:sql-write",
            "orion:chat:history:log",
            "orion:chat:history:turn",
            "orion:chat:social:turn",
            "orion:chat:response:feedback",
            "orion:bridge:social:room:intake",
            "orion:bridge:social:room:delivery",
            "orion:bridge:social:room:skipped",
            "orion:bridge:social:participant",
            "orion:chat:gpt:log",
            "orion:chat:gpt:turn",
            "orion:chat:gpt:message:log",
            "orion:chat:gpt:import:run",
            "orion:chat:gpt:conversation",
            "orion:chat:gpt:example",
            "orion:chat:response:feedback",
            "orion:dream:log",
            "orion:telemetry:biometrics",
            "orion:biometrics:summary",
            "orion:biometrics:induction",
            "orion:spark:telemetry",
            "orion:cognition:trace",
            "orion:thought:artifact",
            "orion:harness:run:artifact",
            "orion:harness:verdict:artifact",
            "orion:substrate:turn_outcome",
            "orion:substrate:post_turn_closure",
            "orion:metacognition:tick",
            "orion:equilibrium:metacog:trigger",
            "orion:metacog:trace",
            "orion:notify:persistence:request",
            "orion:notify:persistence:receipt",
            "orion:journal:write",
            "orion:journal:index",
            "orion:evidence:index:upsert",
            "orion:evidence:markdown:ingest",
            "orion:evidence:parsed:ingest",
            "orion:endogenous:runtime:record",
            "orion:endogenous:runtime:audit",
            "orion:calibration:profile:audit",
            "orion:world_pulse:run:result",
            "orion:world_pulse:digest:created",
            "orion:world_pulse:digest:item",
            "orion:world_pulse:article:emit",
            "orion:world_pulse:cluster:emit",
            "orion:world_pulse:claim:emit",
            "orion:world_pulse:event:emit",
            "orion:world_pulse:entity:emit",
            "orion:world_pulse:situation:brief:upsert",
            "orion:world_pulse:situation:change:emit",
            "orion:world_pulse:learning:emit",
            "orion:world_pulse:worth:reading",
            "orion:world_pulse:worth:watching",
            "orion:world_context:daily_capsule",
            "orion:world_pulse:publish:status",
            "orion:hub:messages:create",
            "orion:mind:artifact",
            "orion:grammar:event",
            "orion:equilibrium:transition",
            "orion:chat:history:spark_meta:patch",
            "orion:autonomy:action:outcome",
            "orion:self:phi_reward",
            "orion:causal_geometry:snapshot",
            "orion:debug:attention:streak_tick",
            "orion:substrate:dev_economics_ledger",
            "orion:substrate:doc_semantic_drift","orion:substrate:juniper_affective_state",
        ],
        alias="SQL_WRITER_SUBSCRIBE_CHANNELS"
    )
    sql_writer_enable_spark_snapshot_channel: bool = Field(
        False,
        alias="SQL_WRITER_ENABLE_SPARK_SNAPSHOT_CHANNEL",
    )
    sql_writer_enable_grammar_channel: bool = Field(
        True,
        alias="SQL_WRITER_ENABLE_GRAMMAR_CHANNEL",
    )

    spark_legacy_mode: str = Field("accept", alias="SPARK_LEGACY_MODE")

    # JSON mapping from envelope.kind -> destination table (or internal model key)
    sql_writer_route_map_json: str = Field(
        default=json.dumps(DEFAULT_ROUTE_MAP),
        alias="SQL_WRITER_ROUTE_MAP_JSON"
    )
    sql_writer_emit_journal_created: bool = Field(True, alias="SQL_WRITER_EMIT_JOURNAL_CREATED")
    sql_writer_journal_created_channel: str = Field("orion:journal:created", alias="SQL_WRITER_JOURNAL_CREATED_CHANNEL")
    sql_writer_emit_social_turn_stored: bool = Field(True, alias="SQL_WRITER_EMIT_SOCIAL_TURN_STORED")
    sql_writer_social_turn_stored_channel: str = Field(
        "orion:chat:social:stored",
        alias="SQL_WRITER_SOCIAL_TURN_STORED_CHANNEL",
    )
    sql_writer_emit_memory_turn_persisted: bool = Field(True, alias="SQL_WRITER_EMIT_MEMORY_TURN_PERSISTED")
    channel_memory_turn_persisted: str = Field(
        "orion:memory:turn:persisted", alias="CHANNEL_MEMORY_TURN_PERSISTED"
    )
    channel_chat_history_spark_meta_patch: str = Field(
        "orion:chat:history:spark_meta:patch", alias="CHANNEL_CHAT_HISTORY_SPARK_META_PATCH"
    )
    # AI Town chat-history table split (docs/superpowers/specs/
    # 2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md).
    # Retired SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED (PR #1734, 2026-08-19) --
    # that was an additive bridge (every AI-Town row landed in BOTH tables),
    # built for a live AI Town world. Retired same-day with AI Town's own
    # backend confirmed dead (Convex connection refused, zero concurrent
    # writes) -- no fallback to the dual-write it replaced (CLAUDE.md's
    # "kill means kill"). This flag controls ROUTING instead: an AI-Town
    # row lands in exactly one table, chosen once
    # (worker.py::_resolve_chat_history_model_cls), never duplicated.
    # Defaults True -- unlike the dual-write bridge this replaces, routing
    # carries no first-time-in-production write-path risk (chat_history_log
    # itself is untouched for every non-AI-Town row either way), so there is
    # no reason to ship it off. False is a real rollback path: every row
    # (AI-Town or not) goes to chat_history_log only, the pre-split
    # behavior, if routing is ever suspected of picking the wrong table.
    sql_writer_aitown_routing_enabled: bool = Field(
        True, alias="SQL_WRITER_AITOWN_ROUTING_ENABLED"
    )
    metacog_trace_retention_days: int = Field(14, alias="METACOG_TRACE_RETENTION_DAYS")
    # drive_audits_retention_days removed 2026-08-13: the drive_audits table
    # (and orion-sql-writer's whole write path for it) was fully removed that
    # day, so there is nothing left to prune. See docs/superpowers/pr-reports/
    # 2026-08-13-untangle-drive-audit-sql-writer-pr.md.

    # goal_provenance_streak_ticks (2026-08-11, Part H debug telemetry, review fix): unlike
    # the now-removed drive_audits above, this table has a real, currently-live producer and no natural
    # ceiling -- ~1 row per real orion-attention-runtime field tick, matching
    # substrate_attention_frames' cadence (~43k rows/day per that table's own retention
    # comment). Defaults ON (unlike drive_audits' 0/disabled default) precisely because this
    # is meant to be temporary, collect-then-decide instrumentation: CLAUDE.md's own
    # transport_prediction_error incident is the exact "known temporary but nobody removed
    # it" failure mode this default exists to avoid. 14 days comfortably spans the "collect a
    # few days, then run measure_goal_provenance_streak_distribution.py and decide" window
    # this instrumentation exists for, with room to spare if that decision takes longer than
    # planned. 0 disables pruning.
    goal_provenance_streak_ticks_retention_days: int = Field(
        14, alias="GOAL_PROVENANCE_STREAK_TICKS_RETENTION_DAYS"
    )

    # Guardrails: grammar lane is async-isolated; operational writes use concurrent Hunter + pool limits.
    sql_writer_concurrent_handlers: bool = Field(True, alias="SQL_WRITER_CONCURRENT_HANDLERS")
    sql_writer_max_inflight: int = Field(12, alias="SQL_WRITER_MAX_INFLIGHT")
    sql_writer_db_pool_size: int = Field(10, alias="SQL_WRITER_DB_POOL_SIZE")
    sql_writer_db_max_overflow: int = Field(20, alias="SQL_WRITER_DB_MAX_OVERFLOW")
    sql_writer_db_statement_timeout_ms: int = Field(30_000, alias="SQL_WRITER_DB_STATEMENT_TIMEOUT_MS")
    sql_writer_db_lock_timeout_ms: int = Field(10_000, alias="SQL_WRITER_DB_LOCK_TIMEOUT_MS")
    sql_writer_grammar_persist_timeout_sec: float = Field(15.0, alias="SQL_WRITER_GRAMMAR_PERSIST_TIMEOUT_SEC")
    sql_writer_grammar_workers: int = Field(4, alias="SQL_WRITER_GRAMMAR_WORKERS")
    sql_writer_grammar_pool_size: int = Field(4, alias="SQL_WRITER_GRAMMAR_POOL_SIZE")
    sql_writer_grammar_pool_max_overflow: int = Field(4, alias="SQL_WRITER_GRAMMAR_POOL_MAX_OVERFLOW")
    sql_writer_grammar_statement_timeout_ms: int = Field(10_000, alias="SQL_WRITER_GRAMMAR_STATEMENT_TIMEOUT_MS")
    sql_writer_grammar_lock_timeout_ms: int = Field(3_000, alias="SQL_WRITER_GRAMMAR_LOCK_TIMEOUT_MS")
    # 2026-08-19: dropped 30->15 days. The 30-day default had never actually been
    # enforced -- grammar_events retention was silently failing on every startup
    # (missing index -> full-table scan -> statement timeout, see
    # idx_grammar_events_created_at below) -- so this is a real policy change, not
    # just re-enabling what was already the intended live behavior.
    grammar_events_retention_days: int = Field(15, alias="GRAMMAR_EVENTS_RETENTION_DAYS")
    # batch_size 5000->1000, max_batches 20->100 (2026-08-19, live-verified): with the
    # new created_at index, the SELECT half of the batched DELETE is fast (~13ms for
    # 5000 rows), but the DELETE itself still needs one primary-key-index lookup per
    # matched row to actually remove it -- on grammar_events (8 indexes, 16GB, poorly
    # cached at this scale) that's real random I/O, observed at ~4.1ms/row under load
    # (EXPLAIN ANALYZE: a real 5000-row DELETE took 20.6s, well past the 10s grammar
    # statement timeout). 1000 rows/batch leaves real margin under that timeout; 100
    # max batches removes the batch-count cap as the binding constraint so
    # max_elapsed_sec below is what actually stops a run -- gracefully
    # (capped_by_elapsed_limit=True, real committed progress, no exception) instead of
    # occasionally dying mid-batch with a swallowed QueryCanceled and losing whatever
    # that batch's rows would have been (previous behavior, confirmed live: grammar_edges
    # committed 25000 rows across 5 successful batches, then batch 6 timed out and the
    # whole run reported as a failure despite real progress having been made).
    grammar_events_retention_batch_size: int = Field(1000, alias="GRAMMAR_EVENTS_RETENTION_BATCH_SIZE")
    grammar_events_retention_max_batches_per_startup: int = Field(
        100,
        alias="GRAMMAR_EVENTS_RETENTION_MAX_BATCHES_PER_STARTUP",
    )
    grammar_events_retention_max_elapsed_sec: float = Field(
        120.0,
        alias="GRAMMAR_EVENTS_RETENTION_MAX_ELAPSED_SEC",
    )
    # grammar_edges/grammar_atoms/substrate_organ_emissions had NO retention at all
    # until this patch (confirmed live 2026-08-19: unbounded growth, ~13GB combined,
    # zero deletes ever). Each gets its own retention_days knob, same precedent as
    # goal_provenance_streak_ticks_retention_days above, but deliberately reuses the
    # grammar_events batch/cap knobs above rather than adding a near-duplicate set
    # per table -- same scale, same startup-bounded-batch shape, nothing about these
    # three tables needs independently tunable batching.
    grammar_edges_retention_days: int = Field(15, alias="GRAMMAR_EDGES_RETENTION_DAYS")
    grammar_atoms_retention_days: int = Field(15, alias="GRAMMAR_ATOMS_RETENTION_DAYS")
    substrate_organ_emissions_retention_days: int = Field(
        15, alias="SUBSTRATE_ORGAN_EMISSIONS_RETENTION_DAYS"
    )
    sql_writer_allow_accepted_pressure_ingest: bool = Field(
        False,
        alias="SQL_WRITER_ALLOW_ACCEPTED_PRESSURE_INGEST",
    )
    sql_writer_grammar_trace_batch_max: int = Field(64, alias="SQL_WRITER_GRAMMAR_TRACE_BATCH_MAX")
    sql_writer_grammar_trace_batch_timeout_sec: float = Field(45.0, alias="SQL_WRITER_GRAMMAR_TRACE_BATCH_TIMEOUT_SEC")

    @property
    def route_map(self) -> dict[str, str]:
        try:
            overrides = json.loads(self.sql_writer_route_map_json)
        except Exception:
            overrides = {}
        return {**DEFAULT_ROUTE_MAP, **overrides}

    @property
    def effective_subscribe_channels(self) -> list[str]:
        """Back-compat alias.

        Some refactor branches referenced `effective_subscribe_channels`.
        We keep env as source of truth and simply expose the configured list.
        """
        channels = list(self.sql_writer_subscribe_channels)
        if not self.sql_writer_enable_grammar_channel:
            channels = [c for c in channels if c != "orion:grammar:event"]
        if self.sql_writer_enable_spark_snapshot_channel and "orion:spark:state:snapshot" not in channels:
            channels.append("orion:spark:state:snapshot")
        # action.outcome.emit.v1 is a code-default route with no feature toggle; guarantee its
        # channel is always subscribed so a stale operator env list cannot silently drop the
        # autonomy feedback lane before it can even reach the fallback log.
        if "orion:autonomy:action:outcome" not in channels:
            channels.append("orion:autonomy:action:outcome")
        return channels

    @property
    def spark_legacy_mode_normalized(self) -> str:
        mode = (self.spark_legacy_mode or "accept").strip().lower()
        if mode not in {"accept", "warn", "drop"}:
            logger.warning("Invalid SPARK_LEGACY_MODE=%s; defaulting to accept", self.spark_legacy_mode)
            return "accept"
        return mode


    # bus_fallback_log backlog watcher (app/fallback_watch.py).
    #
    # Nothing watched the fallback log until 2026-08-14, which is how two
    # separate routing failures each ran for hours looking healthy. On by
    # default: a monitor that ships disabled is a monitor that is off in
    # production, and the whole failure mode here is silence being mistaken
    # for health.
    sql_writer_fallback_watch_enabled: bool = Field(
        True, alias="SQL_WRITER_FALLBACK_WATCH_ENABLED"
    )
    sql_writer_fallback_watch_interval_sec: int = Field(
        300, alias="SQL_WRITER_FALLBACK_WATCH_INTERVAL_SEC"
    )
    # Trailing window the count is taken over. 24h rather than something short:
    # the observed failures dribbled a handful of events per hour for hours, and
    # a 15-minute window would never have accumulated enough to cross a
    # threshold before someone noticed by hand anyway.
    sql_writer_fallback_watch_window_sec: int = Field(
        86400, alias="SQL_WRITER_FALLBACK_WATCH_WINDOW_SEC"
    )
    # Alerts fire at each multiple of this: 5, 10, 15, ...
    sql_writer_fallback_watch_threshold_step: int = Field(
        5, alias="SQL_WRITER_FALLBACK_WATCH_THRESHOLD_STEP"
    )

    # Notify service, used only by the watcher above. This service already
    # PERSISTS notify records (models/notify_models.py) but had never SENT one,
    # so these keys are new here.
    notify_service_url: str = Field("", alias="NOTIFY_SERVICE_URL")
    notify_api_token: str = Field("", alias="NOTIFY_API_TOKEN")

    # DB
    # Ensure default matches prod environment (Postgres), not SQLite.
    postgres_uri: str = Field("postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney", alias="POSTGRES_URI")
    database_url: str = Field("postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney", alias="DATABASE_URL")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

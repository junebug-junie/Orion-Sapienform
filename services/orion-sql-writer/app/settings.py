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
    "collapse.enrichment": "CollapseEnrichment",
    "tags.enriched": "CollapseEnrichment",
    "chat.history": "ChatHistoryLogSQL",
    "chat.log": "ChatHistoryLogSQL",
    "chat.history.message.v1": "ChatMessageSQL",
    "chat.response.feedback.v1": "ChatResponseFeedbackSQL",
    "chat.gpt.log.v1": "ChatGptLogSQL",
    "chat.gpt.turn.v1": "ChatGptLogSQL",
    "chat.gpt.message.v1": "ChatGptMessageSQL",
    "dream.log": "Dream",
    "dream.result.v1": "Dream",
    "biometrics.telemetry": "BiometricsTelemetry",
    "biometrics.summary.v1": "BiometricsSummarySQL",
    "biometrics.induction.v1": "BiometricsInductionSQL",
    "spark.telemetry": "SparkTelemetrySQL",
    "spark.state.snapshot.v1": "SparkTelemetrySQL",
    "cognition.trace": "CognitionTraceSQL",
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
            "orion:chat:response:feedback",
            "orion:dream:log",
            "orion:telemetry:biometrics",
            "orion:biometrics:summary",
            "orion:biometrics:induction",
            "orion:spark:telemetry",
            "orion:cognition:trace",
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
        ],
        alias="SQL_WRITER_SUBSCRIBE_CHANNELS"
    )
    sql_writer_enable_spark_snapshot_channel: bool = Field(
        False,
        alias="SQL_WRITER_ENABLE_SPARK_SNAPSHOT_CHANNEL",
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
    metacog_trace_retention_days: int = Field(14, alias="METACOG_TRACE_RETENTION_DAYS")

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
        if self.sql_writer_enable_spark_snapshot_channel and "orion:spark:state:snapshot" not in channels:
            channels.append("orion:spark:state:snapshot")
        return channels

    @property
    def spark_legacy_mode_normalized(self) -> str:
        mode = (self.spark_legacy_mode or "accept").strip().lower()
        if mode not in {"accept", "warn", "drop"}:
            logger.warning("Invalid SPARK_LEGACY_MODE=%s; defaulting to accept", self.spark_legacy_mode)
            return "accept"
        return mode


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

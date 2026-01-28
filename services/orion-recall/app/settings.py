# services/orion-recall/app/settings.py
from __future__ import annotations

from typing import Optional

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Recall service settings.

    Source-of-truth precedence:
      1) Environment variables (docker-compose passes these explicitly)
      2) Local .env file (only for local dev / direct uvicorn runs)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Service Metadata ──────────────────────────────────────────────
    SERVICE_NAME: str = Field(default="recall", validation_alias=AliasChoices("SERVICE_NAME"))
    SERVICE_VERSION: str = Field(default="0.1.0", validation_alias=AliasChoices("SERVICE_VERSION"))
    NODE_NAME: str = Field(default="unknown", validation_alias=AliasChoices("NODE_NAME", "HOSTNAME"))
    PORT: int = Field(default=8260, validation_alias=AliasChoices("PORT"))

    # ── Orion Bus ─────────────────────────────────────────────────────
    ORION_BUS_ENABLED: bool = Field(default=True, validation_alias=AliasChoices("ORION_BUS_ENABLED"))
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False, validation_alias=AliasChoices("ORION_BUS_ENFORCE_CATALOG"))
    ORION_BUS_URL: str = Field(
        default="redis://127.0.0.1:6379/0",
        validation_alias=AliasChoices("ORION_BUS_URL", "REDIS_URL"),
    )

    # RPC intake + reply + telemetry
    RECALL_BUS_INTAKE: str = Field(
        default="orion:exec:request:RecallService",
        validation_alias=AliasChoices("RECALL_BUS_INTAKE", "CHANNEL_RECALL_REQUEST"),
    )
    RECALL_BUS_REPLY_DEFAULT: str = Field(
        default="orion:exec:result:RecallService",
        validation_alias=AliasChoices("RECALL_BUS_REPLY_DEFAULT", "CHANNEL_RECALL_DEFAULT_REPLY_PREFIX"),
    )
    RECALL_BUS_TELEMETRY: str = Field(
        default="orion:recall:telemetry",
        validation_alias=AliasChoices("RECALL_BUS_TELEMETRY"),
    )

    # ── Chassis / Runtime ─────────────────────────────────────────────
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0, validation_alias=AliasChoices("HEARTBEAT_INTERVAL_SEC"))
    ORION_HEALTH_CHANNEL: str = Field(default="orion:system:health", validation_alias=AliasChoices("ORION_HEALTH_CHANNEL"))
    ERROR_CHANNEL: str = Field(default="orion:system:error", validation_alias=AliasChoices("ERROR_CHANNEL"))
    SHUTDOWN_GRACE_SEC: float = Field(default=10.0, validation_alias=AliasChoices("SHUTDOWN_GRACE_SEC"))

    # ── Default Recall Behavior ───────────────────────────────────────
    RECALL_DEFAULT_MAX_ITEMS: int = Field(default=16, validation_alias=AliasChoices("RECALL_DEFAULT_MAX_ITEMS"))
    RECALL_DEFAULT_TIME_WINDOW_DAYS: int = Field(
        default=30, validation_alias=AliasChoices("RECALL_DEFAULT_TIME_WINDOW_DAYS")
    )
    RECALL_DEFAULT_MODE: str = Field(default="hybrid", validation_alias=AliasChoices("RECALL_DEFAULT_MODE"))
    RECALL_DEFAULT_PROFILE: str = Field(default="reflect.v1", validation_alias=AliasChoices("RECALL_DEFAULT_PROFILE"))

    # ── Source Toggles ────────────────────────────────────────────────
    RECALL_ENABLE_SQL_CHAT: bool = Field(default=True, validation_alias=AliasChoices("RECALL_ENABLE_SQL_CHAT"))
    RECALL_ENABLE_SQL_MIRRORS: bool = Field(default=True, validation_alias=AliasChoices("RECALL_ENABLE_SQL_MIRRORS"))
    RECALL_ENABLE_VECTOR: bool = Field(default=True, validation_alias=AliasChoices("RECALL_ENABLE_VECTOR"))
    RECALL_ENABLE_RDF: bool = Field(default=False, validation_alias=AliasChoices("RECALL_ENABLE_RDF"))

    # ── Postgres / SQL ────────────────────────────────────────────────
    RECALL_PG_DSN: str = Field(
        default="postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        validation_alias=AliasChoices("RECALL_PG_DSN", "POSTGRES_URI", "POSTGRES_DSN"),
    )

    # Chat history
    RECALL_SQL_CHAT_TABLE: str = Field(default="chat_history_log", validation_alias=AliasChoices("RECALL_SQL_CHAT_TABLE"))
    RECALL_SQL_CHAT_TEXT_COL: str = Field(default="prompt", validation_alias=AliasChoices("RECALL_SQL_CHAT_TEXT_COL"))
    RECALL_SQL_CHAT_RESPONSE_COL: str = Field(
        default="response", validation_alias=AliasChoices("RECALL_SQL_CHAT_RESPONSE_COL")
    )
    RECALL_SQL_CHAT_CREATED_AT_COL: str = Field(
        default="created_at", validation_alias=AliasChoices("RECALL_SQL_CHAT_CREATED_AT_COL")
    )

    # Collapse mirror base + semantic fields
    RECALL_SQL_MIRROR_TABLE: str = Field(default="collapse_mirror", validation_alias=AliasChoices("RECALL_SQL_MIRROR_TABLE"))
    RECALL_SQL_MIRROR_SUMMARY_COL: str = Field(default="summary", validation_alias=AliasChoices("RECALL_SQL_MIRROR_SUMMARY_COL"))
    RECALL_SQL_MIRROR_TRIGGER_COL: str = Field(default="trigger", validation_alias=AliasChoices("RECALL_SQL_MIRROR_TRIGGER_COL"))
    RECALL_SQL_MIRROR_OBSERVER_COL: str = Field(default="observer", validation_alias=AliasChoices("RECALL_SQL_MIRROR_OBSERVER_COL"))
    RECALL_SQL_MIRROR_OBSERVER_STATE_COL: str = Field(
        default="observer_state", validation_alias=AliasChoices("RECALL_SQL_MIRROR_OBSERVER_STATE_COL")
    )
    RECALL_SQL_MIRROR_FIELD_RESONANCE_COL: str = Field(
        default="field_resonance", validation_alias=AliasChoices("RECALL_SQL_MIRROR_FIELD_RESONANCE_COL")
    )
    RECALL_SQL_MIRROR_INTENT_COL: str = Field(default="intent", validation_alias=AliasChoices("RECALL_SQL_MIRROR_INTENT_COL"))
    RECALL_SQL_MIRROR_TYPE_COL: str = Field(default="type", validation_alias=AliasChoices("RECALL_SQL_MIRROR_TYPE_COL"))
    RECALL_SQL_MIRROR_ENTITY_COL: str = Field(
        default="emergent_entity", validation_alias=AliasChoices("RECALL_SQL_MIRROR_ENTITY_COL")
    )
    RECALL_SQL_MIRROR_MANTRA_COL: str = Field(default="mantra", validation_alias=AliasChoices("RECALL_SQL_MIRROR_MANTRA_COL"))
    RECALL_SQL_MIRROR_CAUSAL_ECHO_COL: str = Field(
        default="causal_echo", validation_alias=AliasChoices("RECALL_SQL_MIRROR_CAUSAL_ECHO_COL")
    )
    RECALL_SQL_MIRROR_TS_COL: str = Field(default="timestamp", validation_alias=AliasChoices("RECALL_SQL_MIRROR_TS_COL"))

    # Collapse enrichment
    RECALL_SQL_ENRICH_TABLE: str = Field(default="collapse_enrichment", validation_alias=AliasChoices("RECALL_SQL_ENRICH_TABLE"))
    RECALL_SQL_ENRICH_COLLAPSE_ID_COL: str = Field(
        default="collapse_id", validation_alias=AliasChoices("RECALL_SQL_ENRICH_COLLAPSE_ID_COL")
    )
    RECALL_SQL_ENRICH_TAGS_COL: str = Field(default="tags", validation_alias=AliasChoices("RECALL_SQL_ENRICH_TAGS_COL"))
    RECALL_SQL_ENRICH_ENTITIES_COL: str = Field(
        default="entities", validation_alias=AliasChoices("RECALL_SQL_ENRICH_ENTITIES_COL")
    )
    RECALL_SQL_ENRICH_SALIENCE_COL: str = Field(
        default="salience", validation_alias=AliasChoices("RECALL_SQL_ENRICH_SALIENCE_COL")
    )
    RECALL_SQL_ENRICH_TS_COL: str = Field(default="ts", validation_alias=AliasChoices("RECALL_SQL_ENRICH_TS_COL"))

    # ── Global Vector / RDF knobs ─────────────────────────────────────
    VECTOR_DB_HOST: str = Field(default="orion-athena-vector-db", validation_alias=AliasChoices("VECTOR_DB_HOST"))
    VECTOR_DB_PORT: int = Field(default=8000, validation_alias=AliasChoices("VECTOR_DB_PORT"))
    VECTOR_DB_COLLECTION: str = Field(default="orion_main_store", validation_alias=AliasChoices("VECTOR_DB_COLLECTION"))

    GRAPHDB_URL: str = Field(default="http://orion-athena-graphdb:7200", validation_alias=AliasChoices("GRAPHDB_URL"))
    GRAPHDB_REPO: str = Field(default="collapse", validation_alias=AliasChoices("GRAPHDB_REPO"))
    GRAPHDB_USER: str = Field(default="admin", validation_alias=AliasChoices("GRAPHDB_USER"))
    GRAPHDB_PASS: str = Field(default="admin", validation_alias=AliasChoices("GRAPHDB_PASS"))

    # ── Vector backend (recall-specific overrides) ────────────────────
    RECALL_VECTOR_BASE_URL: Optional[str] = Field(default=None, validation_alias=AliasChoices("RECALL_VECTOR_BASE_URL"))
    RECALL_VECTOR_COLLECTIONS: Optional[str] = Field(default=None, validation_alias=AliasChoices("RECALL_VECTOR_COLLECTIONS"))
    RECALL_VECTOR_EMBEDDING_URL: Optional[str] = Field(
        default="http://orion-vector-host:8320/embedding",
        validation_alias=AliasChoices("RECALL_VECTOR_EMBEDDING_URL"),
    )
    RECALL_VECTOR_TIMEOUT_SEC: float = Field(default=5.0, validation_alias=AliasChoices("RECALL_VECTOR_TIMEOUT_SEC"))
    RECALL_VECTOR_MAX_ITEMS: int = Field(default=24, validation_alias=AliasChoices("RECALL_VECTOR_MAX_ITEMS"))
    RECALL_EXCLUDE_REJECTED: bool = Field(default=True, validation_alias=AliasChoices("RECALL_EXCLUDE_REJECTED"))
    RECALL_DURABLE_ONLY: bool = Field(default=False, validation_alias=AliasChoices("RECALL_DURABLE_ONLY"))
    RECALL_DEBUG_DUMP_TOP_N: int = Field(default=0, validation_alias=AliasChoices("RECALL_DEBUG_DUMP_TOP_N"))

    # ── RDF / GraphDB (recall-specific) ───────────────────────────────
    RECALL_RDF_ENDPOINT_URL: Optional[str] = Field(default=None, validation_alias=AliasChoices("RECALL_RDF_ENDPOINT_URL"))
    RECALL_RDF_TIMEOUT_SEC: float = Field(default=5.0, validation_alias=AliasChoices("RECALL_RDF_TIMEOUT_SEC"))
    RECALL_RDF_USER: str = Field(default="admin", validation_alias=AliasChoices("RECALL_RDF_USER", "GRAPHDB_USER"))
    RECALL_RDF_PASS: str = Field(default="admin", validation_alias=AliasChoices("RECALL_RDF_PASS", "GRAPHDB_PASS"))
    RECALL_RDF_ENABLE_SUMMARIES: bool = Field(
        default=False, validation_alias=AliasChoices("RECALL_RDF_ENABLE_SUMMARIES")
    )

    # ── SQL timeline knobs ────────────────────────────────────────────
    RECALL_ENABLE_SQL_TIMELINE: bool = Field(default=True, validation_alias=AliasChoices("RECALL_ENABLE_SQL_TIMELINE"))
    RECALL_SQL_SINCE_MINUTES: int = Field(default=180, validation_alias=AliasChoices("RECALL_SQL_SINCE_MINUTES"))
    RECALL_SQL_TOP_K: int = Field(default=10, validation_alias=AliasChoices("RECALL_SQL_TOP_K"))
    # Default timeline source is chat history. When RECALL_SQL_TIMELINE_TABLE == RECALL_SQL_CHAT_TABLE,
    # sql_timeline uses RECALL_SQL_CHAT_* columns to build "User/Orion" turns and ignores TIMELINE_* columns.
    # When RECALL_SQL_TIMELINE_TABLE is another table (e.g. collapse_mirror), TIMELINE_* columns are used.
    RECALL_SQL_TIMELINE_TABLE: str = Field(default="chat_history_log", validation_alias=AliasChoices("RECALL_SQL_TIMELINE_TABLE"))
    RECALL_SQL_TIMELINE_TS_COL: str = Field(default="timestamp", validation_alias=AliasChoices("RECALL_SQL_TIMELINE_TS_COL"))
    RECALL_SQL_TIMELINE_TEXT_COL: str = Field(default="summary", validation_alias=AliasChoices("RECALL_SQL_TIMELINE_TEXT_COL"))
    RECALL_SQL_TIMELINE_SESSION_COL: str = Field(default="observer", validation_alias=AliasChoices("RECALL_SQL_TIMELINE_SESSION_COL"))
    RECALL_SQL_TIMELINE_NODE_COL: str = Field(default="observer_state", validation_alias=AliasChoices("RECALL_SQL_TIMELINE_NODE_COL"))
    RECALL_SQL_TIMELINE_TAGS_COL: str = Field(default="tags", validation_alias=AliasChoices("RECALL_SQL_TIMELINE_TAGS_COL"))
    # Juniper observer filter is intended for collapse_mirror timelines only.
    RECALL_SQL_TIMELINE_REQUIRE_JUNIPER_OBSERVER: Optional[bool] = Field(
        default=None, validation_alias=AliasChoices("RECALL_SQL_TIMELINE_REQUIRE_JUNIPER_OBSERVER")
    )

    # ── SQL message table (optional) ──────────────────────────────────
    RECALL_SQL_MESSAGE_TABLE: str = Field(default="", validation_alias=AliasChoices("RECALL_SQL_MESSAGE_TABLE"))
    RECALL_SQL_MESSAGE_ROLE_COL: str = Field(
        default="role", validation_alias=AliasChoices("RECALL_SQL_MESSAGE_ROLE_COL")
    )
    RECALL_SQL_MESSAGE_TEXT_COL: str = Field(
        default="text", validation_alias=AliasChoices("RECALL_SQL_MESSAGE_TEXT_COL")
    )
    RECALL_SQL_MESSAGE_CREATED_AT_COL: str = Field(
        default="created_at", validation_alias=AliasChoices("RECALL_SQL_MESSAGE_CREATED_AT_COL")
    )

    # ── Future tensor / ranker toggles ────────────────────────────────
    RECALL_TENSOR_RANKER_ENABLED: bool = Field(default=False, validation_alias=AliasChoices("RECALL_TENSOR_RANKER_ENABLED"))
    RECALL_TENSOR_RANKER_MODEL_PATH: str = Field(
        default="/mnt/storage-warm/orion/recall/tensor-ranker.pt",
        validation_alias=AliasChoices("RECALL_TENSOR_RANKER_MODEL_PATH"),
    )

    # ── Helpers ───────────────────────────────────────────────────────
    @field_validator(
        "RECALL_VECTOR_BASE_URL",
        "RECALL_VECTOR_COLLECTIONS",
        "RECALL_VECTOR_EMBEDDING_URL",
        "RECALL_RDF_ENDPOINT_URL",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @model_validator(mode="after")
    def _derive_endpoints(self):
        # If recall-specific base URL is not set, build from VECTOR_DB_HOST/PORT
        if not self.RECALL_VECTOR_BASE_URL:
            self.RECALL_VECTOR_BASE_URL = f"http://{self.VECTOR_DB_HOST}:{self.VECTOR_DB_PORT}"

        # If recall-specific collections are not set, fall back to the chat collection
        if not self.RECALL_VECTOR_COLLECTIONS:
            self.RECALL_VECTOR_COLLECTIONS = "orion_chat_turns,orion_chat"

        # Default Juniper filter only for collapse_mirror timelines.
        if self.RECALL_SQL_TIMELINE_REQUIRE_JUNIPER_OBSERVER is None:
            self.RECALL_SQL_TIMELINE_REQUIRE_JUNIPER_OBSERVER = (
                self.RECALL_SQL_TIMELINE_TABLE == "collapse_mirror"
            )

        # If endpoint override isn't set, build from GraphDB URL + repo
        if not self.RECALL_RDF_ENDPOINT_URL:
            base = self.GRAPHDB_URL.rstrip("/")
            self.RECALL_RDF_ENDPOINT_URL = f"{base}/repositories/{self.GRAPHDB_REPO}"

        return self


settings = Settings()

# services/orion-recall/app/settings.py
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Pydantic / Settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 👈 ignore any env vars we don't explicitly model
    )

    # ── Service Metadata ──────────────────────────────────────────────
    SERVICE_NAME: str = "recall"
    SERVICE_VERSION: str = "0.1.0"
    NODE_NAME: str = Field(default="unknown")
    PORT: int = 8260

    # ── Orion Bus ─────────────────────────────────────────────────────
    ORION_BUS_ENABLED: bool = True
    ORION_BUS_URL: str = "redis://100.92.216.81:6379/0"

    # These channels are for Rabbit usage
    RECALL_BUS_INTAKE: str = "orion-exec:request:RecallService"
    CHANNEL_RECALL_REQUEST: str = "orion-exec:request:RecallService"
    CHANNEL_RECALL_DEFAULT_REPLY_PREFIX: str = "orion-exec:result:RecallService"

    # ── Chassis / Runtime ─────────────────────────────────────────────
    HEARTBEAT_INTERVAL_SEC: float = 10.0
    HEALTH_CHANNEL: str = "system.health"
    ERROR_CHANNEL: str = "system.error"
    SHUTDOWN_GRACE_SEC: float = 10.0

    # ── Default Recall Behavior ───────────────────────────────────────
    RECALL_DEFAULT_MAX_ITEMS: int = 16
    RECALL_DEFAULT_TIME_WINDOW_DAYS: int = 30
    RECALL_DEFAULT_MODE: str = "hybrid"  # short_term | deep | hybrid

    # ── Source Toggles ────────────────────────────────────────────────
    RECALL_ENABLE_SQL_CHAT: bool = True
    RECALL_ENABLE_SQL_MIRRORS: bool = True
    RECALL_ENABLE_VECTOR: bool = True
    RECALL_ENABLE_RDF: bool = False

    # ── Postgres / SQL ────────────────────────────────────────────────
    RECALL_PG_DSN: str = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"

    # Chat history
    RECALL_SQL_CHAT_TABLE: str = "chat_history_log"
    RECALL_SQL_CHAT_TEXT_COL: str = "prompt"
    RECALL_SQL_CHAT_RESPONSE_COL: str = "response"
    RECALL_SQL_CHAT_CREATED_AT_COL: str = "created_at"

    # Collapse mirror base + semantic fields
    RECALL_SQL_MIRROR_TABLE: str = "collapse_mirror"
    RECALL_SQL_MIRROR_SUMMARY_COL: str = "summary"
    RECALL_SQL_MIRROR_TRIGGER_COL: str = "trigger"
    RECALL_SQL_MIRROR_OBSERVER_COL: str = "observer"
    RECALL_SQL_MIRROR_OBSERVER_STATE_COL: str = "observer_state"
    RECALL_SQL_MIRROR_FIELD_RESONANCE_COL: str = "field_resonance"
    RECALL_SQL_MIRROR_INTENT_COL: str = "intent"
    RECALL_SQL_MIRROR_TYPE_COL: str = "type"
    RECALL_SQL_MIRROR_ENTITY_COL: str = "emergent_entity"
    RECALL_SQL_MIRROR_MANTRA_COL: str = "mantra"
    RECALL_SQL_MIRROR_CAUSAL_ECHO_COL: str = "causal_echo"
    RECALL_SQL_MIRROR_TS_COL: str = "timestamp"

    # Collapse enrichment
    RECALL_SQL_ENRICH_TABLE: str = "collapse_enrichment"
    RECALL_SQL_ENRICH_COLLAPSE_ID_COL: str = "collapse_id"
    RECALL_SQL_ENRICH_TAGS_COL: str = "tags"
    RECALL_SQL_ENRICH_ENTITIES_COL: str = "entities"
    RECALL_SQL_ENRICH_SALIENCE_COL: str = "salience"
    RECALL_SQL_ENRICH_TS_COL: str = "ts"

    # ── Global Vector / RDF knobs (for compatibility with your env) ───
    # These mirror the project-wide envs you already have.
    VECTOR_DB_HOST: str = "orion-athena-vector-db"
    VECTOR_DB_PORT: int = 8000
    VECTOR_DB_COLLECTION: str = "orion_main_store"

    GRAPHDB_URL: str = "http://orion-athena-graphdb:7200"
    GRAPHDB_REPO: str = "collapse"
    GRAPHDB_USER: str = "admin"
    GRAPHDB_PASS: str = "admin"

    # ── Vector backend (recall-specific overrides) ────────────────────
    RECALL_VECTOR_BASE_URL: str = "http://orion-athena-vector-db:8000"
    RECALL_VECTOR_COLLECTIONS: str = "docs_design,docs_research,memory_summaries"
    RECALL_VECTOR_TIMEOUT_SEC: float = 5.0
    RECALL_VECTOR_MAX_ITEMS: int = 24

    # ── RDF / GraphDB (optional, recall-specific) ─────────────────────
    RECALL_RDF_ENDPOINT_URL: str = "http://orion-athena-graphdb:7200/repositories/collapse"
    RECALL_RDF_TIMEOUT_SEC: float = 5.0
    RECALL_RDF_ENABLE_SUMMARIES: bool = False

    # ── Future tensor / ranker toggles ────────────────────────────────
    RECALL_TENSOR_RANKER_ENABLED: bool = False
    RECALL_TENSOR_RANKER_MODEL_PATH: str = "/mnt/storage-warm/orion/recall/tensor-ranker.pt"


settings = Settings()

# app/settings.py
from __future__ import annotations

from pydantic import BaseSettings


class Settings(BaseSettings):
    # ── Service Metadata ──────────────────────────────────────────────
    SERVICE_NAME: str = "recall"
    SERVICE_VERSION: str = "0.1.0"
    PORT: int = 8260

    # ── Orion Bus ─────────────────────────────────────────────────────
    ORION_BUS_ENABLED: bool = True
    ORION_BUS_URL: str = "redis://localhost:6379/0"

    CHANNEL_RECALL_REQUEST: str = "orion:recall:request"
    CHANNEL_RECALL_DEFAULT_REPLY_PREFIX: str = "orion:recall:reply"

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
    RECALL_PG_DSN: str = "postgresql://postgres:postgres@localhost:5432/conjourney"

    # Chat history
    RECALL_SQL_CHAT_TABLE: str = "chat_history_log"
    RECALL_SQL_CHAT_TEXT_COL: str = "prompt"
    RECALL_SQL_CHAT_RESPONSE_COL: str = "response"
    RECALL_SQL_CHAT_CREATED_AT_COL: str = "created_at"

    # Collapse mirror
    RECALL_SQL_MIRROR_TABLE: str = "collapse_mirror"
    RECALL_SQL_MIRROR_SUMMARY_COL: str = "summary"
    RECALL_SQL_MIRROR_TRIGGER_COL: str = "trigger"
    RECALL_SQL_MIRROR_OBSERVER_COL: str = "observer"
    RECALL_SQL_MIRROR_OBSERVER_STATE_COL: str = "observer_state"
    RECALL_SQL_MIRROR_TYPE_COL: str = "type"
    RECALL_SQL_MIRROR_MANTRA_COL: str = "mantra"
    RECALL_SQL_MIRROR_INTENT_COL: str = "intent"
    RECALL_SQL_MIRROR_ENTITY_COL: str = "emergent_entity"
    RECALL_SQL_MIRROR_CAUSAL_ECHO_COL: str = "causal_echo"
    RECALL_SQL_MIRROR_TS_COL: str = "timestamp"

    # Collapse enrichment
    RECALL_SQL_ENRICH_TABLE: str = "collapse_enrichment"
    RECALL_SQL_ENRICH_COLLAPSE_ID_COL: str = "collapse_id"
    RECALL_SQL_ENRICH_TAGS_COL: str = "tags"
    RECALL_SQL_ENRICH_SALIENCE_COL: str = "salience"
    RECALL_SQL_ENRICH_TS_COL: str = "ts"

    # ── Vector backend (Chroma / orion-vector-db) ─────────────────────
    RECALL_VECTOR_BASE_URL: str = "http://orion-vector-db:8000"
    RECALL_VECTOR_COLLECTIONS: str = "docs_design,docs_research,memory_summaries"
    RECALL_VECTOR_TIMEOUT_SEC: float = 5.0
    RECALL_VECTOR_MAX_ITEMS: int = 24

    # ── RDF / GraphDB (optional) ──────────────────────────────────────
    RECALL_RDF_ENDPOINT_URL: str = "http://graphdb:7200/repositories/collapse"
    RECALL_RDF_TIMEOUT_SEC: float = 5.0
    RECALL_RDF_ENABLE_SUMMARIES: bool = False
    RECALL_RDF_USER: str = "admin"
    RECALL_RDF_PASS: str = "admin"

    # ── Future tensor / ranker toggles ────────────────────────────────
    RECALL_TENSOR_RANKER_ENABLED: bool = False
    RECALL_TENSOR_RANKER_MODEL_PATH: str = "/mnt/storage-warm/orion/recall/tensor-ranker.pt"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

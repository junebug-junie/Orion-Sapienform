# services/orion-recall/app/settings.py
from __future__ import annotations


from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # ── Service Metadata ──────────────────────────────────────────────
    SERVICE_NAME: str = "recall"
    SERVICE_VERSION: str = "0.1.0"
    PORT: int = 8260

    # ── Orion Bus ─────────────────────────────────────────────────────
    ORION_BUS_ENABLED: bool = True
    ORION_BUS_URL: str = "redis://localhost:6379/0"

    # ── Bus Channels ──────────────────────────────────────────────────
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

    # ── Postgres / SQL base DSN ───────────────────────────────────────
    RECALL_PG_DSN: str = "postgresql://postgres:postgres@localhost:5432/conjourney"

    # ── Chat history table ────────────────────────────────────────────
    RECALL_SQL_CHAT_TABLE: str = "chat_history_log"
    RECALL_SQL_CHAT_TEXT_COL: str = "prompt"
    RECALL_SQL_CHAT_RESPONSE_COL: str = "response"
    RECALL_SQL_CHAT_CREATED_AT_COL: str = "created_at"

    # ── Collapse Mirror base table ────────────────────────────────────
    RECALL_SQL_MIRROR_TABLE: str = "collapse_mirror"

    # Core textual fields (as in your Collapse Mirror schema)
    RECALL_SQL_MIRROR_SUMMARY_COL: str = "summary"
    RECALL_SQL_MIRROR_TRIGGER_COL: str = "trigger"
    RECALL_SQL_MIRROR_OBSERVER_COL: str = "observer"
    RECALL_SQL_MIRROR_OBSERVER_STATE_COL: str = "observer_state"
    RECALL_SQL_MIRROR_FIELD_RESONANCE_COL: str = "field_resonance"
    RECALL_SQL_MIRROR_INTENT_COL: str = "intent"
    RECALL_SQL_MIRROR_TYPE_COL: str = "type"
    RECALL_SQL_MIRROR_EMERGENT_ENTITY_COL: str = "emergent_entity"
    RECALL_SQL_MIRROR_MANTRA_COL: str = "mantra"
    RECALL_SQL_MIRROR_CAUSAL_ECHO_COL: str = "causal_echo"

    # Timestamp / created-at column
    # .env uses RECALL_SQL_MIRROR_CREATED_AT_COL=timestamp
    RECALL_SQL_MIRROR_CREATED_AT_COL: str = "timestamp"

    # Legacy alias (if any code still uses TS_COL)
    RECALL_SQL_MIRROR_TS_COL: str = "timestamp"

    # Optional "title / body / tags" aliases (for future use)
    RECALL_SQL_MIRROR_TITLE_COL: str = "title"
    RECALL_SQL_MIRROR_BODY_COL: str = "body"
    RECALL_SQL_MIRROR_TAGS_COL: str = "tags"

    # ── Collapse Enrichment table (semantic tags, salience) ───────────
    # Canonical names that match your schema:
    #   __tablename__ = "collapse_enrichment"
    #   collapse_id, tags, entities, salience, ts
    RECALL_SQL_MIRROR_ENRICH_TABLE: str = "collapse_enrichment"
    RECALL_SQL_MIRROR_ENRICH_FK_COL: str = "collapse_id"
    RECALL_SQL_MIRROR_ENRICH_TAGS_COL: str = "tags"
    RECALL_SQL_MIRROR_ENRICH_ENTITIES_COL: str = "entities"
    RECALL_SQL_MIRROR_ENRICH_SALIENCE_COL: str = "salience"
    RECALL_SQL_MIRROR_ENRICH_TS_COL: str = "ts"

    # Backwards-compat alias block (in case any earlier code uses these)
    RECALL_SQL_ENRICH_TABLE: str = "collapse_enrichment"
    RECALL_SQL_ENRICH_COLLAPSE_ID_COL: str = "collapse_id"
    RECALL_SQL_ENRICH_TAGS_COL: str = "tags"
    RECALL_SQL_ENRICH_SALIENCE_COL: str = "salience"
    RECALL_SQL_ENRICH_TS_COL: str = "ts"

    # ── Vector backend (Chroma / orion-vector-db) ─────────────────────
    # Project-wide base (used by Dream-style aggregators)
    VECTOR_DB_HOST: str = "orion-vector-db"
    VECTOR_DB_PORT: int = 8000
    VECTOR_DB_COLLECTION: str = "orion_main_store"

    # Recall-specific overrides (your .env can leave these blank)
    RECALL_VECTOR_BASE_URL: str = ""  # if empty, build from host/port/collection
    RECALL_VECTOR_COLLECTIONS: str = "docs_design,docs_research,memory_summaries"
    RECALL_VECTOR_TIMEOUT_SEC: float = 5.0
    RECALL_VECTOR_MAX_ITEMS: int = 24

    # ── GraphDB / RDF (shared style with Dream) ───────────────────────
    GRAPHDB_URL: str = "http://graphdb:7200"
    GRAPHDB_REPO: str = "collapse"
    GRAPHDB_USER: str = "admin"
    GRAPHDB_PASS: str = "admin"

    # Recall-specific RDF endpoint override
    RECALL_RDF_ENDPOINT_URL: str = ""  # if empty, build from GRAPHDB_URL/REPO
    RECALL_RDF_TIMEOUT_SEC: float = 5.0
    RECALL_RDF_ENABLE_SUMMARIES: bool = False

    # ── Future tensor / ranker toggles ────────────────────────────────
    RECALL_TENSOR_RANKER_ENABLED: bool = False
    RECALL_TENSOR_RANKER_MODEL_PATH: str = "/mnt/storage-warm/orion/recall/tensor-ranker.pt"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

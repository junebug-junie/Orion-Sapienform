# ==================================================
# settings.py
# ==================================================
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # --- Metadata ---
    PROJECT: str = Field(default="orion-janus")
    SERVICE_NAME: str = Field(default="orion-dream")
    SERVICE_VERSION: str = Field(default="1.0.0")
    NODE_NAME: str = Field(default="unknown")
    ENVIRONMENT: str = Field(default="prod")
    PORT: int = Field(default=8620)

    # --- Redis ---
    ORION_BUS_URL: str = Field(default="redis://redis:6379/0")
    ORION_BUS_ENABLED: bool = Field(default=True)
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False)

    # --- Channels ---
    CHANNEL_DREAM_TRIGGER: str = Field(default="orion:dream:trigger")
    CHANNEL_DREAM_BUFFER: str = Field(default="orion:dream:buffer")
    CHANNEL_DREAM_COMPLETE: str = Field(default="orion:dream:complete")
    CHANNEL_DREAM_STATUS: str = Field(default="orion:dream:status")
    CHANNEL_BRAIN_INTAKE: str = Field(default="orion:brain:intake") # Legacy

    # --- REM compaction (Phase F, default-off, staged — applies nothing) ---
    # When on, REM narration reads the Phase-E compaction-request queue + recent
    # episodes/motifs and emits a MemoryCompactionDeltaV1 (proposal_marked=true)
    # on CHANNEL_DREAM_COMPACTION_DELTA. No service applies it; the hub previews it.
    ORION_DREAM_REM_ENABLED: bool = Field(default=False)
    CHANNEL_DREAM_COMPACTION_DELTA: str = Field(default="orion:dream:compaction-delta")
    # Cap on requests drained per REM pass (§cap-all-collections).
    DREAM_REM_MAX_REQUESTS: int = Field(default=50)

    # --- Compaction APPLIER (Phase G — the hot gate, hard-off) ---
    # THIS MUTATES MEMORY. It stays off pending explicit proposal-mode sign-off +
    # a live §14 backfill verification. Even when on it applies ONLY deltas whose
    # proposal was policy-approved for execution (reverie proposals carry
    # operator_review, so they require a human). Snapshot precedes every apply.
    ORION_DREAM_COMPACTION_APPLY_ENABLED: bool = Field(default=False)
    # Safer subset first: apply downscale-renormalize only; prune stays gated
    # behind this flag being flipped false (never prune before downscale is trusted).
    ORION_DREAM_COMPACTION_DOWNSCALE_ONLY: bool = Field(default=True)
    # §14 snapshot destination (before/after + rollback artifact).
    DREAM_COMPACTION_SNAPSHOT_DIR: str = Field(default="/tmp/dream-compaction-apply")

    CHANNEL_CORTEX_GATEWAY_REQUEST: str = Field(default="orion:cortex:gateway:request", alias="CORTEX_GATEWAY_REQUEST_CHANNEL")
    CHANNEL_DREAM_REPLY_PREFIX: str = Field(default="orion:dream:reply", alias="DREAM_REPLY_PREFIX")
    DREAM_VERB: str = Field(default="dream_cycle", alias="DREAM_VERB")

    # --- Memory streams ---
    CHANNEL_COLLAPSE_SQL_PUBLISH: str = Field(default="orion:collapse:sql-write")
    CHANNEL_COLLAPSE_TAGS_PUBLISH: str = Field(default="orion:tags:enriched")
    CHANNEL_TELEMETRY_PUBLISH: str = Field(default="orion:biometrics:telemetry")
    CHANNEL_CHAT: str = Field(default="orion:chat:history:log")

    # --- Stores ---
    POSTGRES_URI: str = Field(default="postgresql://postgres:postgres@postgres:5432/conjourney")
    VECTOR_DB_HOST: str = Field(default="vector-db")
    VECTOR_DB_PORT: int = Field(default=8000)
    VECTOR_DB_COLLECTION: str = Field(default="orion_main_store")

    RDF_STORE_QUERY_URL: str = Field(default="")
    RDF_STORE_USER: str = Field(default="admin")
    RDF_STORE_PASS: str = Field(default="orion")
    GRAPHDB_URL: str = Field(default="http://graphdb:7200")
    GRAPHDB_REPO: str = Field(default="collapse")
    GRAPHDB_USER: str = Field(default="admin")
    GRAPHDB_PASS: str = Field(default="admin")

    @property
    def rdf_sparql_endpoint(self) -> str:
        q = (self.RDF_STORE_QUERY_URL or "").strip()
        if q:
            return q
        base = (self.GRAPHDB_URL or "").strip().rstrip("/")
        if not base:
            return ""
        repo = (self.GRAPHDB_REPO or "collapse").strip() or "collapse"
        return f"{base}/repositories/{repo}"

    @property
    def rdf_sparql_auth(self) -> tuple[str, str]:
        if (self.RDF_STORE_QUERY_URL or "").strip():
            return (self.RDF_STORE_USER or "admin", self.RDF_STORE_PASS or "orion")
        return (self.GRAPHDB_USER, self.GRAPHDB_PASS)

    # --- Brain ---
    BRAIN_URL: str = Field(default="http://brain:8088")
    LLM_MODEL: str = Field(default="mistral:instruct")

    DREAM_LOG_DIR: str = Field(default="/app/logs/dreams")

    # --- Chassis Defaults ---
    HEARTBEAT_INTERVAL_SEC: float = 10.0
    ORION_HEALTH_CHANNEL: str = "orion:system:health"
    ERROR_CHANNEL: str = "orion:system:error"
    SHUTDOWN_GRACE_SEC: float = 10.0

settings = Settings()

if settings.DREAM_LOG_DIR:
    # Best-effort: importing settings must not crash where the log dir isn't
    # writable (tests, constrained hosts). The container mounts a writable path.
    try:
        os.makedirs(settings.DREAM_LOG_DIR, exist_ok=True)
    except OSError:
        pass

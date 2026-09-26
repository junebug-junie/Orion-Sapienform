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

    # --- REM compaction (Phase F, default-off, staged — applies nothing) ---
    # When on, REM narration reads the Phase-E compaction-request queue + recent
    # episodes/motifs and emits a MemoryCompactionDeltaV1 (proposal_marked=true)
    # on CHANNEL_DREAM_COMPACTION_DELTA. No service applies it; the hub previews it.
    # With the dream cycle on, this pass also runs once inside each sleep.
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

    # --- Dream cycle v2 (sleep pressure -> replay -> REM recombination) ---
    # Default off. When on, a background loop checks sleep pressure every
    # DREAM_CYCLE_CHECK_INTERVAL_SEC and sleeps when pressure >= threshold AND
    # no chat turn for DREAM_IDLE_MINUTES AND the last cycle ended at least
    # DREAM_MIN_INTERVAL_HOURS ago. Writes only dream_cycle / dream_replay_item /
    # dream_hypothesis (services/orion-sql-db/manual_migration_dream_cycle_v2.sql).
    ORION_DREAM_CYCLE_ENABLED: bool = Field(default=False)
    DREAM_CYCLE_CHECK_INTERVAL_SEC: float = Field(default=600.0, gt=0.0)
    DREAM_SLEEP_PRESSURE_THRESHOLD: float = Field(default=3.0, ge=0.0)
    DREAM_IDLE_MINUTES: float = Field(default=45.0, ge=0.0)
    DREAM_MIN_INTERVAL_HOURS: float = Field(default=6.0, ge=0.0)
    DREAM_LOOKBACK_HOURS: float = Field(default=48.0, gt=0.0)
    DREAM_CANDIDATES_PER_SOURCE: int = Field(default=50, ge=1)
    DREAM_REPLAY_MAX: int = Field(default=12, ge=0, le=24)
    # Arms: dream pairs come from the replay set; control pairs are random pairs
    # from the whole candidate pool through the same prompt (the baseline).
    DREAM_HYPOTHESES_PER_CYCLE: int = Field(default=3, ge=0, le=8)
    DREAM_CONTROL_PER_CYCLE: int = Field(default=1, ge=0, le=4)
    DREAM_HYPOTHESIS_TTL_HOURS: float = Field(default=72.0, gt=0.0)
    CHANNEL_LLM_INTAKE: str = Field(default="orion:exec:request:LLMGatewayService")
    DREAM_LLM_ROUTE: str = Field(default="metacog_background")
    DREAM_LLM_TIMEOUT_SEC: float = Field(default=90.0, gt=0.0)

    # --- Stores ---
    POSTGRES_URI: str = Field(default="postgresql://postgres:postgres@postgres:5432/conjourney")

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

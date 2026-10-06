from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = Field(default="orion-memory-consolidation", alias="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.1.0", alias="SERVICE_VERSION")
    NODE_NAME: str = Field(default="athena", alias="NODE_NAME")
    LOG_LEVEL: str = Field(default="INFO", alias="LOG_LEVEL")
    PORT: int = Field(default=8635, alias="PORT")

    ORION_BUS_URL: str = Field(default="redis://127.0.0.1:6379/0", alias="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, alias="ORION_BUS_ENABLED")
    ORION_HEALTH_CHANNEL: str = Field(default="orion:system:health", alias="ORION_HEALTH_CHANNEL")
    ERROR_CHANNEL: str = Field(default="orion:system:error", alias="ERROR_CHANNEL")
    HEARTBEAT_INTERVAL_SEC: int = Field(default=30, alias="HEARTBEAT_INTERVAL_SEC")

    CHANNEL_MEMORY_TURN_PERSISTED: str = Field(
        default="orion:memory:turn:persisted", alias="CHANNEL_MEMORY_TURN_PERSISTED"
    )
    CHANNEL_CHAT_HISTORY_SPARK_META_PATCH: str = Field(
        default="orion:chat:history:spark_meta:patch", alias="CHANNEL_CHAT_HISTORY_SPARK_META_PATCH"
    )
    CHANNEL_LLM_INTAKE: str = Field(
        default="orion:exec:request:LLMGatewayService", alias="CHANNEL_LLM_INTAKE"
    )
    CHANNEL_CORTEX_REQUEST: str = Field(
        default="orion:cortex:request", alias="CHANNEL_CORTEX_REQUEST"
    )
    CHANNEL_CORTEX_RESULT_PREFIX: str = Field(
        default="orion:cortex:result", alias="CHANNEL_CORTEX_RESULT_PREFIX"
    )

    POSTGRES_URI: str = Field(default="", alias="POSTGRES_URI")
    MEMORY_CONSOLIDATION_ENABLED: bool = Field(default=True, alias="MEMORY_CONSOLIDATION_ENABLED")
    MEMORY_CLASSIFY_TIMEOUT_SEC: float = Field(default=8.0, alias="MEMORY_CLASSIFY_TIMEOUT_SEC")
    # Gateway route for turn-change classify RPC (metacog = instruct-only; avoid thinking lanes).
    # metacog_background (2026-09-07, not plain metacog): this is background turn
    # classification, not live-turn work -- it should yield slot slack to Mind's
    # now-live metacog traffic via the gateway's priority_admission.py, not compete
    # evenly with it. See docs/superpowers/pr-reports/ for this patch.
    TURN_CHANGE_CLASSIFY_ROUTE: str = Field(default="metacog_background", alias="TURN_CHANGE_CLASSIFY_ROUTE")
    # Margin on novelty_score (0-1) for session-window reappraisal; also minimum confidence for substrate emit.
    TURN_CHANGE_CONFIDENCE_MARGIN: float = Field(default=0.15, alias="TURN_CHANGE_CONFIDENCE_MARGIN")
    TURN_CHANGE_SUBSTRATE_THRESHOLD: float = Field(default=0.65, alias="TURN_CHANGE_SUBSTRATE_THRESHOLD")
    TURN_CHANGE_WINDOW_TURNS: int = Field(default=3, alias="TURN_CHANGE_WINDOW_TURNS")
    CHANNEL_SIGNALS_PREFIX: str = Field(default="orion:signals", alias="CHANNEL_SIGNALS_PREFIX")
    MEMORY_BOUNDARY_SCORE_THRESHOLD: float = Field(default=0.70, alias="MEMORY_BOUNDARY_SCORE_THRESHOLD")
    MEMORY_BOUNDARY_LLM_ONLY_THRESHOLD: float = Field(default=0.85, alias="MEMORY_BOUNDARY_LLM_ONLY_THRESHOLD")
    MEMORY_BOUNDARY_OVERRIDE_THRESHOLD: float = Field(default=0.92, alias="MEMORY_BOUNDARY_OVERRIDE_THRESHOLD")
    MEMORY_SUGGEST_TIMEOUT_SEC: float = Field(default=180.0, alias="MEMORY_SUGGEST_TIMEOUT_SEC")
    MEMORY_GRAPH_SUGGEST_MAX_TOKENS: int = Field(default=4096, alias="MEMORY_GRAPH_SUGGEST_MAX_TOKENS")
    MEMORY_GRAPH_SUGGEST_CTX_TOKENS: int = Field(default=4096, alias="MEMORY_GRAPH_SUGGEST_CTX_TOKENS")
    MEMORY_GRAPH_SUGGEST_PROMPT_OVERHEAD_TOKENS: int = Field(
        default=1800, alias="MEMORY_GRAPH_SUGGEST_PROMPT_OVERHEAD_TOKENS"
    )
    MEMORY_GRAPH_SUGGEST_MIN_COMPLETION_TOKENS: int = Field(
        default=768, alias="MEMORY_GRAPH_SUGGEST_MIN_COMPLETION_TOKENS"
    )
    MEMORY_GRAPH_SUGGEST_CHARS_PER_TOKEN: int = Field(default=3, alias="MEMORY_GRAPH_SUGGEST_CHARS_PER_TOKEN")
    MEMORY_GRAPH_SUGGEST_MIN_PROMPT_TOKENS_ESTIMATE: int = Field(
        default=400, alias="MEMORY_GRAPH_SUGGEST_MIN_PROMPT_TOKENS_ESTIMATE"
    )
    MEMORY_WINDOW_FALLBACK_GAP_SEC: int = Field(default=5400, alias="MEMORY_WINDOW_FALLBACK_GAP_SEC")
    # Memory episode redesign Stage 1 (2026-10-02): boundary Rule 3 runs in
    # SHADOW beside the live windows and writes memory_episode_shadow, then
    # publishes memory.episode.closed.v1. Kill switch: false stops both; the
    # live windows are unaffected either way.
    MEMORY_EPISODE_SHADOW_ENABLED: bool = Field(default=True, alias="MEMORY_EPISODE_SHADOW_ENABLED")
    # Whether the LIVE (legacy) window rule and the turn classify prompt see the
    # Hub's new spark_meta.conversation_phase stamp (boundary Fix 1). Default
    # false keeps live window closing exactly as it was: the stamp is recorded
    # and drives the Rule 3 shadow only. Turning it on changes live windows
    # (30-day replay: ~98 windows -> ~33-35) and the classify prompt's phase
    # line; see the Stage 1 PR report before flipping it.
    MEMORY_LEGACY_BOUNDARY_USE_PHASE: bool = Field(default=False, alias="MEMORY_LEGACY_BOUNDARY_USE_PHASE")
    # Daily old-vs-new memory report (Stage 1 PR 2): yesterday's shadow episodes, legacy
    # crystallization rows next to the shadow distiller's memories, as markdown on a named
    # volume. Read-only; no notification. Quotes Juniper's conversation: never commit it.
    MEMORY_EPISODE_REPORT_ENABLED: bool = Field(default=True, alias="MEMORY_EPISODE_REPORT_ENABLED")
    MEMORY_EPISODE_REPORT_DIR: str = Field(
        default="/data/memory-episode-reports", alias="MEMORY_EPISODE_REPORT_DIR"
    )
    MEMORY_EPISODE_REPORT_TZ: str = Field(default="America/Denver", alias="MEMORY_EPISODE_REPORT_TZ")
    # Memory confirmation loop (2026-10-06): open "Orion is asking" cards for high-stakes shadow
    # memories (max 5 open, 7-day expiry) and apply Juniper's answers from
    # orion:attention:loop_outcome plus a table catch-up. Kill switch: false stops all three.
    MEMORY_CONFIRMATION_LOOP_ENABLED: bool = Field(default=True, alias="MEMORY_CONFIRMATION_LOOP_ENABLED")
    MEMORY_CONFIRMATION_TICK_SEC: float = Field(default=60.0, gt=0.0, alias="MEMORY_CONFIRMATION_TICK_SEC")
    CHANNEL_ATTENTION_LOOP_OUTCOME: str = Field(
        default="orion:attention:loop_outcome", alias="CHANNEL_ATTENTION_LOOP_OUTCOME"
    )
    CHANNEL_MEMORY_EPISODE_CLOSED: str = Field(
        default="orion:memory:episode:closed", alias="CHANNEL_MEMORY_EPISODE_CLOSED"
    )
    MEMORY_FAILED_RETRY_INTERVAL_SEC: int = Field(default=1800, alias="MEMORY_FAILED_RETRY_INTERVAL_SEC")
    MEMORY_CLASSIFY_RETRY_INTERVAL_SEC: int = Field(default=120, alias="MEMORY_CLASSIFY_RETRY_INTERVAL_SEC")
    MEMORY_CONSOLIDATION_OUTPUT: str = Field(
        default="crystallization_propose", alias="MEMORY_CONSOLIDATION_OUTPUT"
    )
    MEMORY_CONSOLIDATION_MIN_NOVELTY: float = Field(default=0.35, alias="MEMORY_CONSOLIDATION_MIN_NOVELTY")
    MEMORY_CONSOLIDATION_MIN_SIGNIFICANCE: float = Field(
        default=0.40, alias="MEMORY_CONSOLIDATION_MIN_SIGNIFICANCE"
    )
    MEMORY_CONSOLIDATION_FETCH_GRAMMAR_EVIDENCE: bool = Field(
        default=True, alias="MEMORY_CONSOLIDATION_FETCH_GRAMMAR_EVIDENCE"
    )
    MEMORY_CONSOLIDATION_GRAMMAR_DSN: str = Field(default="", alias="MEMORY_CONSOLIDATION_GRAMMAR_DSN")
    MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED: bool = Field(
        default=False, alias="MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED"
    )
    MEMORY_FORMATION_AUTO_ENCODE_ACTIVATION_RATIO: float = Field(
        default=0.4, alias="MEMORY_FORMATION_AUTO_ENCODE_ACTIVATION_RATIO"
    )
    # Comma-separated external platforms (chat_history_log client_meta.external_room
    # .platform) whose windows never become a crystallization at all -- no governor
    # queue, no auto-activation, no graph/vector projection. The raw chat turns are
    # unaffected (orion-sql-writer already wrote them to chat_history_log
    # independently of this pipeline). Empty string disables the gate entirely.
    # Renamed 2026-08-16 from MEMORY_FORMATION_AUTO_ACTIVATE_PLATFORMS: that name
    # described the first cut of this gate, which still formed and projected the
    # memory, just skipped review. Juniper's direct correction ("I just don't want
    # it on the graphs and crystalizations") made that name inaccurate.
    MEMORY_FORMATION_DISCARD_PLATFORMS: str = Field(
        default="aitown", alias="MEMORY_FORMATION_DISCARD_PLATFORMS"
    )

    @property
    def discard_platforms(self) -> frozenset[str]:
        raw = self.MEMORY_FORMATION_DISCARD_PLATFORMS or ""
        return frozenset(p.strip() for p in raw.split(",") if p.strip())

    # Cross-window concept-relation resolution (candidate retrieval + typed link dispatch).
    # Off by default pending review -- see orion/memory/crystallization/concept_relation.py.
    CRYSTALLIZER_VECTOR_COLLECTION: str = Field(
        default="orion_memory_crystallizations", alias="CRYSTALLIZER_VECTOR_COLLECTION"
    )
    CRYSTALLIZER_EMBED_HOST_URL: str = Field(default="", alias="CRYSTALLIZER_EMBED_HOST_URL")
    CRYSTALLIZER_EMBED_TIMEOUT_MS: int = Field(default=8000, alias="CRYSTALLIZER_EMBED_TIMEOUT_MS")
    CHROMA_HOST: str = Field(default="", alias="CHROMA_HOST")
    CHROMA_PORT: int = Field(default=8000, alias="CHROMA_PORT")

    CONCEPT_RELATION_RESOLUTION_ENABLED: bool = Field(
        default=False, alias="CONCEPT_RELATION_RESOLUTION_ENABLED"
    )
    CONCEPT_RELATION_CONFIDENCE_FLOOR: float = Field(
        default=0.6, alias="CONCEPT_RELATION_CONFIDENCE_FLOOR"
    )
    CONCEPT_RELATION_CANDIDATE_LIMIT: int = Field(
        default=5, alias="CONCEPT_RELATION_CANDIDATE_LIMIT"
    )
    CONCEPT_RELATION_TIMEOUT_SEC: float = Field(
        default=8.0, alias="CONCEPT_RELATION_TIMEOUT_SEC"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


settings = Settings()

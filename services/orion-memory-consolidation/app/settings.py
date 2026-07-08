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
    TURN_CHANGE_CLASSIFY_ROUTE: str = Field(default="metacog", alias="TURN_CHANGE_CLASSIFY_ROUTE")
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

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


settings = Settings()

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
    MEMORY_BOUNDARY_SCORE_THRESHOLD: float = Field(default=0.70, alias="MEMORY_BOUNDARY_SCORE_THRESHOLD")
    MEMORY_BOUNDARY_LLM_ONLY_THRESHOLD: float = Field(default=0.85, alias="MEMORY_BOUNDARY_LLM_ONLY_THRESHOLD")
    MEMORY_BOUNDARY_OVERRIDE_THRESHOLD: float = Field(default=0.92, alias="MEMORY_BOUNDARY_OVERRIDE_THRESHOLD")
    MEMORY_SUGGEST_TIMEOUT_SEC: float = Field(default=120.0, alias="MEMORY_SUGGEST_TIMEOUT_SEC")
    MEMORY_WINDOW_FALLBACK_GAP_SEC: int = Field(default=5400, alias="MEMORY_WINDOW_FALLBACK_GAP_SEC")
    MEMORY_FAILED_RETRY_INTERVAL_SEC: int = Field(default=1800, alias="MEMORY_FAILED_RETRY_INTERVAL_SEC")

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


settings = Settings()

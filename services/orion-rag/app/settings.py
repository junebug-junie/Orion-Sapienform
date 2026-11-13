from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Configuration for the Orion RAG service, loaded from environment variables.
    This service now acts as a client to the standalone vector database.
    """
    # --- Service Identity ---
    SERVICE_NAME: str = Field(..., env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(..., env="SERVICE_VERSION")
    PORT: int = Field(..., env="PORT")
    # Host this process runs on; comes from the root .env (NODE_NAME=athena etc.).
    NODE_NAME: str = Field(default="athena", env="NODE_NAME")

    # --- Orion Bus Integration ---
    ORION_BUS_ENABLED: bool = Field(..., env="ORION_BUS_ENABLED")
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False, env="ORION_BUS_ENFORCE_CATALOG")
    ORION_BUS_URL: str = Field(..., env="ORION_BUS_URL")

    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health). See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0, env="HEARTBEAT_INTERVAL_SEC")

    # --- Bus Channels ---
    SUBSCRIBE_CHANNEL_RAG_REQUEST: str = Field(..., env="SUBSCRIBE_CHANNEL_RAG_REQUEST")
    PUBLISH_CHANNEL_BRAIN_INTAKE: str = Field(..., env="PUBLISH_CHANNEL_BRAIN_INTAKE")

    # --- Vector Store Configuration ---
    VECTOR_DB_URL: str = Field(..., env="VECTOR_DB_URL")
    VECTOR_DB_COLLECTION: str = Field(..., env="VECTOR_DB_COLLECTION")
    EMBEDDING_MODEL: str = Field(..., env="EMBEDDING_MODEL")

    # --- Runtime Settings ---
    STARTUP_DELAY: int = Field(default=5, env="STARTUP_DELAY")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

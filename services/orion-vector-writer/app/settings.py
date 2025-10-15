from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Configuration for the Orion Vector Writer service.
    """
    # --- Service Identity ---
    SERVICE_NAME: str = Field(..., env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(..., env="SERVICE_VERSION")
    PORT: int = Field(..., env="PORT")

    # --- Orion Bus ---
    ORION_BUS_ENABLED: bool = Field(..., env="ORION_BUS_ENABLED")
    ORION_BUS_URL: str = Field(..., env="ORION_BUS_URL")

    # --- Subscription Channels ---
    SUBSCRIBE_CHANNEL_COLLAPSE: str = Field(..., env="SUBSCRIBE_CHANNEL_COLLAPSE")
    SUBSCRIBE_CHANNEL_CHAT: str = Field(..., env="SUBSCRIBE_CHANNEL_CHAT")
    SUBSCRIBE_CHANNEL_RAG_DOC: str = Field(..., env="SUBSCRIBE_CHANNEL_RAG_DOC")

    # --- Publish Channel ---
    PUBLISH_CHANNEL_VECTOR_CONFIRM: str = Field(..., env="PUBLISH_CHANNEL_VECTOR_CONFIRM")

    # --- Vector Store ---
    # Now uses explicit host and port for a reliable connection.
    VECTOR_DB_HOST: str = Field(..., env="VECTOR_DB_HOST")
    VECTOR_DB_PORT: int = Field(..., env="VECTOR_DB_PORT")
    VECTOR_DB_COLLECTION: str = Field(..., env="VECTOR_DB_COLLECTION")
    VECTOR_DB_CREATE_IF_MISSING: bool = Field(default=True, env="VECTOR_DB_CREATE_IF_MISSING")
    EMBEDDING_MODEL: str = Field(..., env="EMBEDDING_MODEL")

    # --- Runtime ---
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    BATCH_SIZE: int = Field(default=10, env="BATCH_SIZE")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

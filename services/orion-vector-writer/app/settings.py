from typing import List, Union
from pydantic_settings import BaseSettings
from pydantic import Field
import json

class Settings(BaseSettings):
    SERVICE_NAME: str = Field(default="orion-vector-writer", alias="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.2.0", alias="SERVICE_VERSION")
    NODE_NAME: str = Field(default="athena", alias="NODE_NAME")
    LOG_LEVEL: str = Field(default="INFO", alias="LOG_LEVEL")

    # Bus
    ORION_BUS_URL: str = Field(..., alias="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, alias="ORION_BUS_ENABLED")
    HEALTH_CHANNEL: str = "system.health"
    ERROR_CHANNEL: str = "system.error"

    # Subscriptions
    # We accept a string (JSON or comma-separated) and convert it, or a list if passed directly
    VECTOR_WRITER_SUBSCRIBE_CHANNELS: Union[str, List[str]] = Field(
        default='["orion:collapse:sql-write", "orion:chat:history:log", "orion:rag:ingest"]',
        alias="VECTOR_WRITER_SUBSCRIBE_CHANNELS"
    )

    @property
    def SUBSCRIBE_CHANNELS(self) -> List[str]:
        """Helper to parse the subscription channels from env var."""
        val = self.VECTOR_WRITER_SUBSCRIBE_CHANNELS
        if isinstance(val, list):
            return val
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return [x.strip() for x in val.split(",") if x.strip()]

    # Chroma / Vector DB Configuration
    CHROMA_HOST: str = Field(default="orion-vector-db", alias="VECTOR_DB_HOST")
    CHROMA_PORT: int = Field(default=8000, alias="VECTOR_DB_PORT")

    # Capture the collection from .env. Defaults to 'orion_general' if missing.
    CHROMA_COLLECTION_DEFAULT: str = Field(default="orion_general", alias="VECTOR_DB_COLLECTION")

    # Embedding
    EMBEDDING_MODEL_NAME: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    class Config:
        env_file = ".env"
        extra = "ignore" 
        # This allows populating by alias (e.g. VECTOR_DB_HOST -> CHROMA_HOST)
        populate_by_name = True

settings = Settings()

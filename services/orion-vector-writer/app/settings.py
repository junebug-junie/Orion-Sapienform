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
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False, alias="ORION_BUS_ENFORCE_CATALOG")
    ORION_HEALTH_CHANNEL: str = "orion:system:health"
    ERROR_CHANNEL: str = "orion:system:error"
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0, alias="HEARTBEAT_INTERVAL_SEC")

    # Subscriptions
    # We accept a string (JSON or comma-separated) and convert it, or a list if passed directly
    VECTOR_WRITER_SUBSCRIBE_CHANNELS: Union[str, List[str]] = Field(
        default='["orion:vector:semantic:upsert", "orion:vector:latent:upsert", "orion:vector:write", "orion:memory:vector:upsert", "orion:metacog:trace"]',
        alias="VECTOR_WRITER_SUBSCRIBE_CHANNELS"
    )
    # VECTOR_WRITER_CHAT_HISTORY_CHANNEL / VECTOR_WRITER_CHAT_COLLECTION
    # removed 2026-08-14: the dedicated chat-history normalization path
    # (`app/chat_history.py`, deleted) was already dead code -- this service
    # never subscribed to `orion:chat:history:log` by default, so
    # `chat_history_envelope_to_request` was never reachable. See
    # orion-vector-host's README for the live half of this kill.
    VECTOR_WRITER_METACOG_COLLECTION: str = Field(
        default="orion_metacog_store", alias="VECTOR_WRITER_METACOG_COLLECTION"
    )
    VECTOR_WRITER_METACOG_SUMMARY_MAX_CHARS: int = Field(
        default=512, alias="VECTOR_WRITER_METACOG_SUMMARY_MAX_CHARS"
    )
    VECTOR_WRITER_REQUIRE_EMBEDDINGS: bool = Field(
        default=False, alias="VECTOR_WRITER_REQUIRE_EMBEDDINGS"
    )

    @property
    def SUBSCRIBE_CHANNELS(self) -> List[str]:
        """Helper to parse the subscription channels from env var."""
        val = self.VECTOR_WRITER_SUBSCRIBE_CHANNELS
        channels: List[str]
        if isinstance(val, list):
            channels = list(val)
        else:
            try:
                channels = json.loads(val)
            except json.JSONDecodeError:
                channels = [x.strip() for x in val.split(",") if x.strip()]

        # Ensure canonical memory upsert channel is always present
        if "orion:memory:vector:upsert" not in channels:
            channels.append("orion:memory:vector:upsert")
        return channels

    # Chroma / Vector DB Configuration
    CHROMA_HOST: str = Field(default="orion-vector-db", alias="VECTOR_DB_HOST")
    CHROMA_PORT: int = Field(default=8000, alias="VECTOR_DB_PORT")

    # Capture the collection from .env. Defaults to 'orion_general' if missing.
    CHROMA_COLLECTION_DEFAULT: str = Field(default="orion_general", alias="VECTOR_DB_COLLECTION")
    CHROMA_COLLECTION_LATENT: str = Field(default="orion_latent_store", alias="VECTOR_DB_COLLECTION_LATENT")

    class Config:
        env_file = ".env"
        extra = "ignore" 
        # This allows populating by alias (e.g. VECTOR_DB_HOST -> CHROMA_HOST)
        populate_by_name = True

settings = Settings()

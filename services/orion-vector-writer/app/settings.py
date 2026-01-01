from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Dict
import json

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

    # --- Routing ---
    # List of channels to subscribe to
    VECTOR_WRITER_SUBSCRIBE_CHANNELS: List[str] = Field(
        default=[
            "orion:collapse:triage",
            "orion:chat:message",
            "orion:rag:doc"
        ],
        env="VECTOR_WRITER_SUBSCRIBE_CHANNELS"
    )

    # JSON mapping from envelope.kind -> collection (or internal model hint)
    # The current logic uses one collection for everything but different models.
    # We'll map kind -> collection name, defaulting to "orion_context" if not specified.
    # OR we map kind -> model type to validation.
    VECTOR_WRITER_ROUTE_MAP_JSON: str = Field(
        default=json.dumps({
            "collapse.triage": "CollapseTriageEvent",
            "chat.message": "ChatMessageEvent",
            "rag.document": "RAGDocumentEvent"
        }),
        env="VECTOR_WRITER_ROUTE_MAP_JSON"
    )

    @property
    def route_map(self) -> Dict[str, str]:
        try:
            return json.loads(self.VECTOR_WRITER_ROUTE_MAP_JSON)
        except Exception:
            return {}

    # --- Publish Channel ---
    PUBLISH_CHANNEL_VECTOR_CONFIRM: str = Field("orion:vector:confirm", env="PUBLISH_CHANNEL_VECTOR_CONFIRM")

    # --- Vector Store ---
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

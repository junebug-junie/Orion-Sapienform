from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = Field(default="orion-memory-crystallizer", alias="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.1.0", alias="SERVICE_VERSION")
    NODE_NAME: str = Field(default="athena", alias="NODE_NAME")
    LOG_LEVEL: str = Field(default="INFO", alias="LOG_LEVEL")

    ORION_BUS_URL: str = Field(..., alias="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, alias="ORION_BUS_ENABLED")
    ORION_HEALTH_CHANNEL: str = Field(default="orion:system:health", alias="ORION_HEALTH_CHANNEL")
    ERROR_CHANNEL: str = Field(default="orion:system:error", alias="ERROR_CHANNEL")
    HEARTBEAT_INTERVAL_SEC: int = Field(default=30, alias="HEARTBEAT_INTERVAL_SEC")

    CRYSTALLIZER_CHANNEL_PROPOSED: str = Field(
        default="orion:memory:crystallization:proposed", alias="CRYSTALLIZER_CHANNEL_PROPOSED"
    )
    CRYSTALLIZER_CHANNEL_VALIDATED: str = Field(
        default="orion:memory:crystallization:validated", alias="CRYSTALLIZER_CHANNEL_VALIDATED"
    )
    CRYSTALLIZER_CHANNEL_APPROVED: str = Field(
        default="orion:memory:crystallization:approved", alias="CRYSTALLIZER_CHANNEL_APPROVED"
    )
    CRYSTALLIZER_CHANNEL_REJECTED: str = Field(
        default="orion:memory:crystallization:rejected", alias="CRYSTALLIZER_CHANNEL_REJECTED"
    )
    CRYSTALLIZER_CHANNEL_QUARANTINED: str = Field(
        default="orion:memory:crystallization:quarantined", alias="CRYSTALLIZER_CHANNEL_QUARANTINED"
    )
    CRYSTALLIZER_CHANNEL_PROJECT: str = Field(
        default="orion:memory:crystallization:project", alias="CRYSTALLIZER_CHANNEL_PROJECT"
    )
    CRYSTALLIZER_CHANNEL_RETRIEVED: str = Field(
        default="orion:memory:crystallization:retrieved", alias="CRYSTALLIZER_CHANNEL_RETRIEVED"
    )
    CRYSTALLIZER_CHANNEL_VECTOR_UPSERT: str = Field(
        default="orion:memory:vector:upsert", alias="CRYSTALLIZER_CHANNEL_VECTOR_UPSERT"
    )

    POSTGRES_URI: str = Field(default="", alias="POSTGRES_URI")
    CRYSTALLIZER_AUTO_APPLY_SCHEMA: bool = Field(default=True, alias="CRYSTALLIZER_AUTO_APPLY_SCHEMA")
    CRYSTALLIZER_VECTOR_COLLECTION: str = Field(
        default="orion_memory_crystallizations", alias="CRYSTALLIZER_VECTOR_COLLECTION"
    )
    CRYSTALLIZER_EMBED_HOST_URL: str = Field(default="", alias="CRYSTALLIZER_EMBED_HOST_URL")
    CRYSTALLIZER_EMBED_TIMEOUT_MS: int = Field(default=8000, alias="CRYSTALLIZER_EMBED_TIMEOUT_MS")

    GRAPHITI_ENABLED: bool = Field(default=False, alias="GRAPHITI_ENABLED")
    GRAPHITI_URL: str = Field(default="", alias="GRAPHITI_URL")
    FALKORDB_URI: str = Field(default="", alias="FALKORDB_URI")

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


settings = Settings()

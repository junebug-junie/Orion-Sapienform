from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # RDF + GraphDB
    GRAPHDB_URL: str = Field(default="http://graphdb:7200")
    GRAPHDB_REPO: str =Field(default="collapse")
    GRAPHDB_USER: str | None = None
    GRAPHDB_PASS: str | None = None

    # === ORION BUS (Shared Core) ===
    ORION_BUS_URL: str = Field(..., env="ORION_BUS_URL")
    ORION_BUS_CHANNEL: str = Field(default="orion:events:tagged", env="ORION_BUS_CHANNEL")
    ORION_CORE_EVENTS: str = Field(default="orion:core:events", env="ORION_CORE_EVENTS")
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")

    # === PUBLISH CHANNELS ===
    CHANNEL_EVENTS_TAGGED: str = Field(default="orion:events:tagged", env="CHANNEL_EVENTS_TAGGED")
    CHANNEL_RDF_ENQUEUE: str = Field(default="orion:rdf-collapse:enqueue", env="CHANNEL_RDF_ENQUEUE")
    CHANNEL_EVENTS_TRIAGE: str = Field(default="orion:collapse:triage", env="CHANNEL_EVENTS_TRIAGE")

    # === LISTENER CHANNELS ===
    CHANNEL_RDF_CONFIRM: str = Field(default="orion:rdf:confirm", env="CHANNEL_RDF_CONFIRM")
    CHANNEL_RDF_ERROR: str = Field(default="orion:rdf:error", env="CHANNEL_RDF_ERROR")

    # Service configuration
    SERVICE_NAME: str = Field(default="orion-rdf-writer")
    LOG_LEVEL: str = Field(default="INFO")
    BATCH_SIZE: int = Field(default=10)
    RETRY_LIMIT: int = Field(default=3)
    RETRY_INTERVAL: int = Field(default=2)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

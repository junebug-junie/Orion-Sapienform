from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # RDF + GraphDB
    GRAPHDB_URL: str = Field(..., env="GRAPHDB_URL")
    GRAPHDB_REPO: str = Field(..., env="GRAPHDB_REPO")
    GRAPHDB_USER: str | None = Field(None, env="GRAPHDB_USER")
    GRAPHDB_PASS: str | None = Field(None, env="GRAPHDB_PASS")

    # === ORION BUS (Shared Core) ===
    ORION_BUS_URL: str = Field(..., env="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")

    # === LISTENER CHANNELS ===
    # Defines all channels this service will subscribe to.
    CHANNEL_EVENTS_TRIAGE: str = Field(..., env="CHANNEL_EVENTS_TRIAGE")
    CHANNEL_EVENTS_TAGGED: str = Field(..., env="CHANNEL_EVENTS_TAGGED")
    CHANNEL_RDF_ENQUEUE: str = Field(..., env="CHANNEL_RDF_ENQUEUE")
    ORION_CORE_EVENTS: str = Field(..., env="ORION_CORE_EVENTS")

    # === PUBLISH CHANNELS ===
    CHANNEL_RDF_CONFIRM: str = Field(..., env="CHANNEL_RDF_CONFIRM")
    CHANNEL_RDF_ERROR: str = Field(..., env="CHANNEL_RDF_ERROR")

    # Service configuration
    SERVICE_NAME: str = Field(default="orion-rdf-writer", env="SERVICE_NAME")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    RETRY_LIMIT: int = Field(default=3, env="RETRY_LIMIT")
    RETRY_INTERVAL: int = Field(default=2, env="RETRY_INTERVAL")

    def get_all_subscribe_channels(self) -> list[str]:
        """Returns a list of all channels this service should subscribe to."""
        return [
            self.CHANNEL_EVENTS_TRIAGE,
            self.CHANNEL_EVENTS_TAGGED,
            self.CHANNEL_RDF_ENQUEUE,
            self.ORION_CORE_EVENTS,
        ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()


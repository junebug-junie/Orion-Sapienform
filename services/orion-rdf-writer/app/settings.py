from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List

class Settings(BaseSettings):
    # RDF + GraphDB
    GRAPHDB_URL: str = Field(default="http://graphdb:7200", env="GRAPHDB_URL")
    GRAPHDB_REPO: str = Field(default="orion", env="GRAPHDB_REPO")
    GRAPHDB_USER: str | None = Field(None, env="GRAPHDB_USER")
    GRAPHDB_PASS: str | None = Field(None, env="GRAPHDB_PASS")

    # === ORION BUS (Shared Core) ===
    ORION_BUS_URL: str = Field(default="redis://orion-janus-bus-core:6379/0", env="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")

    # === LISTENER CHANNELS ===
    # Enqueue (Direct writes)
    CHANNEL_RDF_ENQUEUE: str = Field(default="orion:rdf:enqueue", env="CHANNEL_RDF_ENQUEUE")
    # Collapse (Raw)
    CHANNEL_EVENTS_COLLAPSE: str = Field(default="orion:collapse:intake", env="CHANNEL_EVENTS_COLLAPSE")
    # Tagged/Enriched
    CHANNEL_EVENTS_TAGGED: str = Field(default="orion:events:tagged", env="CHANNEL_EVENTS_TAGGED")
    # Core Events (Filtering targets="rdf")
    CHANNEL_CORE_EVENTS: str = Field(default="orion:core:events", env="CHANNEL_CORE_EVENTS")

    # Worker intake for Cortex (RPC)
    CHANNEL_WORKER_RDF: str = Field(default="orion:rdf:worker", env="CHANNEL_WORKER_RDF")

    # === PUBLISH CHANNELS ===
    CHANNEL_RDF_CONFIRM: str = Field(default="orion:rdf:confirm", env="CHANNEL_RDF_CONFIRM")
    CHANNEL_RDF_ERROR: str = Field(default="orion:rdf:error", env="CHANNEL_RDF_ERROR")

    # Service configuration
    SERVICE_NAME: str = Field(default="orion-rdf-writer", env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.2.0", env="SERVICE_VERSION")
    NODE_NAME: str = Field(default="unknown")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_all_subscribe_channels(self) -> List[str]:
        """Returns a list of all channels this service should subscribe to."""
        return [
            self.CHANNEL_RDF_ENQUEUE,
            self.CHANNEL_EVENTS_COLLAPSE,
            self.CHANNEL_EVENTS_TAGGED,
            self.CHANNEL_CORE_EVENTS,
            self.CHANNEL_WORKER_RDF
        ]

settings = Settings()

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # RDF + GraphDB
    GRAPHDB_URL: str = Field(default="http://graphdb:7200")
    GRAPHDB_REPO: str =Field(default="collapse")
    GRAPHDB_USER: str | None = None
    GRAPHDB_PASS: str | None = None

    # Core bus (shared)
    ORION_BUS_URL: str = Field(default="redis://orion-redis:6379/0")
    ORION_BUS_CHANNEL: str = Field(default="orion:events:tagged")

    # Bus channels (multi-domain)
    CHANNEL_EVENTS_TAGGED: str = Field(default="orion:events:tagged")
    CHANNEL_RDF_ENQUEUE: str = Field(default="orion:rdf:enqueue")
    CHANNEL_CORE_EVENTS: str = Field(default="orion:core:events")
    CHANNEL_RDF_CONFIRM: str = Field(default="orion:rdf:confirm")
    CHANNEL_RDF_ERROR: str = Field(default="orion:rdf:error")

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

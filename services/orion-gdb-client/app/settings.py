from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Orion GDB Client configuration.
    Handles GraphDB connectivity, Redis bus, and runtime metadata.
    """

    # === Core Identity ===
    SERVICE_NAME: str = Field(default="gdb-client", env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="1.0.1", env="SERVICE_VERSION")
    PORT: int = Field(default=8000, env="PORT")

    # === Orion Bus ===
    ORION_BUS_URL: str = Field(default="redis://orion-janus-bus-core:6379/0", env="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")

    # This service now listens on two channels to handle both raw and enriched data.
    CHANNEL_COLLAPSE_TRIAGE: str = Field(default="orion:collapse:triage", env="CHANNEL_COLLAPSE_TRIAGE")
    CHANNEL_TAGS_ENRICHED: str = Field(default="orion:tags:enriched", env="CHANNEL_TAGS_ENRICHED")
    
    # Channels for publishing confirmations/errors
    CHANNEL_RDF_CONFIRM: str = Field(default="orion:rdf:confirm", env="CHANNEL_RDF_CONFIRM")
    CHANNEL_RDF_ERROR: str = Field(default="orion:rdf:error", env="CHANNEL_RDF_ERROR")

    # === GraphDB ===
    GRAPHDB_VERSION: str = Field(default="11.0.0", env="GRAPHDB_VERSION")
    GRAPHDB_URL: str = Field(default="http://orion-janus-graphdb:7200", env="GRAPHDB_URL")
    GRAPHDB_REPO: str = Field(default="collapse", env="GRAPHDB_REPO")
    GRAPHDB_HOME: str = Field(default="/mnt/storage/collapse-mirrors", env="GRAPHDB_HOME")
    GRAPHDB_IMPORT: str = Field(default="/mnt/storage/collapse-mirrors/import", env="GRAPHDB_IMPORT")

    # === Runtime ===
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    @field_validator("ORION_BUS_ENABLED", mode="before")
    def parse_bool(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


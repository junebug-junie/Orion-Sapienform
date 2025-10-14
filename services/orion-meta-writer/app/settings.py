from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    PROJECT: str = Field(default="orion", env="PROJECT")

    SERVICE_NAME: str = Field(default="orion-meta-writer", env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.1.0", env="SERVICE_VERSION")
    PORT: int = Field(default=8210, env="PORT")

    ORION_BUS_URL: str
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")

    # bus channels
    CHANNEL_EVENTS_TAGGED: str = Field(default="orion:tags:raw", env="CHANNEL_EVENTS_TAGGED")
    CHANNEL_EVENTS_ENRICHED: str = Field(default="orion:tags:enriched", env="CHANNEL_EVENTS_ENRICHED")
    CHANNEL_EVENTS_TRIAGE: str = Field(default="orion:collapse:triage", env="CHANNEL_EVENTS_TRIAGE")
    CHANNEL_RDF_ENQUEUE: str = Field(default="orion:rdf-collapse:enqueue", env="CHANNEL_RDF_ENQUEUE")

    STARTUP_DELAY: int = Field(default=5, env="STARTUP_DELAY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

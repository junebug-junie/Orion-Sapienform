from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Configuration for the Orion Vector Writer service.
    """
    # === Core Identity ===
    SERVICE_NAME: str = Field(..., env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(..., env="SERVICE_VERSION")
    PORT: int = Field(..., env="PORT")

    # === Orion Bus ===
    ORION_BUS_URL: str = Field(..., env="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(..., env="ORION_BUS_ENABLED")
    CHANNEL_COLLAPSE_TRIAGE: str = Field(..., env="CHANNEL_COLLAPSE_TRIAGE")
    CHANNEL_TAGS_ENRICHED: str = Field(..., env="CHANNEL_TAGS_ENRICHED")

    # === Vector Store ===
    VECTOR_DB_URL: str = Field(..., env="VECTOR_DB_URL")
    EMBEDDING_MODEL: str = Field(..., env="EMBEDDING_MODEL")
    VECTOR_DB_COLLECTION: str = Field(..., env="VECTOR_DB_COLLECTION")

    # === Runtime ===
    STARTUP_DELAY: int = Field(default=5, env="STARTUP_DELAY")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    def get_subscribe_channels(self) -> list[str]:
        """Returns a list of all channels this service should listen to."""
        return [
            self.CHANNEL_COLLAPSE_TRIAGE,
            self.CHANNEL_TAGS_ENRICHED,
        ]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

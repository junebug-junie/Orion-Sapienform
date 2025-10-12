from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # --- Orion Bus Configuration ---
    ORION_BUS_URL: str = "redis://orion-janus-bus-core:6379/0"
    ORION_BUS_ENABLED: bool = True
    SUBSCRIBE_CHANNEL: str = "orion.tags"
    PUBLISH_CHANNEL: str = "orion.tags.enriched"

    # --- Service Identity ---
    SERVICE_NAME: str = "orion-meta-writer"
    SERVICE_VERSION: str = "0.1.0"

    class Config:
       env_file = ".env"

settings = Settings()

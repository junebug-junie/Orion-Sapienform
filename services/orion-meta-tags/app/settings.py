from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ORION_BUS_URL: str = "redis://${PROJECT}-bus-core:6379/0"
    ORION_BUS_ENABLED: bool = True
    SPA_MODEL: str = "en_core_web_trf"
    SERVICE_NAME: str = "orion-meta-tags"
    SERVICE_VERSION: str = "0.0.0"
    SUBSCRIBE_CHANNEL: str = "collapse.events.raw"
    PUBLISH_CHANNEL: str = "orion.tags"

    class Config:
        env_file = ".env"


settings = Settings()

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ORION_BUS_URL: str = "redis://orion-redis:6379/0"
    ORION_BUS_ENABLED: bool = True
    SUBSCRIBE_CHANNEL: str = "orion.tags"
    SERVICE_NAME: str = "orion-enrichment-writer"
    SERVICE_VERSION: str = "0.1.0"

    POSTGRES_URI: str = "postgresql://postgres:postgres@postgres:5432/conjourney"
    CHROMA_PATH: str = "/mnt/storage/collapse-mirrors/chroma"

    class Config:
        env_file = ".env"

settings = Settings()

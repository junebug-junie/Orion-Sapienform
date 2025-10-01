from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    POSTGRES_URI: str = "postgresql://postgres:postgres@orion-sql-db:5432/conjourney"
    ORION_BUS_URL: str = "redis://orion-redis:6379/0"
    ORION_BUS_ENABLED: bool = True

    SERVICE_NAME: str = "orion-collapse-mirror"
    SERVICE_VERSION: str = "0.3.0"
    CHRONICLE_ENVIRONMENT: str = "dev"
    CHRONICLE_MODE: str = "local"

    @field_validator("ORION_BUS_ENABLED", mode="before")
    def parse_bool(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v)

    class Config:
        env_file = ".env"

settings = Settings()

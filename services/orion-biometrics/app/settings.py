from pydantic_settings import BaseSettings
from pydantic import Field

from typing import Optional

class Settings(BaseSettings):
    """
    Loads and validates configuration from environment variables.
    """
    # --- Service Identity ---
    PROJECT: str = Field(..., env="PROJECT")
    SERVICE_NAME: str = Field(..., env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(..., env="SERVICE_VERSION")
    NET: str = Field(..., env="NET")

    # --- Database & Bus ---
    POSTGRES_URI: str = Field(..., env="POSTGRES_URI")
    TABLE_NAME: str = Field(..., env="TABLE_NAME")
    ORION_BUS_URL: str = Field(..., env="ORION_BUS_URL")
    TELEMETRY_PUBLISH_CHANNEL: str = Field(..., env="TELEMETRY_PUBLISH_CHANNEL")
    EXTERNAL_SUBSCRIBE_CHANNEL: Optional[str] = Field(default=None, env="EXTERNAL_SUBSCRIBE_CHANNEL")

    # --- Runtime ---
    TELEMETRY_INTERVAL: int = Field(..., env="TELEMETRY_INTERVAL")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        # This allows pydantic to read from .env files if needed,
        # but docker-compose will inject them directly.
        env_file = ".env"
        extra = "ignore"

settings = Settings()

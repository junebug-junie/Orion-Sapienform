from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Configuration for the Orion Biometrics service.
    """
    # --- Service Identity ---
    SERVICE_NAME: str = Field(..., env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(..., env="SERVICE_VERSION")
    PORT: int = Field(..., env="PORT")

    # --- Database & Bus ---
    POSTGRES_URI: str = Field(..., env="POSTGRES_URI")
    ORION_BUS_URL: str = Field(..., env="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(..., env="ORION_BUS_ENABLED")
    PUBLISH_CHANNEL_BIOMETRICS_NEW: str = Field(..., env="PUBLISH_CHANNEL_BIOMETRICS_NEW")

    # --- Database Table Name ---
    TABLE_NAME: str = Field(default="biometric_records", env="TABLE_NAME")

    # --- Runtime ---
    STARTUP_DELAY: int = Field(default=5, env="STARTUP_DELAY")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()


# services/orion-collapse-mirror/app/settings.py

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

class Settings(BaseSettings):
    """
    Orion Collapse Mirror settings.
    No DB or Chroma dependencies — event-driven only.
    """

    # === Core Identity ===
    SERVICE_NAME: str = Field(default="collapse-mirror", env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.4.0", env="SERVICE_VERSION")
    PORT: int = Field(default=8087, env="PORT")

    # === Bus Configuration ===
    ORION_BUS_URL: str = Field(default="redis://orion-bus-core:6379/0", env="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")

    # === Exec-step request prefix (so this service can act as a Cortex step target) ===
    EXEC_REQUEST_PREFIX: str = Field(default="orion-exec:request", env="EXEC_REQUEST_PREFIX")

    # === Bus Channels ===
    CHANNEL_COLLAPSE_INTAKE: str = Field(default="orion:collapse:intake", env="CHANNEL_COLLAPSE_INTAKE")
    CHANNEL_COLLAPSE_TRIAGE: str = Field(default="orion:collapse:triage", env="CHANNEL_COLLAPSE_TRIAGE")

    # === Environment ===
    CHRONICLE_ENVIRONMENT: str = Field(default="dev", env="CHRONICLE_ENVIRONMENT")
    CHRONICLE_MODE: str = Field(default="local", env="CHRONICLE_MODE")
    DEV_MODE: bool = Field(default=True, env="DEV_MODE")

    @field_validator("ORION_BUS_ENABLED", mode="before")
    @classmethod
    def parse_bool(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

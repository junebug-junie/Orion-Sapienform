from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    SERVICE_NAME: str = Field(default="orion-bus-tap")
    SERVICE_VERSION: str = Field(default="0.1.0")
    NODE_NAME: str = Field(default="unknown")
    INSTANCE_ID: str = Field(default="local")

    ORION_BUS_URL: str = Field(default="redis://localhost:6379/0")
    TAP_PATTERN: str = Field(default="orion:*")

    UI_BIND_HOST: str = Field(default="0.0.0.0")
    UI_BIND_PORT: int = Field(default=8101)


settings = Settings()

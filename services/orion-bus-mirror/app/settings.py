from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ORION_BUS_URL: str = Field(default="redis://localhost:6379/0")
    MIRROR_PATTERN: str = Field(default="orion:*")
    MIRROR_SQLITE_PATH: str = Field(default="/data/bus_mirror.sqlite")
    MIRROR_PARQUET_DIR: str = Field(default="/data/parquet")


settings = Settings()

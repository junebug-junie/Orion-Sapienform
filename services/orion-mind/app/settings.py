from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository layout: services/orion-mind/app/settings.py → service root is parents[1]
_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _SERVICE_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.is_file() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    SERVICE_NAME: str = Field(default="orion-mind", alias="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.1.0", alias="SERVICE_VERSION")
    NODE_NAME: str = Field(default="unknown", alias="NODE_NAME")
    LOG_LEVEL: str = Field(default="INFO", alias="LOG_LEVEL")

    PORT: int = Field(default=6611, alias="PORT")
    MIND_SNAPSHOT_MAX_BYTES: int = Field(default=512_000, alias="MIND_SNAPSHOT_MAX_BYTES")
    MIND_WALL_MS_DEFAULT: int = Field(default=120_000, alias="MIND_WALL_MS_DEFAULT")
    MIND_N_LOOPS_DEFAULT: int = Field(default=1, alias="MIND_N_LOOPS_DEFAULT")
    MIND_ROUTER_PROFILES_PATH: str = Field(
        default="",
        alias="MIND_ROUTER_PROFILES_PATH",
        description="Directory containing router_profiles.yaml; empty = app/config beside settings",
    )

    @property
    def router_profiles_dir(self) -> Path:
        if (self.MIND_ROUTER_PROFILES_PATH or "").strip():
            return Path(self.MIND_ROUTER_PROFILES_PATH)
        return Path(__file__).resolve().parent / "config"


settings = Settings()

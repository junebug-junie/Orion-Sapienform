from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    SERVICE_NAME: str = Field(default="orion-pageindex")
    SERVICE_VERSION: str = Field(default="0.1.0")
    NODE_NAME: str = Field(default="unknown")
    PORT: int = Field(default=8360)

    # Bus-native SystemHealthV1 heartbeat (orion:system:health). PageIndex had no bus
    # connection at all before this -- these fields exist solely to carry an independent
    # heartbeat chassis. See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    ORION_BUS_URL: str = Field(default="redis://100.92.216.81:6379/0")
    ORION_BUS_ENABLED: bool = Field(default=True)
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0)

    PAGEINDEX_IMPL: str = Field(default="actual")
    PAGEINDEX_INSTALLATION_MODE: str = Field(default="cli")
    PAGEINDEX_REPO_PATH: str = Field(default="/opt/PageIndex")
    PAGEINDEX_RUN_SCRIPT: str = Field(default="run_pageindex.py")
    PAGEINDEX_PYTHON_BIN: str = Field(default="python3")
    PAGEINDEX_BUILD_ARGS: str = Field(default="--md_path {md_path}")
    PAGEINDEX_QUERY_ARGS: str = Field(default="--md_path {md_path}")
    PAGEINDEX_TIMEOUT_SEC: int = Field(default=120)
    PAGEINDEX_ALLOW_EMPTY_REBUILD: bool = Field(default=False)

    JOURNAL_PG_DSN: str = Field(
        default="postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
    )
    PAGEINDEX_SQL_DATABASE_URL: Optional[str] = Field(default=None)
    # Denormalized journal retrieval table consumed by pageindex corpus export.
    JOURNAL_INDEX_TABLE: str = Field(default="journal_entry_index")

    PAGEINDEX_DATA_DIR: str = Field(default="/data/pageindex")
    CHAT_EPISODES_MARKDOWN_PATH: str = Field(default="/data/pageindex/chat_episodes/chat_episode_corpus.md")


settings = Settings()

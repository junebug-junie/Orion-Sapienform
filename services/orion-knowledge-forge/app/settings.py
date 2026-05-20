from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from orion.knowledge_forge.paths import resolve_corpus_root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = Field("orion-knowledge-forge", alias="SERVICE_NAME")
    service_version: str = Field("1.0.0", alias="SERVICE_VERSION")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    port: int = Field(8630, alias="PORT")

    knowledge_forge_enabled: bool = Field(True, alias="KNOWLEDGE_FORGE_ENABLED")
    knowledge_forge_repo_root: Path | None = Field(None, alias="KNOWLEDGE_FORGE_REPO_ROOT")
    knowledge_forge_write_enabled: bool = Field(False, alias="KNOWLEDGE_FORGE_WRITE_ENABLED")
    knowledge_forge_operator_token: str | None = Field(None, alias="KNOWLEDGE_FORGE_OPERATOR_TOKEN")
    knowledge_forge_max_search_results: int = Field(50, alias="KNOWLEDGE_FORGE_MAX_SEARCH_RESULTS")

    @model_validator(mode="after")
    def resolve_repo_root(self) -> Settings:
        if self.knowledge_forge_repo_root is None:
            env_root = os.environ.get("ORION_KNOWLEDGE_ROOT")
            if env_root:
                self.knowledge_forge_repo_root = Path(env_root).expanduser().resolve()
            else:
                self.knowledge_forge_repo_root = resolve_corpus_root()
        else:
            self.knowledge_forge_repo_root = Path(self.knowledge_forge_repo_root).expanduser().resolve()
        return self


settings = Settings()

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    MIND_EVIDENCE_MAX_CHARS: int = Field(default=12_000, alias="MIND_EVIDENCE_MAX_CHARS")
    MIND_EVIDENCE_MAX_MESSAGES: int = Field(default=8, alias="MIND_EVIDENCE_MAX_MESSAGES")
    MIND_EVIDENCE_MAX_RECALL_FRAGMENTS: int = Field(default=8, alias="MIND_EVIDENCE_MAX_RECALL_FRAGMENTS")
    MIND_EVIDENCE_MAX_PROJECTION_ITEMS: int = Field(default=16, alias="MIND_EVIDENCE_MAX_PROJECTION_ITEMS")

    MIND_LLM_SYNTHESIS_ENABLED: bool = Field(default=False, alias="MIND_LLM_SYNTHESIS_ENABLED")
    MIND_SEMANTIC_MODEL_ROUTE: str = Field(default="quick", alias="MIND_SEMANTIC_MODEL_ROUTE")
    MIND_APPRAISAL_MODEL_ROUTE: str = Field(default="metacog", alias="MIND_APPRAISAL_MODEL_ROUTE")
    MIND_STANCE_MODEL_ROUTE: str = Field(default="chat", alias="MIND_STANCE_MODEL_ROUTE")
    MIND_LLM_TIMEOUT_SEC: float = Field(default=90.0, alias="MIND_LLM_TIMEOUT_SEC")
    MIND_LLM_MAX_TOKENS_SEMANTIC: int = Field(default=2048, alias="MIND_LLM_MAX_TOKENS_SEMANTIC")
    MIND_LLM_MAX_TOKENS_APPRAISAL: int = Field(default=3072, alias="MIND_LLM_MAX_TOKENS_APPRAISAL")
    MIND_LLM_MAX_TOKENS_STANCE: int = Field(default=1536, alias="MIND_LLM_MAX_TOKENS_STANCE")
    MIND_LLM_THINKING_APPRAISAL: bool = Field(default=True, alias="MIND_LLM_THINKING_APPRAISAL")
    MIND_LLM_FAIL_OPEN_LEGACY: bool = Field(default=True, alias="MIND_LLM_FAIL_OPEN_LEGACY")

    MIND_LLM_USE_BUS: bool = Field(default=True, alias="MIND_LLM_USE_BUS")
    ORION_BUS_ENABLED: bool = Field(default=True, alias="ORION_BUS_ENABLED")
    ORION_BUS_URL: str = Field(default="redis://redis:6379/0", alias="ORION_BUS_URL")
    MIND_LLM_INTAKE_CHANNEL: str = Field(
        default="orion:exec:request:LLMGatewayService",
        alias="MIND_LLM_INTAKE_CHANNEL",
    )
    MIND_LLM_REPLY_PREFIX: str = Field(default="orion:mind:llm:reply", alias="MIND_LLM_REPLY_PREFIX")

    @property
    def router_profiles_dir(self) -> Path:
        if (self.MIND_ROUTER_PROFILES_PATH or "").strip():
            return Path(self.MIND_ROUTER_PROFILES_PATH)
        return Path(__file__).resolve().parent / "config"


settings = Settings()

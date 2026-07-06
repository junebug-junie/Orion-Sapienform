from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-proposal-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    proposal_policy_path: str = Field(
        "config/proposals/proposal_policy.v1.yaml",
        alias="PROPOSAL_POLICY_PATH",
    )
    proposal_poll_interval_sec: float = Field(2.0, alias="PROPOSAL_POLL_INTERVAL_SEC")
    enable_proposal_runtime: bool = Field(True, alias="ENABLE_PROPOSAL_RUNTIME")
    enable_transport_proposals: bool = Field(False, alias="ENABLE_TRANSPORT_PROPOSALS")
    transport_proposal_mode: str = Field("read_only", alias="TRANSPORT_PROPOSAL_MODE")
    # Phase B: incorporate spontaneous-thought (reverie) proposals into Layer 7.
    # Default-off; candidates are operator_review-gated (never auto-dispatch).
    reverie_propose_enabled: bool = Field(False, alias="ORION_REVERIE_PROPOSE_ENABLED")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

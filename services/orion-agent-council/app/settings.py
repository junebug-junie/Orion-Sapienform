# services/orion-agent-council/app/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # --- Service identity ---
    service_name: str = Field("agent-council", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")

    # --- Orion bus ---
    orion_bus_url: str = Field("redis://orion-redis:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")

    # --- Council channels (bus) ---
    # Intake = where others send council_deliberation requests
    channel_intake: str = Field("orion:council:intake", alias="CHANNEL_COUNCIL_INTAKE")
    # Reply prefix = we publish replies to `${prefix}:${trace_id}` if caller didn’t give a channel
    channel_reply_prefix: str = Field(
        "orion:council:reply",
        alias="CHANNEL_COUNCIL_REPLY_PREFIX",
    )

    # --- LLM Gateway routing (bus) ---
    llm_intake_channel: str = Field(
        "orion:llm-gateway:intake",
        alias="CHANNEL_LLM_GATEWAY_INTAKE",
    )
    llm_reply_prefix: str = Field(
        "orion:llm-gateway:reply",
        alias="CHANNEL_LLM_GATEWAY_REPLY_PREFIX",
    )
    llm_service_name: str = Field(
        "LLMGateway",
        alias="LLM_GATEWAY_SERVICE_NAME",
    )

    council_llm_timeout_sec: float = Field(30.0, alias="COUNCIL_LLM_TIMEOUT_SEC")

    # --- Deliberation loop config ---
    max_rounds: int = Field(2, alias="COUNCIL_MAX_ROUNDS")

    # Thresholds for auditor decisions
    disagreement_threshold: float = Field(0.7, alias="COUNCIL_DISAGREEMENT_THRESHOLD")
    risk_threshold: float = Field(0.7, alias="COUNCIL_RISK_THRESHOLD")
    min_coherence: float = Field(0.5, alias="COUNCIL_MIN_COHERENCE")
    min_faithfulness: float = Field(0.6, alias="COUNCIL_MIN_FAITHFULNESS")

    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
    )


settings = Settings()

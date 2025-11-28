# services/orion-llm-gateway/app/settings.py

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # Service identity
    service_name: str = Field("llm-gateway", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")

    # Bus config
    orion_bus_url: str = Field("redis://orion-redis:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")

    # Intake from other services
    channel_llm_intake: str = Field(
        "orion-exec:request:LLMGatewayService", alias="CHANNEL_LLM_INTAKE"
    )
    # Where callers expect replies (prefix)
    channel_llm_reply_prefix: str = Field(
        "orion:llm:reply", alias="CHANNEL_LLM_REPLY_PREFIX"
    )

    # Polling / timing
    poll_timeout: float = Field(1.0, alias="POLL_TIMEOUT")

    # Default model
    default_model: str = Field("llama3.1:8b-instruct-q8_0", alias="ORION_DEFAULT_LLM_MODEL")

    # Default backend name if caller doesn't specify options["backend"]
    # Supported: "ollama", "vllm", "brain"
    default_backend: str = Field("ollama", alias="ORION_LLM_DEFAULT_BACKEND")

    # Backend endpoints (can be None if not used yet)
    # 1) Ollama model server (current brain-llm)
    ollama_url: str = Field("http://llm-brain:11434", alias="ORION_LLM_OLLAMA_URL")

    # 2) vLLM server (future; usually OpenAI-compatible)
    vllm_url: Optional[str] = Field(None, alias="ORION_LLM_VLLM_URL")

    # 3) Brain HTTP /chat endpoint (optional fallback / special path)
    brain_url: Optional[str] = Field(None, alias="ORION_LLM_BRAIN_URL")

    # Timeout knobs (shared across backends)
    connect_timeout_sec: float = Field(10.0, alias="CONNECT_TIMEOUT_SEC")
    read_timeout_sec: float = Field(60.0, alias="READ_TIMEOUT_SEC")

    # Service label used in reply messages
    llm_service_name: str = Field("LLMGatewayService", alias="ORION_LLM_SERVICE_NAME")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field

from .profiles import LLMProfileRegistry, LLMProfile
import yaml


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
    CHANNEL_SPARK_INTROSPECT_CANDIDATE: str = Field(
        "orion:spark:introspect:candidate",
        env="CHANNEL_SPARK_INTROSPECT_CANDIDATE",
    )

    # Polling / timing
    poll_timeout: float = Field(1.0, alias="POLL_TIMEOUT")

    # Default backend name if caller doesn't specify options["backend"]
    default_backend: str = Field("vllm", alias="ORION_LLM_DEFAULT_BACKEND")

    # 🔹 NEW: default model name for when we have no profile + no body.model
    # For llama.cpp this is mostly ignored, but required by OpenAI spec.
    default_model: str = Field(
        "Active-GGUF-Model",
        alias="ORION_DEFAULT_LLM_MODEL",
        description="Fallback model name when no profile/model is provided.",
    )

    # Backend endpoints
    vllm_url: Optional[str] = Field(None, alias="ORION_LLM_VLLM_URL")
    llamacpp_url: Optional[str] = Field(None, alias="ORION_LLM_LLAMACPP_URL")

    # Timeout knobs (shared across backends)
    connect_timeout_sec: float = Field(10.0, alias="CONNECT_TIMEOUT_SEC")
    read_timeout_sec: float = Field(60.0, alias="READ_TIMEOUT_SEC")

    # Service label used in reply messages
    llm_service_name: str = Field(
        "LLMGatewayService", alias="ORION_LLM_SERVICE_NAME"
    )

    # Profiles config
    llm_profiles_config_path: Optional[Path] = Field(
        None,
        alias="LLM_PROFILES_CONFIG_PATH",
        description="Path to YAML defining LLM profiles (optional)",
    )
    llm_default_profile_name: Optional[str] = Field(
        None,
        alias="LLM_DEFAULT_PROFILE_NAME",
        description="Default profile name when profiles are enabled",
    )

    class Config:
        env_file = ".env"
        extra = "ignore"

    def load_profile_registry(self) -> LLMProfileRegistry:
        """
        Load profiles from YAML if configured. If not configured,
        returns an empty registry and the gateway falls back to
        default_model/default_backend.
        """
        if not self.llm_profiles_config_path:
            return LLMProfileRegistry(profiles={})

        try:
            with self.llm_profiles_config_path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            return LLMProfileRegistry(profiles={})

        profiles_dict = raw.get("profiles", {}) or {}
        parsed = {
            name: LLMProfile(name=name, **data)
            for name, data in profiles_dict.items()
        }
        return LLMProfileRegistry(profiles=parsed)


settings = Settings()

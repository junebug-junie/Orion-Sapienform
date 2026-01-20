from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field, model_validator, AliasChoices
from pydantic_settings import BaseSettings

from .profiles import LLMProfile, LLMProfileRegistry


class Settings(BaseSettings):
    # Service identity
    service_name: str = Field("llm-gateway", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: Optional[str] = Field(None, alias="NODE_NAME")

    # Bus config
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    # Intake from other services
    channel_llm_intake: str = Field("orion:exec:request:LLMGatewayService", alias="CHANNEL_LLM_INTAKE")
    channel_vector_latent_upsert: str = Field(
        "orion:vector:latent:upsert",
        alias="CHANNEL_VECTOR_LATENT_UPSERT",
    )
    channel_embedding_generate: str = Field(
        "orion:embedding:generate",
        alias="CHANNEL_EMBEDDING_GENERATE",
    )

    # Spark
    channel_spark_introspect_candidate: str = Field(
        "orion:spark:introspect:candidate",
        alias="CHANNEL_SPARK_INTROSPECT_CANDIDATE",
    )

    # Backend routing defaults
    default_backend: str = Field("vllm", alias="ORION_LLM_DEFAULT_BACKEND")
    default_model: str = Field("Active-GGUF-Model", alias="ORION_DEFAULT_LLM_MODEL")

    # Backend endpoints
    vllm_url: Optional[str] = Field(None, alias="ORION_LLM_VLLM_URL")
    ollama_url: Optional[str] = Field(None, alias="ORION_LLM_OLLAMA_URL")
    ollama_use_openai_compat: bool = Field(False, alias="ORION_LLM_OLLAMA_USE_OPENAI")
    llamacpp_url: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("ORION_LLM_LLAMACPP_URL", "ORION_LLM_LLAMA_CPP_URL"),
    )
    llama_cola_url: Optional[str] = Field(None, alias="ORION_LLM_LLAMA_COLA_URL")

    # If false, the gateway will NOT attempt a secondary embedding call.
    # If true, and the backend response did not already include an embedding/vector,
    # the gateway will try to fetch one from the embedding URLs (cola/llamacpp/vllm/ollama).
    include_embeddings: bool = Field(False, alias="ORION_LLM_INCLUDE_EMBEDDINGS")

    # Vector collections
    orion_vector_latent_collection: str = Field(
        "orion_latent_store",
        alias="ORION_VECTOR_LATENT_COLLECTION",
    )

    # Timeout knobs (shared across backends)
    connect_timeout_sec: float = Field(10.0, alias="CONNECT_TIMEOUT_SEC")
    read_timeout_sec: float = Field(60.0, alias="READ_TIMEOUT_SEC")

    # Profiles config
    llm_profiles_config_path: Optional[Path] = Field(None, alias="LLM_PROFILES_CONFIG_PATH")
    llm_default_profile_name: Optional[str] = Field(None, alias="LLM_DEFAULT_PROFILE_NAME")

    # Route table (single-subscriber routing)
    llm_route_table_json: Optional[str] = Field(None, alias="LLM_GATEWAY_ROUTE_TABLE_JSON")
    llm_route_default: str = Field("chat", alias="LLM_ROUTE_DEFAULT")
    llm_route_chat_url: Optional[str] = Field(None, alias="LLM_ROUTE_CHAT_URL")
    llm_route_metacog_url: Optional[str] = Field(None, alias="LLM_ROUTE_METACOG_URL")
    llm_route_latents_url: Optional[str] = Field(None, alias="LLM_ROUTE_LATENTS_URL")
    llm_route_specialist_url: Optional[str] = Field(None, alias="LLM_ROUTE_SPECIALIST_URL")
    llm_route_chat_served_by: Optional[str] = Field(None, alias="LLM_ROUTE_CHAT_SERVED_BY")
    llm_route_metacog_served_by: Optional[str] = Field(None, alias="LLM_ROUTE_METACOG_SERVED_BY")
    llm_route_latents_served_by: Optional[str] = Field(None, alias="LLM_ROUTE_LATENTS_SERVED_BY")
    llm_route_specialist_served_by: Optional[str] = Field(None, alias="LLM_ROUTE_SPECIALIST_SERVED_BY")
    llm_route_health_timeout_sec: float = Field(1.5, alias="LLM_ROUTE_HEALTH_TIMEOUT_SEC")
    llm_gateway_health_port: int = Field(8210, alias="LLM_GATEWAY_HEALTH_PORT")

    class Config:
        env_file = ".env"
        extra = "ignore"

    @model_validator(mode='after')
    def enforce_no_embeddings(self) -> "Settings":
        if self.include_embeddings:
            self.include_embeddings = False
        return self

    def load_profile_registry(self) -> LLMProfileRegistry:
        if not self.llm_profiles_config_path:
            return LLMProfileRegistry(profiles={})

        try:
            with self.llm_profiles_config_path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            return LLMProfileRegistry(profiles={})

        profiles_dict = raw.get("profiles", {}) or {}
        parsed = {name: LLMProfile(name=name, **data) for name, data in profiles_dict.items()}
        return LLMProfileRegistry(profiles=parsed)


settings = Settings()

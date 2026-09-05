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
    orion_vector_host_url: Optional[str] = Field(
        "http://orion-athena-vector-host:8320",
        alias="ORION_VECTOR_HOST_URL",
    )

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
    llm_logprob_summary_enabled: bool = Field(False, alias="LLM_LOGPROB_SUMMARY_ENABLED")
    llm_logprob_top_k_default: int = Field(5, alias="LLM_LOGPROB_TOP_K_DEFAULT")
    llm_logprob_low_margin_threshold: float = Field(0.5, alias="LLM_LOGPROB_LOW_MARGIN_THRESHOLD")
    llm_logprob_low_logprob_threshold: float = Field(-2.0, alias="LLM_LOGPROB_LOW_LOGPROB_THRESHOLD")
    llm_logprob_unstable_span_min_len: int = Field(3, alias="LLM_LOGPROB_UNSTABLE_SPAN_MIN_LEN")
    llm_logprob_native_completion_enabled: bool = Field(
        False, alias="LLM_LOGPROB_NATIVE_COMPLETION_ENABLED"
    )
    llm_logprob_native_completion_max_tokens: int = Field(
        256, alias="LLM_LOGPROB_NATIVE_COMPLETION_MAX_TOKENS"
    )

    # Profiles config
    llm_profiles_config_path: Optional[Path] = Field(None, alias="LLM_PROFILES_CONFIG_PATH")
    llm_default_profile_name: Optional[str] = Field(None, alias="LLM_DEFAULT_PROFILE_NAME")

    # Route table (single-subscriber routing)
    llm_route_table_json: Optional[str] = Field(None, alias="LLM_GATEWAY_ROUTE_TABLE_JSON")
    llm_route_default: str = Field("chat", alias="LLM_ROUTE_DEFAULT")

    # Background-priority routes (RouteTarget.priority == "background"): wait
    # for upstream /slots slack before dispatch instead of competing evenly
    # with foreground traffic on the same llama.cpp process. See
    # priority_admission.py and README.md's "Background-priority routes".
    llm_gateway_background_max_wait_sec: float = Field(
        30.0, alias="LLM_GATEWAY_BACKGROUND_MAX_WAIT_SEC"
    )
    llm_gateway_background_poll_interval_sec: float = Field(
        0.5, alias="LLM_GATEWAY_BACKGROUND_POLL_INTERVAL_SEC"
    )
    llm_gateway_background_concurrency: int = Field(
        1, alias="LLM_GATEWAY_BACKGROUND_CONCURRENCY"
    )
    # Per-upstream in-flight cap on the bus chat path (upstream_admission.py). Each
    # distinct route-table URL may hold at most this many executor threads at once;
    # the executor is sized to (distinct upstreams x this) + headroom, so a flood on
    # one lane cannot take the threads another lane's requests need. A request that
    # cannot get its lane's permit inside its own read-timeout budget is shed with
    # `gateway_overloaded` instead of being generated for a caller that already
    # gave up. Incident: 2026-09-05 stance_react starvation.
    llm_gateway_upstream_max_inflight: int = Field(
        8, alias="LLM_GATEWAY_UPSTREAM_MAX_INFLIGHT"
    )
    llm_route_chat_url: Optional[str] = Field(None, alias="LLM_ROUTE_CHAT_URL")
    llm_route_metacog_url: Optional[str] = Field(None, alias="LLM_ROUTE_METACOG_URL")
    llm_route_latents_url: Optional[str] = Field(None, alias="LLM_ROUTE_LATENTS_URL")
    llm_route_specialist_url: Optional[str] = Field(None, alias="LLM_ROUTE_SPECIALIST_URL")
    llm_route_chat_served_by: Optional[str] = Field(None, alias="LLM_ROUTE_CHAT_SERVED_BY")
    # Phase 3: optional lane labels used to match route-table entries by served_by
    llm_route_spark_served_by: Optional[str] = Field(None, alias="LLM_ROUTE_SPARK_SERVED_BY")
    llm_route_background_served_by: Optional[str] = Field(None, alias="LLM_ROUTE_BACKGROUND_SERVED_BY")
    llm_route_agent_served_by: Optional[str] = Field(None, alias="LLM_ROUTE_AGENT_SERVED_BY")
    llm_allow_background_to_chat_fallback: bool = Field(False, alias="LLM_ALLOW_BACKGROUND_TO_CHAT_FALLBACK")
    llm_lane_default: str = Field("chat", alias="LLM_LANE_DEFAULT")
    llm_lane_routing_enabled: bool = Field(False, alias="LLM_LANE_ROUTING_ENABLED")
    llm_route_metacog_served_by: Optional[str] = Field(None, alias="LLM_ROUTE_METACOG_SERVED_BY")
    llm_route_latents_served_by: Optional[str] = Field(None, alias="LLM_ROUTE_LATENTS_SERVED_BY")
    llm_route_specialist_served_by: Optional[str] = Field(None, alias="LLM_ROUTE_SPECIALIST_SERVED_BY")
    llm_route_health_timeout_sec: float = Field(1.5, alias="LLM_ROUTE_HEALTH_TIMEOUT_SEC")
    llm_gateway_health_port: int = Field(8210, alias="LLM_GATEWAY_HEALTH_PORT")
    llm_gateway_concurrent_handlers: bool = Field(True, alias="LLM_GATEWAY_CONCURRENT_HANDLERS")
    llm_gateway_anthropic_passthrough_enabled: bool = Field(
        True, alias="LLM_GATEWAY_ANTHROPIC_PASSTHROUGH_ENABLED"
    )
    llm_gateway_anthropic_passthrough_timeout_sec: float = Field(
        900.0, alias="LLM_GATEWAY_ANTHROPIC_PASSTHROUGH_TIMEOUT_SEC"
    )
    llm_gateway_openai_passthrough_enabled: bool = Field(
        True, alias="LLM_GATEWAY_OPENAI_PASSTHROUGH_ENABLED"
    )
    atlas_metacog_service_name: str = Field("atlas-worker-2", alias="ATLAS_METACOG_SERVICE_NAME")
    atlas_metacog_profile_name: Optional[str] = Field(None, alias="ATLAS_METACOG_PROFILE_NAME")
    atlas_metacog_cuda_visible_devices: Optional[str] = Field(None, alias="ATLAS_METACOG_CUDA_VISIBLE_DEVICES")
    atlas_metacog_host_port: int = Field(8012, alias="ATLAS_METACOG_HOST_PORT")

    # Default structured-output method when options.structured_output_method is auto/unset
    llm_structured_output_method: str = Field(
        default="none",
        alias="LLM_STRUCTURED_OUTPUT_METHOD",
        description=(
            "Gateway default for llama.cpp response_format builder: json_object_schema, "
            "json_schema_schema, json_object_only, none, or auto."
        ),
    )

    # --- Vision / chat attachments ---
    # Capability is read from the worker's own /props, never from the profile
    # registry's supports_vision claim -- see app/vision.py for why.
    llm_gateway_vision_enabled: bool = Field(
        default=True,
        alias="LLM_GATEWAY_VISION_ENABLED",
        description="Master switch. When false, every route is treated as blind and images are refused.",
    )
    llm_gateway_vision_props_cache_ttl_sec: float = Field(
        default=300.0,
        alias="LLM_GATEWAY_VISION_PROPS_CACHE_TTL_SEC",
        description="How long a worker's /props modality report is trusted before re-probing.",
    )
    llm_gateway_vision_props_timeout_sec: float = Field(
        default=5.0,
        alias="LLM_GATEWAY_VISION_PROPS_TIMEOUT_SEC",
    )
    llm_gateway_attachment_base_url: str = Field(
        default="",
        alias="LLM_GATEWAY_ATTACHMENT_BASE_URL",
        description=(
            "Trusted base the gateway builds attachment fetch URLs from, as "
            "'<base>/<sha256>'. The ref's own source_url is client-controlled and is "
            "deliberately NOT used. Empty means refuse to fetch attachments at all."
        ),
    )
    llm_gateway_percept_base_url: str = Field(
        default="",
        alias="LLM_GATEWAY_PERCEPT_BASE_URL",
        description=(
            "Trusted base for attachments with kind='percept' -- camera frames, "
            "served by orion-percept-store. Separate from the chat attachment base "
            "on purpose: percepts are frames of a private home with their own short "
            "retention, and the chat store is served by Hub, which also holds the "
            "docker socket. Empty means refuse to fetch percepts; it must NOT fall "
            "back to the chat base."
        ),
    )
    llm_gateway_attachment_allowed_hosts: str = Field(
        default="",
        alias="LLM_GATEWAY_ATTACHMENT_ALLOWED_HOSTS",
        description=(
            "Comma-separated hostnames the derived fetch URL may resolve to. "
            "Defence in depth over the base URL above, not the primary control. "
            "Empty means refuse everything."
        ),
    )
    llm_gateway_attachment_max_bytes: int = Field(
        default=8_388_608,
        alias="LLM_GATEWAY_ATTACHMENT_MAX_BYTES",
    )
    llm_gateway_attachment_fetch_timeout_sec: float = Field(
        default=10.0,
        alias="LLM_GATEWAY_ATTACHMENT_FETCH_TIMEOUT_SEC",
    )

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

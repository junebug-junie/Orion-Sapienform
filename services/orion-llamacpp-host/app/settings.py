# services/orion-llamacpp-host/app/settings.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices

from .profiles import LLMProfileRegistry, LLMProfile

logger = logging.getLogger("orion-llamacpp-host.settings")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,   # ✅ empty strings become "unset" instead of parsing errors
    )

    # Identity
    service_name: str = Field("orion-llamacpp-host", env="SERVICE_NAME")
    service_version: str = Field("0.1.0", env="SERVICE_VERSION")

    # Profiles
    llm_profiles_config_path: Path = Field(
        Path("/app/config/llm_profiles.yaml"),
        env="LLM_PROFILES_CONFIG_PATH",
    )
    llm_profile_name: str = Field(..., env="LLM_PROFILE_NAME")

    # Token (optional; only needed for gated/private repos)
    hf_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "hf_token"),
    )

    # --- Optional hard overrides (take precedence over profile.llamacpp) ---
    llamacpp_model_path_override: Optional[str] = Field(None, env="LLAMACPP_MODEL_PATH_OVERRIDE")

    # runtime knob overrides (optional)
    llamacpp_host_override: Optional[str] = Field(None, env="LLAMACPP_HOST_OVERRIDE")
    llamacpp_port_override: Optional[int] = Field(None, env="LLAMACPP_PORT_OVERRIDE")
    llamacpp_ctx_size_override: Optional[int] = Field(None, env="LLAMACPP_CTX_SIZE_OVERRIDE")
    llamacpp_n_gpu_layers_override: Optional[int] = Field(None, env="LLAMACPP_N_GPU_LAYERS_OVERRIDE")
    llamacpp_threads_override: Optional[int] = Field(None, env="LLAMACPP_THREADS_OVERRIDE")
    llamacpp_n_parallel_override: Optional[int] = Field(None, env="LLAMACPP_N_PARALLEL_OVERRIDE")
    llamacpp_batch_size_override: Optional[int] = Field(None, env="LLAMACPP_BATCH_SIZE_OVERRIDE")

    # CUDA override (optional; otherwise profile.gpu.device_ids is used)
    cuda_visible_devices_override: Optional[str] = Field(None, env="CUDA_VISIBLE_DEVICES")

    # Embedding Lobe Configuration
    embedding_model_path: Optional[str] = Field(None, env="EMBEDDING_MODEL_PATH")
    embedding_host: str = Field("0.0.0.0", env="EMBEDDING_HOST")
    embedding_port: int = Field(8001, env="EMBEDDING_PORT")
    embedding_ctx_size: int = Field(2048, env="EMBEDDING_CTX_SIZE")
    embedding_n_gpu_layers: int = Field(99, env="EMBEDDING_N_GPU_LAYERS")

    _registry: Optional[LLMProfileRegistry] = None

    def load_profile_registry(self) -> LLMProfileRegistry:
        if self._registry is not None:
            return self._registry

        path = self.llm_profiles_config_path
        if not path.exists():
            raise FileNotFoundError(f"llm_profiles.yaml not found at {path}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        profiles_raw = raw.get("profiles") if isinstance(raw, dict) else None
        if not isinstance(profiles_raw, dict):
            raise ValueError("llm_profiles.yaml must be a dict with top-level key: 'profiles:'")

        profiles: Dict[str, LLMProfile] = {}
        for name, cfg in profiles_raw.items():
            if not isinstance(cfg, dict):
                continue
            cfg = dict(cfg)
            cfg.setdefault("name", name)
            profiles[name] = LLMProfile(**cfg)

        self._registry = LLMProfileRegistry(profiles=profiles)
        logger.info("Loaded %d profiles from %s", len(profiles), path)
        return self._registry

    def resolve_profile(self) -> LLMProfile:
        reg = self.load_profile_registry()
        return reg.get(self.llm_profile_name)


settings = Settings()

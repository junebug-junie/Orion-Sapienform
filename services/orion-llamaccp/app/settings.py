# services/orion-llamaccp/app/settings.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

from .profiles import LLMProfileRegistry, LLMProfile

logger = logging.getLogger("orion-llamacpp.settings")


class Settings(BaseSettings):
    # Identity
    service_name: str = Field("orion-llamacpp", env="SERVICE_NAME")
    service_version: str = Field("0.1.0", env="SERVICE_VERSION")

    # HTTP bind
    host: str = Field("0.0.0.0", env="LLAMACPP_HOST")
    port: int = Field(8000, env="LLAMACPP_PORT")

    # Profiles
    llm_profiles_config_path: Path = Field(
        Path("/app/config/llm_profiles.yaml"),
        env="LLM_PROFILES_CONFIG_PATH",
    )
    llm_profile_name: Optional[str] = Field(None, env="LLM_PROFILE_NAME")

    # Model overrides (highest priority first)
    llamacpp_model: Optional[str] = Field(
        None,
        env="LLAMACPP_MODEL",
        description="Optional absolute GGUF path inside container, e.g. /models/foo.gguf",
    )
    llm_model_id: Optional[str] = Field(
        None,
        env="LLM_MODEL_ID",
        description="Optional logical model id or GGUF path",
    )

    # llama.cpp generic defaults (can be refined via profile.gpu)
    ctx_size: int = Field(8192, env="LLAMACPP_CTX_SIZE")
    n_gpu_layers: int = Field(80, env="LLAMACPP_N_GPU_LAYERS")
    threads: int = Field(16, env="LLAMACPP_THREADS")
    n_parallel: int = Field(2, env="LLAMACPP_N_PARALLEL")
    batch_size: int = Field(512, env="LLAMACPP_BATCH_SIZE")

    class Config:
        extra = "ignore"
        env_file = ".env"
        env_file_encoding = "utf-8"

    # --- Runtime cache ---
    _profile_registry: Optional[LLMProfileRegistry] = None

    def load_profile_registry(self) -> LLMProfileRegistry:
        if self._profile_registry is not None:
            return self._profile_registry

        path = self.llm_profiles_config_path
        if not path.exists():
            raise FileNotFoundError(f"llm_profiles.yaml not found at {path}")

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        profiles: Dict[str, LLMProfile] = {}
        for name, cfg in raw.items():
            if not isinstance(cfg, dict):
                logger.warning("Skipping non-dict profile entry: %s", name)
                continue
            cfg = dict(cfg)
            cfg.setdefault("name", name)
            try:
                profiles[name] = LLMProfile(**cfg)
            except Exception as e:
                logger.error("Failed to parse LLM profile '%s': %s", name, e, exc_info=True)
                raise

        self._profile_registry = LLMProfileRegistry(profiles=profiles)
        logger.info(
            "Loaded %d LLM profiles from %s",
            len(self._profile_registry.profiles),
            path,
        )
        return self._profile_registry

    def resolve_model_and_gpu(self) -> Tuple[str, Dict[str, Any]]:
        """
        Resolve the concrete model_id (usually GGUF path) and GPU config dict.

        Priority:
          1) env LLAMACPP_MODEL if set (absolute GGUF path)
          2) env LLM_MODEL_ID if set
          3) profile.model_id from LLM_PROFILE_NAME
        """
        registry = self.load_profile_registry()

        profile: Optional[LLMProfile] = None
        if self.llm_profile_name:
            try:
                profile = registry.get(self.llm_profile_name)
            except KeyError as e:
                logger.error("Unknown LLM_PROFILE_NAME '%s'", self.llm_profile_name)
                raise e

        # Model resolution priority
        if self.llamacpp_model:
            model_id = self.llamacpp_model
            logger.info("Using model from LLAMACPP_MODEL env: %s", model_id)
        elif self.llm_model_id:
            model_id = self.llm_model_id
            logger.info("Using model from LLM_MODEL_ID env: %s", model_id)
        elif profile is not None:
            model_id = profile.model_id
            logger.info(
                "Using model from profile '%s': %s",
                self.llm_profile_name,
                model_id,
            )
        else:
            raise RuntimeError(
                "No model configured: set LLAMACPP_MODEL, LLM_MODEL_ID, or LLM_PROFILE_NAME."
            )

        # GPU configuration: start from profile.gpu if present
        gpu_cfg: Dict[str, Any] = {}
        if profile is not None and profile.gpu is not None:
            gpu_cfg = profile.gpu.model_dump()

        # Ensure we have some context length even if profile omits it
        gpu_cfg.setdefault("max_model_len", self.ctx_size)

        return model_id, gpu_cfg


settings = Settings()

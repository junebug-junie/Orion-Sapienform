from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices

from .profiles import LLMProfileRegistry, LLMProfile

logger = logging.getLogger("neural-host.settings")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
    )

    # Identity
    service_name: str = Field("llama-cola-host", env="SERVICE_NAME")
    service_version: str = Field("0.1.0", env="SERVICE_VERSION")

    # Bus / Heartbeat
    bus_url: str = Field("redis://100.92.216.81:6379/0", env="ORION_BUS_URL")
    node_name: str = Field("unknown", env="NODE_NAME")
    instance_id: str = Field("default", env="INSTANCE_ID")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")

    # Profiles
    llm_profiles_config_path: Path = Field(
        Path("/app/config/llm_profiles.yaml"),
        env="LLM_PROFILES_CONFIG_PATH",
    )
    llm_profile_name: str = Field(..., env="LLM_PROFILE_NAME")

    # Token
    hf_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "hf_token"),
    )

    # --- Optional hard overrides (take precedence over profile.llama_cola) ---
    llama_cola_model_path_override: Optional[str] = Field(None, env="LLAMA_COLA_MODEL_PATH_OVERRIDE")
    llama_cola_revision_override: Optional[str] = Field(None, env="LLAMA_COLA_REVISION_OVERRIDE")

    # CUDA override
    cuda_visible_devices_override: Optional[str] = Field(None, env="CUDA_VISIBLE_DEVICES_OVERRIDE")

    # Behavior
    ensure_model_download: bool = Field(True, env="ENSURE_MODEL_DOWNLOAD")
    wait_for_model_seconds: int = Field(0, env="WAIT_FOR_MODEL_SECONDS")

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
            backend = str(cfg.get("backend") or "").strip()
            if backend and backend != "llama-cola":
                logger.info("Skipping non-llama-cola profile '%s' (backend=%s)", name, backend)
                continue
            profiles[name] = LLMProfile(**cfg)

        self._registry = LLMProfileRegistry(profiles=profiles)
        logger.info("Loaded %d profiles from %s", len(profiles), path)
        return self._registry

    def resolve_profile(self) -> LLMProfile:
        reg = self.load_profile_registry()
        return reg.get(self.llm_profile_name)


settings = Settings()

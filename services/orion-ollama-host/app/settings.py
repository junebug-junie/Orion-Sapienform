# services/orion-ollama-host/app/settings.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

from pydantic_settings import BaseSettings
from pydantic import Field
import yaml
import logging

logger = logging.getLogger("orion-ollama-host.settings")

class Settings(BaseSettings):
    service_name: str = Field("orion-ollama-host", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health), published on its own
    # independent bus connection, separate from the ollama server subprocess this service
    # launches. See docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    llm_profiles_config_path: Path = Field(
        default=Path("/app/config/llm_profiles.yaml"),
        alias="LLM_PROFILES_CONFIG_PATH",
    )

    profile_name: Optional[str] = Field(default=None, alias="OLLAMA_PROFILE_NAME")
    model_id: Optional[str] = Field(default=None, alias="OLLAMA_MODEL_ID")

    class Config:
        env_file = ".env"
        extra = "ignore"

    def _load_profiles(self) -> Dict[str, Any]:
        path = self.llm_profiles_config_path
        try:
            if not path.exists():
                return {}
            with path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            return raw.get("profiles", {}) or {}
        except Exception as e:
            logger.warning("Failed to load profiles from %s: %s", path, e)
            return {}

    def resolve_model(self) -> Optional[str]:
        # 1. Env override wins
        if self.model_id:
            return self.model_id

        # 2. Profile lookup
        if self.profile_name:
            profiles = self._load_profiles()
            profile = profiles.get(self.profile_name)
            if profile:
                return profile.get("model_id")

        return None

settings = Settings()

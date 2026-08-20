"""Skeleton settings for orion-diffusion-host (Patch 1).

No field on this class is read by anything that loads a model yet -- see
README.md "Current skeleton status". Shape follows
services/orion-vision-host/app/settings.py's Settings class.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "diffusion-host"
    SERVICE_VERSION: str = "0.1.0"
    NODE_NAME: str = "athena"
    LOG_LEVEL: str = "INFO"

    # Bus. Declared but not consumed in this patch -- README.md.
    ORION_BUS_ENABLED: bool = False
    ORION_BUS_ENFORCE_CATALOG: bool = True
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    HEARTBEAT_INTERVAL_SEC: float = 10.0

    # Model cache dir convention: storage-warm, not vision-host's telemetry
    # convention -- see README.md "Model cache dir convention" for why.
    # Unused in this patch (no model is loaded).
    MODEL_CACHE_DIR: str = "/mnt/storage-warm/models/diffusion"
    HF_HOME: str = "/mnt/storage-warm/models/diffusion/hf"
    TRANSFORMERS_CACHE: str = "/mnt/storage-warm/models/diffusion/hf/transformers"

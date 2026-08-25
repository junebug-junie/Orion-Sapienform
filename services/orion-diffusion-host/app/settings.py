"""Settings for orion-diffusion-host.

Patch 1 (docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md)
shipped this class with no field read by anything that loads a model. This
patch (the "wire a real model" follow-up, PR TBD) adds the DIFFUSION_* block
below and is the first patch where MODEL_CACHE_DIR/HF_HOME/TRANSFORMERS_CACHE
actually get read. Shape follows services/orion-vision-host/app/settings.py's
Settings class.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "diffusion-host"
    SERVICE_VERSION: str = "0.2.0"
    NODE_NAME: str = "circe"
    LOG_LEVEL: str = "INFO"

    # Bus. Declared but not consumed yet -- README.md. Still no chain
    # orchestration in this patch (that's Patch 2); this patch only wires
    # /generate as a plain HTTP endpoint.
    ORION_BUS_ENABLED: bool = False
    ORION_BUS_ENFORCE_CATALOG: bool = True
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    HEARTBEAT_INTERVAL_SEC: float = 10.0

    # Model cache dir convention: storage-warm, not vision-host's telemetry
    # convention -- see README.md "Model cache dir convention" for why.
    MODEL_CACHE_DIR: str = "/mnt/storage-warm/models/diffusion"
    HF_HOME: str = "/mnt/storage-warm/models/diffusion/hf"
    TRANSFORMERS_CACHE: str = "/mnt/storage-warm/models/diffusion/hf/transformers"

    # Model. sdxl-turbo is a distilled, single-step SDXL variant -- picked
    # over a full SDXL checkpoint (25-50 steps) because this card is shared
    # with nothing else but also serves a slow, capacity-gated cadence
    # (design doc §4): a several-second generation is fine, a
    # multi-minute one is not, and there is no batching/queueing here to
    # absorb the difference. Trained at 512x512 with cfg effectively
    # disabled -- DIFFUSION_GUIDANCE_SCALE=0.0 and
    # DIFFUSION_NUM_INFERENCE_STEPS=1 below are not arbitrary defaults, they
    # are this model's documented operating point (non-zero guidance or
    # >4 steps on a turbo checkpoint produces degraded output, not just
    # slower output).
    DIFFUSION_MODEL_ID: str = "stabilityai/sdxl-turbo"
    DIFFUSION_DEVICE: str = "cuda:0"
    DIFFUSION_DTYPE: str = "fp16"
    DIFFUSION_NUM_INFERENCE_STEPS: int = 1
    DIFFUSION_GUIDANCE_SCALE: float = 0.0
    DIFFUSION_DEFAULT_WIDTH: int = 512
    DIFFUSION_DEFAULT_HEIGHT: int = 512
    DIFFUSION_MAX_PROMPT_CHARS: int = 2000

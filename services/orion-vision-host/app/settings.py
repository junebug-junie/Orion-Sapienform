from __future__ import annotations

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "vision-host"
    SERVICE_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    # Bus
    ORION_BUS_ENABLED: bool = True
    ORION_BUS_ENFORCE_CATALOG: bool = False
    ORION_BUS_URL: str = "redis://localhost:6379/0"

    # Channels
    CHANNEL_VISIONHOST_INTAKE: str = "orion:exec:request:VisionHostService"
    CHANNEL_VISIONHOST_REPLY_PREFIX: str = "orion:vision:reply"
    CHANNEL_VISIONHOST_PUB: str = "orion:vision:artifacts"

    # Caches
    MODEL_CACHE_DIR: str = "/mnt/telemetry/models/vision"
    HF_HOME: str = "/mnt/telemetry/models/vision/hf"
    TRANSFORMERS_CACHE: str = "/mnt/telemetry/models/hf/transformers"

    # Profiles
    VISION_PROFILES_PATH: str = "/app/config/vision_profiles.yaml"
    VISION_ENABLED_PROFILES: str = "pipeline_retina_fast,retina_detect_open_vocab,embed_image,vlm_caption"

    # VLM / Captioning
    VISION_VLM_MODEL_ID: str = "HuggingFaceM4/idefics2-8b" # Placeholder default, likely overridden by env
    VISION_VLM_MAX_TOKENS: int = 128
    VISION_VLM_TEMPERATURE: float = 0.4

    # Multi-GPU scheduling
    VISION_DEVICE_STRATEGY: str = Field(default="best_free_vram", description="best_free_vram|fixed|round_robin")
    VISION_DEFAULT_DEVICE: str = "cuda:0"
    VISION_DEVICES: str = "cuda:0"
    VISION_PICK_GPU_METRIC: str = Field(default="free_vram_mb", description="free_vram_mb|free_fraction")

    # Runtime
    VISION_DTYPE: str = Field(default="auto", description="auto|fp16|bf16|fp32")
    VISION_TIMEOUT_S: int = 30

    # Concurrency / queueing
    VISION_MAX_INFLIGHT: int = 4
    VISION_MAX_INFLIGHT_PER_GPU: int = 1
    VISION_QUEUE_WHEN_BUSY: bool = True
    VISION_MAX_QUEUE: int = 200

    # VRAM pressure
    VISION_VRAM_RESERVE_MB: int = 3500
    VISION_VRAM_SOFT_FLOOR_MB: int = 2200
    VISION_VRAM_HARD_FLOOR_MB: int = 1400

    @property
    def enabled_profiles(self) -> List[str]:
        return _split_csv(self.VISION_ENABLED_PROFILES)

    @property
    def devices(self) -> List[str]:
        d = _split_csv(self.VISION_DEVICES)
        return d if d else [self.VISION_DEFAULT_DEVICE]

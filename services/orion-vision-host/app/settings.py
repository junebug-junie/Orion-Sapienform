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
    NODE_NAME: str = "athena"
    LOG_LEVEL: str = "INFO"

    # Bus
    ORION_BUS_ENABLED: bool = True
    ORION_BUS_ENFORCE_CATALOG: bool = False
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health). See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    HEARTBEAT_INTERVAL_SEC: float = 10.0

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

    # VLM / Captioning (default is ~2.7B BLIP-2; override for your VRAM — avoid 8B-class defaults on small GPUs)
    VISION_VLM_MODEL_ID: str = "Salesforce/blip2-opt-2.7b"
    VISION_VLM_MAX_TOKENS: int = 128
    VISION_VLM_TEMPERATURE: float = 0.4

    # Fresh-capture resolution (P1's "on-demand capture, bypassing window/
    # council" work -- docs/superpowers/specs/2026-08-12-perception-frontier-
    # design.md). orion-vision-edge writes every captured frame here
    # (services/orion-vision-edge's own FRAME_STORAGE_DIR default matches);
    # mounted read-only into this container (see docker-compose.yml). A
    # caller opts in per-request via `request.use_latest_frame: true`
    # instead of supplying `image_path` -- see _resolve_latest_frame_path.
    VISION_FRAMES_DIR: str = "/mnt/telemetry/vision/frames"

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

    # VRAM pressure.
    #
    # These defaults were 3500/2200/1400 -- correct for the Tesla P100 16GB this
    # service was originally written for, and fatal on the Tesla P4 7.68GB that
    # replaced it when the P100 moved to circe (2026-08-21). app/gpu.py computes
    # `effective_free = free_mb - reserve_mb` and refuses below hard_floor, so
    # with ~3.2GB of warm resident models on the P4, 4191 - 3500 = 691 < 1400 was
    # unsatisfiable and the service refused every task for ~21 hours while
    # reporting healthy. A host with no .env override would have inherited that.
    # Re-derive against the RESIDENT footprint whenever a card changes.
    #
    # These defaults are gated by test_vram_budget_matches_env_example.py::
    # test_settings_defaults_match_env_example. That test was ADDED for this,
    # because the original comment here claimed coverage that did not exist:
    # the file only compared config/vision_profiles.yaml to .env_example, so
    # settings.py could (and did) carry 3500/2200/1400 while .env_example
    # carried 1200/1000/800 with the suite fully green -- including throughout
    # the 21-hour outage.
    VISION_VRAM_RESERVE_MB: int = 1200
    VISION_VRAM_SOFT_FLOOR_MB: int = 1000
    VISION_VRAM_HARD_FLOOR_MB: int = 800

    # Content-addressed frames from nodes with no shared filesystem.
    # Empty means this host can only read frames off its own disk, which is
    # correct for a single-machine deployment and wrong the moment a second
    # camera exists anywhere else.
    VISION_PERCEPT_STORE_URL: str = ""
    VISION_PERCEPT_STORE_TOKEN: str = ""
    VISION_PERCEPT_TIMEOUT_SEC: float = 10.0

    # Liveness alerting -- see app/liveness.py for why this exists.
    VISION_LIVENESS_ALERT_ENABLED: bool = True
    VISION_LIVENESS_WINDOW_SEC: float = 300.0
    VISION_LIVENESS_MIN_SAMPLES: int = 10
    VISION_LIVENESS_ARM_FAIL_RATE: float = 0.8
    VISION_LIVENESS_CLEAR_FAIL_RATE: float = 0.2
    VISION_LIVENESS_SUSTAIN_SEC: float = 180.0
    VISION_LIVENESS_COOLDOWN_SEC: float = 3600.0
    NOTIFY_BASE_URL: str = ""
    NOTIFY_API_TOKEN: str = ""
    # NODE_NAME is already declared above -- a second declaration silently
    # shadows the first (review finding).

    # identity_face (2026-08-25, docs/superpowers/specs/2026-08-21-seeing-
    # juniper-identity-and-situated-observation-design.md section 4).
    # ONE enrolled subject, gallery does not grow at runtime -- app/
    # identity_gallery.py's save path is only ever called from
    # scripts/enroll_identity_face.py, never from the request-handling path.
    # Reuses the existing /mnt/telemetry/orion-vision-host mount (see
    # docker-compose.yml) rather than adding a new volume.
    IDENTITY_GALLERY_DIR: str = "/mnt/telemetry/orion-vision-host/identity_gallery"
    IDENTITY_ENROLLED_SUBJECT: str = "juniper"

    @property
    def enabled_profiles(self) -> List[str]:
        return _split_csv(self.VISION_ENABLED_PROFILES)

    @property
    def devices(self) -> List[str]:
        d = _split_csv(self.VISION_DEVICES)
        return d if d else [self.VISION_DEFAULT_DEVICE]

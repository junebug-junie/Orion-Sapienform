from __future__ import annotations

from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "world-model"
    SERVICE_VERSION: str = "0.1.0"
    # Default is athena so a fresh operator .env still boots locally; the real
    # deployment target for this service is `circe` (see README "Node" and
    # docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md §3) --
    # circe provisioning itself is out of scope for this patch.
    NODE_NAME: str = "athena"
    LOG_LEVEL: str = "INFO"

    # Bus
    ORION_BUS_ENABLED: bool = True
    ORION_BUS_ENFORCE_CATALOG: bool = True
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    HEARTBEAT_INTERVAL_SEC: float = 10.0

    # Channels (orion/bus/channels.yaml)
    CHANNEL_WORLDMODEL_INTAKE: str = "orion:exec:request:WorldModelService"
    CHANNEL_WORLDMODEL_REPLY_PREFIX: str = "orion:worldmodel:reply"
    CHANNEL_WORLDMODEL_PUB: str = "orion:worldmodel:predictions"

    # Caches -- storage-warm convention (orion-diffusion-host, orion-vllm-host,
    # orion-recall), not vision-host's mixed-purpose /mnt/telemetry: this
    # service's weights are a single-purpose checkpoint download, not
    # telemetry-adjacent data.
    # NOT YET WIRED (code review, 2026-08-20): no code in app/ reads this
    # path. Weights are always randomly initialized in-process in this patch
    # (README "Status: untrained inference-capable scaffolding") -- there is
    # no trained checkpoint to load or save yet. Declared now so the path
    # convention is settled and the volume mount (docker-compose.yml) is in
    # place before a future patch adds real checkpoint loading.
    MODEL_CACHE_DIR: str = "/mnt/storage-warm/models/world-model"

    # --- Feature group dims (encoder input shape) ---
    # ASSUMPTION (README "Assumptions"): exact dims were not specified before
    # context was lost to compaction. These are placeholder defaults sized
    # for a small fusion encoder; an operator wiring a real upstream producer
    # must set these to match that producer's actual vector lengths. A
    # trajectory step whose vector length does not match its declared `dim`
    # (or whose `dim` does not match the configured group dim below) is
    # rejected at forward-pass time -- see app/model.py.
    WM_DIM_BIOMETRICS: int = 32
    WM_DIM_AFFECT: int = 16
    WM_DIM_EXECUTION_CONTEXT: int = 16
    WM_DIM_MEMORY_POINTERS: int = 32
    WM_DIM_TEMPORAL: int = 8
    # Vision embeddings come from orion-vision-host's VisionEmbedding.vector
    # (orion/schemas/vision.py). 512 matches common CLIP-class embedding dims;
    # override if the deployed vision profile emits a different width.
    WM_DIM_VISION_EMBEDDING: int = 512

    # --- Model architecture ---
    # ASSUMPTION (README "~150M param assumption"): the exact target size was
    # not nailed down before context was lost to compaction. 150M is a
    # concrete default within the requested ~100-350M range, not a spec.
    WM_FUSION_DIM: int = 512
    WM_DYNAMICS_D_MODEL: int = 1024
    WM_DYNAMICS_NHEAD: int = 16
    WM_DYNAMICS_NUM_LAYERS: int = 12
    WM_DYNAMICS_DIM_FEEDFORWARD: int = 4096
    WM_DYNAMICS_MAX_WINDOW: int = 32
    WM_DYNAMICS_DROPOUT: float = 0.1
    # None => state_dim defaults to WM_FUSION_DIM (predict the next fused
    # state, not a separately-sized readout). Left blank in .env_example on
    # purpose -- pydantic-settings raises on an empty string for a plain
    # `int | None` field, so the validator below treats "" the same as unset.
    WM_STATE_DIM: int | None = None

    @field_validator("WM_STATE_DIM", mode="before")
    @classmethod
    def _blank_state_dim_is_none(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    # --- Runtime ---
    # Both read by app/main.py::_select_device -- "fixed" restricts the
    # candidate pool to WM_DEFAULT_DEVICE only (still floor-checked);
    # "best_free_vram" (default) scans every cuda:* entry in WM_DEVICES.
    WM_DEVICE_STRATEGY: str = Field(default="best_free_vram", description="best_free_vram|fixed")
    WM_DEFAULT_DEVICE: str = "cuda:0"
    WM_DEVICES: str = "cuda:0"
    WM_PICK_GPU_METRIC: str = Field(default="free_vram_mb", description="free_vram_mb|free_fraction")
    # Read by app/main.py::resolve_dtype. "auto" resolves to fp32, not a
    # device-conditional guess -- see resolve_dtype's docstring.
    WM_DTYPE: str = Field(default="auto", description="auto|fp16|bf16|fp32")
    # Read by app/main.py::run_prediction_task (asyncio.wait_for around the
    # forward pass). Like vision-host's VISION_TIMEOUT_S, this bounds how
    # long the *caller* waits for a reply -- it cannot forcibly kill an
    # already-dispatched asyncio.to_thread worker thread (same limitation as
    # vision-host's identical wait_for(to_thread(...)) pattern).
    WM_TIMEOUT_S: int = 30
    WM_MAX_INFLIGHT: int = 2

    # VRAM pressure -- sized well under a 32GB card so this service's own
    # ceiling leaves real headroom for the separately-scheduled Infinity 2B
    # diffusion process sharing the same physical card via MPS (README "GPU
    # scheduling"). A ~150M-param fp32 model is itself only ~600MB of
    # weights; the floors below are deliberately generous for activation
    # memory + MPS overhead, not because this model needs that much.
    WM_VRAM_RESERVE_MB: int = 4000
    # NOT YET WIRED (code review, 2026-08-20): only WM_VRAM_RESERVE_MB and
    # WM_VRAM_HARD_FLOOR_MB are read (app/main.py::_select_device ->
    # GpuInspector.pick_best_gpu). vision-host's soft floor gates a
    # queue-vs-reject decision in its async VisionScheduler; this scaffold
    # has no queue -- a single synchronous forward pass per request, capped
    # by WM_MAX_INFLIGHT -- so there is no queue-vs-reject decision for a
    # soft floor to gate yet. Declared for parity/future use, not faked.
    WM_VRAM_SOFT_FLOOR_MB: int = 3000
    WM_VRAM_HARD_FLOOR_MB: int = 2000

    @property
    def devices(self) -> List[str]:
        d = _split_csv(self.WM_DEVICES)
        return d if d else [self.WM_DEFAULT_DEVICE]

    @property
    def state_dim(self) -> int:
        return self.WM_STATE_DIM if self.WM_STATE_DIM is not None else self.WM_FUSION_DIM

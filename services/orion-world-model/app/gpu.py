"""VRAM-aware GPU picker.

Adapted from `services/orion-vision-host/app/gpu.py` (same reserve/hard-floor
shape) -- deliberately not imported cross-service (CLAUDE.md §5: each service
is self-contained). This copy is sized/tuned via `app/settings.py`'s
`WM_VRAM_*` env keys so this service caps itself well under a 32GB card,
leaving room for the separately-scheduled Infinity 2B diffusion process that
will share circe's 4th V100 via MPS (see `orion/schemas/world_model.py`
module docstring).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from loguru import logger

try:
    import pynvml  # provided by nvidia-ml-py
except Exception:  # pragma: no cover - absent on non-GPU dev/test hosts
    pynvml = None


_UNSET = object()


def physical_to_visible_cuda_index(
    physical_idx: int, *, cuda_visible_devices=_UNSET
) -> Optional[int]:
    """Translate a PHYSICAL GPU index (what pynvml/NVML reports, and what
    operator-facing config like WM_DEFAULT_DEVICE/WM_DEVICES names -- same
    convention as orion-diffusion-host's DIFFUSION_POWER_INTENT_GPU_INDEX,
    "the PHYSICAL nvidia-smi index...NOT the container's cuda:N") into the
    index torch's CUDA runtime will actually see for it.

    Without this, `f"cuda:{physical_idx}"` is only correct when
    CUDA_VISIBLE_DEVICES is unset. Confirmed live 2026-09-24: locking this
    container to one physical GPU (CUDA_VISIBLE_DEVICES=2) broke device
    selection with `RuntimeError: CUDA error: invalid device ordinal` --
    pynvml keeps enumerating every physical GPU regardless of
    CUDA_VISIBLE_DEVICES (confirmed live: nvmlDeviceGetCount() still
    reported 4, and index 0 was a different physical card entirely, a
    V100-PCIE-32GB, not the PG500-216 this service targets), while torch's
    CUDA runtime only sees the remapped subset. `pick_best_gpu` below picks
    correctly using physical indices (NVML-consistent); this function is
    the missing translation step before that pick becomes a torch device
    string.

    `cuda_visible_devices=None` vs `""` are deliberately NOT the same thing
    (review finding, caught before this shipped): the variable being
    genuinely ABSENT from the environment means "no restriction" (today's
    behavior when this service isn't locked to a card at all), but a
    container that explicitly sets `CUDA_VISIBLE_DEVICES=` to an empty
    string -- which docker-compose's `${VAR}` interpolation produces
    whenever an operator's `.env` defines the key with no value, exactly
    what `.env_example` ships by default for a non-circe host -- is real
    CUDA/nvidia-container-toolkit behavior for "zero GPUs visible", not
    "unrestricted". Conflating the two would silently pass an unavailable
    physical index straight to torch as if nothing were scoped.

    Entries may be plain physical indices (this repo's only live usage
    today) or GPU UUIDs (`GPU-...`/`MIG-...`), both documented-valid
    CUDA_VISIBLE_DEVICES forms -- resolved via NVML when present, so a
    UUID-based operator config doesn't get misreported as "not visible."

    Returns the unchanged index when CUDA_VISIBLE_DEVICES is genuinely
    unset. Returns None when it IS set (even to empty) but does not make
    this physical index visible -- a real condition to surface, not a case
    to silently guess through.
    """
    if cuda_visible_devices is _UNSET:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is None:
        return physical_idx
    raw = cuda_visible_devices.strip()
    if not raw:
        return None
    visible = [v.strip() for v in raw.split(",") if v.strip()]
    try:
        return visible.index(str(physical_idx))
    except ValueError:
        pass
    if pynvml is not None and any(v.upper().startswith(("GPU-", "MIG-")) for v in visible):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            return visible.index(uuid)
        except Exception:
            return None
    return None


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mb: int
    free_mb: int
    used_mb: int


class GpuInspector:
    def __init__(self):
        self._initialized = False

    def init(self) -> None:
        if pynvml is None:
            raise RuntimeError("pynvml not available (install nvidia-ml-py)")
        if not self._initialized:
            pynvml.nvmlInit()
            self._initialized = True
            logger.info("[GPU] NVML initialized")

    def shutdown(self) -> None:
        if pynvml is None:
            return
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False

    def list_gpus(self) -> List[GpuInfo]:
        if pynvml is None:
            return []
        self.init()
        count = pynvml.nvmlDeviceGetCount()
        out: List[GpuInfo] = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            total_mb = int(mem.total / (1024 * 1024))
            free_mb = int(mem.free / (1024 * 1024))
            used_mb = int(mem.used / (1024 * 1024))
            out.append(GpuInfo(index=i, name=str(name), total_mb=total_mb, free_mb=free_mb, used_mb=used_mb))
        return out

    def pick_best_gpu(
        self,
        candidates: List[int],
        reserve_mb: int,
        hard_floor_mb: int,
        metric: str = "free_vram_mb",
    ) -> Optional[Tuple[int, GpuInfo]]:
        """
        Pick GPU with highest free VRAM among candidates, honoring
        reserve/hard_floor. `reserve_mb` is headroom deliberately withheld
        for co-hosted workloads (e.g. Infinity 2B sharing the same MPS card)
        -- not this service's own expected usage.
        """
        infos = self.list_gpus()
        by_index = {g.index: g for g in infos}

        best: Optional[Tuple[int, GpuInfo, float]] = None
        for idx in candidates:
            g = by_index.get(idx)
            if not g:
                continue

            effective_free = g.free_mb - reserve_mb
            if effective_free < hard_floor_mb:
                continue

            if metric == "free_fraction":
                score = effective_free / max(g.total_mb, 1)
            else:
                score = float(effective_free)

            if best is None or score > best[2]:
                best = (idx, g, score)

        if best is None:
            return None

        return best[0], best[1]

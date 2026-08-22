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

from dataclasses import dataclass
from typing import List, Optional, Tuple

from loguru import logger

try:
    import pynvml  # provided by nvidia-ml-py
except Exception:  # pragma: no cover - absent on non-GPU dev/test hosts
    pynvml = None


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

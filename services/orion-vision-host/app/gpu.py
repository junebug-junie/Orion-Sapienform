from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

try:
    import pynvml  # provided by nvidia-ml-py
except Exception:  # pragma: no cover
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
        self.last_pick_detail: List[Dict[str, Any]] = []

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
        own_reusable_fn: Optional[Callable[[int], int]] = None,
    ) -> Optional[Tuple[int, GpuInfo]]:
        """
        Pick GPU with highest free VRAM among candidates, honoring reserve/hard_floor.

        "Free" = NVML free + this process's own reusable CUDA cache on that
        device (torch reserved - allocated). NVML counts PyTorch's caching
        allocator as "used", so without this term a process that is the only
        tenant of its GPU starves itself once its cache grows (2026-10-15
        incident: 5610 MiB held by our own PID, 0 other tenants, 100% refusals).
        Other processes' memory is still counted as used, so the gate still
        protects against neighbours growing.

        Per-candidate breakdown of the last call is kept in
        ``self.last_pick_detail`` so a refusal can say why.
        """
        if own_reusable_fn is None:
            own_reusable_fn = own_reusable_cache_mb
        infos = self.list_gpus()
        by_index = {g.index: g for g in infos}

        detail: List[Dict[str, Any]] = []
        best: Optional[Tuple[int, GpuInfo, float]] = None
        for idx in candidates:
            g = by_index.get(idx)
            if not g:
                detail.append({"gpu": idx, "visible": False})
                continue

            try:
                own_reusable = max(0, int(own_reusable_fn(idx)))
            except Exception:
                own_reusable = 0
            available = g.free_mb + own_reusable
            effective_free = available - reserve_mb
            admitted = effective_free >= hard_floor_mb
            row = {
                "gpu": idx,
                "nvml_free_mb": g.free_mb,
                "own_reusable_mb": own_reusable,
                "reserve_mb": reserve_mb,
                "effective_free_mb": effective_free,
                "hard_floor_mb": hard_floor_mb,
                "admitted": admitted,
            }
            detail.append(row)
            logger.debug(
                f"[GPU] admit_check gpu={idx} nvml_free={g.free_mb}MB own_reusable={own_reusable}MB "
                f"reserve={reserve_mb}MB effective={effective_free}MB floor={hard_floor_mb}MB admitted={admitted}"
            )
            if not admitted:
                continue

            if metric == "free_fraction":
                score = effective_free / max(g.total_mb, 1)
            else:
                score = float(effective_free)

            if best is None or score > best[2]:
                best = (idx, g, score)

        self.last_pick_detail = detail

        if best is None:
            return None

        return best[0], best[1]


def nvml_to_torch_index(nvml_index: int) -> Optional[int]:
    """Map an NVML (PCI-bus-order) index to this process's torch device index.

    Torch indices are relative to CUDA_VISIBLE_DEVICES; NVML sees every card
    the container exposes. With CUDA_DEVICE_ORDER=PCI_BUS_ID (pinned in the
    Dockerfile and compose), visible device k is the card at the k-th entry
    of CUDA_VISIBLE_DEVICES. Unset means identity. UUID / MIG entries or an
    index not in the list -> None, so the caller falls back to NVML only and
    never adds one card's cache to another card's free figure.
    """
    if nvml_index < 0:
        return None
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return nvml_index
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    try:
        ints = [int(e) for e in entries]
    except ValueError:
        return None
    try:
        return ints.index(nvml_index)
    except ValueError:
        return None


def torch_to_nvml_index(torch_index: int) -> Optional[int]:
    """Inverse of nvml_to_torch_index: this process's torch index -> NVML index.

    VISION_DEVICES names torch devices ("cuda:N"), but the admission gate reads
    NVML, which sees every card the container exposes regardless of
    CUDA_VISIBLE_DEVICES. Without this, a container that sees two cards and
    sets CUDA_VISIBLE_DEVICES=1 would run on the second card while checking
    free memory on the first (the qwen lane did exactly that on athena).
    Unset means identity. UUID / MIG entries or an out-of-range index -> None.
    """
    if torch_index < 0:
        return None
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return torch_index
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    try:
        ints = [int(e) for e in entries]
    except ValueError:
        return None
    if torch_index >= len(ints):
        return None
    return ints[torch_index]


def own_reusable_cache_mb(gpu_index: int) -> int:
    """MiB this process holds in PyTorch's caching allocator but is not using.

    That memory is reusable by our own next task, yet NVML reports it as used.
    Returns 0 whenever torch/CUDA is missing, CUDA was never initialised in
    this process, or the NVML index is not visible to this process -- i.e. the old
    behaviour. Never initialises CUDA itself.
    """
    try:
        import torch  # local import: tests and CPU-only paths need not have it
    except Exception:
        return 0
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return 0
        torch_index = nvml_to_torch_index(gpu_index)
        if torch_index is None or torch_index >= torch.cuda.device_count():
            return 0
        reserved = int(torch.cuda.memory_reserved(torch_index))
        allocated = int(torch.cuda.memory_allocated(torch_index))
    except Exception:
        return 0
    return max(0, reserved - allocated) // (1024 * 1024)

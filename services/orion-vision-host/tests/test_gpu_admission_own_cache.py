"""Gate: the VRAM admission check counts this process's own reusable CUDA cache.

Live incident 2026-10-15 from 18:54 UTC: orion-athena-vision-host refused every
task with ``gpu_hard_floor``. It was the ONLY process on GPU 0 (Tesla P4,
7680 MiB); nvidia-smi showed its own PID holding 5610 MiB. NVML counts
PyTorch's caching allocator (reserved but not allocated, i.e. reusable by our
own next task) as "used", so ``free_mb - reserve(1200) < hard_floor(800)`` and
the service starved itself. These tests pin the fix and pin that memory held
by OTHER processes still blocks admission.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from app import gpu as gpu_mod
from app.gpu import GpuInfo, GpuInspector, own_reusable_cache_mb
from app.scheduler import VisionScheduler

RESERVE = 1200
FLOOR = 800


def _inspector(free_mb: int, total_mb: int = 7680) -> GpuInspector:
    insp = GpuInspector()
    insp.list_gpus = lambda: [  # type: ignore[method-assign]
        GpuInfo(index=0, name="Tesla P4", total_mb=total_mb, free_mb=free_mb, used_mb=total_mb - free_mb)
    ]
    return insp


def test_live_incident_numbers_are_admitted_when_own_cache_is_reusable():
    # Live NVML free was 1988 (gate needs 2000). How much of the 5610 MiB our PID
    # holds is idle cache was NOT measured live -- 1000 here is a fixture value,
    # and free is set tighter than live so the own-cache term must carry it.
    insp = _inspector(free_mb=1467)
    pick = insp.pick_best_gpu([0], RESERVE, FLOOR, own_reusable_fn=lambda idx: 1000)
    assert pick is not None and pick[0] == 0
    row = insp.last_pick_detail[0]
    assert row["nvml_free_mb"] == 1467
    assert row["own_reusable_mb"] == 1000
    assert row["effective_free_mb"] == 1467 + 1000 - RESERVE
    assert row["admitted"] is True


def test_same_numbers_without_own_cache_are_refused():
    insp = _inspector(free_mb=1467)
    assert insp.pick_best_gpu([0], RESERVE, FLOOR, own_reusable_fn=lambda idx: 0) is None
    assert insp.last_pick_detail[0]["admitted"] is False


def test_other_process_eating_memory_is_still_refused():
    # Another tenant took the card down to 900 MiB free; we hold no idle cache.
    insp = _inspector(free_mb=900)
    assert insp.pick_best_gpu([0], RESERVE, FLOOR, own_reusable_fn=lambda idx: 0) is None
    # A tiny own cache does not paper over a neighbour's growth either.
    assert insp.pick_best_gpu([0], RESERVE, FLOOR, own_reusable_fn=lambda idx: 50) is None


def test_own_reusable_fn_failure_degrades_to_old_behaviour():
    def boom(idx):
        raise RuntimeError("cuda gone")

    insp = _inspector(free_mb=1467)
    assert insp.pick_best_gpu([0], RESERVE, FLOOR, own_reusable_fn=boom) is None
    insp = _inspector(free_mb=2500)
    assert insp.pick_best_gpu([0], RESERVE, FLOOR, own_reusable_fn=boom) is not None


def _fake_torch(*, available=True, initialized=True, count=1, reserved=0, allocated=0):
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        is_initialized=lambda: initialized,
        device_count=lambda: count,
        memory_reserved=lambda idx: reserved,
        memory_allocated=lambda idx: allocated,
    )
    return types.SimpleNamespace(cuda=cuda)


MIB = 1024 * 1024


def test_own_reusable_cache_mb_reads_reserved_minus_allocated(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(reserved=5000 * MIB, allocated=3800 * MIB))
    assert own_reusable_cache_mb(0) == 1200


@pytest.mark.parametrize(
    "kwargs,idx",
    [
        ({"available": False}, 0),
        ({"initialized": False}, 0),
        ({"count": 1}, 1),  # index not visible to this process
        ({"reserved": 100 * MIB, "allocated": 200 * MIB}, 0),  # never negative
    ],
)
def test_own_reusable_cache_mb_is_zero_when_not_applicable(monkeypatch, kwargs, idx):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    base = {"reserved": 5000 * MIB, "allocated": 1000 * MIB}
    base.update(kwargs)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(**base))
    assert own_reusable_cache_mb(idx) == 0


def test_no_torch_means_zero(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    assert own_reusable_cache_mb(0) == 0
    insp = _inspector(free_mb=1467)
    assert insp.pick_best_gpu([0], RESERVE, FLOOR) is None  # default fn, old behaviour


def _sched() -> VisionScheduler:
    return VisionScheduler(
        devices=["cuda:0"],
        pick_metric="free_vram_mb",
        max_inflight=2,
        max_inflight_per_gpu=1,
        queue_when_busy=False,
        max_queue=10,
        reserve_mb=RESERVE,
        soft_floor_mb=1000,
        hard_floor_mb=FLOOR,
    )


def test_refusal_streak_is_counted_and_explained(monkeypatch):
    sched = _sched()
    free = {"mb": 900}
    sched.gpu.list_gpus = lambda: [  # type: ignore[method-assign]
        GpuInfo(index=0, name="Tesla P4", total_mb=7680, free_mb=free["mb"], used_mb=7680 - free["mb"])
    ]
    monkeypatch.setattr(gpu_mod, "own_reusable_cache_mb", lambda idx: 0)

    seen = []

    async def handler(pick):
        seen.append(pick)
        return pick.device

    async def body():
        for _ in range(3):
            await sched.submit(handler)
        free["mb"] = 4000
        await sched.submit(handler)

    asyncio.run(body())

    assert [p.device for p in seen] == ["cpu", "cpu", "cpu", "cuda:0"]
    reason = seen[0].refusal_reason
    assert "nvml_free=900MB" in reason and "own_reusable=0MB" in reason and "hard_floor=800MB" in reason
    stats = sched.refusal_stats()
    assert stats["gpu_hard_floor_refusals_total"] == 3
    assert stats["gpu_hard_floor_refusal_streak"] == 0  # reset by the admit
    assert stats["last_refusal_reason"] == reason
    assert stats["last_admit_at"] is not None


def test_readiness_probe_does_not_count_as_refusal(monkeypatch):
    sched = _sched()
    sched.gpu.list_gpus = lambda: [  # type: ignore[method-assign]
        GpuInfo(index=0, name="Tesla P4", total_mb=7680, free_mb=100, used_mb=7580)
    ]
    monkeypatch.setattr(gpu_mod, "own_reusable_cache_mb", lambda idx: 0)
    assert sched.can_pick_gpu() is False
    assert sched.refusal_stats()["gpu_hard_floor_refusals_total"] == 0


def test_health_surfaces_refusal_stats(monkeypatch):
    from app import main as main_mod

    sched = _sched()
    sched.hard_floor_refusals_total = 7
    sched.hard_floor_refusal_streak = 7
    sched.last_refusal_reason = "gpu=0 nvml_free=900MB ..."
    monkeypatch.setattr(main_mod.service, "sched", sched)
    body = asyncio.run(main_mod.health())
    assert body["scheduler"]["gpu_hard_floor_refusals_total"] == 7
    assert body["scheduler"]["gpu_hard_floor_refusal_streak"] == 7
    assert body["scheduler"]["last_refusal_reason"].startswith("gpu=0")


@pytest.mark.parametrize(
    "cvd,nvml_idx,expected",
    [
        (None, 0, 0),
        (None, 2, 2),
        ("0", 0, 0),
        ("0", 1, None),  # NVML card 1 is not ours: never borrow torch device 0's cache for it
        ("4", 4, 0),
        ("4", 0, None),
        ("3,1", 1, 1),
        ("GPU-abc", 0, None),
        ("", 0, None),
    ],
)
def test_nvml_to_torch_index(monkeypatch, cvd, nvml_idx, expected):
    if cvd is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd)
    assert gpu_mod.nvml_to_torch_index(nvml_idx) == expected


def test_own_cache_is_read_from_the_mapped_torch_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    seen = []
    torch_stub = _fake_torch(reserved=3000 * MIB, allocated=1000 * MIB)
    torch_stub.cuda.memory_reserved = lambda i: (seen.append(i), 3000 * MIB)[1]
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    assert own_reusable_cache_mb(4) == 2000
    assert seen == [0]
    assert own_reusable_cache_mb(0) == 0  # NVML card 0 is someone else's

"""Gate: the VRAM admission check reads the card the work runs on.

2026-10-15, athena GPU swap. ``VISION_DEVICES=cuda:N`` names a torch device,
but the gate reads NVML, which sees every card the container exposes and
ignores CUDA_VISIBLE_DEVICES. Under ``gpus: all`` the two index spaces only
agreed by luck: the qwen lane (CUDA_VISIBLE_DEVICES=1, VISION_DEVICES unset ->
cuda:0) ran on the T10 while checking the P4's free memory. The shared host
now gets one card via compose ``device_ids``; these tests pin both layouts
and the expected T10 numbers with the qwen lane as a neighbour.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app import gpu as gpu_mod
from app.gpu import GpuInfo, nvml_to_torch_index, torch_to_nvml_index
from app.scheduler import VisionScheduler

SERVICE_DIR = Path(__file__).resolve().parents[1]
T10_TOTAL = 16384
P4_TOTAL = 7680
RESERVE = 1200  # .env_example VISION_VRAM_RESERVE_MB
FLOOR = 800  # .env_example VISION_VRAM_HARD_FLOOR_MB


def _sched(devices=("cuda:0",)) -> VisionScheduler:
    return VisionScheduler(
        devices=list(devices),
        pick_metric="free_vram_mb",
        max_inflight=2,
        max_inflight_per_gpu=1,
        queue_when_busy=False,
        max_queue=10,
        reserve_mb=RESERVE,
        soft_floor_mb=1000,
        hard_floor_mb=FLOOR,
    )


def _run(sched: VisionScheduler):
    async def handler(pick):
        return pick

    return asyncio.run(sched.submit(handler))


@pytest.mark.parametrize(
    "cvd,torch_idx,expected",
    [
        (None, 0, 0),
        (None, 3, 3),
        ("0", 0, 0),
        ("1", 0, 1),
        ("1", 1, None),
        ("3,1", 1, 1),
        ("GPU-abc", 0, None),
        ("", 0, None),
    ],
)
def test_torch_to_nvml_index(monkeypatch, cvd, torch_idx, expected):
    if cvd is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd)
    assert torch_to_nvml_index(torch_idx) == expected
    if expected is not None:
        assert nvml_to_torch_index(expected) == torch_idx  # round trip


def test_device_ids_pin_one_card_is_index_zero_in_both_spaces(monkeypatch):
    """New athena layout: device_ids exposes only the T10. NVML lists it as
    index 0, torch calls it cuda:0, CUDA_VISIBLE_DEVICES=0."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    sched = _sched()
    sched.gpu.list_gpus = lambda: [  # type: ignore[method-assign]
        GpuInfo(index=0, name="Tesla T10", total_mb=T10_TOTAL, free_mb=4190, used_mb=T10_TOTAL - 4190)
    ]
    monkeypatch.setattr(gpu_mod, "own_reusable_cache_mb", lambda idx: 0)
    pick = _run(sched)
    assert pick.device == "cuda:0" and pick.gpu_index == 0
    assert sched.gpu.last_pick_detail[0]["gpu"] == 0


def test_all_cards_visible_gate_checks_the_card_torch_runs_on(monkeypatch):
    """Qwen-lane layout: container sees both cards, CUDA_VISIBLE_DEVICES=1.
    torch cuda:0 is the T10 (NVML 1); the gate must read NVML 1, not the P4."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    sched = _sched()
    sched.gpu.list_gpus = lambda: [  # type: ignore[method-assign]
        # P4 roomy, T10 full: the old code admitted here off the P4's numbers.
        GpuInfo(index=0, name="Tesla P4", total_mb=P4_TOTAL, free_mb=4500, used_mb=P4_TOTAL - 4500),
        GpuInfo(index=1, name="Tesla T10", total_mb=T10_TOTAL, free_mb=300, used_mb=T10_TOTAL - 300),
    ]
    monkeypatch.setattr(gpu_mod, "own_reusable_cache_mb", lambda idx: 0)
    pick = _run(sched)
    assert pick.device == "cpu"
    assert [row["gpu"] for row in sched.gpu.last_pick_detail] == [1]

    sched.gpu.list_gpus = lambda: [  # type: ignore[method-assign]
        GpuInfo(index=0, name="Tesla P4", total_mb=P4_TOTAL, free_mb=300, used_mb=P4_TOTAL - 300),
        GpuInfo(index=1, name="Tesla T10", total_mb=T10_TOTAL, free_mb=4190, used_mb=T10_TOTAL - 4190),
    ]
    pick = _run(sched)
    assert pick.device == "cuda:0" and pick.gpu_index == 0  # torch index, not NVML 1


def test_own_cache_is_read_from_the_torch_index_of_the_checked_card(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    sched = _sched()
    sched.gpu.list_gpus = lambda: [  # type: ignore[method-assign]
        GpuInfo(index=0, name="Tesla P4", total_mb=P4_TOTAL, free_mb=5000, used_mb=P4_TOTAL - 5000),
        GpuInfo(index=1, name="Tesla T10", total_mb=T10_TOTAL, free_mb=1500, used_mb=T10_TOTAL - 1500),
    ]
    asked = []

    def own(nvml_idx):
        asked.append(nvml_idx)
        return 600 if nvml_to_torch_index(nvml_idx) == 0 else 0

    monkeypatch.setattr(gpu_mod, "own_reusable_cache_mb", own)
    pick = _run(sched)
    assert asked == [1]
    assert pick.device == "cuda:0"  # 1500 + 600 - 1200 = 900 >= 800


@pytest.mark.parametrize(
    "t10_free_mb,admitted",
    [
        # Expected after the move: 16384 - qwen 6556 - vision host 5632 ~= 4190 free.
        (4190, True),
        # Qwen grows ~2.2GB: free + own idle cache must stay >= reserve + floor = 2000.
        (2000, True),
        (1999, False),
    ],
)
def test_expected_t10_numbers_with_qwen_neighbour(monkeypatch, t10_free_mb, admitted):
    """Qwen's memory is another process's, so it counts as used; only our own
    idle torch cache is added back (0 here, the worst case)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    sched = _sched()
    sched.gpu.list_gpus = lambda: [  # type: ignore[method-assign]
        GpuInfo(index=0, name="Tesla T10", total_mb=T10_TOTAL, free_mb=t10_free_mb, used_mb=T10_TOTAL - t10_free_mb)
    ]
    monkeypatch.setattr(gpu_mod, "own_reusable_cache_mb", lambda idx: 0)
    assert (_run(sched).device == "cuda:0") is admitted


def test_unmappable_cvd_falls_back_to_identity(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-49b63420")
    sched = _sched()
    assert sched._nvml_by_torch == {0: 0}


def _env_example() -> dict:
    out = {}
    for line in (SERVICE_DIR / ".env_example").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def test_compose_pins_one_card_by_device_ids():
    text = (SERVICE_DIR / "docker-compose.yml").read_text()
    live = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    body = "\n".join(live)
    assert "gpus: all" not in body, "gpus: all exposes every card; NVML and torch index spaces diverge"
    assert "runtime: nvidia" not in body, "CDI-mode nvidia runtime rejects a PCI bus id in device_ids"
    assert 'device_ids: ["${VISION_HOST_GPU_DEVICE_ID:?' in body, "pin must be required, no default card"


def test_env_example_device_keys_are_container_relative():
    env = _env_example()
    assert env["VISION_HOST_GPU_DEVICE_ID"] == "00000000:D8:00.0"  # athena Tesla T10
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["VISION_DEVICES"] == "cuda:0"
    assert env["VISION_DEFAULT_DEVICE"] == "cuda:0"

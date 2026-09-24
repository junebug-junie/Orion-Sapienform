"""GPU picker unit tests -- mocked pynvml, no real CUDA required.

Mirrors services/orion-vision-host's scheduler test style (mock
`GpuInspector` internals rather than requiring real hardware).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.gpu import GpuInfo, GpuInspector, physical_to_visible_cuda_index


def _fake_gpus():
    return [
        GpuInfo(index=0, name="Tesla V100", total_mb=32768, free_mb=20000, used_mb=12768),
        GpuInfo(index=1, name="Tesla V100", total_mb=32768, free_mb=5000, used_mb=27768),
    ]


def test_pick_best_gpu_picks_highest_free_vram():
    gpu = GpuInspector()
    with patch.object(gpu, "list_gpus", return_value=_fake_gpus()):
        picked = gpu.pick_best_gpu([0, 1], reserve_mb=1000, hard_floor_mb=2000, metric="free_vram_mb")
    assert picked is not None
    idx, info = picked
    assert idx == 0
    assert info.name == "Tesla V100"


def test_pick_best_gpu_honors_hard_floor_and_reserve():
    """GPU 0 has 20000MB free; reserve 19000 leaves 1000 effective, below a
    2000MB hard floor -- must be excluded. GPU 1 (5000 free) is even worse,
    so no candidate should be returned."""
    gpu = GpuInspector()
    with patch.object(gpu, "list_gpus", return_value=_fake_gpus()):
        picked = gpu.pick_best_gpu([0, 1], reserve_mb=19000, hard_floor_mb=2000, metric="free_vram_mb")
    assert picked is None


def test_pick_best_gpu_no_candidates_returns_none():
    gpu = GpuInspector()
    with patch.object(gpu, "list_gpus", return_value=_fake_gpus()):
        picked = gpu.pick_best_gpu([], reserve_mb=0, hard_floor_mb=0, metric="free_vram_mb")
    assert picked is None


def test_pick_best_gpu_unknown_index_skipped():
    gpu = GpuInspector()
    with patch.object(gpu, "list_gpus", return_value=_fake_gpus()):
        picked = gpu.pick_best_gpu([5], reserve_mb=0, hard_floor_mb=0, metric="free_vram_mb")
    assert picked is None


def test_pick_best_gpu_free_fraction_metric():
    """GPU 0: (20000-0)/32768 ~= 0.61. GPU 1: (5000-0)/32768 ~= 0.15. GPU 0 wins either metric here,
    so use unequal totals to actually distinguish the metrics."""
    gpus = [
        GpuInfo(index=0, name="A", total_mb=100000, free_mb=10000, used_mb=90000),  # fraction 0.10
        GpuInfo(index=1, name="B", total_mb=10000, free_mb=9000, used_mb=1000),  # fraction 0.90
    ]
    gpu = GpuInspector()
    with patch.object(gpu, "list_gpus", return_value=gpus):
        by_free_mb = gpu.pick_best_gpu([0, 1], reserve_mb=0, hard_floor_mb=0, metric="free_vram_mb")
        by_fraction = gpu.pick_best_gpu([0, 1], reserve_mb=0, hard_floor_mb=0, metric="free_fraction")
    assert by_free_mb[0] == 0  # more absolute free MB
    assert by_fraction[0] == 1  # more free as a fraction of its own total


def test_list_gpus_returns_empty_without_pynvml():
    gpu = GpuInspector()
    with patch("app.gpu.pynvml", None):
        assert gpu.list_gpus() == []


def test_physical_to_visible_cuda_index_unset_is_identity():
    """No CUDA_VISIBLE_DEVICES -- today's behavior, physical index IS the
    torch index. Passing the env var explicitly (not monkeypatching
    os.environ) keeps this test independent of the real process env."""
    assert physical_to_visible_cuda_index(2, cuda_visible_devices="") == 2
    assert physical_to_visible_cuda_index(0, cuda_visible_devices=None) == 0


def test_physical_to_visible_cuda_index_translates_scoped_container():
    """Regression: live-caught 2026-09-24. CUDA_VISIBLE_DEVICES=2 means
    torch's cuda:0 IS physical GPU 2 -- pick_best_gpu (NVML-based) correctly
    returns the physical index 2, and this must translate it to 0, the only
    index torch's CUDA runtime actually sees."""
    assert physical_to_visible_cuda_index(2, cuda_visible_devices="2") == 0


def test_physical_to_visible_cuda_index_multi_gpu_container():
    assert physical_to_visible_cuda_index(1, cuda_visible_devices="0,1,3") == 1
    assert physical_to_visible_cuda_index(3, cuda_visible_devices="0,1,3") == 2
    assert physical_to_visible_cuda_index(0, cuda_visible_devices="0,1,3") == 0


def test_physical_to_visible_cuda_index_not_visible_returns_none():
    """A real misconfiguration (asking for a physical index the container
    was never given) must surface as None, not silently return a wrong
    index or the raw physical one."""
    assert physical_to_visible_cuda_index(2, cuda_visible_devices="0,1") is None


def test_list_gpus_uses_pynvml_when_available():
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlDeviceGetCount.return_value = 1
    fake_handle = MagicMock()
    fake_pynvml.nvmlDeviceGetHandleByIndex.return_value = fake_handle
    fake_pynvml.nvmlDeviceGetName.return_value = "Tesla V100"
    fake_mem = MagicMock(total=32 * 1024 * 1024 * 1024, free=20 * 1024 * 1024 * 1024, used=12 * 1024 * 1024 * 1024)
    fake_pynvml.nvmlDeviceGetMemoryInfo.return_value = fake_mem

    gpu = GpuInspector()
    with patch("app.gpu.pynvml", fake_pynvml):
        infos = gpu.list_gpus()
    assert len(infos) == 1
    assert infos[0].name == "Tesla V100"
    assert infos[0].total_mb == 32768
    fake_pynvml.nvmlInit.assert_called_once()

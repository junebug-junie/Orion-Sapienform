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
    """The variable genuinely ABSENT (None) -- today's unlocked-container
    behavior, physical index IS the torch index."""
    assert physical_to_visible_cuda_index(0, cuda_visible_devices=None) == 0
    assert physical_to_visible_cuda_index(2, cuda_visible_devices=None) == 2


def test_physical_to_visible_cuda_index_explicit_empty_is_zero_devices_not_unset():
    """Regression (review finding, caught before this shipped): an
    operator's .env defining `CUDA_VISIBLE_DEVICES=` with no value makes
    docker-compose set the container's env var to an EXPLICIT empty string
    -- real CUDA/nvidia-container-toolkit semantics for "no GPUs visible",
    NOT the same as the variable being absent. Conflating the two would
    silently pass an unavailable index straight to torch on any fresh
    operator host using .env_example's own generic default."""
    assert physical_to_visible_cuda_index(0, cuda_visible_devices="") is None
    assert physical_to_visible_cuda_index(2, cuda_visible_devices="") is None


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


def test_physical_to_visible_cuda_index_falls_back_to_real_env_when_not_passed(monkeypatch):
    """Default call site behavior (app/main.py doesn't pass the kwarg) --
    must read the real process environment, distinguishing a missing key
    (os.environ.get returns None) from one present but blank."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert physical_to_visible_cuda_index(2) == 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert physical_to_visible_cuda_index(2) == 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert physical_to_visible_cuda_index(2) is None


def test_physical_to_visible_cuda_index_resolves_gpu_uuid_form():
    """CUDA_VISIBLE_DEVICES may list GPU UUIDs instead of bare indices --
    also documented-valid, and must not be misreported as 'not visible.'"""
    fake_pynvml = MagicMock()
    fake_handle = object()
    fake_pynvml.nvmlDeviceGetHandleByIndex.return_value = fake_handle
    fake_pynvml.nvmlDeviceGetUUID.return_value = "GPU-00e3b626-de2a-e344-796e-017b7a1b3f40"
    with patch("app.gpu.pynvml", fake_pynvml):
        result = physical_to_visible_cuda_index(
            2, cuda_visible_devices="GPU-00e3b626-de2a-e344-796e-017b7a1b3f40"
        )
    assert result == 0
    fake_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(2)


def test_physical_to_visible_cuda_index_uuid_form_no_pynvml_returns_none():
    """Without pynvml available at all, a UUID-form CUDA_VISIBLE_DEVICES
    can't be resolved -- must fail closed (None), not guess."""
    with patch("app.gpu.pynvml", None):
        result = physical_to_visible_cuda_index(2, cuda_visible_devices="GPU-xxxx")
    assert result is None


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

"""Tests for the previously-declared-but-unwired knobs found by code review
(2026-08-20): WM_DEVICE_STRATEGY and WM_DTYPE are now actually read by
app/main.py -- these pin that they do the right thing, not just that the
settings field exists."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from app.gpu import GpuInfo, GpuInspector
from app.main import _select_device, resolve_dtype
from app.settings import Settings


def _fake_gpus():
    return [
        GpuInfo(index=0, name="Tesla V100", total_mb=32768, free_mb=20000, used_mb=12768),
        GpuInfo(index=1, name="Tesla V100", total_mb=32768, free_mb=25000, used_mb=7768),
    ]


def test_select_device_best_free_vram_scans_all_devices():
    s = Settings(WM_DEVICE_STRATEGY="best_free_vram", WM_DEVICES="cuda:0,cuda:1")
    gpu = GpuInspector()
    with patch.object(gpu, "list_gpus", return_value=_fake_gpus()):
        device = _select_device(gpu, s)
    assert device == "cuda:1"  # more free VRAM than cuda:0


def test_select_device_fixed_strategy_only_considers_default_device():
    """cuda:1 has more free VRAM, but strategy=fixed must pin to
    WM_DEFAULT_DEVICE (cuda:0) rather than scanning for the best one."""
    s = Settings(WM_DEVICE_STRATEGY="fixed", WM_DEFAULT_DEVICE="cuda:0", WM_DEVICES="cuda:0,cuda:1")
    gpu = GpuInspector()
    with patch.object(gpu, "list_gpus", return_value=_fake_gpus()):
        device = _select_device(gpu, s)
    assert device == "cuda:0"


def test_select_device_fixed_strategy_still_honors_hard_floor():
    """fixed strategy pins the candidate pool, not the floor check."""
    s = Settings(
        WM_DEVICE_STRATEGY="fixed",
        WM_DEFAULT_DEVICE="cuda:0",
        WM_VRAM_RESERVE_MB=19000,
        WM_VRAM_HARD_FLOOR_MB=2000,
    )
    gpu = GpuInspector()
    with patch.object(gpu, "list_gpus", return_value=_fake_gpus()):
        # cuda:0 free=20000, reserve=19000 -> effective 1000 < hard_floor 2000
        device = _select_device(gpu, s)
    assert device is None


def test_select_device_fixed_strategy_non_cuda_default_returns_none():
    s = Settings(WM_DEVICE_STRATEGY="fixed", WM_DEFAULT_DEVICE="cpu")
    gpu = GpuInspector()
    with patch.object(gpu, "list_gpus", return_value=_fake_gpus()):
        device = _select_device(gpu, s)
    assert device is None


def test_select_device_translates_physical_index_when_cuda_visible_devices_scoped(monkeypatch):
    """Regression, live-caught 2026-09-24: WM_DEFAULT_DEVICE names the
    PHYSICAL GPU (pynvml/NVML-consistent, same convention as
    orion-diffusion-host's own physical-index env vars) -- when the
    container is scoped via CUDA_VISIBLE_DEVICES, torch only sees a
    remapped index 0 for it, not the physical one. Fixed by returning
    'cuda:1' (list_gpus reports physical index 1 as best-free-VRAM), NOT
    'cuda:1' unscoped -- here CUDA_VISIBLE_DEVICES="0,1" makes physical 1
    the SECOND visible device, i.e. torch's cuda:1 -- same number by
    coincidence in this fixture, so also assert the genuinely-different
    case right below where they diverge."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    s = Settings(WM_DEVICE_STRATEGY="best_free_vram", WM_DEVICES="cuda:0,cuda:1")
    gpu = GpuInspector()
    with patch.object(gpu, "list_gpus", return_value=_fake_gpus()):
        device = _select_device(gpu, s)
    assert device == "cuda:1"


def test_select_device_translates_when_physical_and_visible_indices_diverge(monkeypatch):
    """The case that actually broke in production: WM_DEFAULT_DEVICE=cuda:2
    (physical GPU 2), CUDA_VISIBLE_DEVICES=2 (container scoped to ONLY that
    card) -- torch's only visible index is 0, not 2. Returning 'cuda:2'
    here previously crashed with 'CUDA error: invalid device ordinal'."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    gpus = [GpuInfo(index=2, name="PG500-216", total_mb=32768, free_mb=20000, used_mb=12768)]
    s = Settings(WM_DEVICE_STRATEGY="fixed", WM_DEFAULT_DEVICE="cuda:2")
    gpu = GpuInspector()
    with patch.object(gpu, "list_gpus", return_value=gpus):
        device = _select_device(gpu, s)
    assert device == "cuda:0"


def test_select_device_returns_none_when_physical_index_not_in_cuda_visible_devices(monkeypatch):
    """A real misconfiguration (WM_DEFAULT_DEVICE names a card the container
    was never given) must fall back to cpu, not silently pick a wrong card."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    gpus = [GpuInfo(index=2, name="PG500-216", total_mb=32768, free_mb=20000, used_mb=12768)]
    s = Settings(WM_DEVICE_STRATEGY="fixed", WM_DEFAULT_DEVICE="cuda:2")
    gpu = GpuInspector()
    with patch.object(gpu, "list_gpus", return_value=gpus):
        device = _select_device(gpu, s)
    assert device is None


@pytest.mark.parametrize(
    "wm_dtype,expected",
    [
        ("auto", torch.float32),
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
        ("FP16", torch.float16),  # case-insensitive
    ],
)
def test_resolve_dtype_valid(wm_dtype, expected):
    assert resolve_dtype(wm_dtype) is expected


def test_resolve_dtype_unknown_raises():
    with pytest.raises(ValueError, match="unknown WM_DTYPE"):
        resolve_dtype("int8")

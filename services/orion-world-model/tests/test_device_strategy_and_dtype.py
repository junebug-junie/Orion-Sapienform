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

"""Regression guard for the CUDA_DEVICE_ORDER fix (2026-08-25).

This cannot catch the underlying bug itself -- that requires real
heterogeneous multi-GPU CUDA-runtime enumeration divergence between torch
and NVML, which no CPU-only/mocked-pynvml test harness can reproduce (same
reason orion-llamacpp-host/orion-vllm-host, which apply the equivalent
env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID") fix for their spawned
subprocesses, have no dedicated test either -- confirmed live 2026-08-25:
app/gpu.py::GpuInspector picks a candidate index via NVML/pynvml, always
PCI-bus-id order like nvidia-smi, then app/main.py::_select_device hands
that same integer straight to torch as cuda:{idx}; on circe this silently
loaded the model onto a different physical card than the one GpuInspector
picked, while /health still reported the intended device string).

What this test DOES catch: a future refactor of the Dockerfile silently
dropping the fix. Plain text assertion, no Docker/CUDA/torch import needed.
"""

from __future__ import annotations

from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
DOCKERFILE_PATH = SERVICE_DIR / "Dockerfile"


def test_dockerfile_sets_cuda_device_order_pci_bus_id():
    text = DOCKERFILE_PATH.read_text()
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in text, (
        "Dockerfile must set CUDA_DEVICE_ORDER=PCI_BUS_ID so torch's CUDA "
        "device index space matches NVML's (what app/gpu.py::GpuInspector "
        "uses to pick a candidate) -- without it, cuda:{idx} can silently "
        "refer to a different physical GPU than the one GpuInspector picked. "
        "See README.md 'Operator checklist' item 1 for the live incident."
    )

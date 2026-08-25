"""Regression guard for the CUDA_DEVICE_ORDER fix (2026-08-25).

This cannot catch the underlying bug itself -- that requires real
heterogeneous multi-GPU CUDA-runtime enumeration divergence, which no
CPU-only/mocked test harness can reproduce. It was found live on this
service's own first real deploy on Circe: `CUDA_VISIBLE_DEVICES=2`
(intended: the empty Tesla PG500-216, physical index 2) instead loaded
`stabilityai/sdxl-turbo` onto physical index 3, a busy Tesla V100-32GB
already serving a llama.cpp worker, while `/health`/`/ready` both reported
success with no indication anything was wrong -- torch's default CUDA
device enumeration is "fastest first," not PCI-bus-id order, so
`CUDA_VISIBLE_DEVICES=2` silently means a different physical card than
`nvidia-smi`'s index 2 unless the two orderings are forced to agree.

Same root cause and same fix as `services/orion-world-model/Dockerfile`
(found there first, same day, same host) and the equivalent
`env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")` every spawned
subprocess in `orion-llamacpp-host`/`orion-vllm-host` already applies.

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
        "device index space matches nvidia-smi's -- without it, "
        "CUDA_VISIBLE_DEVICES=N / cuda:0 can silently refer to a different "
        "physical GPU than the one intended. Confirmed live on Circe, "
        "2026-08-25 -- see README.md's GPU-2 collision section for the "
        "verification command to run after every deploy."
    )

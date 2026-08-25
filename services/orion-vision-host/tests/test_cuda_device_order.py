"""Gate: the image must pin CUDA_DEVICE_ORDER=PCI_BUS_ID.

Why this exists. Live-verified on circe (7-GPU host) 2026-08-25:
CUDA_VISIBLE_DEVICES=4 was set correctly in the container's environment
(confirmed via `docker exec ... printenv`), and `nvidia-smi` index 4 really
is the Tesla P100. But `torch.cuda.get_device_name(0)` inside that same
container reported a **Tesla V100-16GB** -- a completely different physical
card. Setting CUDA_VISIBLE_DEVICES alone is not enough: without
CUDA_DEVICE_ORDER=PCI_BUS_ID, the CUDA runtime enumerates devices in its own
FASTEST_FIRST order, which does not have to match nvidia-smi's PCI-bus-order
index. Re-running the exact same check with CUDA_DEVICE_ORDER=PCI_BUS_ID
injected fixed it immediately (confirmed: torch then reported the real P100).

This was silently dormant on athena this whole time because athena has
exactly one GPU -- with device_count()==1, "the wrong ordering" and "the
right ordering" pick the same (only) card, so the bug had zero observable
effect until the first multi-GPU host. `orion-vllm-host` and
`orion-llamacpp-host` already guard against this in their own launch code
(`env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")` -- see
services/orion-vllm-host/app/main.py and
services/orion-llamacpp-host/app/main.py); orion-vision-host had no
equivalent anywhere.

No GPU is required to run this test -- it is a static gate on the Dockerfile
text, not a runtime CUDA check (a real multi-GPU-host check isn't available
in CI). Mutation-tested: deleting the ENV line makes this test fail.
"""

from __future__ import annotations

from pathlib import Path

DOCKERFILE_PATH = Path(__file__).resolve().parents[1] / "Dockerfile"


def test_dockerfile_pins_cuda_device_order_pci_bus_id():
    text = DOCKERFILE_PATH.read_text()
    assert "ENV CUDA_DEVICE_ORDER=PCI_BUS_ID" in text, (
        "orion-vision-host's Dockerfile must set CUDA_DEVICE_ORDER=PCI_BUS_ID "
        "(baked into the image, not left as an operator .env key) -- without "
        "it, CUDA_VISIBLE_DEVICES=<nvidia-smi index> is not guaranteed to "
        "select the physical card nvidia-smi reports at that index on a "
        "multi-GPU host. See this file's module docstring for the live "
        "incident that found this."
    )


def test_cuda_device_order_is_set_before_any_pip_install():
    """The ENV must land before requirements are installed / app code is
    copied, so it's unconditionally present for every process the image ever
    runs -- not just guarded by import order inside app code."""
    text = DOCKERFILE_PATH.read_text()
    env_idx = text.index("ENV CUDA_DEVICE_ORDER=PCI_BUS_ID")
    copy_app_idx = text.index("COPY services/orion-vision-host/app")
    assert env_idx < copy_app_idx, (
        "CUDA_DEVICE_ORDER must be set before app code is copied in, so "
        "every entrypoint inherits it unconditionally from the image env."
    )

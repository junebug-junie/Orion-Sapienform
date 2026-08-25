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

Two layers pin this, not one -- a code-review pass on this same PR correctly
caught that a Dockerfile ENV default alone can be silently overridden: docker
compose `environment:` entries win over `env_file:`-sourced values, which in
turn win over a Dockerfile ENV default. So `docker-compose.yml` ALSO pins
`CUDA_DEVICE_ORDER=PCI_BUS_ID` as a literal (not `${CUDA_DEVICE_ORDER}`-
interpolated) compose `environment:` entry -- that one wins even if some
future `.env` adds a conflicting key. Both layers are gated below.

No GPU is required to run this test -- it is a static gate on the Dockerfile
and compose-file text, not a runtime CUDA check (a real multi-GPU-host check
isn't available in CI). Mutation-tested: deleting either line makes the
matching test fail.
"""

from __future__ import annotations

from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]
DOCKERFILE_PATH = SERVICE_DIR / "Dockerfile"
COMPOSE_PATH = SERVICE_DIR / "docker-compose.yml"


def test_dockerfile_pins_cuda_device_order_pci_bus_id():
    text = DOCKERFILE_PATH.read_text()
    assert "ENV CUDA_DEVICE_ORDER=PCI_BUS_ID" in text, (
        "orion-vision-host's Dockerfile must set CUDA_DEVICE_ORDER=PCI_BUS_ID "
        "-- without it, CUDA_VISIBLE_DEVICES=<nvidia-smi index> is not "
        "guaranteed to select the physical card nvidia-smi reports at that "
        "index on a multi-GPU host. See this file's module docstring for the "
        "live incident that found this."
    )


def test_compose_also_pins_cuda_device_order_literally():
    """The Dockerfile ENV default alone is not sufficient: compose
    `environment:` entries override both `env_file:` values and a Dockerfile
    ENV default for the same key, so a future `.env` adding CUDA_DEVICE_ORDER
    would silently win over the Dockerfile default with no guard. The compose
    file must pin the same value directly (a literal, not a `${VAR}`
    interpolation of something an operator's .env could set)."""
    text = COMPOSE_PATH.read_text()
    assert "- CUDA_DEVICE_ORDER=PCI_BUS_ID" in text, (
        "docker-compose.yml's `environment:` list must pin a literal "
        "CUDA_DEVICE_ORDER=PCI_BUS_ID (not ${CUDA_DEVICE_ORDER}) -- compose "
        "`environment:` entries take precedence over the Dockerfile's ENV "
        "default, so this is the layer that actually wins at container "
        "start if the two ever disagree."
    )


def test_cuda_device_order_is_set_before_first_pip_install():
    """The ENV must land before the first `RUN pip3 install`, so torch (and
    everything else) is compiled/imported into a layer that already has the
    right device order baked in -- not just before app code is copied, which
    would still be "early enough" today but says nothing about pip-install
    ordering, the property this test's name actually promises."""
    text = DOCKERFILE_PATH.read_text()
    env_idx = text.index("ENV CUDA_DEVICE_ORDER=PCI_BUS_ID")
    pip_install_idxs = [
        i for i in range(len(text)) if text.startswith("RUN pip3 install", i)
    ]
    assert pip_install_idxs, "expected at least one 'RUN pip3 install' line in the Dockerfile"
    first_pip_install_idx = min(pip_install_idxs)
    assert env_idx < first_pip_install_idx, (
        "CUDA_DEVICE_ORDER must be set before the first RUN pip3 install line, "
        "not merely before the COPY of app code."
    )

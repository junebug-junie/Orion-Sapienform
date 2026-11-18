"""Gate: vision-edge is pinned to one host card by compose device_ids.

2026-10-15: the old form (`runtime: nvidia` + a device request with no
device_ids) exposed every athena card, and YOLO_DEVICE=0 landed on the T10 by
CUDA's fastest-first order. device_ids exposes one card, so YOLO_DEVICE=0 is
that card. Checked live in the built image pinned to the P4: torch
device_count 1, YOLO on cuda:0 = Tesla P4.
"""

from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]


def _live_compose() -> str:
    text = (SERVICE_DIR / "docker-compose.yml").read_text()
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


def _env_example() -> dict:
    out = {}
    for line in (SERVICE_DIR / ".env_example").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def test_compose_pins_one_card_by_required_device_id():
    body = _live_compose()
    assert 'device_ids: ["${VISION_EDGE_GPU_DEVICE_ID:?' in body
    assert "runtime: nvidia" not in body, "CDI-mode nvidia runtime rejects a PCI bus id"
    assert "gpus: all" not in body


def test_env_example_pins_the_p4_and_keeps_yolo_container_relative():
    env = _env_example()
    assert env["VISION_EDGE_GPU_DEVICE_ID"] == "00000000:86:00.0"  # athena Tesla P4
    assert env["YOLO_DEVICE"] == "0"

"""Gate checks for the orion-kev ops contract (no GPU / no network)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_pins_gpu_and_restart() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    svc = compose["services"]["kev"]
    assert svc["restart"] == "unless-stopped"
    assert svc["container_name"] == "${PROJECT}-kev"
    devices = svc["deploy"]["resources"]["reservations"]["devices"]
    assert devices[0]["driver"] == "nvidia"
    assert "KEV_GPU_DEVICE_ID" in devices[0]["device_ids"][0]
    assert compose["networks"]["app-net"]["external"] is True


def test_env_example_has_required_keys() -> None:
    text = (ROOT / ".env_example").read_text(encoding="utf-8")
    for key in (
        "PROJECT=",
        "KEV_GPU_DEVICE_ID=",
        "KEV_HF_CACHE_HOST_DIR=",
        "KEV_HOST_PORT=",
        "KEV_MODEL=",
    ):
        assert key in text, key


def test_dockerfile_binds_all_interfaces() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'host="0.0.0.0"' in text
    assert "KEV_GIT_SHA=" in text

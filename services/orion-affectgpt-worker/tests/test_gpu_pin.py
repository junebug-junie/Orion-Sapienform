"""Gate: affectgpt-worker is pinned by GPU UUID through a required env key.

2026-10-14..23: the pin was the bare index "1". circe's GPU numbering changed, index 1 became the
agent llama-server's card, and the worker hit CUDA OOM at load and restarted 601 times. An index
can move; a UUID cannot. No card had room on 2026-10-23, so the checked-in value is empty and
compose refuses to start the worker until Juniper sets one.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

SERVICE_DIR = Path(__file__).resolve().parents[1]
UUID = re.compile(r"GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def _env_example() -> dict[str, str]:
    out = {}
    for line in (SERVICE_DIR / ".env_example").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v.split("#", 1)[0].strip()
    return out


def _device_ids() -> list[str]:
    compose = yaml.safe_load((SERVICE_DIR / "docker-compose.yml").read_text())
    ids = []
    for svc in compose["services"].values():
        for dev in svc.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices", []):
            ids += [str(i) for i in dev.get("device_ids", [])]
    return ids


def test_every_device_id_is_the_required_env_key():
    ids = _device_ids()
    assert ids, "no GPU reservation found"
    assert all(i.startswith("${AFFECTGPT_GPU_DEVICE_ID:?") for i in ids), ids


def test_env_example_key_is_empty_or_a_uuid():
    value = _env_example()["AFFECTGPT_GPU_DEVICE_ID"]
    assert value == "" or UUID.fullmatch(value), value

"""Gate: affectgpt-worker is pinned by GPU UUID through a required env key.

2026-10-14..23: the pin was the bare index "1". circe's GPU numbering changed, index 1 became the
agent llama-server's card, and the worker hit CUDA OOM at load and restarted 601 times. An index
can move; a UUID cannot. No card had room on 2026-10-23, so the checked-in value is empty and
compose refuses to start the worker until Juniper sets one.
"""
from __future__ import annotations

import re
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]


def _live(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


def _env_example() -> dict[str, str]:
    out = {}
    for line in _live((SERVICE_DIR / ".env_example").read_text()).splitlines():
        if "=" in line:
            k, v = line.strip().split("=", 1)
            out[k] = v
    return out


def test_compose_pin_is_a_required_env_key():
    text = _live((SERVICE_DIR / "docker-compose.yml").read_text())
    assert 'device_ids: ["${AFFECTGPT_GPU_DEVICE_ID:?' in text


def test_no_bare_index_pin():
    text = _live((SERVICE_DIR / "docker-compose.yml").read_text())
    assert not re.search(r'device_ids:\s*\[\s*"\d+"\s*\]', text)


def test_env_example_key_is_empty_or_a_uuid():
    value = _env_example()["AFFECTGPT_GPU_DEVICE_ID"]
    assert value == "" or re.fullmatch(r"GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", value), value

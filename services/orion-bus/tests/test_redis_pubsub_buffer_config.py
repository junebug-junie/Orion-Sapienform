from __future__ import annotations

from pathlib import Path

import yaml


def test_bus_core_redis_pubsub_output_buffer_limit() -> None:
    compose_path = Path(__file__).resolve().parents[1] / "docker-compose.yml"
    doc = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    command = doc["services"]["bus-core"]["command"]
    assert "redis-server" in command
    idx = command.index("--client-output-buffer-limit")
    assert command[idx : idx + 5] == [
        "--client-output-buffer-limit",
        "pubsub",
        "64mb",
        "32mb",
        "120",
    ]

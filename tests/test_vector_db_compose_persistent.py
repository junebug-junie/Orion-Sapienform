"""orion-vector-db must run Chroma persistently.

chromadb 0.4.x defaults to an in-memory store; PERSIST_DIRECTORY is ignored unless
IS_PERSISTENT is truthy. With it commented out, every vector-db restart silently
wiped every collection -- found live 2026-10-11 when the concept-relation writer's
candidate collection read 0 docs vs ~760 active crystallizations.
"""
from __future__ import annotations

from pathlib import Path

import yaml

COMPOSE = Path(__file__).resolve().parents[1] / "services" / "orion-vector-db" / "docker-compose.yml"


def _env() -> dict[str, str]:
    svc = yaml.safe_load(COMPOSE.read_text())["services"]["vector-db"]
    env = svc.get("environment") or {}
    if isinstance(env, list):
        env = dict(item.split("=", 1) for item in env if "=" in item)
    return {k: str(v) for k, v in env.items()}


def test_chroma_is_persistent() -> None:
    assert _env().get("IS_PERSISTENT", "").strip().lower() in {"true", "1"}


def test_persist_directory_is_the_mounted_volume() -> None:
    svc = yaml.safe_load(COMPOSE.read_text())["services"]["vector-db"]
    persist = _env().get("PERSIST_DIRECTORY")
    assert persist
    assert any(str(v).endswith(f":{persist}") for v in svc.get("volumes") or [])

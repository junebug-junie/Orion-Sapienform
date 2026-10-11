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


ENV_EXAMPLE = COMPOSE.parent / ".env_example"
BACKUP = COMPOSE.parents[2] / "scripts" / "backup" / "orion_backup_databases.py"


def test_chroma_data_lives_on_the_graphdb_drive() -> None:
    lines = [l for l in ENV_EXAMPLE.read_text().splitlines() if l.startswith("CHROMA_DATA_PATH=")]
    assert lines == ["CHROMA_DATA_PATH=${ORION_DATA_ROOT}/chroma"]


def test_backup_follows_the_live_chroma_mount() -> None:
    # The backup must resolve the folder from the container's mount at
    # PERSIST_DIRECTORY, never a hardcoded host path that can drift.
    persist = _env()["PERSIST_DIRECTORY"]
    src = BACKUP.read_text()
    assert f'container_mount_source("orion-athena-vector-db", "{persist}"' in src
    assert "collapse-mirrors/chroma" not in src

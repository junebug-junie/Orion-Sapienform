"""Regression gate: never bind-mount the operator's ~/.claude.json into a
container that spawns Orion's claude subprocess.

Claude Code rewrites ~/.claude.json by atomic replace, so a single-file bind
mount pins the inode from container start and goes stale silently. It also
leaks host-global MCP servers into Orion's turns (2026-09-06: `caveman`,
installed on the host only, produced a "failed to connect" banner in every
Orion session and the chat model misread it as a GitHub outage).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[3]
_COMPOSE_FILES = [
    _REPO / "services/orion-harness-governor/docker-compose.yml",
    _REPO / "services/orion-hub/docker-compose.yml",
]


def _bind_mount_sources(compose_path: Path) -> list[str]:
    doc = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
    sources: list[str] = []
    for svc in (doc.get("services") or {}).values():
        for vol in (svc or {}).get("volumes") or []:
            if isinstance(vol, str):
                sources.append(vol.split(":", 1)[0])
            elif isinstance(vol, dict) and vol.get("source"):
                sources.append(str(vol["source"]))
    return sources


@pytest.mark.parametrize("compose_path", _COMPOSE_FILES, ids=lambda p: p.parent.name)
def test_no_host_claude_json_bind_mount(compose_path: Path) -> None:
    offenders = [s for s in _bind_mount_sources(compose_path) if s.rstrip("/").endswith("/.claude.json")]
    assert not offenders, f"{compose_path}: ~/.claude.json must not be bind-mounted: {offenders}"


def test_gate_detects_the_removed_mount(tmp_path: Path) -> None:
    """The gate must fail on the exact line this patch removed."""
    bad = tmp_path / "docker-compose.yml"
    bad.write_text(
        "services:\n  x:\n    volumes:\n      - ${HOME}/.fcc:/root/.fcc\n"
        "      - ${HOME}/.claude.json:/root/.claude.json:ro\n",
        encoding="utf-8",
    )
    assert [s for s in _bind_mount_sources(bad) if s.endswith("/.claude.json")] == ["${HOME}/.claude.json"]

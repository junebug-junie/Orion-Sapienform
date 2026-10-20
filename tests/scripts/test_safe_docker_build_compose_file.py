"""safe_docker_build.sh --compose-file targets a service's second compose file
INSTEAD of docker-compose.yml (2026-10-20: the post-pull rebuild could not reach
services/orion-vision-host/docker-compose.circe-qwen.yml, so the Qwen caption lane
ran stale for days)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tests.scripts.test_safe_docker_build_heartbeat import SCRIPT, _init_primary_repo, _make_fake_docker


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    primary = _init_primary_repo(tmp_path, "orion-fake-service")
    (primary / "services/orion-fake-service/docker-compose.lane.yml").write_text("services: {}\n")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _make_fake_docker(fake_bin)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "ORION_AGENT_BOARD_PATH": str(tmp_path / "board.jsonl"),
    }
    return subprocess.run(
        ["sh", str(SCRIPT), "orion-fake-service", *args],
        cwd=primary, env=env, capture_output=True, text=True, timeout=30,
    )


def test_compose_file_replaces_the_default_file(tmp_path: Path) -> None:
    proc = _run(tmp_path, "--compose-file", "docker-compose.lane.yml", "config")
    assert proc.returncode == 0, proc.stderr
    assert "-f services/orion-fake-service/docker-compose.lane.yml config" in proc.stdout
    assert "docker-compose.yml" not in proc.stdout  # never merged with the main file


def test_default_file_unchanged_without_the_flag(tmp_path: Path) -> None:
    proc = _run(tmp_path, "config")
    assert proc.returncode == 0, proc.stderr
    assert "-f services/orion-fake-service/docker-compose.yml config" in proc.stdout


def test_compose_file_must_be_a_docker_compose_basename(tmp_path: Path) -> None:
    for i, bad in enumerate(("../x/docker-compose.lane.yml", "lane.yml", "docker-compose.a/b.yml", "docker-compose..yml")):
        case = tmp_path / f"case{i}"
        case.mkdir()
        proc = _run(case, "--compose-file", bad, "config")
        assert proc.returncode == 1, bad
        assert "fake-docker" not in proc.stdout


def test_missing_compose_file_is_refused(tmp_path: Path) -> None:
    proc = _run(tmp_path, "--compose-file", "docker-compose.nope.yml", "config")
    assert proc.returncode == 1
    assert "docker-compose.nope.yml" in proc.stderr

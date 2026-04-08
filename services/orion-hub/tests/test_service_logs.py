from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from scripts.service_logs import (
    ServiceLogConfig,
    build_compose_logs_command,
    collect_service_inventory,
    discover_loggable_services,
    resolve_repo_root_details,
)


def test_discover_loggable_services_only_returns_dirs_with_compose(tmp_path: Path) -> None:
    services_root = tmp_path / "services"
    services_root.mkdir()

    valid = services_root / "orion-valid"
    valid.mkdir()
    (valid / "docker-compose.yml").write_text("services: {}", encoding="utf-8")
    (valid / ".env").write_text("FOO=bar", encoding="utf-8")

    invalid_no_compose = services_root / "orion-missing"
    invalid_no_compose.mkdir()

    invalid_name = services_root / "bad name"
    invalid_name.mkdir()
    (invalid_name / "docker-compose.yml").write_text("services: {}", encoding="utf-8")

    discovered = discover_loggable_services(tmp_path)

    assert [item.name for item in discovered] == ["orion-valid"]
    assert discovered[0].service_env_file is not None


def test_build_compose_logs_command_includes_root_and_optional_service_env_files(tmp_path: Path) -> None:
    svc_dir = tmp_path / "services" / "orion-hub"
    svc_dir.mkdir(parents=True)
    compose = svc_dir / "docker-compose.yml"
    compose.write_text("services: {}", encoding="utf-8")

    cfg_no_env = ServiceLogConfig(name="orion-hub", compose_file=compose, service_env_file=None)
    cmd_no_env = build_compose_logs_command(cfg_no_env, tmp_path)
    assert cmd_no_env[:4] == ["docker", "compose", "--env-file", ".env"]
    assert "services/orion-hub/.env" not in cmd_no_env

    svc_env = svc_dir / ".env"
    svc_env.write_text("FOO=bar", encoding="utf-8")
    cfg_with_env = ServiceLogConfig(name="orion-hub", compose_file=compose, service_env_file=svc_env)
    cmd_with_env = build_compose_logs_command(cfg_with_env, tmp_path)

    assert "services/orion-hub/.env" in cmd_with_env
    assert "services/orion-hub/docker-compose.yml" in cmd_with_env
    assert cmd_with_env[-3:] == ["-f", "--no-color", "--timestamps"]


def test_discovery_uses_orion_repo_root_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    services_root = tmp_path / "services"
    service_dir = services_root / "orion-env-root"
    service_dir.mkdir(parents=True)
    (service_dir / "docker-compose.yml").write_text("services: {}", encoding="utf-8")

    monkeypatch.setenv("ORION_REPO_ROOT", str(tmp_path))

    discovered = discover_loggable_services()
    assert any(item.name == "orion-env-root" for item in discovered)


def test_repo_root_resolution_tracks_invalid_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad_root = tmp_path / "bad-root"
    bad_root.mkdir()
    monkeypatch.setenv("ORION_REPO_ROOT", str(bad_root))

    details = resolve_repo_root_details()

    assert any(str(bad_root) in checked for checked in details.checked)
    assert details.strategy in {"module-ancestor", "cwd", "fallback:/repo", "fallback:module_dir"}


def test_inventory_reports_diagnostics_when_zero_services(tmp_path: Path) -> None:
    payload = collect_service_inventory(repo_root=tmp_path)

    assert payload["services"] == []
    meta = payload["meta"]
    assert meta["count"] == 0
    assert meta["repo_root"] == os.fspath(tmp_path)
    assert meta["services_root_exists"] is False
    assert isinstance(meta["docker_available"], bool)
    assert isinstance(meta["docker_socket_exists"], bool)
    assert isinstance(meta["scan_preview"], list)

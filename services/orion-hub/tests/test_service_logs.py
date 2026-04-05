from __future__ import annotations

import sys
from pathlib import Path


HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from scripts.service_logs import ServiceLogConfig, build_compose_logs_command, discover_loggable_services


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


def test_build_compose_logs_command_includes_root_and_service_env_files(tmp_path: Path) -> None:
    svc_dir = tmp_path / "services" / "orion-hub"
    svc_dir.mkdir(parents=True)
    compose = svc_dir / "docker-compose.yml"
    compose.write_text("services: {}", encoding="utf-8")
    svc_env = svc_dir / ".env"
    svc_env.write_text("FOO=bar", encoding="utf-8")

    cfg = ServiceLogConfig(name="orion-hub", compose_file=compose, service_env_file=svc_env)
    cmd = build_compose_logs_command(cfg, tmp_path)

    assert cmd[:4] == ["docker", "compose", "--env-file", ".env"]
    assert "services/orion-hub/.env" in cmd
    assert "services/orion-hub/docker-compose.yml" in cmd
    assert cmd[-3:] == ["-f", "--no-color", "--timestamps"]

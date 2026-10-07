"""Remediation shells out to `docker compose` against the host daemon. Two
ways that silently broke it, both live 2026-10-02 (dormant only because
MESH_GUARDIAN_AUTO_REMEDIATE=false):

- the image had no docker CLI (Debian trixie's docker.io stopped shipping it)
- run from /repo, compose hands the host daemon /repo/... paths that do not
  exist on the host and relabels each recreated container's working_dir
"""
from __future__ import annotations

from pathlib import Path

import yaml

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def test_image_installs_docker_cli_and_compose_plugin() -> None:
    dockerfile = (SERVICE_ROOT / "Dockerfile").read_text()
    assert "docker-cli" in dockerfile
    assert "docker-compose" in dockerfile
    assert "docker-buildx" in dockerfile  # tier-2 `compose build`
    assert "docker compose version" in dockerfile  # build fails if the plugin is missing


def test_remediation_runs_against_the_repo_at_its_host_path() -> None:
    compose = yaml.safe_load((SERVICE_ROOT / "docker-compose.yml").read_text())
    svc = next(iter(compose["services"].values()))
    host_repo = "${ORION_HOST_REPO_ROOT:-/mnt/scripts/Orion-Sapienform}"
    assert svc["environment"]["ORION_REPO_ROOT"] == host_repo
    assert f"{host_repo}:{host_repo}:ro" in svc["volumes"]

"""Orion must act on GitHub as its own user, never as the operator.

Regression gate for the 2026-11-05 identity split: the Hub and harness-governor
used to bind-mount the operator's ~/.config/gh and ~/.ssh, so every Orion PR,
comment and push was junebug-junie (an admin that main's protection did not bind).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ORION_CONTAINERS = ("orion-hub", "orion-harness-governor")


def _volumes(service: str) -> list[str]:
    doc = yaml.safe_load((ROOT / "services" / service / "docker-compose.yml").read_text())
    # str() of a long-form dict volume still contains its source path.
    return [str(v) for svc in doc["services"].values() for v in svc.get("volumes", [])]


def _environment(service: str) -> list[str]:
    doc = yaml.safe_load((ROOT / "services" / service / "docker-compose.yml").read_text())
    out: list[str] = []
    for svc in doc["services"].values():
        env = svc.get("environment", [])
        out += [f"{k}={v}" for k, v in env.items()] if isinstance(env, dict) else [str(e) for e in env]
    return out


def test_orion_containers_do_not_mount_the_operators_github_login() -> None:
    for service in ORION_CONTAINERS:
        for vol in _volumes(service):
            assert ".config/gh" not in vol, f"{service} mounts the operator's gh login: {vol}"
            assert ".ssh" not in vol, f"{service} mounts the operator's SSH keys: {vol}"


def test_orion_containers_get_their_own_token() -> None:
    for service in ORION_CONTAINERS:
        env = _environment(service)
        assert "ORION_GITHUB_TOKEN=${ORION_GITHUB_TOKEN:-}" in env
        assert "GH_TOKEN=${ORION_GITHUB_TOKEN:-}" in env


def test_entrypoint_never_reads_the_operators_gh_hosts_file(tmp_path: Path) -> None:
    script = (ROOT / "services/orion-harness-governor/entrypoint.sh").read_text()
    code = [l for l in script.splitlines() if not l.lstrip().startswith("#")]
    assert not any("hosts.yml" in l or ".config/gh" in l for l in code)


def _run_entrypoint(home: Path, token: str | None) -> str:
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    if token is not None:
        env["ORION_GITHUB_TOKEN"] = token
    subprocess.run(
        ["sh", str(ROOT / "services/orion-harness-governor/entrypoint.sh"), "true"],
        env=env, check=True, capture_output=True,
    )
    return (home / ".fcc" / ".env").read_text()


def test_entrypoint_replaces_a_stale_operator_pat_with_orions(tmp_path: Path) -> None:
    (tmp_path / ".fcc").mkdir()
    (tmp_path / ".fcc" / ".env").write_text("KEEP=1\nGITHUB_PAT=gho_operator_old\n")
    out = _run_entrypoint(tmp_path, "ghp_orion_own")
    assert "gho_operator_old" not in out
    assert out.splitlines().count("GITHUB_PAT=ghp_orion_own") == 1
    assert "KEEP=1" in out


def test_entrypoint_without_a_token_leaves_no_pat_at_all(tmp_path: Path) -> None:
    (tmp_path / ".fcc").mkdir()
    (tmp_path / ".fcc" / ".env").write_text("KEEP=1\nGITHUB_PAT=gho_operator_old\n")
    out = _run_entrypoint(tmp_path, None)
    assert "GITHUB_PAT" not in out and "KEEP=1" in out


def test_codeowners_has_a_catch_all_for_the_repo_owner() -> None:
    lines = [l.split() for l in (ROOT / ".github/CODEOWNERS").read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    assert lines[0][0] == "*" and "@junebug-junie" in lines[0]

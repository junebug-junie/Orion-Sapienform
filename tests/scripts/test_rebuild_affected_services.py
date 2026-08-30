from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "scripts" / "rebuild_affected_services.py"


def _resolve(paths: list[str], repo_root: Path) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "--paths",
            *paths,
            "--json",
            "--repo-root",
            str(repo_root),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """Minimal repo skeleton for path-rule unit tests (no full import index)."""
    root = tmp_path / "repo"
    (root / "services" / "orion-actions").mkdir(parents=True)
    (root / "services" / "orion-actions" / "docker-compose.yml").write_text("services: {}\n")
    (root / "scripts").mkdir()
    (root / "scripts" / "rebuild_affected_services.py").write_text("# stub\n")
    shutil_copy_mapping(root)
    return root


def shutil_copy_mapping(root: Path) -> None:
    mapping_src = ROOT / "scripts" / "service_rebuild_paths.yaml"
    (root / "scripts" / "service_rebuild_paths.yaml").write_text(
        mapping_src.read_text(encoding="utf-8"), encoding="utf-8"
    )


def test_direct_service_path(mini_repo: Path) -> None:
    payload = _resolve(["services/orion-actions/app/main.py"], mini_repo)
    assert payload["services"] == ["orion-actions"]


def test_orion_curiosity_import_closure() -> None:
    payload = _resolve(["orion/curiosity/worldview.py"], ROOT)
    assert "orion-hub" in payload["services"]
    assert "orion-harness-governor" in payload["services"]


def test_orion_bus_contract_broadcast() -> None:
    payload = _resolve(["orion/bus/channels.yaml"], ROOT)
    assert "orion-hub" in payload["services"]
    assert "orion-actions" in payload["services"]
    assert len(payload["services"]) > 10


def test_root_tests_skipped() -> None:
    payload = _resolve(["tests/test_curiosity_worldview.py"], ROOT)
    assert payload["services"] == []
    assert "tests/test_curiosity_worldview.py" in payload["skipped"]


def test_sample_pull_diff() -> None:
    """User example: orion/curiosity, services/orion-actions, root tests."""
    payload = _resolve(
        [
            "orion/curiosity/worldview.py",
            "services/orion-actions/app/main.py",
            "tests/test_curiosity_worldview.py",
        ],
        ROOT,
    )
    services = set(payload["services"])
    assert services == {"orion-actions", "orion-hub", "orion-harness-governor"}


def test_unmapped_script_skipped(mini_repo: Path) -> None:
    payload = _resolve(["scripts/agent_board.py"], mini_repo)
    assert payload["services"] == []
    assert "scripts/agent_board.py" in payload["skipped"]
    assert "no script_services mapping" in payload["reasons"]["scripts/agent_board.py"]


def test_script_mapping(mini_repo: Path) -> None:
    payload = _resolve(["scripts/smoke_llm_gateway_routes.py"], mini_repo)
    # mini_repo only has orion-actions; unknown mapped services are ignored.
    assert payload["services"] == []


def test_script_mapping_real_repo() -> None:
    payload = _resolve(["scripts/smoke_llm_gateway_routes.py"], ROOT)
    assert "orion-llm-gateway" in payload["services"]
    assert "orion-hub" in payload["services"]


def test_resolve_base_ref_prefers_orig_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / "a.txt").write_text("1", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "one"], cwd=repo, check=True)
    first = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    (repo / "a.txt").write_text("2", encoding="utf-8")
    subprocess.run(["git", "commit", "-aq", "-m", "two"], cwd=repo, check=True)
    subprocess.run(["git", "update-ref", "ORIG_HEAD", first], cwd=repo, check=True)

    proc = subprocess.run(
        [sys.executable, str(MODULE), "--base", "ORIG_HEAD", "--repo-root", str(repo), "--list-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

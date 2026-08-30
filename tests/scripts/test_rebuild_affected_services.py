from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "scripts" / "rebuild_affected_services.py"


def _resolve(paths: list[str], repo_root: Path, host: str | None = None) -> dict:
    cmd = [
        sys.executable,
        str(MODULE),
        "--paths",
        *paths,
        "--json",
        "--repo-root",
        str(repo_root),
    ]
    if host:
        cmd.extend(["--host", host])
    proc = subprocess.run(
        cmd,
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
    assert payload["services"] == []
    assert "orion/bus/channels.yaml" in payload["skipped"]


def test_orion_graph_not_in_import_allowlist() -> None:
    payload = _resolve(["orion/graph/analytics.py"], ROOT)
    assert payload["services"] == []
    assert "not in orion_import_packages" in payload["reasons"]["orion/graph/analytics.py"]


def test_exclude_services_respected(mini_repo: Path) -> None:
    excludes = ROOT / "mesh-utilities" / "common" / "exclude_services.txt"
    (mini_repo / "mesh-utilities" / "common").mkdir(parents=True)
    (mini_repo / "mesh-utilities" / "common" / "exclude_services.txt").write_text(
        "orion-actions\n", encoding="utf-8"
    )
    payload = _resolve(["services/orion-actions/app/main.py"], mini_repo)
    assert payload["services"] == []


def test_actual_pull_scope() -> None:
    """Atlas graph pull: direct service touches only (no bus blast, no graph fan-out)."""
    paths = [
        "orion/bus/channels.yaml",
        "orion/graph/analytics.py",
        "services/orion-embodiment/app/worker.py",
        "services/orion-hub/scripts/concept_atlas_routes.py",
        "services/orion-social-memory/app/main.py",
        "services/orion-ai-town/scripts/generate_descriptions.py",
    ]
    payload = _resolve(paths, ROOT)
    assert payload["services"] == ["orion-embodiment", "orion-hub", "orion-social-memory"]


def test_circe_host_allowlist_rebuilds_ai_town() -> None:
    payload = _resolve(["services/orion-ai-town/scripts/generate_descriptions.py"], ROOT, host="circe")
    assert payload["services"] == ["orion-ai-town"]
    assert payload["mesh_host"] == "circe"


def test_circe_host_allowlist_skips_athena_services() -> None:
    paths = [
        "services/orion-embodiment/app/worker.py",
        "services/orion-hub/scripts/concept_atlas_routes.py",
        "services/orion-ai-town/scripts/generate_descriptions.py",
    ]
    payload = _resolve(paths, ROOT, host="circe")
    assert payload["services"] == ["orion-ai-town"]
    assert "orion-hub" in payload["host_filtered_out"]
    assert "orion-embodiment" in payload["host_filtered_out"]


def test_athena_has_no_host_allowlist_by_default() -> None:
    payload = _resolve(["services/orion-ai-town/scripts/generate_descriptions.py"], ROOT, host="athena")
    assert payload["services"] == []
    assert payload["host_allowlist"] == []


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

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


def test_sample_pull_diff(mini_repo: Path) -> None:
    """Keep exact fan-out assertions on an explicit import graph, not growing ROOT."""
    (mini_repo / "orion/curiosity").mkdir(parents=True)
    (mini_repo / "orion/curiosity/worldview.py").write_text("# changed package\n")
    for name in ("orion-cortex-exec", "orion-hub", "orion-harness-governor"):
        service = mini_repo / "services" / name
        service.mkdir()
        (service / "docker-compose.yml").write_text("services: {}\n")
        (service / "consumer.py").write_text("import orion.curiosity\n")
    payload = _resolve(
        [
            "orion/curiosity/worldview.py",
            "services/orion-actions/app/main.py",
            "tests/test_curiosity_worldview.py",
        ],
        mini_repo,
    )
    services = set(payload["services"])
    assert services == {"orion-actions", "orion-cortex-exec", "orion-hub", "orion-harness-governor"}


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


def test_athena_pull_never_rebuilds_hecate_vllm_worker() -> None:
    payload = _resolve([
        "services/orion-vllm-host/app/main.py",
        "services/orion-vllm-host/Dockerfile.sm70",
        "services/orion-vllm-host/docker-compose.yml",
        "config/llm_profiles.yaml",
    ], ROOT, host="athena")
    assert "orion-vllm-host" not in payload["services"]


def test_vllm_worker_requires_explicit_compose_profile() -> None:
    import yaml
    compose = yaml.safe_load((ROOT / "services/orion-vllm-host/docker-compose.yml").read_text())
    assert compose["services"]["vllm-host"]["profiles"] == ["vllm-worker"]


# --- Baked / bind-mounted repo paths (PR #2677 regression) -----------------


def test_pr2677_baked_config_rebuilds_every_image_that_copies_it() -> None:
    """config/vision_zones.yaml is COPY'd into three images; the old classifier
    skipped it as `under config` and rebuilt none of them."""
    payload = _resolve(["config/vision_zones.yaml"], ROOT, host="athena")
    for svc in ("orion-vision-host", "orion-sql-writer", "orion-vision-council"):
        assert svc in payload["services"], payload["reasons"]
    assert "config/vision_zones.yaml" not in payload["skipped"]
    assert "orion-vision-host" not in payload["recreate"]


def test_mounted_repo_config_marks_service_for_recreate() -> None:
    payload = _resolve(["config/vision_frame_router.yaml"], ROOT, host="athena")
    assert "orion-vision-frame-router" in payload["services"]
    assert "orion-vision-frame-router" in payload["recreate"]


def _service(root: Path, name: str, compose: str, dockerfile: str | None = None) -> None:
    svc = root / "services" / name
    svc.mkdir(parents=True, exist_ok=True)
    (svc / "docker-compose.yml").write_text(compose)
    if dockerfile is not None:
        (svc / "Dockerfile").write_text(dockerfile)


def test_dependency_sources_parse_copy_add_and_mounts(mini_repo: Path) -> None:
    _service(
        mini_repo,
        "orion-baker",
        "services:\n"
        "  baker:\n"
        "    build:\n"
        "      context: ../..\n"
        "      dockerfile: services/orion-baker/Dockerfile\n",
        "FROM python:3.12 AS base\n"
        "COPY services/orion-baker/app ./app\n"
        "COPY orion ./orion\n"
        "COPY --chown=1000:1000 config/zones.yaml \\\n"
        "     config/other.yaml /app/config/\n"
        "ADD [\"config/rules/*.yaml\", \"/app/rules/\"]\n"
        "COPY --from=base /opt/thing /opt/thing\n"
        "COPY\tconfig/tabbed.yaml /app/\n"
        "COPY config/a.yaml \\\n"
        "# a comment inside a continuation\n"
        "     /app/a.yaml\n"
        "COPY services/${SVC}/x config/${NAME}.yaml /app/\n"
        "COPY config /app/config\n"
        "COPY . /repo\n",
    )
    _service(
        mini_repo,
        "orion-mounter",
        "services:\n"
        "  mounter:\n"
        "    image: python:3.12\n"
        "    volumes:\n"
        "      - ../../config/router.yaml:/app/config/router.yaml:ro\n"
        "      - ${ORION_HOST_REPO_ROOT:-/mnt/scripts/Orion-Sapienform}/config/corpus:/app/corpus:ro\n"
        "      - ${ORION_HOST_REPO_ROOT:-/mnt/scripts/Orion-Sapienform}/orion:/app/orion\n"
        "      - ${ORION_HOST_REPO_ROOT:-/mnt/scripts/Orion-Sapienform}:/repo:ro\n"
        "      - ${SOME_OTHER_DIR}/x:/x\n"
        "      - named_volume:/data\n"
        "      - type: bind\n"
        "        source: ../../config/long.yaml\n"
        "        target: /app/long.yaml\n",
    )

    def services_for(path: str) -> tuple[list[str], list[str]]:
        payload = _resolve([path], mini_repo, host="athena")
        return payload["services"], payload["recreate"]

    assert services_for("config/zones.yaml") == (["orion-baker"], [])
    assert services_for("config/other.yaml") == (["orion-baker"], [])
    assert services_for("config/rules/a.yaml") == (["orion-baker"], [])
    assert services_for("config/tabbed.yaml") == (["orion-baker"], [])
    assert services_for("config/a.yaml") == (["orion-baker"], [])
    assert services_for("config/router.yaml") == (["orion-mounter"], ["orion-mounter"])
    assert services_for("config/corpus/doc.md") == (["orion-mounter"], ["orion-mounter"])
    assert services_for("config/long.yaml") == (["orion-mounter"], ["orion-mounter"])
    # --from stage copies, whole-repo COPY/mounts and orion/ do not fan out here;
    # orion/ keeps its own narrow import-graph policy.
    assert services_for("config/unrelated.yaml") == ([], [])
    assert services_for("docs/readme.md") == ([], [])
    assert services_for("orion/vision/zones.py") == ([], [])


def test_dockerfile_in_other_service_dir_is_a_dependency(mini_repo: Path) -> None:
    _service(
        mini_repo,
        "orion-twin",
        "services:\n"
        "  twin:\n"
        "    build:\n"
        "      context: ../..\n"
        "      dockerfile: services/orion-actions/Dockerfile\n",
    )
    (mini_repo / "services" / "orion-actions" / "Dockerfile").write_text("FROM python:3.12\n")
    payload = _resolve(["services/orion-actions/Dockerfile"], mini_repo, host="athena")
    assert payload["services"] == ["orion-actions", "orion-twin"]


# --- Running profile-gated services -----------------------------------------


def _profile_services(repo: Path, service: str, ps_lines: list[str], tmp_path: Path, *skip: str) -> list[str]:
    ps_file = tmp_path / "docker-ps.txt"
    ps_file.write_text("\n".join(ps_lines) + "\n")
    cmd = [
        sys.executable,
        str(MODULE),
        "--repo-root",
        str(repo),
        "--running-profile-services",
        service,
        "--docker-ps-file",
        str(ps_file),
    ]
    for profile in skip:
        cmd.extend(["--skip-profile", profile])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


_RETINA_COMPOSE = (ROOT / "services/orion-vision-retina/docker-compose.yml").resolve().as_posix()


def test_running_walkway_profile_service_is_reported(tmp_path: Path) -> None:
    ps = [
        f"orion-vision-retina\torion-vision-retina\t{_RETINA_COMPOSE}",
        f"orion-vision-retina\torion-vision-retina-walkway\t{_RETINA_COMPOSE}",
    ]
    assert _profile_services(ROOT, "orion-vision-retina", ps, tmp_path) == [
        "walkway\torion-vision-retina-walkway"
    ]


def test_stopped_profile_service_is_never_started(tmp_path: Path) -> None:
    ps = [f"orion-vision-retina\torion-vision-retina\t{_RETINA_COMPOSE}"]
    assert _profile_services(ROOT, "orion-vision-retina", ps, tmp_path) == []


def test_profile_match_requires_the_same_compose_file(tmp_path: Path) -> None:
    # Same compose service name, but from another stack's compose file.
    ps = ["other\torion-vision-retina-walkway\t/srv/services/orion-other/docker-compose.yml"]
    assert _profile_services(ROOT, "orion-vision-retina", ps, tmp_path) == []


def test_skip_profile_excludes_analytics(tmp_path: Path) -> None:
    compose = (ROOT / "services/orion-analytics/docker-compose.yml").resolve().as_posix()
    ps = [f"orion-analytics\tlightdash\t{compose}"]
    assert _profile_services(ROOT, "orion-analytics", ps, tmp_path) == ["analytics\tlightdash"]
    assert _profile_services(ROOT, "orion-analytics", ps, tmp_path, "analytics") == []


def test_plan_output_marks_recreate(mini_repo: Path) -> None:
    _service(
        mini_repo,
        "orion-mounter",
        "services:\n  m:\n    image: x\n    volumes:\n      - ../../config/r.yaml:/r.yaml:ro\n",
    )
    proc = subprocess.run(
        [
            sys.executable, str(MODULE), "--repo-root", str(mini_repo), "--host", "athena",
            "--plan", "--paths", "config/r.yaml", "services/orion-actions/app/main.py",
        ],
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == ["orion-actions\tbuild", "orion-mounter\trecreate"]


def test_whole_config_folder_copy_does_not_fan_out() -> None:
    """mesh-guardian/llamacpp hosts `COPY config` whole; that is not evidence
    they read any given file, so an unrelated config edit must not rebuild them."""
    payload = _resolve(["config/vision_zones.yaml"], ROOT, host="athena")
    assert "orion-mesh-guardian" not in payload["services"]
    payload = _resolve(["config/some_unrelated_new_file.yaml"], ROOT, host="athena")
    assert payload["services"] == []


def test_own_directory_mount_is_not_force_recreated(mini_repo: Path) -> None:
    _service(
        mini_repo,
        "orion-selfmount",
        "services:\n"
        "  s:\n"
        "    image: x\n"
        "    volumes:\n"
        "      - ./static:/app/static:ro\n"
        "      - ./otel/cfg.yaml:/etc/cfg.yaml:ro\n",
    )
    svc = mini_repo / "services" / "orion-selfmount"
    (svc / "static").mkdir()
    (svc / "static" / "app.js").write_text("// x\n")
    (svc / "otel").mkdir()
    (svc / "otel" / "cfg.yaml").write_text("a: 1\n")
    payload = _resolve(["services/orion-selfmount/static/app.js"], mini_repo, host="athena")
    assert payload["services"] == ["orion-selfmount"]
    assert payload["recreate"] == []
    # A single-file own mount keeps the old inode after a pull: recreate it.
    payload = _resolve(["services/orion-selfmount/otel/cfg.yaml"], mini_repo, host="athena")
    assert payload["recreate"] == ["orion-selfmount"]


def test_profile_service_started_from_another_checkout_is_left_alone(tmp_path: Path) -> None:
    other = "/srv/worktrees/Orion-x/services/orion-vision-retina/docker-compose.yml"
    ps = [f"orion-vision-retina\torion-vision-retina-walkway\t{other}"]
    assert _profile_services(ROOT, "orion-vision-retina", ps, tmp_path) == []
    overlay = f"{_RETINA_COMPOSE},/tmp/overlay.yml"
    ps = [f"orion-vision-retina\torion-vision-retina-walkway\t{overlay}"]
    assert _profile_services(ROOT, "orion-vision-retina", ps, tmp_path) == []


def test_docker_unavailable_starts_no_profile_services(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text("#!/bin/sh\nexit 1\n")
    docker.chmod(0o755)
    import os

    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}")
    env.pop("ORION_REBUILD_DOCKER_PS_FILE", None)
    proc = subprocess.run(
        [sys.executable, str(MODULE), "--running-profile-services", "orion-vision-retina"],
        capture_output=True, text=True, check=False, timeout=60, env=env,
    )
    assert proc.returncode == 0
    assert proc.stdout == ""
    assert "docker ps unavailable" in proc.stderr

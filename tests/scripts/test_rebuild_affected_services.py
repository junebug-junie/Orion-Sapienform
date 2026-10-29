from __future__ import annotations

import json
import os
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


def test_any_orion_package_fans_out_through_the_import_graph() -> None:
    """The old orion_import_packages allowlist (curiosity only) is gone."""
    payload = _resolve(["orion/graph/analytics.py"], ROOT, host="athena")
    assert "orion/graph/analytics.py" not in payload["skipped"]
    assert payload["reasons"]["orion/graph/analytics.py"].startswith("orion.graph.analytics (module)")


def test_pr2684_shared_code_rebuilds_its_importers() -> None:
    """#2684 changed only orion/situational + orion/schemas and rebuilt nothing."""
    paths = [
        "orion/situational/perception_reader.py",
        "orion/schemas/situation.py",
        "orion/situational/tests/test_street_summary.py",
        "docs/superpowers/pr-reports/2026-10-15-walkway-street-into-mind-pr.md",
    ]
    payload = _resolve(paths, ROOT, host="athena")
    assert "orion-cortex-exec" in payload["services"], payload["reasons"]
    assert "orion-hub" in payload["services"], payload["reasons"]
    # Importers only -- not the 60+ COPY-orion fleet.
    assert len(payload["services"]) < 10, payload["services"]
    assert "orion/schemas/situation.py" not in payload["skipped"]


def test_orion_tests_select_nothing() -> None:
    paths = [
        "orion/situational/tests/test_street_summary.py",
        "orion/core/bus/tests/test_anything.py",
        "orion/curiosity/test_loose_file.py",
        "orion/curiosity/conftest.py",
    ]
    payload = _resolve(paths, ROOT, host="athena")
    assert payload["services"] == []
    for path in paths:
        assert payload["reasons"][path] == "orion package tests (no auto rebuild)"


def test_orion_markdown_selects_nothing() -> None:
    payload = _resolve(["orion/curiosity/README.md"], ROOT, host="athena")
    assert payload["services"] == []


def _orion_graph_repo(root: Path) -> None:
    """orion.schemas.alpha <- orion.hubmod <- orion.farmod; one service per edge.

    NOTE: the classifier always reads the REAL scripts/service_rebuild_paths.yaml
    (not one under --repo-root). These synthetic tests rely on its
    contract_paths (orion/bus/, orion/schemas/), orion_import_hops: 1 and the
    orion.schemas.registry barrier; changing those keys changes their meaning.
    """
    files = {
        "orion/__init__.py": "",
        "orion/schemas/__init__.py": "",
        "orion/schemas/alpha.py": "X = 1\n",
        "orion/schemas/beta.py": "Y = 1\n",
        "orion/schemas/gamma.json": "{}\n",
        "orion/hubmod/__init__.py": "",
        "orion/hubmod/core.py": "from orion.schemas.alpha import X\n",
        "orion/hubmod/rel.py": "from ..schemas import beta\n",
        "orion/farmod/__init__.py": "from orion.hubmod.core import X\n",
        "orion/datapkg/__init__.py": "",
        "orion/datapkg/loader.py": "LOADER = 1\n",
        "orion/datapkg/prompts/thing.j2": "hi\n",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    imports = {
        "svc-direct": "from orion.schemas.alpha import X\n",
        "svc-one-hop": "import orion.hubmod.core\n",
        "svc-two-hops": "from orion.farmod import X\n",
        "svc-other-schema": "from orion.schemas import beta\n",
        "svc-rel-hop": "from orion.hubmod import rel\n",
        "svc-loader": "from orion.datapkg.loader import LOADER\n",
        "svc-evals-only": None,
    }
    for name, text in imports.items():
        svc = root / "services" / name
        (svc / "app").mkdir(parents=True)
        (svc / "docker-compose.yml").write_text("services: {}\n")
        (svc / "Dockerfile").write_text("FROM python:3.12\nCOPY orion /app/orion\n")
        if text is not None:
            (svc / "app" / "main.py").write_text(text)
    evals = root / "services" / "svc-evals-only"
    (evals / "evals").mkdir()
    (evals / "evals" / "run_eval.py").write_text("from orion.schemas.alpha import X\n")
    (evals / "tests").mkdir()
    (evals / "tests" / "test_x.py").write_text("from orion.schemas.alpha import X\n")


def test_schema_module_selects_only_its_importers(mini_repo: Path) -> None:
    _orion_graph_repo(mini_repo)
    payload = _resolve(["orion/schemas/alpha.py"], mini_repo, host="athena")
    # Direct importer + importer of a module that imports it (one hop).
    # Not two hops, not a sibling schema's importer, not evals/tests-only.
    assert payload["services"] == ["svc-direct", "svc-one-hop"], payload["reasons"]


def test_own_package_closure_then_one_hop_and_registry_barrier(mini_repo: Path) -> None:
    """A package is one unit (deep intra-package chains count); the schema
    registry is a consumer but never walked through (it reaches every bus
    service via orion.core.bus)."""
    _orion_graph_repo(mini_repo)
    extra = {
        "orion/schemas/delta.py": "from orion.schemas.alpha import X\n",
        "orion/schemas/eps.py": "from orion.schemas.delta import X\n",
        "orion/otherpkg/__init__.py": "",
        "orion/otherpkg/deep.py": "from orion.schemas.eps import X\n",
        "orion/schemas/registry.py": "from orion.schemas.alpha import X\n",
        "orion/core/__init__.py": "",
        "orion/core/bus.py": "import orion.schemas.registry\n",
    }
    for rel, text in extra.items():
        (mini_repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (mini_repo / rel).write_text(text)
    for name, text in {
        "svc-deep": "from orion.otherpkg.deep import X\n",
        "svc-registry": "from orion.schemas.registry import X\n",
        "svc-bus": "import orion.core.bus\n",
    }.items():
        svc = mini_repo / "services" / name
        (svc / "app").mkdir(parents=True)
        (svc / "docker-compose.yml").write_text("services: {}\n")
        (svc / "app" / "main.py").write_text(text)
    payload = _resolve(["orion/schemas/alpha.py"], mini_repo, host="athena")
    assert payload["services"] == ["svc-deep", "svc-direct", "svc-one-hop", "svc-registry"], payload["reasons"]


def test_hop_is_closed_inside_the_package_it_lands_in(mini_repo: Path) -> None:
    """pkgb.x imports alpha (the hop); pkgb.y reaches it through pkgb.x. A
    service importing only pkgb.y runs alpha and must rebuild (review finding:
    orion.core.activation_decay <- substrate.activation <- falkor_store)."""
    _orion_graph_repo(mini_repo)
    for rel, text in {
        "orion/pkgb/__init__.py": "",
        "orion/pkgb/x.py": "from orion.schemas.alpha import X\n",
        "orion/pkgb/y.py": "from orion.pkgb.x import X\n",
    }.items():
        (mini_repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (mini_repo / rel).write_text(text)
    svc = mini_repo / "services" / "svc-sibling"
    (svc / "app").mkdir(parents=True)
    (svc / "docker-compose.yml").write_text("services: {}\n")
    (svc / "app" / "main.py").write_text("from orion.pkgb.y import X\n")
    payload = _resolve(["orion/schemas/alpha.py"], mini_repo, host="athena")
    assert "svc-sibling" in payload["services"], payload["reasons"]
    assert "svc-two-hops" not in payload["services"]


def test_real_repo_hop_closure_reaches_cortex_exec() -> None:
    payload = _resolve(["orion/core/activation_decay.py"], ROOT, host="athena")
    assert "orion-cortex-exec" in payload["services"], payload["reasons"]


def test_core_data_file_without_reader_entry_does_not_storm() -> None:
    payload = _resolve(["orion/core/storage/sql/some_new_table.sql"], ROOT, host="athena")
    assert payload["services"] == []
    assert "bus core" in payload["reasons"]["orion/core/storage/sql/some_new_table.sql"]


def test_data_reader_entries_can_name_services() -> None:
    payload = _resolve(["orion/core/storage/sql/graphiti_projection.sql"], ROOT, host="athena")
    assert payload["services"] == ["orion-graphiti-adapter"], payload["reasons"]


def test_orion_eval_fixtures_and_makefiles_select_nothing() -> None:
    paths = [
        "orion/substrate/evals/fixtures/repair_pressure/grounding_negative.json",
        "orion/autonomy/evals/fixtures/agency_episode_metadata.json",
        "orion/cognition/Makefile",
    ]
    payload = _resolve(paths, ROOT, host="athena")
    assert payload["services"] == []


def test_missing_pyyaml_exits_2_instead_of_dropping_skip_rules() -> None:
    code = (
        "import sys, runpy; sys.modules['yaml'] = None; "
        f"sys.argv = ['x', '--paths', 'orion/bus/channels.yaml', '--list-only']; "
        f"runpy.run_path({str(MODULE)!r}, run_name='__main__')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "PyYAML is required" in proc.stderr
    assert proc.stdout == ""


def test_relative_import_counts_as_an_edge(mini_repo: Path) -> None:
    _orion_graph_repo(mini_repo)
    payload = _resolve(["orion/schemas/beta.py"], mini_repo, host="athena")
    assert payload["services"] == ["svc-other-schema", "svc-rel-hop"], payload["reasons"]


def test_package_init_change_reaches_every_submodule_importer(mini_repo: Path) -> None:
    _orion_graph_repo(mini_repo)
    payload = _resolve(["orion/hubmod/__init__.py"], mini_repo, host="athena")
    # farmod imports orion.hubmod.core, which runs orion/hubmod/__init__.py: one hop.
    assert payload["services"] == ["svc-one-hop", "svc-rel-hop", "svc-two-hops"], payload["reasons"]


def test_data_file_maps_to_its_top_level_package(mini_repo: Path) -> None:
    _orion_graph_repo(mini_repo)
    payload = _resolve(["orion/datapkg/prompts/thing.j2"], mini_repo, host="athena")
    assert payload["services"] == ["svc-loader"], payload["reasons"]


def test_unresolvable_contract_data_file_falls_back_to_copy_orion_fleet(mini_repo: Path) -> None:
    _orion_graph_repo(mini_repo)
    proc = subprocess.run(
        [sys.executable, str(MODULE), "--paths", "orion/schemas/gamma.json", "--json",
         "--repo-root", str(mini_repo), "--host", "athena"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload["services"]) == 7
    assert "readers unresolvable" in proc.stderr


def test_recall_profiles_rebuild_their_cross_package_readers() -> None:
    payload = _resolve(["orion/recall/profiles/chat.general.v1.yaml"], ROOT, host="athena")
    assert "orion-cortex-exec" in payload["services"], payload["reasons"]


def test_dot_prefixed_paths_are_not_mangled() -> None:
    payload = _resolve([".github/workflows/x.yml", "./.env"], ROOT, host="athena")
    assert payload["services"] == []
    assert payload["reasons"][".github/workflows/x.yml"] == "under .github (no auto rebuild)"
    assert ".env" in payload["reasons"]


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
        (service / "consumer.py").write_text("import orion.curiosity.worldview\n")
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


@pytest.mark.parametrize("path", ["orion/dig/__main__.py", "orion/body_atlas/names.py", "orion/self_inventory.py"])
def test_cli_packages_orion_runs_in_the_harness_rebuild_it(path: str) -> None:
    # No service imports these; Orion runs them through Bash in the harness. PR #2951 rebuilt nothing.
    assert "orion-harness-governor" in _resolve([path], ROOT)["services"]


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


# --- GPU-pool launch targets held stopped (2026-10-16 diffusion-host incident) ---


def _pool_held(
    repo: Path, service: str, ps_lines: list[str] | None, tmp_path: Path, expect_rc: int = 0
) -> list[str]:
    cmd = [sys.executable, str(MODULE), "--repo-root", str(repo), "--pool-held-services", service]
    env = None
    if ps_lines is not None:
        ps_file = tmp_path / "docker-ps.txt"
        ps_file.write_text("\n".join(ps_lines) + "\n")
        cmd.extend(["--docker-ps-file", str(ps_file)])
    else:
        # docker unreadable: point docker at a binary that fails.
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        (fake_bin / "docker").write_text("#!/bin/sh\nexit 1\n")
        (fake_bin / "docker").chmod(0o755)
        env = {**os.environ, "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}"}
        env.pop("ORION_REBUILD_DOCKER_PS_FILE", None)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60, env=env)
    assert proc.returncode == expect_rc, proc.stderr
    return proc.stdout.splitlines()


_DIFFUSION_COMPOSE = (ROOT / "services/orion-diffusion-host/docker-compose.yml").resolve().as_posix()


def test_real_pool_config_names_diffusion_host_as_a_launch_target(tmp_path: Path) -> None:
    # Pins the live wiring: config/gpu_pool.yaml roles.diffusion.launch -> diffusion-host.
    assert _pool_held(ROOT, "orion-diffusion-host", [], tmp_path) == ["hold\tdiffusion-host"]


def test_running_pool_target_is_not_held(tmp_path: Path) -> None:
    ps = [f"orion-diffusion-host\tdiffusion-host\t{_DIFFUSION_COMPOSE}"]
    assert _pool_held(ROOT, "orion-diffusion-host", ps, tmp_path) == ["running\tdiffusion-host"]


def test_running_target_from_another_checkout_path_counts_as_running(tmp_path: Path) -> None:
    # circe's controller runs compose from a /repo mount; the label then carries that prefix.
    ps = ["orion-diffusion-host\tdiffusion-host\t/repo/services/orion-diffusion-host/docker-compose.yml"]
    assert _pool_held(ROOT, "orion-diffusion-host", ps, tmp_path) == ["running\tdiffusion-host"]


def test_same_name_from_another_stack_does_not_count_as_running(tmp_path: Path) -> None:
    ps = ["other\tdiffusion-host\t/srv/services/orion-other/docker-compose.yml"]
    assert _pool_held(ROOT, "orion-diffusion-host", ps, tmp_path) == ["hold\tdiffusion-host"]


def test_malformed_pool_config_exits_3(mini_repo: Path, tmp_path: Path) -> None:
    (mini_repo / "config").mkdir(exist_ok=True)
    (mini_repo / "config" / "gpu_pool.yaml").write_text("roles: [unclosed\n")
    _pool_held(mini_repo, "orion-actions", [], tmp_path, expect_rc=3)


def test_docker_unreadable_holds_every_pool_target(tmp_path: Path) -> None:
    assert _pool_held(ROOT, "orion-diffusion-host", None, tmp_path) == ["hold\tdiffusion-host"]


def test_profile_launch_target_is_left_to_the_profile_rule(tmp_path: Path) -> None:
    # atlas-agent-burst lives behind compose profile agent-burst: a plain `up` never starts it.
    assert _pool_held(ROOT, "orion-llamacpp-host", [], tmp_path) == []


def test_non_pool_service_prints_nothing(tmp_path: Path) -> None:
    assert _pool_held(ROOT, "orion-vision-retina", [], tmp_path) == []


def test_hold_lists_the_stacks_other_plain_services_to_start(mini_repo: Path, tmp_path: Path) -> None:
    (mini_repo / "config").mkdir(exist_ok=True)
    (mini_repo / "config" / "gpu_pool.yaml").write_text(
        "roles:\n  gen:\n    launch: {actuator: h, compose: services/orion-gen/docker-compose.yml, service: gen}\n"
    )
    _service(
        mini_repo,
        "orion-gen",
        "services:\n  gen:\n    image: x\n  sidecar:\n    image: y\n  extra:\n    image: z\n    profiles: [p]\n",
    )
    assert _pool_held(mini_repo, "orion-gen", [], tmp_path) == ["hold\tgen", "start\tsidecar"]


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


def test_profile_service_from_a_deleted_worktree_is_adopted(tmp_path: Path) -> None:
    """2026-10-20: orion-energy-portal ran from a pruned worktree and was skipped on
    every pull; only main can ever redeploy it, so it is adopted."""
    gone = (ROOT.parent / f"{ROOT.name}-pruned-worktree").as_posix()
    ps = [f"orion-vision-retina\torion-vision-retina-walkway\t{gone}/services/orion-vision-retina/docker-compose.yml"]
    assert _profile_services(ROOT, "orion-vision-retina", ps, tmp_path) == ["walkway\torion-vision-retina-walkway"]
    # Another compose project would clash with main's on container_name: left alone.
    ps = [f"custom\torion-vision-retina-walkway\t{gone}/services/orion-vision-retina/docker-compose.yml"]
    assert _profile_services(ROOT, "orion-vision-retina", ps, tmp_path) == []


# --- Second compose files (2026-10-20 Qwen caption lane incident) -------------


def _extra(service: str, ps_lines: list[str], tmp_path: Path, repo: Path = ROOT) -> list[str]:
    ps_file = tmp_path / "docker-ps.txt"
    ps_file.write_text("\n".join(ps_lines) + "\n")
    proc = subprocess.run(
        [sys.executable, str(MODULE), "--repo-root", str(repo), "--running-extra-compose", service,
         "--docker-ps-file", str(ps_file)],
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


_QWEN = (ROOT / "services/orion-vision-host/docker-compose.circe-qwen.yml").resolve().as_posix()
_QWEN_PROJECT = "orion-vision-host-qwen"  # the file's own `name:`
_ATLAS = (ROOT / "services/orion-llamacpp-host/docker-compose.atlas-workers.yml").resolve().as_posix()


def test_running_qwen_lane_is_rebuilt_from_its_own_file(tmp_path: Path) -> None:
    ps = [f"{_QWEN_PROJECT}\tvision-host-circe-qwen\t{_QWEN}"]
    assert _extra("orion-vision-host", ps, tmp_path) == ["docker-compose.circe-qwen.yml\tvision-host-circe-qwen"]


def test_stopped_extra_compose_lane_is_never_started(tmp_path: Path) -> None:
    main = (ROOT / "services/orion-vision-host/docker-compose.yml").resolve().as_posix()
    assert _extra("orion-vision-host", [f"orion-vision-host\tvision-host\t{main}"], tmp_path) == []


def test_extra_compose_from_another_place_or_project_is_left_alone(tmp_path: Path) -> None:
    for project, files in (
        (_QWEN_PROJECT, "/srv/elsewhere/services/orion-vision-host/docker-compose.circe-qwen.yml"),
        (_QWEN_PROJECT, f"{_QWEN},/tmp/overlay.yml"),  # overlay
        ("orion-vision-host", _QWEN),  # a different project: main's up would clash
    ):
        assert _extra("orion-vision-host", [f"{project}\tvision-host-circe-qwen\t{files}"], tmp_path) == []


def test_extra_compose_from_a_deleted_worktree_is_adopted(tmp_path: Path) -> None:
    gone = (ROOT.parent / f"{ROOT.name}-pruned-worktree").as_posix()
    ps = [f"{_QWEN_PROJECT}\tvision-host-circe-qwen\t{gone}/services/orion-vision-host/docker-compose.circe-qwen.yml"]
    assert _extra("orion-vision-host", ps, tmp_path) == ["docker-compose.circe-qwen.yml\tvision-host-circe-qwen"]
    # No separator after the repo name: an unrelated clone, not a worktree of this repo.
    other = (ROOT.parent / f"{ROOT.name}X").as_posix()
    ps = [f"{_QWEN_PROJECT}\tvision-host-circe-qwen\t{other}/services/orion-vision-host/docker-compose.circe-qwen.yml"]
    assert _extra("orion-vision-host", ps, tmp_path) == []


def _sibling_repo(tmp_path: Path) -> tuple[Path, Path]:
    """tmp/Repo (main) + tmp/Repo-wt (a LIVE worktree), each with one two-service lane file."""
    lane = "services:\n  lane-a: {image: x}\n  lane-b: {image: x}\n"
    roots = []
    for name in ("Repo", "Repo-wt"):
        d = tmp_path / name / "services" / "orion-x"
        d.mkdir(parents=True)
        (d / "docker-compose.yml").write_text("services:\n  main: {image: x}\n  side: {image: x, profiles: [p]}\n")
        (d / "docker-compose.lane.yml").write_text(lane)
        roots.append(tmp_path / name)
    return roots[0], roots[1]


def test_live_sibling_worktree_deploys_are_never_taken_over(tmp_path: Path) -> None:
    """A path under the repo's worktree prefix that still EXISTS is a live worktree
    deploy: main's up would silently replace it."""
    repo, wt = _sibling_repo(tmp_path)
    lane = (wt / "services/orion-x/docker-compose.lane.yml").as_posix()
    assert _extra("orion-x", [f"orion-x\tlane-a\t{lane}"], tmp_path, repo) == []
    prof = (wt / "services/orion-x/docker-compose.yml").as_posix()
    assert _profile_services(repo, "orion-x", [f"orion-x\tside\t{prof}"], tmp_path) == []


def test_only_the_running_services_of_an_extra_file_are_named(tmp_path: Path) -> None:
    repo, _wt = _sibling_repo(tmp_path)
    lane = (repo / "services/orion-x/docker-compose.lane.yml").resolve().as_posix()
    assert _extra("orion-x", [f"orion-x\tlane-b\t{lane}"], tmp_path, repo) == ["docker-compose.lane.yml\tlane-b"]


def test_extra_file_that_needs_its_own_env_file_is_left_alone(tmp_path: Path) -> None:
    """orion-bus/docker-compose.atlas-standby.yml runs with .env.atlas-standby; the
    wrapper always loads services/<svc>/.env, so it must never redeploy it."""
    standby = (ROOT / "services/orion-bus/docker-compose.atlas-standby.yml").resolve().as_posix()
    import yaml

    project = (yaml.safe_load(Path(standby).read_text()) or {}).get("name") or "orion-bus"
    assert _extra("orion-bus", [f"{project}\tbus-core-standby\t{standby}"], tmp_path) == []


def test_gpu_pool_launch_files_are_left_to_the_lane_controller(tmp_path: Path) -> None:
    ps = [
        f"orion-llamacpp-host\tatlas-agent\t{_ATLAS}",
        # The controller's own container path: missing on the host, but not an orphan.
        "orion-llamacpp-host\tatlas-agent-burst\t/repo/services/orion-llamacpp-host/docker-compose.atlas-workers.yml",
    ]
    assert _extra("orion-llamacpp-host", ps, tmp_path) == []


def test_controller_path_is_not_an_orphan_even_without_the_pool_skip() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import rebuild_affected_services as rebuild

    rows = [("orion-llamacpp-host", "atlas-agent-burst", "/repo/services/orion-llamacpp-host/docker-compose.atlas-workers.yml")]
    assert rebuild.running_extra_compose_files(ROOT, "orion-llamacpp-host", rows, set()) == []


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


# 2026-10-15: PR #2698 only ADDED a hand-applied migration file, and the
# post-merge rebuild recreated orion-athena-sql-db (unclean Postgres shutdown,
# fleet-wide "not yet accepting connections"). A migration is applied with
# docker exec ... psql, never by a container.
_INCIDENT_MIGRATION = "services/orion-sql-db/manual_migration_vision_confirm_asks_v1.sql"


def test_migration_only_change_rebuilds_nothing() -> None:
    payload = _resolve([_INCIDENT_MIGRATION], ROOT, host="athena")
    assert payload["services"] == []
    assert payload["skipped"] == [_INCIDENT_MIGRATION]
    reason = payload["reasons"][_INCIDENT_MIGRATION]
    assert "hand-applied SQL migration" in reason
    assert "never by recreating the DB container" in reason
    assert payload["migrations"] == [_INCIDENT_MIGRATION]


@pytest.mark.parametrize(
    "path",
    [
        "services/orion-sql-db/collapse_dump.sql",
        "services/orion-sql-db/README.md",
        "services/orion-sql-db/.gitignore",
    ],
)
def test_files_the_db_container_never_reads_rebuild_nothing(path: str) -> None:
    payload = _resolve([path], ROOT, host="athena")
    assert payload["services"] == []
    assert payload["migrations"] == []
    assert "orion-sql-db only rebuilds on docker-compose.yml" in payload["reasons"][path]


def test_sql_db_env_template_does_not_recreate_postgres() -> None:
    path = "services/orion-sql-db/.env_example"
    payload = _resolve([path], ROOT, host="athena")
    assert payload["services"] == []
    assert "sync_local_env_from_example.py" in payload["reasons"][path]


def test_rollback_migration_rebuilds_nothing() -> None:
    path = "services/orion-sql-db/manual_migration_episode_memory_v1_rollback.sql"
    payload = _resolve([path], ROOT, host="athena")
    assert payload["services"] == []
    # Reported (the post-pull report labels it never-apply), never rebuilt.
    assert payload["migrations"] == [path]


def test_migration_glob_star_does_not_cross_a_directory(mini_repo: Path) -> None:
    path = "services/orion-sql-db/sub/manual_migration_x.sql"
    (mini_repo / "services" / "orion-sql-db").mkdir(parents=True)
    (mini_repo / "services" / "orion-sql-db" / "docker-compose.yml").write_text("services: {}\n")
    payload = _resolve([path], mini_repo, host="athena")
    assert payload["migrations"] == []


def test_another_services_mount_does_not_drag_postgres_along(mini_repo: Path) -> None:
    """Only the owning service's own mount/COPY makes a skipped sql-db file count as consumed."""
    db = mini_repo / "services" / "orion-sql-db"
    db.mkdir(parents=True)
    (db / "docker-compose.yml").write_text("services:\n  orion-sql-db:\n    image: postgres:15\n")
    (db / "seed.sql").write_text("SELECT 1;\n")
    other = mini_repo / "services" / "orion-seeder"
    other.mkdir(parents=True)
    (other / "docker-compose.yml").write_text(
        "services:\n"
        "  orion-seeder:\n"
        "    image: alpine\n"
        "    volumes:\n"
        "      - ../orion-sql-db/seed.sql:/seed.sql:ro\n"
    )
    payload = _resolve(["services/orion-sql-db/seed.sql"], mini_repo, host="athena")
    assert payload["services"] == ["orion-seeder"]


def test_migrations_are_not_reported_on_a_host_without_the_database() -> None:
    payload = _resolve([_INCIDENT_MIGRATION], ROOT, host="circe")
    assert payload["services"] == []
    assert payload["migrations"] == []


@pytest.mark.parametrize(
    "path",
    [
        "services/orion-sql-db/docker-compose.yml",
        "services/orion-sql-db/Dockerfile",
    ],
)
def test_sql_db_container_inputs_still_rebuild(path: str) -> None:
    payload = _resolve([path, _INCIDENT_MIGRATION], ROOT, host="athena")
    assert payload["services"] == ["orion-sql-db"]
    assert payload["migrations"] == [_INCIDENT_MIGRATION]


def test_a_bind_mounted_sql_file_still_rebuilds_despite_the_skip_glob(mini_repo: Path) -> None:
    """If orion-sql-db ever mounts init SQL, the mount wins over the glob skips."""
    svc = mini_repo / "services" / "orion-sql-db"
    (svc / "initdb").mkdir(parents=True)
    (svc / "initdb" / "01_init.sql").write_text("SELECT 1;\n")
    (svc / "docker-compose.yml").write_text(
        "services:\n"
        "  orion-sql-db:\n"
        "    image: postgres:15\n"
        "    volumes:\n"
        "      - ./initdb:/docker-entrypoint-initdb.d:ro\n"
    )
    payload = _resolve(["services/orion-sql-db/initdb/01_init.sql"], mini_repo, host="athena")
    assert payload["services"] == ["orion-sql-db"]
    assert payload["skipped"] == []


def test_a_mounted_migration_glob_path_still_rebuilds(mini_repo: Path) -> None:
    """Same guard for hand_applied_migration_globs: a mounted migration is consumed."""
    svc = mini_repo / "services" / "orion-sql-db"
    svc.mkdir(parents=True)
    (svc / "manual_migration_x.sql").write_text("SELECT 1;\n")
    (svc / "docker-compose.yml").write_text(
        "services:\n"
        "  orion-sql-db:\n"
        "    image: postgres:15\n"
        "    volumes:\n"
        "      - ./manual_migration_x.sql:/docker-entrypoint-initdb.d/x.sql:ro\n"
    )
    payload = _resolve(["services/orion-sql-db/manual_migration_x.sql"], mini_repo, host="athena")
    assert payload["services"] == ["orion-sql-db"]
    assert payload["recreate"] == ["orion-sql-db"]
    assert payload["skipped"] == []


def test_migrations_flag_prints_paths() -> None:
    proc = subprocess.run(
        [sys.executable, str(MODULE), "--paths", _INCIDENT_MIGRATION, "--migrations", "--host", "athena"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [_INCIDENT_MIGRATION]


def test_migrations_flag_omits_a_deleted_migration(mini_repo: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "--paths",
            "services/orion-sql-db/manual_migration_gone.sql",
            "--migrations",
            "--repo-root",
            str(mini_repo),
            "--host",
            "athena",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


def _commit_pair(tmp_path: Path, before: str, after: str, name: str = "orion/mod.py") -> tuple[Path, str]:
    repo = tmp_path / "repo"
    (repo / name).parent.mkdir(parents=True)
    for cmd in (["init", "-q", "-b", "main"], ["config", "user.email", "t@e.com"], ["config", "user.name", "T"]):
        subprocess.run(["git", *cmd], cwd=repo, check=True)
    (repo / name).write_text(before, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "one"], cwd=repo, check=True)
    first = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    (repo / name).write_text(after, encoding="utf-8")
    subprocess.run(["git", "commit", "-aq", "-m", "two"], cwd=repo, check=True)
    return repo, first


def _changed(tmp_path: Path, before: str, after: str) -> list[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("rebuild_affected_services_under_test", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    repo, first = _commit_pair(tmp_path, before, after)
    return mod.changed_files(repo, first)


def test_comment_only_python_change_is_not_a_change(tmp_path: Path) -> None:
    assert _changed(tmp_path, "X = 1\n", "# note\nX = 1  # why\n") == []


def test_real_python_change_is_still_a_change(tmp_path: Path) -> None:
    assert _changed(tmp_path, "X = 1\n", "X = 2\n") == ["orion/mod.py"]


def test_docstring_change_still_counts(tmp_path: Path) -> None:
    assert _changed(tmp_path, '"""a"""\nX = 1\n', '"""b"""\nX = 1\n') == ["orion/mod.py"]


def test_unparseable_python_is_kept(tmp_path: Path) -> None:
    assert _changed(tmp_path, "X = (\n", "X = (  # c\n") == ["orion/mod.py"]


def test_non_utf8_python_does_not_crash_planner(tmp_path: Path) -> None:
    head = "# -*- coding: latin-1 -*-\n"
    before = (head + "X = '\xe9'\n").encode("latin-1").decode("utf-8", "surrogateescape")
    after = (head + "X = '\xe9'  # c\n").encode("latin-1").decode("utf-8", "surrogateescape")
    # comment-only edit to a latin-1 file is still skipped, and nothing raises
    assert _changed_bytes(tmp_path, before, after) == []
    assert _changed_bytes(tmp_path / "b", before, after.replace("'\udce9'", "'z'")) == ["orion/mod.py"]


def _changed_bytes(tmp_path: Path, before: str, after: str) -> list[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("rebuild_affected_services_under_test", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "repo"
    (repo / "orion").mkdir(parents=True)
    for cmd in (["init", "-q", "-b", "main"], ["config", "user.email", "t@e.com"], ["config", "user.name", "T"]):
        subprocess.run(["git", *cmd], cwd=repo, check=True)
    f = repo / "orion" / "mod.py"
    f.write_bytes(before.encode("utf-8", "surrogateescape"))
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "one"], cwd=repo, check=True)
    first = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    f.write_bytes(after.encode("utf-8", "surrogateescape"))
    subprocess.run(["git", "commit", "-aq", "-m", "two"], cwd=repo, check=True)
    return mod.changed_files(repo, first)


_PR2728_PATHS = [
    "services/orion-vision-retina/app/frame_corruption.py",
    "services/orion-vision-retina/app/health.py",
    "services/orion-vision-retina/app/main.py",
    "services/orion-vision-retina/app/settings.py",
    "services/orion-vision-retina/app/sources.py",
    "services/orion-vision-retina/docker-compose.yml",
    "services/orion-vision-retina/.env_example",
]


def test_pr2728_retina_app_change_plans_a_build_that_changes_the_image() -> None:
    """PR #2728 (2026-10-16) changed retina app/*.py, bind-mounted at /app/app.

    The plan says `build`, not `recreate`. That is enough only because the
    Dockerfile also COPYs app/, so the image changes and compose recreates
    every container of the stack, including the walkway profile one.
    """
    proc = subprocess.run(
        [sys.executable, str(MODULE), "--plan", "--paths", *_PR2728_PATHS, "--repo-root", str(ROOT),
         "--host", "athena"],  # retina runs on athena; circe excludes it
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == ["orion-vision-retina\tbuild"]
    dockerfile = (ROOT / "services/orion-vision-retina/Dockerfile").read_text(encoding="utf-8")
    assert "COPY services/orion-vision-retina/app ./app" in dockerfile


# Own-directory bind mounts that hold .py files a long-running process does
# not import, so "live on disk" really is live. Anything else must be baked.
_OWN_MOUNT_PY_NOT_IMPORTED = {
    # dbt project dir; its one .py is an operator helper (scripts/init_local_env.py).
    "services/orion-analytics",
}


def test_own_python_code_mounts_are_also_baked_into_the_image() -> None:
    """A bind-mounted code dir changes on `git pull`, but the running process
    keeps the modules it already imported. The planner treats an own-dir
    directory mount as live and plans `build`, not `recreate`; that only
    restarts the process if the same files are COPY'd into the image. A
    mount-only Python app would keep running old code after every pull.

    Limits: checked per services/<svc> directory, not per compose service, so
    a sibling compose service that mounts the code onto a prebuilt `image:`
    would pass. A repo-root `COPY . /app` yields no own-dir build prefix and
    fails conservatively; add an allowlist entry with a reason if that occurs.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import rebuild_affected_services as planner
    finally:
        sys.path.pop(0)
    deps = planner.build_dependency_sources(str(ROOT), include_own=True)
    baked: dict[str, list[str]] = {}
    for dep in deps:
        if dep.mode == planner.MODE_BUILD:
            baked.setdefault(dep.service, []).append(dep.prefix)
    offenders = []
    for dep in deps:
        own = f"services/{dep.service}"
        if dep.mode != planner.MODE_RECREATE or not (dep.prefix == own or dep.prefix.startswith(own + "/")):
            continue
        mount = ROOT / dep.prefix
        if not mount.is_dir() or dep.prefix in _OWN_MOUNT_PY_NOT_IMPORTED:
            continue
        if not any(p.suffix == ".py" and "tests" not in p.parts for p in mount.rglob("*.py")):
            continue
        prefixes = baked.get(dep.service, [])
        if not any(dep.prefix == b or dep.prefix.startswith(b + "/") for b in prefixes):
            offenders.append(dep.prefix)
    assert offenders == [], (
        "bind-mounted Python code with no matching Dockerfile COPY; a pull would "
        f"leave the old process running: {offenders}"
    )

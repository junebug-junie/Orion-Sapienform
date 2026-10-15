"""Gate checks for the orion-kev ops contract (no GPU / no network)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_pins_gpu_and_restart() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    svc = compose["services"]["kev"]
    assert svc["restart"] == "unless-stopped"
    assert svc["container_name"] == "${PROJECT}-kev"
    devices = svc["deploy"]["resources"]["reservations"]["devices"]
    assert devices[0]["driver"] == "nvidia"
    assert "KEV_GPU_DEVICE_ID" in devices[0]["device_ids"][0]
    assert compose["networks"]["app-net"]["external"] is True


def test_env_example_has_required_keys() -> None:
    text = (ROOT / ".env_example").read_text(encoding="utf-8")
    for key in (
        "PROJECT=",
        "KEV_GPU_DEVICE_ID=",
        "KEV_HF_CACHE_HOST_DIR=",
        "KEV_HOST_PORT=",
        "KEV_MODEL=",
    ):
        assert key in text, key


def test_dockerfile_binds_all_interfaces() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'host="0.0.0.0"' in text
    assert "KEV_GIT_SHA=" in text


def _env_value(path: Path, key: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{key} missing from {path}")


def test_gpu_pin_matches_gpu_pool_kev_role() -> None:
    """KEV_GPU_DEVICE_ID and the pool's `kev` role must name the same card and port,
    or the pool counts Kev's VRAM on a card it is not on."""
    repo = ROOT.parents[1]
    pool = yaml.safe_load((repo / "config" / "gpu_pool.yaml").read_text(encoding="utf-8"))
    role = pool["roles"]["kev"]
    assert role["kind"] == "service"
    indices = {str(pool["cards"][card]["index"]) for card in role["cards"]}
    assert indices == {_env_value(ROOT / ".env_example", "KEV_GPU_DEVICE_ID")}
    assert str(role["port"]) == _env_value(ROOT / ".env_example", "KEV_HOST_PORT")


def test_substrate_points_at_kev_pool_address() -> None:
    """Substrate's Kev URL must be the pool's own address for the kev role (node + port),
    so moving the card to another host without moving the URL fails here."""
    from orion.gpu_pool.config import load_pool_config

    repo = ROOT.parents[1]
    url = _env_value(repo / "services" / "orion-substrate-runtime" / ".env_example",
                     "SUBSTRATE_SYSTEM_ONE_BASE_URL")
    assert url == load_pool_config(repo / "config" / "gpu_pool.yaml").url("kev")


def _service_list(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def test_rebuild_lists_place_kev_on_circe_only() -> None:
    """The post-merge rebuild must refresh Kev on circe and never recreate it on athena.
    circe reads its allowlist; every host without one (athena) reads the exclude file."""
    common = ROOT.parents[1] / "mesh-utilities" / "common"
    assert "orion-kev" in _service_list(common / "include_services_circe.txt")
    assert "orion-kev" in _service_list(common / "exclude_services.txt")


def test_post_merge_resolver_rebuilds_kev_on_circe_not_athena() -> None:
    """Pins the resolver's real output, so a change to include/exclude precedence fails here.
    Runs it as a subprocess with --json, the same way tests/scripts/test_rebuild_affected_services.py does."""
    import json
    import subprocess
    import sys

    repo = ROOT.parents[1]

    def resolve(host: str) -> list[str]:
        proc = subprocess.run(
            [sys.executable, str(repo / "scripts" / "rebuild_affected_services.py"),
             "--paths", "services/orion-kev/entrypoint.sh", "--json",
             "--repo-root", str(repo), "--host", host],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)["services"]

    assert "orion-kev" in resolve("circe")
    assert "orion-kev" not in resolve("athena")

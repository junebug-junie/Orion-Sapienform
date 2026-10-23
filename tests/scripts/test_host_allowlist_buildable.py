"""A host's rebuild allowlist must not name a service that cannot start from its defaults.

Live 2026-10-23: PR #2894 made orion-affectgpt-worker's compose require
``${AFFECTGPT_GPU_DEVICE_ID:?...}`` and left the key empty in .env_example on
purpose (no circe card has room for it yet). include_services_circe.txt still
listed it, so every circe post-pull rebuild that picked it ended
``FAILED orion-affectgpt-worker (exit 1)``. A required key that ships empty means
"paused until an operator acts" -- such a service belongs off the allowlist.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import rebuild_affected_services as planner  # noqa: E402

# Only `${VAR:?}` refuses an empty value; `${VAR?}` accepts it.
REQUIRED_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):\?")


def _example_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.split(" #", 1)[0].strip().strip('"').strip("'")
    return values


def empty_required_keys(service_dir: Path) -> list[str]:
    compose = service_dir / "docker-compose.yml"
    if not compose.exists():
        return []
    required = sorted(set(REQUIRED_VAR.findall(compose.read_text())))
    # safe_docker_build passes the root .env first, then the service's own.
    examples = {**_example_values(ROOT / ".env_example"), **_example_values(service_dir / ".env_example")}
    return [key for key in required if not examples.get(key)]


def test_detector_flags_an_empty_required_key(tmp_path: Path) -> None:
    svc = tmp_path / "svc"
    svc.mkdir()
    (svc / "docker-compose.yml").write_text('device_ids: ["${GPU_ID:?pick a card}"]\nx: ${SET_ID:?y}\nz: ${LOOSE?ok}\n')
    (svc / ".env_example").write_text("GPU_ID=\nSET_ID=abc\n")
    assert empty_required_keys(svc) == ["GPU_ID"]


def _allowlist_hosts() -> list[str]:
    common = ROOT / "mesh-utilities" / "common"
    hosts = {p.stem.removeprefix("include_services_") for p in common.glob("include_services_*.txt")}
    hosts |= {p.stem for p in (common / "include_services").glob("*.txt")}
    return sorted(hosts)


def test_no_host_allowlist_names_a_service_paused_by_an_empty_required_key(monkeypatch) -> None:
    # The loader honours an operator override; the gate must read the committed lists.
    monkeypatch.delenv("ORION_INCLUDE_SERVICES_FILE", raising=False)
    offenders = []
    assert _allowlist_hosts(), "no host allowlists found"
    for host in _allowlist_hosts():
        for service in sorted(planner.load_host_allowlist(ROOT, host) or ()):
            empty = empty_required_keys(ROOT / "services" / service)
            if empty:
                offenders.append(f"{host}: {service} requires {empty} but .env_example leaves it empty")
    assert not offenders, "\n".join(offenders)

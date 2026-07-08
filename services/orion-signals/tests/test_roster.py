"""Gate tests for orion-signals roster contract."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SIGNALS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SIGNALS_DIR.parents[1]
ROSTER_FILE = SIGNALS_DIR / "roster.v1.yaml"

TIER1_EXPECTED_IDS = {
    "orion-biometrics",
    "orion-equilibrium-service",
    "orion-collapse-mirror",
    "orion-cortex-exec",
    "orion-recall",
    "orion-spark-introspector",
    "orion-memory-consolidation",
}

CORE_EXPECTED_IDS = {"orion-bus", "orion-signal-gateway"}


def _load_roster() -> dict:
    assert ROSTER_FILE.is_file(), f"missing roster: {ROSTER_FILE}"
    with ROSTER_FILE.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    assert isinstance(doc, dict)
    return doc


def _compose_service_names(compose_path: Path) -> set[str]:
    text = compose_path.read_text(encoding="utf-8")
    # Top-level service keys under `services:` (indent 2, not nested keys).
    return set(re.findall(r"(?m)^  ([a-zA-Z][a-zA-Z0-9_-]*):\s*$", text))


@pytest.fixture(scope="module")
def roster() -> dict:
    return _load_roster()


def test_roster_schema_version(roster: dict) -> None:
    assert roster["schema_version"] == "orion_signals_roster.v1"


def test_roster_tiers_present(roster: dict) -> None:
    for tier in ("core", "tier1", "tier2", "routing"):
        assert tier in roster
        assert isinstance(roster[tier], list)


def test_every_compose_dir_has_compose_file(roster: dict) -> None:
    for tier in ("core", "tier1", "tier2", "routing"):
        for entry in roster[tier]:
            compose_dir = entry["compose_dir"]
            compose_path = REPO_ROOT / "services" / compose_dir / "docker-compose.yml"
            assert compose_path.is_file(), f"{entry['id']}: missing {compose_path}"


def test_every_compose_service_in_compose_file(roster: dict) -> None:
    for tier in ("core", "tier1", "tier2", "routing"):
        for entry in roster[tier]:
            compose_dir = entry["compose_dir"]
            compose_service = entry["compose_service"]
            compose_path = REPO_ROOT / "services" / compose_dir / "docker-compose.yml"
            names = _compose_service_names(compose_path)
            assert compose_service in names, (
                f"{entry['id']}: compose_service '{compose_service}' "
                f"not in {compose_path} (found: {sorted(names)})"
            )


def test_core_has_bus_and_signal_gateway(roster: dict) -> None:
    core_ids = {entry["id"] for entry in roster["core"]}
    assert CORE_EXPECTED_IDS.issubset(core_ids)


def test_tier1_has_expected_service_ids(roster: dict) -> None:
    tier1_ids = {entry["id"] for entry in roster["tier1"]}
    assert tier1_ids == TIER1_EXPECTED_IDS


def test_roster_entry_required_fields(roster: dict) -> None:
    for tier in ("core", "tier1", "tier2", "routing"):
        for entry in roster[tier]:
            assert "id" in entry
            assert "compose_dir" in entry
            assert "compose_service" in entry
            assert entry.get("required", True) is True

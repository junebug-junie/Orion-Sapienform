"""The operator template's expected-services list (what counts as "down" if silent).

2026-10-20: gpu-cluster-power, vision-edge, vision-frame-router and whisper-tts were
added -- each runs always-on on athena and heartbeats on orion:system:health under
exactly these names. orion-rag is left out on purpose (excluded from athena's
rebuild, running nowhere), so listing it would be a permanent false "down".
Being expected never triggers a restart: orion-mesh-guardian only acts on
config/mesh_remediation_roster.yaml, which this change does not touch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.settings import Settings  # noqa: E402

ADDED = ("gpu-cluster-power", "vision-edge", "vision-frame-router", "whisper-tts")


def _template_value(key: str) -> str:
    for line in (ROOT / ".env_example").read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
    raise AssertionError(f"{key} missing from .env_example")


def _expected(monkeypatch) -> list[str]:
    monkeypatch.setenv("EQUILIBRIUM_EXPECTED_SERVICES", _template_value("EQUILIBRIUM_EXPECTED_SERVICES"))
    monkeypatch.setenv("EQUILIBRIUM_EXPECTED_SERVICES_PATH", "")
    return Settings().expected_services()


def test_always_on_athena_services_are_expected(monkeypatch) -> None:
    expected = _expected(monkeypatch)
    for name in ADDED:
        assert name in expected, name
    assert "recall" in expected  # the original entries are still there


def test_rag_is_not_expected(monkeypatch) -> None:
    expected = _expected(monkeypatch)
    assert "rag" not in expected and "orion-rag" not in expected


def test_added_services_are_not_auto_remediated() -> None:
    roster = yaml.safe_load((REPO_ROOT / "config" / "mesh_remediation_roster.yaml").read_text())
    names = {e.get("heartbeat_name") for e in roster.get("services", [])}
    assert not names.intersection(ADDED)

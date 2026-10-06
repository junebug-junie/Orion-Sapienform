"""Every name in SUBSTRATE_ASSERTION_REQUIRED_READERS must be a name some service
actually advertises (its SERVICE_NAME, or SUBSTRATE_READER_NAME), or the
projector readiness gate can never open.

Regression, 2026-10-06: the list said `orion-hub`, `orion-recall`, ... while the
services advertise `hub`, `recall`, `cortex-exec`, ... so after a full deploy the
gate stayed shut forever.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_NAME_LINE = re.compile(r"^\s*-?\s*(SERVICE_NAME|SUBSTRATE_READER_NAME)\s*[=:]\s*(.+?)\s*$")


def _required_readers() -> list[str]:
    text = (ROOT / "services/orion-memory-consolidation/.env_example").read_text()
    match = re.search(r"^SUBSTRATE_ASSERTION_REQUIRED_READERS=(.*)$", text, re.MULTILINE)
    assert match, "SUBSTRATE_ASSERTION_REQUIRED_READERS missing from .env_example"
    return [name.strip() for name in match.group(1).split(",") if name.strip()]


def _declared_reader_names() -> set[str]:
    names: set[str] = set()
    for service_dir in (ROOT / "services").iterdir():
        for fname in (".env_example", "docker-compose.yml"):
            path = service_dir / fname
            if not path.is_file():
                continue
            for line in path.read_text(errors="ignore").splitlines():
                match = _NAME_LINE.match(line)
                if not match:
                    continue
                value = match.group(2)
                default = re.search(r"\$\{[A-Z_]+:-([^}]+)\}", value)
                if default:
                    value = default.group(1)
                if "${" not in value:
                    names.add(value.strip().strip("'\""))
    return names


def test_every_required_reader_is_a_name_a_service_advertises():
    declared = _declared_reader_names()
    missing = [name for name in _required_readers() if name not in declared]
    assert not missing, f"required readers no service advertises: {missing}"


def test_compose_default_matches_env_example():
    compose = (ROOT / "services/orion-memory-consolidation/docker-compose.yml").read_text()
    match = re.search(r"SUBSTRATE_ASSERTION_REQUIRED_READERS:-([^}]+)\}", compose)
    assert match, "compose default missing"
    assert [n.strip() for n in match.group(1).split(",")] == _required_readers()

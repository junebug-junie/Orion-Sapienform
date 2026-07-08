"""Gate tests for orion-signals launcher scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SIGNALS_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SIGNALS_DIR / "scripts"
ENV_EXAMPLE = SIGNALS_DIR / ".env_example"


@pytest.mark.parametrize(
    "script_name",
    ["up.sh", "down.sh", "smoke.sh"],
)
def test_scripts_pass_bash_syntax_check(script_name: str) -> None:
    script = SCRIPTS_DIR / script_name
    assert script.is_file(), f"missing script: {script}"
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_env_example_has_orion_bus_url() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "ORION_BUS_URL=" in text


def test_up_sh_requires_orion_bus_url() -> None:
    up = SCRIPTS_DIR / "up.sh"
    text = up.read_text(encoding="utf-8")
    assert '[[ -z "${ORION_BUS_URL:-}" ]]' in text
    assert "exit 1" in text


def test_compose_helpers_skip_missing_env_file() -> None:
    for script in ("up.sh", "down.sh"):
        text = (SCRIPTS_DIR / script).read_text(encoding="utf-8")
        assert 'if [[ -f "${service_env}" ]]' in text or 'if [[ -f "${gateway_env}" ]]' in text

"""Runs turn-timer.test.js from the Python suite.

The Hub has a dozen `*.test.js` files and no gate anywhere -- not CI, not the
Makefile -- actually executes them, so "5 formatter cases" is advertised
coverage that nothing runs. This does not fix that repo-wide gap; it just makes
the one module this change added real rather than decorative.

It SKIPS loudly when node is absent instead of passing, so a missing runtime can
never be mistaken for a green formatter.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_JS = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "turn-timer.test.js"


def test_turn_timer_formatter_cases_pass_under_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH -- turn-timer.test.js was NOT executed")
    assert TEST_JS.is_file(), TEST_JS

    result = subprocess.run(
        [node, "--test", str(TEST_JS)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # A node --test run over a file with no tests also exits 0, so pin that the
    # cases actually ran.
    assert "# fail 0" in result.stdout, result.stdout
    assert "# pass 7" in result.stdout, result.stdout

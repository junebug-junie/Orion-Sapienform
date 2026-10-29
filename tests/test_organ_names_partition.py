"""Orion's organ names stay out of recall until battle-tested (design PR #2888, decision 7).

Runs in the repo-wide static gates, not the body-atlas workflow: a new reader added anywhere else
in the repo must fail CI even when no body-atlas file changed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
ALLOWED = ("orion/body_atlas/", "orion/dig/", "tests/test_organ_names_partition.py")


def test_only_the_atlas_and_dig_read_the_organ_names_ledger():
    proc = subprocess.run(["git", "-C", str(_REPO), "grep", "-l", "-e", "organ_names", "-e", "body_atlas.names",
                           "-e", "body_atlas import names", "--", "*.py"], capture_output=True, text=True)
    assert proc.returncode in (0, 1), proc.stderr  # 1 = no match; anything else is a broken check
    hits = proc.stdout.split()
    assert "orion/body_atlas/names.py" in hits or "orion/dig/__main__.py" in hits  # the grep really ran
    assert [h for h in hits if not h.startswith(ALLOWED)] == []

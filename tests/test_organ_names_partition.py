"""Orion's self ledgers stay out of recall until battle-tested (design PR #2888, decision 7): the
organ names (rung 3) and the capacity claims and capacity sentences (rung 4, design
``docs/superpowers/specs/2026-10-30-self-now-capacities-vsm-design.md``, section 6).

Runs in the repo-wide static gates, not the body-atlas workflow: a new reader added anywhere else
in the repo must fail CI even when no body-atlas file changed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
ALLOWED = ("orion/body_atlas/", "orion/dig/", "tests/test_organ_names_partition.py")


def _readers(*terms: str) -> list[str]:
    args = [x for t in terms for x in ("-e", t)]
    proc = subprocess.run(["git", "-C", str(_REPO), "grep", "-l", "-F", *args, "--", "*.py"],
                          capture_output=True, text=True)
    assert proc.returncode in (0, 1), proc.stderr  # 1 = no match; anything else is a broken check
    return proc.stdout.split()


def test_only_the_atlas_and_dig_read_the_organ_names_ledger():
    hits = _readers("organ_names", "body_atlas.names", "body_atlas import names", "names_path")
    assert "orion/body_atlas/names.py" in hits or "orion/dig/__main__.py" in hits  # the grep really ran
    assert [h for h in hits if not h.startswith(ALLOWED)] == []


def test_only_the_atlas_and_dig_read_the_capacity_ledgers():
    hits = _readers("capacity_claims", "capacities.jsonl", "body_atlas.claims", "body_atlas import claims",
                    "import capacities, claims", "claims_path", "capacities_path")
    assert "orion/body_atlas/claims.py" in hits and "orion/dig/__main__.py" in hits  # the grep really ran
    assert [h for h in hits if not h.startswith(ALLOWED)] == []

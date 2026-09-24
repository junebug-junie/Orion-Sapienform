"""pytest wrapper so the capture eval runs as a gate, not just by hand."""
from __future__ import annotations

from orion.metacog.evals.run_capture_eval import run


def test_metacog_capture_acceptance_checks_4_to_6_on_real_fixture():
    assert run() == []

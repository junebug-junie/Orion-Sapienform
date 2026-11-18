"""CI wrapper: the fixture eval must pass (only accepted reading links reach stored seeds)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_curiosity_seed_neighborhood_eval import run_fixture  # noqa: E402


def test_curiosity_seed_neighborhood_eval_fixture_passes():
    report, failures = run_fixture()
    assert not failures, failures
    assert report["seeds_with_accepted_link"] == 3
    assert report["live_replay_seeds_with_link"] == 0

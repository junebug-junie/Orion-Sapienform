"""Person sighting replay eval (run_person_sighting_replay_eval.py): a spread-out week
passes; one person seen every 3 hours fails the 25% domination check."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_person_sighting_replay_eval import synthetic, verdict  # noqa: E402


def test_spread_out_sightings_pass_every_check() -> None:
    r = synthetic()
    assert verdict(r) == []
    assert r["top_share"] <= 0.25
    assert r["flood_writes"] == 0 and r["fenced_changed"] == [] and r["not_resting"] == []
    # Each sighting keeps its person active for hours, not all replay long.
    assert 0 < max(r["active_hours_by_person"].values()) < r["replay_hours"] // 2


def test_one_person_seen_all_day_fails_domination() -> None:
    r = synthetic(dominant=True)
    fails = verdict(r)
    assert r["top_person"] == "A" and r["top_share"] > 0.25
    assert any(f.startswith("domination") for f in fails)

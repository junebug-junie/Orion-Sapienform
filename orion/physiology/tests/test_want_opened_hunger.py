"""#3640: a ``hunger_from: want_opened`` family ranks each want by its own age (durable.intention: the promise's age).

What must hold: the want's hunger counts from its ``opened_at`` even after the family has closed acts; no lane seed
raises it; tau is the family window, not the close rhythm; the rank is that intensity (no close-gap history); a
fresh want still reads ~0; a ``last_close`` family is unchanged; the config value is validated."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.physiology.appetite import ActRecord, build_family_appetite, hunger, rank_value
from orion.physiology.families import FamilyConfig, load_families, parse_families

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
DAY = 86400.0
WANT = FamilyConfig(family="durable.intention", tier=1, deferrable_window_s=DAY, lanes=("agent",),
                    hunger_from="want_opened")
DRIVE = FamilyConfig(family="durable.intention", tier=1, deferrable_window_s=DAY, lanes=("agent",))


def _acts(hours_ago):
    return [ActRecord(act_id=f"a{i}", family="durable.intention", opened_at=NOW - timedelta(hours=h, minutes=30),
                      closed_at=NOW - timedelta(hours=h)) for i, h in enumerate(hours_ago)]


def _rank(cfg, acts, opened_at, *, seed=None):
    app = build_family_appetite(cfg, acts, NOW)
    h = hunger(app, NOW, opened_at=opened_at, seed_sec=seed, seed_lane="agent" if seed else None)
    rank, basis, _ = rank_value(app, h.intensity, None)
    return app, h, rank, basis


@pytest.mark.parametrize("hours", [1, 6, 24, 48])
def test_a_want_ranks_by_its_own_age_not_by_the_last_close(hours):
    # A close 5 minutes ago (another promise's carry) does not reset this promise's clock.
    acts = _acts([0.083, 30, 60, 90, 120, 150, 180])
    app, h, rank, basis = _rank(WANT, acts, NOW - timedelta(hours=hours))
    assert h.basis == "since_opened" and h.unfunded_sec == pytest.approx(hours * 3600)
    assert app.tau_sec == DAY and app.tau_basis == "deferrable_window"    # 6 close gaps, still not the clock
    assert app.values == () and basis == "want_opened:ungraded"
    assert rank == pytest.approx(1 - math.exp(-hours * 3600 / DAY))


def test_a_fresh_want_still_ranks_near_zero_and_no_lane_seed_lifts_it():
    _, h, rank, _ = _rank(WANT, _acts([30]), NOW - timedelta(seconds=5), seed=7200.0)
    assert h.basis == "since_opened" and rank < 1e-4


def test_rank_grows_monotonically_with_age():
    ranks = [_rank(WANT, _acts([30]), NOW - timedelta(hours=h))[2] for h in (0, 1, 6, 24, 72)]
    assert ranks == sorted(ranks) and ranks[0] == 0.0 and ranks[-1] > 0.95


def test_no_opened_at_falls_back_to_the_drive_clock():
    _, h, _, _ = _rank(WANT, _acts([2]), None)
    assert h.basis == "since_last_close" and h.unfunded_sec == pytest.approx(7200)


def test_a_last_close_family_is_unchanged():
    # The drive: a close 5 minutes ago resets it, whatever the want's own age.
    _, h, _, _ = _rank(DRIVE, _acts([0.083, 30]), NOW - timedelta(hours=40))
    assert h.basis == "since_last_close" and h.unfunded_sec == pytest.approx(0.083 * 3600, rel=1e-2)


def test_hunger_from_is_parsed_and_validated():
    base = {"version": 1, "families": {"f": {"tier": 1, "deferrable_window_s": 60}}}
    assert parse_families(base).get("f").hunger_from == "last_close"
    base["families"]["f"]["hunger_from"] = "want_opened"
    assert parse_families(base).get("f").hunger_from == "want_opened"
    base["families"]["f"]["hunger_from"] = "promise"
    with pytest.raises(ValueError, match="hunger_from"):
        parse_families(base)


def test_durable_intention_is_the_one_want_opened_family_in_the_live_config():
    fams, err = load_families(Path(__file__).resolve().parents[3] / "config/physiology/families.yaml")
    assert err is None
    assert [f for f, c in fams.families.items() if c.hunger_from == "want_opened"] == ["durable.intention"]

"""Attention patch 3b: the exploration floor counts acts started, not funded bids (spec 2026-11-07,
decided 2026-11-10 item 4; acceptance check 6)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.physiology.appetite import ActRecord
from orion.physiology.choice import FundedWant, ReservationBook, acts_started
from orion.physiology.families import load_families

NOW = datetime(2026, 11, 11, 6, 0, tzinfo=timezone.utc)
FAMS, _ = load_families(Path(__file__).resolve().parents[3] / "config/physiology/families.yaml")


def act(i, family, opened_days_ago):
    opened = NOW - timedelta(days=opened_days_ago) if opened_days_ago is not None else None
    return ActRecord(act_id=f"a{i}", family=family, opened_at=opened, closed_at=None, outcome_value=None,
                     outcome_source=None, signal_id=None, cost={})


def test_share_equals_a_hand_count_of_acts_started_this_week():
    acts = ([act(i, "hub.world_pulse_read", 1) for i in range(6)]
            + [act(10 + i, "hub.retro_leisure", 2) for i in range(1)]
            + [act(20 + i, "durable.curiosity_investigate", 3) for i in range(3)]
            + [act(30, "durable.curiosity_investigate", 9)]        # outside the week
            + [act(31, "hub.retro_leisure", None)]                  # never opened
            + [act(32, "not.a.family", 1)])                         # not configured
    rows = acts_started(acts, FAMS, NOW)
    assert sorted(r.decision_id for r in rows) == sorted(
        [f"act:a{i}" for i in range(6)] + ["act:a10"] + [f"act:a{20 + i}" for i in range(3)])
    book = ReservationBook()
    book.seed_started(rows, NOW)
    tier2 = {r.family for r in rows if FAMS.get(r.family).tier == 2}
    n2 = sum(1 for r in rows if r.family in tier2)
    assert book.funded_share("hub.retro_leisure", 2, NOW) == pytest.approx(1 / n2)
    assert book.funded_share("hub.world_pulse_read", 2, NOW) == pytest.approx(6 / n2)


def test_own_decisions_count_only_as_floor_funds_and_only_until_the_next_seed():
    book = ReservationBook()
    book.seed_started([FundedWant("act:a", NOW - timedelta(days=1), "durable.self_inquiry", 2)], NOW)
    book.note_funded("d-bid", "hub.retro_leisure", 2, NOW)                # a funded bid: not effort
    assert book.funded_share("hub.retro_leisure", 2, NOW) == pytest.approx(0.0)
    book.note_funded("d-floor", "hub.retro_leisure", 2, NOW, floor=True)  # a floor fund: counts now
    assert book.funded_share("hub.retro_leisure", 2, NOW) == pytest.approx(0.5)
    book.seed_started([FundedWant("act:a", NOW - timedelta(days=1), "durable.self_inquiry", 2)], NOW)
    assert book.funded_share("hub.retro_leisure", 2, NOW) == pytest.approx(0.0)   # its act never opened


def test_the_older_replays_keep_counting_funded_wants():
    book = ReservationBook()
    book.seed_funded([FundedWant("x", NOW, "durable.self_inquiry", 2)], NOW)
    book.note_funded("d-bid", "hub.retro_leisure", 2, NOW)
    assert book.funded_share("hub.retro_leisure", 2, NOW) == pytest.approx(0.5)

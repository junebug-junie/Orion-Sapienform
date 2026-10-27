"""Organ 3b learned weights (Patch C1): shrinkage, missing sources skipped, drift cap, rest at prior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.physiology.appetite import ActRecord
from orion.physiology.families import parse_families
from orion.physiology.weights import (
    KAPPA,
    MAX_DRIFT_PER_DAY,
    UseRecord,
    evidence_for,
    max_daily_move,
    shrunk_target,
    weight_series,
)

NOW = datetime(2026, 10, 27, 6, 0, tzinfo=timezone.utc)
FAMS = parse_families({"version": 1, "families": {
    "used": {"tier": 2, "deferrable_window_s": 3600},
    "unused": {"tier": 2, "deferrable_window_s": 3600},
    "unmeasured": {"tier": 2, "deferrable_window_s": 3600},
    "outreach": {"tier": 2, "deferrable_window_s": 3600},
    "upkeep": {"tier": 1, "deferrable_window_s": 3600},
    "turn": {"tier": 0},
}})


def uses(family, n, used_every, *, start=NOW - timedelta(days=20), status="linked", closed=True):
    return [UseRecord(act_id=f"{family}{i}", family=family, started_at=start + timedelta(hours=i), link_status=status,
                      used=(i % used_every == 0) if used_every else (None if status == "no_link_path" else False),
                      window_closed=closed) for i in range(n)]


def test_a_missing_source_is_skipped_never_zero():
    ev = evidence_for("unmeasured", [], uses("unmeasured", 40, 0, status="no_link_path"),
                      since=NOW - timedelta(days=30), until=NOW)
    assert ev.use_rate is None and ev.mean_rate is None
    assert shrunk_target(1.0, ev) == (1.0, "prior")


def test_open_window_is_not_a_final_not_used():
    ev = evidence_for("used", [], uses("used", 40, 1, closed=False), since=NOW - timedelta(days=30), until=NOW)
    assert ev.use_n == 0


def test_fewer_than_ten_evidenced_acts_stays_at_prior():
    ev = evidence_for("unused", [], uses("unused", 9, 0), since=NOW - timedelta(days=30), until=NOW)
    assert ev.use_rate == 0.0 and shrunk_target(1.0, ev) == (1.0, "prior")


def test_shrinkage_moves_toward_evidence_with_kappa_30():
    ev = evidence_for("unused", [], uses("unused", 30, 0), since=NOW - timedelta(days=30), until=NOW)
    tgt, basis = shrunk_target(1.0, ev)
    assert basis == "evidence" and tgt == pytest.approx(KAPPA / (KAPPA + 30))


def test_juniper_engagement_is_source_b_for_resolver_closed_reply_acts():
    acts = [ActRecord(act_id=f"o{i}", family="outreach", opened_at=NOW - timedelta(days=3, hours=i),
                      closed_at=NOW - timedelta(days=2, hours=i), outcome_value=1.0 if i < 6 else 0.0,
                      outcome_source="resolver", signal_id="juniper_replied") for i in range(12)]
    ev = evidence_for("outreach", acts, [], since=NOW - timedelta(days=30), until=NOW)
    assert ev.engagement_rate == pytest.approx(0.5) and ev.engagement_n == 12 and ev.use_rate is None


def test_drift_is_capped_and_a_zero_use_family_ends_below_its_prior():
    data = uses("used", 60, 1) + uses("unused", 60, 0)
    series = weight_series(FAMS, [], data, start=(NOW - timedelta(days=25)).date(), end=NOW.date())
    assert max_daily_move(series) <= MAX_DRIFT_PER_DAY + 1e-12
    final = series[NOW.date()]
    assert final["unused"].weight < 1.0 < final["used"].weight
    assert "turn" not in final                       # tier 0 never gets a weight: it never bids
    tier2 = [w.target for f, w in final.items() if w.tier == 2]
    assert sum(tier2) / len(tier2) == pytest.approx(1.0)


def test_rest_state_evidence_ages_out_and_the_weight_relaxes_to_prior():
    old = uses("unused", 60, 0, start=NOW - timedelta(days=70))
    series = weight_series(FAMS, [], old, start=(NOW - timedelta(days=45)).date(), end=NOW.date())
    first, last = series[min(series)], series[NOW.date()]
    assert first["unused"].weight < 1.0
    assert last["unused"].weight == pytest.approx(1.0) and last["unused"].basis == "prior"

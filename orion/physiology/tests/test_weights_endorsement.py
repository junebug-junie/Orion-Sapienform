"""Personality rev 4 Move 2: source (c), Orion's endorsement, in the learned family weights."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.personality.graph import DECLINE_TEXT
from orion.physiology.appetite import ActRecord
from orion.physiology.families import parse_families
from orion.physiology.weights import (
    DECLINABLE_LANES,
    KAPPA,
    choice_record,
    endorsement_exclusions,
    endorsement_for,
    endorsement_option_key,
    evidence_for,
    option_family,
    shrunk_target,
    weights_now,
)

NOW = datetime(2026, 11, 20, 6, 0, tzinfo=timezone.utc)
SINCE = NOW - timedelta(days=30)
FAMS = parse_families({"version": 1, "families": {
    "hub.outreach": {"tier": 2, "deferrable_window_s": 3600},
    "hub.world_pulse_read": {"tier": 2, "deferrable_window_s": 3600},
    "durable.curiosity_investigate": {"tier": 2, "deferrable_window_s": 3600},
    "prime": {"tier": 2, "deferrable_window_s": 3600},
}})


def _outreach(i, *, said=True, promoted=False, at=None):
    at = at or NOW - timedelta(days=1, minutes=i)
    return choice_record(f"outreach:{i}", "outreach", at, None if not said else f"topic{i}", [f"t{i}b", f"t{i}c"],
                         declined_all=not said, care_promoted=promoted)


def test_declinable_lanes_match_the_reducers_decline_phrases():
    assert DECLINABLE_LANES == frozenset(DECLINE_TEXT)


def test_option_family_reads_the_key_then_the_lane():
    assert option_family("prime", endorsement_option_key("hub.world_pulse_read", "c1")) == "hub.world_pulse_read"
    assert option_family("prime", "c1") == "prime"
    assert option_family("outreach", "topic") == "hub.outreach"
    assert option_family("chat", "thread") is None
    assert option_family("prime", "::x") == "prime"           # an empty family is not a family


def test_a_family_with_no_choice_point_has_no_source_c_never_zero():
    rate, n, _ = endorsement_for("hub.outreach", [], since=SINCE, until=NOW)
    assert rate is None and n == 0
    ev = evidence_for("hub.outreach", [], [], since=SINCE, until=NOW, choices=[])
    assert ev.endorsement_rate is None and ev.mean_rate is None


def test_a_menu_that_cannot_be_declined_is_not_a_choice_point():
    # Curiosity priors: one family, no "none" option. Picking from it always reads 1.0 -- not evidence.
    cur = [choice_record(f"curiosity:{i}", "curiosity", NOW - timedelta(days=1), f"p{i}", [f"q{i}"])
           for i in range(40)]
    assert endorsement_for("durable.curiosity_investigate", cur, since=SINCE, until=NOW)[0] is None
    # ...but one that did decline proves the menu could be declined.
    declined = choice_record("curiosity:x", "curiosity", NOW - timedelta(days=1), None, ["q"], declined_all=True)
    assert endorsement_for("durable.curiosity_investigate", [declined], since=SINCE, until=NOW)[0] == 0.0


def test_endorsement_is_a_rate_per_offer_and_care_promoted_counts_half():
    rows = [_outreach(i) for i in range(3)] + [_outreach(3, said=False)]
    assert endorsement_for("hub.outreach", rows, since=SINCE, until=NOW)[:2] == (0.75, 4)
    rows2 = [_outreach(0, promoted=True), _outreach(1, said=False)]
    assert endorsement_for("hub.outreach", rows2, since=SINCE, until=NOW)[0] == pytest.approx(0.5 / 1.5)


def test_a_care_promoted_point_counts_half_toward_n_too():
    rows = [_outreach(i, promoted=True) for i in range(20)]
    ev = evidence_for("hub.outreach", [], [], since=SINCE, until=NOW, choices=rows)
    assert ev.endorsement_n == 20 and ev.n == pytest.approx(10.0)


def test_prime_options_spanning_families_endorse_the_chosen_family():
    k = endorsement_option_key
    rows = [choice_record(f"prime:{i}", "prime", NOW - timedelta(days=1), k("hub.world_pulse_read", f"c{i}"),
                          [k("vision.ask", f"d{i}")]) for i in range(4)]
    assert endorsement_for("hub.world_pulse_read", rows, since=SINCE, until=NOW)[0] == 1.0
    assert endorsement_for("vision.ask", rows, since=SINCE, until=NOW)[0] == 0.0
    assert endorsement_for("prime", rows, since=SINCE, until=NOW)[0] is None     # no prime option offered


def test_source_c_joins_the_mean_with_kappa_shrinkage():
    rows = [_outreach(i) for i in range(20)]           # 20 points, every one taken: rate 1.0
    ev = evidence_for("hub.outreach", [], [], since=SINCE, until=NOW, choices=rows)
    assert ev.endorsement_rate == 1.0 and ev.n == 20
    tgt, basis = shrunk_target(1.0, ev)
    assert basis == "evidence" and tgt == pytest.approx((KAPPA * 1.0 + 20 * 1.0) / (KAPPA + 20))
    declines = [_outreach(i, said=False) for i in range(20)]
    w_with = weights_now(FAMS, [], [], NOW, choices=declines)
    w_without = weights_now(FAMS, [], [], NOW)
    assert w_with["hub.outreach"].evidence.endorsement_rate == 0.0
    assert w_with["hub.outreach"].weight < w_without["hub.outreach"].weight
    excluded = weights_now(FAMS, [], [], NOW, choices=declines, endorsement_excluded=frozenset({"hub.outreach"}))
    assert excluded["hub.outreach"].weight == w_without["hub.outreach"].weight


def _engagement(day_offset, value):
    closed = NOW - timedelta(days=day_offset, hours=-1)
    return ActRecord(act_id=f"e{day_offset}", family="hub.outreach", opened_at=closed - timedelta(minutes=5),
                     closed_at=closed, outcome_value=value, outcome_source="resolver", signal_id="juniper_replied")


def _day_choices(day_offset, taken, total):
    at = NOW - timedelta(days=day_offset, hours=-2)
    return [choice_record(f"o{day_offset}-{i}", "outreach", at, f"t{i}" if i < taken else None, ["x"],
                          declined_all=i >= taken) for i in range(total)]


def test_endorsement_tracking_engagement_above_point_eight_is_excluded():
    acts, rows = [], []
    for d in range(1, 9):                       # 8 paired days, (c) and (b) move together
        v = 0.25 if d % 2 else 1.0
        acts.append(_engagement(d, v))
        rows += _day_choices(d, 1 if d % 2 else 4, 4)
    chk = endorsement_exclusions(FAMS, acts, rows, NOW)["hub.outreach"]
    assert chk.paired_days == 8 and chk.correlation > 0.8 and chk.excluded
    assert chk.reason.startswith("corr_with_engagement:")


def test_too_few_paired_days_keeps_source_c():
    acts, rows = [], []
    for d in range(1, 5):
        acts.append(_engagement(d, 0.25 if d % 2 else 1.0))
        rows += _day_choices(d, 1 if d % 2 else 4, 4)
    chk = endorsement_exclusions(FAMS, acts, rows, NOW)["hub.outreach"]
    assert chk.paired_days == 4 and not chk.excluded and chk.reason == ""

"""Gate tests for the ask_claude dry-run trigger.

The population regression at the bottom is the load-bearing one: it pins the
real live prior shape from 2026-09-08 so a later knob change has to face what
it does to selectivity, rather than being judged only against synthetic cases
built to make it pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from orion.autonomy.ask_claude_trigger import (
    MAX_LIMIT_STALENESS_SEC,
    MAX_SETTLED_CONFIDENCE,
    MIN_TIMES_TESTED,
    decide,
)


@dataclass
class FakePrior:
    prior_id: str
    claim: str
    confidence: Optional[float]
    status: str
    times_tested: int


@dataclass
class FakeLimit:
    state: str = "clear"
    observed: bool = True
    event_count: int = 0
    staleness_sec: Optional[float] = 1.0


def _prior(pid="p1", *, tested=5, conf=0.3, status="revised", claim="a claim"):
    return FakePrior(prior_id=pid, claim=claim, confidence=conf, status=status, times_tested=tested)


def _stuck_pop():
    return [_prior()]


# -- budget gate: fails closed, in a stated precedence -------------------


def test_missing_observation_refuses_rather_than_assuming_clear():
    d = decide(priors=_stuck_pop(), limit=None)
    assert d.would_ask is False
    assert d.refused == "budget_observation_missing"


def test_unobserved_window_is_not_permission():
    # event_count == 0 and observed == False is the "mount went away" case. It
    # must not authorise the same action as a genuinely empty window.
    d = decide(priors=_stuck_pop(), limit=FakeLimit(state="unknown", observed=False))
    assert d.refused == "budget_unobserved"


def test_limited_refuses():
    d = decide(priors=_stuck_pop(), limit=FakeLimit(state="limited"))
    assert d.refused == "budget_limited"


def test_unknown_state_with_observation_still_refuses():
    d = decide(priors=_stuck_pop(), limit=FakeLimit(state="unknown", observed=True))
    assert d.refused == "budget_unknown"


def test_stale_clear_reading_refuses():
    d = decide(
        priors=_stuck_pop(),
        limit=FakeLimit(staleness_sec=MAX_LIMIT_STALENESS_SEC + 1),
    )
    assert d.refused == "budget_observation_stale"


def test_none_staleness_on_an_observed_window_refuses():
    # Observed messages but no freshest timestamp is a producer contradiction,
    # not a fresh reading.
    d = decide(priors=_stuck_pop(), limit=FakeLimit(staleness_sec=None))
    assert d.refused == "budget_observation_stale"


def test_fresh_clear_reading_admits():
    d = decide(priors=_stuck_pop(), limit=FakeLimit(staleness_sec=1.0))
    assert d.would_ask is True
    assert d.refused is None


# -- the ordering claim in decide()'s docstring --------------------------


def test_priors_are_scored_even_when_the_budget_refused():
    # This is the whole reason a week of dry-run output can answer "did the
    # trigger ever want to fire" independently of whether the meter allowed it.
    d = decide(priors=_stuck_pop(), limit=FakeLimit(state="limited"))
    assert d.refused == "budget_limited"
    assert len(d.assessments) == 1
    assert d.assessments[0].stuck is True


def test_budget_facts_are_recorded_verbatim_on_a_refusal():
    d = decide(priors=[], limit=FakeLimit(state="limited", event_count=7, staleness_sec=12.5))
    assert (d.limit_state, d.limit_event_count, d.limit_staleness_sec) == ("limited", 7, 12.5)


# -- prior scoring -------------------------------------------------------


def test_no_live_priors_is_distinct_from_no_stuck_prior():
    empty = decide(priors=[], limit=FakeLimit())
    settled = decide(priors=[_prior(tested=9, conf=0.99)], limit=FakeLimit())
    assert empty.refused == "no_live_priors"
    assert settled.refused == "no_stuck_prior"


def test_confidence_none_reads_as_unsettled_not_settled():
    # Prior.uncertainty already treats "Orion never said how sure it was" as
    # maximally uncertain. Treating it as settled here would silently exclude
    # the claims Orion was least sure about.
    d = decide(priors=[_prior(tested=MIN_TIMES_TESTED, conf=None)], limit=FakeLimit())
    assert d.would_ask is True
    assert d.assessments[0].unsettled is True


def test_tested_enough_is_inclusive_at_the_threshold():
    below = decide(priors=[_prior(tested=MIN_TIMES_TESTED - 1)], limit=FakeLimit())
    at = decide(priors=[_prior(tested=MIN_TIMES_TESTED)], limit=FakeLimit())
    assert below.refused == "no_stuck_prior"
    assert at.would_ask is True


def test_settled_confidence_boundary_is_inclusive():
    at = decide(priors=[_prior(tested=5, conf=MAX_SETTLED_CONFIDENCE)], limit=FakeLimit())
    above = decide(priors=[_prior(tested=5, conf=MAX_SETTLED_CONFIDENCE + 0.01)], limit=FakeLimit())
    assert at.would_ask is True
    assert above.refused == "no_stuck_prior"


def test_most_tested_wins_then_least_confident_breaks_the_tie():
    d = decide(
        priors=[
            _prior("few", tested=3, conf=0.1),
            _prior("most", tested=8, conf=0.6),
            _prior("tie_high", tested=8, conf=0.5),
        ],
        limit=FakeLimit(),
    )
    # times_tested dominates -- "few" is less confident but Orion has not
    # worked it as hard, so it is not the one a peer would unstick.
    assert d.subject_prior_id == "tie_high"


def test_every_prior_is_assessed_not_only_the_winner():
    d = decide(priors=[_prior("a", tested=5), _prior("b", tested=9, conf=0.99)], limit=FakeLimit())
    assert {a.prior_id for a in d.assessments} == {"a", "b"}


# -- live population regression ------------------------------------------

# Verbatim from orion_worldview at 2026-09-08 (7 live priors: not refuted, not
# retired_unresolvable). Statuses `supported` and `revised` are LIVE -- only
# `refuted` and `retired_unresolvable` close a prior, and reading `supported`
# as closed was a real accumulation outage on 2026-08-27 (worldview.py:80).
LIVE_2026_09_08 = [
    ("atlas_prediction_error_territory", 10, 0.30, "revised"),
    ("dominant_shift_not_biased", 4, 0.90, "supported"),
    ("substrate_concepts_separate_pipeline", 4, 0.92, "revised"),
    ("stance_gate_is_manual_review", 2, 0.95, "supported"),
    ("formation_policy_gate_before_review", 2, 0.85, "supported"),
    ("auto_activate_kind_routing_only", 2, 0.95, "supported"),
    ("degree_zero_is_expected_topology", 1, 0.80, "revised"),
]


def test_live_population_selects_exactly_one_and_it_is_the_ten_times_tested_claim():
    priors = [
        FakePrior(prior_id=pid, claim=pid, confidence=conf, status=st, times_tested=n)
        for pid, n, conf, st in LIVE_2026_09_08
    ]
    d = decide(priors=priors, limit=FakeLimit())
    stuck = [a.prior_id for a in d.assessments if a.stuck]
    assert stuck == ["atlas_prediction_error_territory"], stuck
    assert d.subject_prior_id == "atlas_prediction_error_territory"


def test_live_population_is_not_all_stuck_and_not_none_stuck():
    # The two degenerate outcomes that would mean the knobs carry no
    # information. Guards against a future retune collapsing to either.
    priors = [
        FakePrior(prior_id=pid, claim=pid, confidence=conf, status=st, times_tested=n)
        for pid, n, conf, st in LIVE_2026_09_08
    ]
    d = decide(priors=priors, limit=FakeLimit())
    n_stuck = sum(1 for a in d.assessments if a.stuck)
    assert 0 < n_stuck < len(priors), n_stuck

"""output_used grading rule (act consequence bets, Patch 1a): pure, no DB."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.grammar.output_used import (
    UNOBSERVABLE_NOTE,
    UNUSED_NOTE,
    UseEvidence,
    output_used_grade,
    producer_close_disposition,
)
from orion.grammar.promise_acts import is_void_note
from orion.self_monitor.adapters import act_outcome

H = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
GRACE = 120.0


def _ev(first=None, observable=True, site="reverie"):
    return UseEvidence(first_use_at=first, use_site=site, observable=observable)


def test_use_before_horizon_closes_used_at_first_use_even_before_horizon():
    used_at = H - timedelta(hours=20)
    out = output_used_grade(_ev(first=used_at), horizon_at=H, now=H - timedelta(hours=19), grace_sec=GRACE)
    assert out is not None and out.value == 1.0 and out.source == "resolver"
    assert out.observed_at == used_at and "reverie" in out.note


def test_open_act_with_no_use_waits_until_horizon_plus_grace():
    assert output_used_grade(_ev(), horizon_at=H, now=H - timedelta(seconds=1), grace_sec=GRACE) is None
    assert output_used_grade(_ev(), horizon_at=H, now=H + timedelta(seconds=GRACE - 1), grace_sec=GRACE) is None


def test_observable_and_unused_at_horizon_is_a_real_zero():
    out = output_used_grade(_ev(observable=True), horizon_at=H, now=H + timedelta(seconds=GRACE), grace_sec=GRACE)
    assert out is not None and out.value == 0.0 and out.source == "expired" and out.note == UNUSED_NOTE
    assert not is_void_note(out.note)
    assert act_outcome({"outcome_source": "expired", "outcome_note": out.note, "outcome_value": 0.0}) == "refuted"


def test_unobservable_at_horizon_is_void_and_never_graded():
    out = output_used_grade(_ev(observable=False), horizon_at=H, now=H + timedelta(seconds=GRACE), grace_sec=GRACE)
    assert out is not None and out.note == UNOBSERVABLE_NOTE and is_void_note(out.note)
    # Not a refuted act for the self-monitor or the body economy's reliability weights.
    assert act_outcome({"outcome_source": "expired", "outcome_note": out.note, "outcome_value": 0.0}) is None


def test_use_after_horizon_is_not_a_hit():
    out = output_used_grade(_ev(first=H + timedelta(minutes=1)), horizon_at=H,
                            now=H + timedelta(seconds=GRACE), grace_sec=GRACE)
    assert out is not None and out.value == 0.0 and out.note == UNUSED_NOTE


def test_producer_close_made_defers_nothing_made_grades():
    assert producer_close_disposition(1.0) == "defer"
    assert producer_close_disposition(0.0) == "no_output"


def test_promise_void_is_also_ungraded_by_shared_act_outcome():
    # Pre-existing gap fixed with Patch 1a: a censored promise void read as "refuted" in reliability weights.
    from orion.grammar.promise_acts import VOID_NOTE
    assert act_outcome({"outcome_source": "expired", "outcome_note": VOID_NOTE, "outcome_value": 0.0}) is None

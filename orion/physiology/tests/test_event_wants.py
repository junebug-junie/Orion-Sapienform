"""Attention as the want competition, patch 1: event wants (event_ref, salience) and weights source (d).

Spec: docs/superpowers/specs/2026-11-04-attention-as-want-competition-design.md.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel, ConfigDict

from orion.physiology.appetite import ActRecord, want_intensity
from orion.physiology.choice import ReservationBook, decide
from orion.physiology.families import parse_families
from orion.physiology.tests.test_choice import NOW, STARVED, body, ctx_for, history, lane, seeded_book, want
from orion.physiology.weights import UseRecord, evidence_for, shrunk_target, weight_series
from orion.schemas.physiology_want import WantDecisionV1, WantV1


# --- Schema -----------------------------------------------------------------------------------------

def test_a_clock_want_serializes_without_the_event_fields():
    """Consumers on extra=forbid that predate patch 1 must keep parsing a want with no event."""
    dumped = want().model_dump(mode="json")
    assert "event_ref" not in dumped and "salience" not in dumped
    assert "event_ref" not in want().model_dump_json()


def test_an_old_consumer_with_forbid_still_parses_a_decision_without_an_event():
    """Stand-in for a not-yet-rebuilt consumer: the pre-patch WantV1 fields with extra=forbid."""
    old_fields = set(WantV1.model_fields) - {"event_ref", "salience"}

    class OldWant(BaseModel):
        model_config = ConfigDict(extra="forbid")
        __annotations__ = {k: object for k in old_fields}

    st = body(lane("agent"))
    d = decide(want(), st, ctx_for(history("durable.self_inquiry", slot={"agent": 600.0}), st), ReservationBook(), NOW)
    payload = json.loads(d.model_dump_json())
    OldWant.model_validate(payload["want"])          # raises on any extra key


def test_an_event_want_round_trips_with_its_event():
    w = want(event_ref="temporal_self_event:42", salience=0.93)
    back = WantV1.model_validate_json(w.model_dump_json())
    assert back.event_ref == "temporal_self_event:42" and back.salience == pytest.approx(0.93)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_salience_is_bounded(bad):
    with pytest.raises(ValueError):
        want(salience=bad)


# --- Appetite: intensity = max(hunger, salience) ----------------------------------------------------

def test_want_intensity_is_max_of_hunger_and_salience():
    assert want_intensity(0.2, None) == (0.2, "hunger")
    assert want_intensity(0.2, 0.9) == (0.9, "salience")
    assert want_intensity(0.7, 0.3) == (0.7, "hunger")
    assert want_intensity(0.0, float("nan")) == (0.0, "hunger")


def test_an_unusual_event_wins_before_the_clock_is_due():
    """A family that just acted is sated on hunger alone; the same want with an unusual event is funded."""
    fresh = history("durable.self_inquiry", slot={"agent": 600.0}, last_h=0.0)
    st = body(lane("agent"))
    clock = decide(want(), st, ctx_for(fresh, st), ReservationBook(), NOW)
    event = decide(want(i=1, event_ref="temporal_self_event:7", salience=0.95), st, ctx_for(fresh, st),
                   ReservationBook(), NOW)
    assert clock.reason == "sated"
    assert event.outcome == "funded" and event.intensity == pytest.approx(0.95)
    assert "event salience 0.95" in event.why
    assert event.want.event_ref == "temporal_self_event:7"


def test_a_dull_event_adds_nothing_over_hunger():
    hungry = history("durable.self_inquiry", slot={"agent": 600.0}, last_h=6.0)
    st = body(lane("agent"))
    a = decide(want(), st, ctx_for(hungry, st), ReservationBook(), NOW)
    b = decide(want(i=1, salience=0.01), st, ctx_for(hungry, st), ReservationBook(), NOW)
    assert a.intensity == b.intensity and a.net == b.net and "event salience" not in b.why


def test_a_tiny_salience_does_not_open_the_exploration_floor():
    """The floor funds families with a drive; an event on a family that just acted competes on its net."""
    fresh = history("durable.self_inquiry", slot={"agent": 600.0}, last_h=0.0)
    st = body(lane("agent", p=5.0, state="binding"))
    d = decide(want(salience=0.01), st, ctx_for(fresh, st), seeded_book(STARVED), NOW)
    assert d.outcome == "deferred" and d.reason != "exploration_floor"


def test_a_producer_intensity_overrides_salience_and_says_so():
    st = body(lane("agent"))
    d = decide(want(intensity=0.4, salience=0.9), st, ctx_for(history("durable.self_inquiry", slot={"agent": 600.0}), st),
               ReservationBook(), NOW)
    assert d.intensity == pytest.approx(0.4) and "salience_ignored:producer_intensity" in d.warnings


def test_the_serialization_schema_keeps_the_wants_shape():
    props = WantV1.model_json_schema(mode="serialization").get("properties") or {}
    assert "family" in props and "salience" in props


# --- Weights: source (d), the self-monitor's reliability ----------------------------------------------

FAMS = parse_families({"version": 1, "families": {
    "good": {"tier": 2, "deferrable_window_s": 3600},
    "bad": {"tier": 2, "deferrable_window_s": 3600},
    "outreach": {"tier": 2, "deferrable_window_s": 3600},
}})
SINCE, UNTIL = NOW - timedelta(days=30), NOW


def closes(family, n, *, value=1.0, source="producer", note="", signal=None, start=NOW - timedelta(days=10)):
    return [ActRecord(act_id=f"{family}-{source}-{start:%j%H}-{i}", family=family, opened_at=start + timedelta(hours=i) - timedelta(minutes=1),
                      closed_at=start + timedelta(hours=i), outcome_value=value, outcome_source=source,
                      outcome_note=note, signal_id=signal) for i in range(n)]


def test_reliability_without_outcome_notes_refuses():
    """Without the note a run that never happened would read as a refuted one."""
    acts = closes("bad", 20, value=0.0, note=None)
    with pytest.raises(ValueError):
        evidence_for("bad", acts, [], since=SINCE, until=UNTIL, reliability=True)


def test_reliability_is_off_unless_asked():
    acts = closes("bad", 40, value=0.0)
    ev = evidence_for("bad", acts, [], since=SINCE, until=UNTIL)
    assert ev.reliability_rate is None and ev.reliability_n == 0 and shrunk_target(1.0, ev) == (1.0, "prior")


def test_reliability_is_the_upheld_share_of_world_settled_closes():
    acts = closes("bad", 30, value=0.0) + closes("bad", 10, value=1.0, start=NOW - timedelta(days=5))
    ev = evidence_for("bad", acts, [], since=SINCE, until=UNTIL, reliability=True)
    assert ev.reliability_rate == pytest.approx(10 / 40) and ev.n == 40
    tgt, basis = shrunk_target(1.0, ev)
    assert basis == "evidence" and tgt == pytest.approx((30 * 1.0 + 40 * 0.25) / 70)


def test_a_run_that_never_ran_is_not_a_grade():
    """failed: ... closes are failed_to_execute in the self-monitor: the claim was never tested."""
    acts = closes("bad", 20, value=0.0, note="failed: rpc timeout") + closes("bad", 12, value=1.0, start=NOW - timedelta(days=3))
    ev = evidence_for("bad", acts, [], since=SINCE, until=UNTIL, reliability=True)
    assert ev.reliability_n == 12 and ev.reliability_rate == 1.0


def test_an_expired_close_is_a_refuted_grade():
    acts = closes("bad", 10, value=None, source="expired") + closes("bad", 10, value=1.0, start=NOW - timedelta(days=3))
    ev = evidence_for("bad", acts, [], since=SINCE, until=UNTIL, reliability=True)
    assert ev.reliability_n == 20 and ev.reliability_rate == pytest.approx(0.5)


def test_resolver_closes_are_source_b_only_never_counted_twice():
    replies = closes("outreach", 20, value=1.0, source="resolver", signal="juniper_replied")
    ev = evidence_for("outreach", replies, [], since=SINCE, until=UNTIL, reliability=True)
    assert ev.engagement_n == 20 and ev.reliability_n == 0 and ev.n == 20


def test_an_act_measured_by_two_sources_counts_once():
    acts = closes("good", 15, value=1.0)
    uses = [UseRecord(act_id=a.act_id, family="good", started_at=a.opened_at, link_status="linked", used=True,
                      window_closed=True) for a in acts]
    ev = evidence_for("good", acts, uses, since=SINCE, until=UNTIL, reliability=True)
    assert ev.use_n == 15 and ev.reliability_n == 15 and ev.n == 15
    assert ev.mean_rate == pytest.approx(1.0)


def test_reliability_moves_weights_within_the_drift_cap_and_rests_to_the_prior():
    acts = closes("good", 60, value=1.0, start=NOW - timedelta(days=25)) + \
        closes("bad", 60, value=0.0, start=NOW - timedelta(days=25))
    end = (NOW + timedelta(days=60)).date()
    start = (NOW - timedelta(days=30)).date()
    off = weight_series(FAMS, acts, [], start=start, end=end)
    on = weight_series(FAMS, acts, [], start=start, end=end, reliability=True)
    assert all(w.weight == 1.0 for day in off.values() for w in day.values())     # nothing read: prior
    peak = on[NOW.date()]
    assert peak["good"].weight > 1.0 > peak["bad"].weight
    days = sorted(on)
    for a, b in zip(days, days[1:]):
        for f in on[b]:
            assert abs(on[b][f].weight - on[a][f].weight) <= 0.1 + 1e-9
    assert all(w.weight == pytest.approx(1.0) and w.basis == "prior" for w in on[end].values())

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


def test_act_output_kinds_maps_actor_to_own_organ_kinds_never_action_outcome():
    from orion.grammar.output_used import act_output_kinds
    raw = {"painting": {"families": ["thought.painting"]},
           "prediction": {"families": ["prediction.experiment"]},
           "body": {"families": None}}
    organ_of = {"visual_run": "painting", "visual_deferral": "painting", "action_outcome": "prediction",
                "experiment_result": "prediction", "gpu_wait": "body"}
    k = act_output_kinds(raw, organ_of)
    pairs = set(zip(k.kind_actors, k.kinds))
    assert ("thought.painting", "visual_run") in pairs and ("thought.painting", "visual_deferral") in pairs
    assert ("prediction.experiment", "experiment_result") in pairs
    assert all(kind != "action_outcome" for kind in k.kinds)
    assert all(kind != "gpu_wait" for actor, kind in pairs if actor == "thought.painting")
    assert dict(zip(k.actors, k.organs)) == {"thought.painting": "painting", "prediction.experiment": "prediction"}


def test_live_registry_loads_and_names_every_switchable_family():
    from orion.grammar.output_used import act_output_kinds
    from orion.integration.organs import load_organs
    reg = load_organs()
    k = act_output_kinds(reg.raw, reg.organ_of)
    for actor in ("thought.painting", "hub.world_pulse_read", "vision.ask", "durable.curiosity_investigate",
                  "durable.self_sense_eval", "hub.chat_reply"):
        assert actor in k.kind_actors, actor


def test_producer_nothing_made_note_keeps_execution_failure_readable():
    from orion.grammar.output_used import NO_OUTPUT_NOTE, producer_close_note
    note = producer_close_note("failed: llm timeout")
    assert note == "failed: llm timeout"
    assert act_outcome({"outcome_source": "producer", "outcome_note": note, "outcome_value": 0.0}) == "failed_to_execute"
    assert act_outcome({"outcome_source": "producer", "outcome_note": producer_close_note("no_planner_decisions"),
                        "outcome_value": 0.0}) == "failed_to_execute"
    assert producer_close_note("  ") == NO_OUTPUT_NOTE


def test_output_used_resolver_close_is_settled_by_the_world_not_juniper():
    from datetime import datetime, timezone
    from orion.self_monitor.adapters import grammar_act
    t = datetime(2026, 11, 25, tzinfo=timezone.utc)
    base = {"act_id": "a1", "actor": "thought.painting", "verb": "make", "object_ref": "", "opened_at": t,
            "closed_at": t, "outcome_source": "resolver", "outcome_value": 1.0, "outcome_note": "used", "predicted": 0.1}
    assert grammar_act({**base, "signal_id": "output_used"}).settled_by == "world"
    assert grammar_act({**base, "signal_id": "juniper_replied"}).settled_by == "juniper"

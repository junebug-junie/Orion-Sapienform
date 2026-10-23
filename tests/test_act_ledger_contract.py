"""Act ledger contract (rung 2, "doing on one track").

Schema: ActV1 extends the grammar (GrammarEventV1.act), closed verb set, expect
required above metabolic, binary outcomes. Grading: closed acts go through the
existing EffectPosterior with constants derived from the binary scale.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import get_args

import pytest
from pydantic import ValidationError

from orion.autonomy.prediction import (
    ACT_BINARY_OBSERVATION_VARIANCE,
    EffectPosterior,
    act_dispatch_kind,
    cold_act_posterior,
    score_act_outcome,
)
from orion.grammar.acts import (
    JUNIPER_REPLY_HORIZON_SEC,
    act_closed_event,
    act_opened_event,
    build_act,
    stable_act_id,
)
from orion.schemas.grammar import ActV1, ActVerb, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.registry import resolve

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)


def _act(**kw):
    base = dict(actor="hub.chat_reply", verb="say", key="c1", signal_id="juniper_replied",
                horizon_sec=JUNIPER_REPLY_HORIZON_SEC, basis="test basis")
    base.update(kw)
    return build_act(**base)


def test_verb_vocabulary_is_the_closed_design_set():
    assert set(get_args(ActVerb)) == {
        "sense", "look", "recall", "infer", "predict", "say", "ask", "wait", "make", "maintain", "regulate", "play"}
    with pytest.raises(ValidationError):
        ActV1(act_id="a", level="faculty", actor="x", verb="ponder")  # type: ignore[arg-type]


def test_expect_required_above_metabolic_only():
    with pytest.raises(ValidationError, match="must carry an expect"):
        ActV1(act_id="a", level="deliberate", actor="x", verb="say")
    ActV1(act_id="a", level="metabolic", actor="x", verb="maintain")


def test_expect_basis_must_not_be_empty():
    with pytest.raises(ValidationError):
        _act(basis="")


def test_act_events_require_act_and_close_requires_outcome():
    prov = GrammarProvenanceV1(source_service="orion-hub")
    with pytest.raises(ValidationError, match="requires act"):
        GrammarEventV1(event_id="e", event_kind="act_opened", trace_id="t", emitted_at=NOW, provenance=prov)
    with pytest.raises(ValidationError, match="requires act.outcome"):
        GrammarEventV1(event_id="e", event_kind="act_closed", trace_id="t", emitted_at=NOW,
                       provenance=prov, act=_act())


def test_outcome_must_be_binary():
    act = _act(signal_id="artifact_produced")
    ev = act_closed_event(act, produced=True, source_service="orion-thought", source_component="t", at=NOW)
    payload = ev.model_dump(mode="json")
    payload["act"]["outcome"]["value"] = 0.5
    with pytest.raises(ValidationError, match="must be 0 or 1"):
        GrammarEventV1.model_validate(payload)


def test_open_and_close_round_trip_and_share_a_trace():
    act = _act()
    opened = act_opened_event(act, source_service="orion-hub", source_component="t", correlation_id="c1", at=NOW)
    closed = act_closed_event(act, produced=False, source_service="orion-hub", source_component="t", at=NOW)
    for ev in (opened, closed):
        again = GrammarEventV1.model_validate(ev.model_dump(mode="json"))
        assert again.act.act_id == act.act_id
    assert opened.trace_id == closed.trace_id == f"act:{act.act_id}"
    assert opened.act.expect.predicted is None, "producers never type the magnitude"
    assert closed.act.outcome.value == 0.0 and closed.act.outcome.source == "producer"


def test_act_id_is_deterministic_per_actor_verb_key():
    assert stable_act_id("a", "say", "k") == stable_act_id("a", "say", "k")
    assert stable_act_id("a", "say", "k") != stable_act_id("a", "ask", "k")


def test_unregistered_producer_identity_is_refused():
    with pytest.raises(ValueError, match="unregistered act producer"):
        act_opened_event(_act(), source_service="orion-somebody-else", source_component="t")


def test_registry_resolves_act_and_grammar_event():
    assert resolve("ActV1") is ActV1
    assert resolve("GrammarEventV1") is GrammarEventV1


def test_act_grading_uses_effect_posterior_on_binary_scale():
    prior = cold_act_posterior()
    assert prior.mean == 0.5 and prior.n == 0
    post, nats, err = score_act_outcome(prior, 1.0)
    assert post.n == 1 and post.mean > 0.5 and nats > 0
    assert err == pytest.approx(0.5)
    # Conjugate update with the binary variance, no new learner.
    precision = 1 / prior.variance + 1 / ACT_BINARY_OBSERVATION_VARIANCE
    assert post.variance == pytest.approx(1 / precision)


def test_repeated_identical_outcomes_stop_paying():
    belief = cold_act_posterior()
    first_nats = None
    for _ in range(200):
        belief, nats, _ = score_act_outcome(belief, 1.0)
        first_nats = nats if first_nats is None else first_nats
    assert nats < first_nats / 50


def test_non_binary_outcome_refused_by_grader():
    with pytest.raises(ValueError):
        score_act_outcome(EffectPosterior(0.5, 0.25, 0), 0.3)


def test_act_dispatch_kind_cannot_collide_with_template_kinds():
    assert act_dispatch_kind("say") == "act:say"


# --- body economy Patch A2: ActCostV2, consumer-first ------------------------------------------------

def test_act_cost_v1_payload_still_parses_as_v1():
    from orion.schemas.grammar import ActCostV1

    closed = act_closed_event(_act(), produced=True, source_service="orion-hub", source_component="t",
                              cost=ActCostV1(seconds=3.0))
    again = GrammarEventV1.model_validate(closed.model_dump(mode="json"))
    assert type(again.act.cost) is ActCostV1 and again.act.cost.seconds == 3.0


def test_act_cost_v2_payload_round_trips_with_new_fields():
    from orion.schemas.grammar import ActCostV2

    cost = ActCostV2(seconds=1.0, gpu_slot_s_by_lane={"fast": 2.5}, wh_by_host={"athena": 0.4},
                     juniper_msgs=1, claude_limit_events=0)
    closed = act_closed_event(_act(), produced=True, source_service="orion-hub", source_component="t", cost=cost)
    again = GrammarEventV1.model_validate(closed.model_dump(mode="json"))
    assert again.act.cost == cost


def test_act_cost_v2_rejects_negative_and_unknown_fields():
    from orion.schemas.grammar import ActCostV2

    with pytest.raises(ValidationError):
        ActCostV2(gpu_slot_s_by_lane={"fast": -1.0})
    with pytest.raises(ValidationError):
        ActV1.model_validate({**_act().model_dump(mode="json"), "cost": {"seconds": 1.0, "dollars": 3}})


def test_act_cost_v2_from_v1_keeps_every_v1_field_and_unknown_stays_none():
    from orion.schemas.grammar import ActCostV1, ActCostV2

    v2 = ActCostV2.from_any(ActCostV1(seconds=1.0, gpu_seconds=2.0, tokens=3, watts=4.0))
    assert (v2.seconds, v2.gpu_seconds, v2.tokens, v2.watts) == (1.0, 2.0, 3, 4.0)
    assert v2.gpu_slot_s_by_lane is None and v2.juniper_msgs is None
    assert ActCostV2.from_any(None) is None


def test_v1_reader_model_rejects_v2_fields_which_is_why_producers_wait():
    """The consumer-first hazard, pinned: V1 alone is extra="forbid"."""
    from orion.schemas.grammar import ActCostV1

    with pytest.raises(ValidationError):
        ActCostV1.model_validate({"seconds": 1.0, "gpu_slot_s_by_lane": {"fast": 1.0}})

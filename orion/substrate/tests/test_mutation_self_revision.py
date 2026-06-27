from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1
from orion.substrate.mutation_pressure import PressureAccumulator, PressurePolicy
from orion.substrate.mutation_proposals import ProposalFactory
from orion.substrate.mutation_self_revision import prediction_error_mutation_signals


def _self_state(prediction_error_scores: dict[str, float]) -> SelfStateV1:
    now = datetime.now(timezone.utc)
    dims = {
        dim: SelfStateDimensionV1(dimension_id=dim, score=0.5, confidence=0.7)
        for dim in prediction_error_scores
    }
    return SelfStateV1(
        self_state_id="ss-revision",
        generated_at=now,
        source_field_tick_id="ft",
        source_field_generated_at=now,
        source_attention_frame_id="af",
        source_attention_generated_at=now,
        overall_condition="strained",
        overall_intensity=0.6,
        overall_confidence=0.6,
        dimensions=dims,
        prediction_error_scores=prediction_error_scores,
    )


def test_sustained_self_model_error_emits_only_supported_cognitive_signals() -> None:
    signals = prediction_error_mutation_signals(
        _self_state({"continuity_pressure": 0.7, "transport_integrity": 0.9})
    )
    # continuity maps to a supported cognitive surface; transport_integrity has no
    # cognitive surface, so it is ignored rather than invented.
    surfaces = {s.target_surface for s in signals}
    assert surfaces == {"cognitive_identity_continuity_adjustment"}
    sig = signals[0]
    assert sig.event_kind == "self_model_drift:continuity_pressure"
    assert sig.strength == 0.7
    assert sig.metadata["source_kind"] == "self_model_prediction_error"


def test_calm_self_model_emits_nothing() -> None:
    # all dimensions below the min_error floor
    assert prediction_error_mutation_signals(
        _self_state({"continuity_pressure": 0.1, "coherence": 0.05})
    ) == []


def test_single_weak_signal_is_below_threshold_but_sustained_error_drafts_a_proposal() -> None:
    accumulator = PressureAccumulator(policy=PressurePolicy())
    factory = ProposalFactory()
    now = datetime.now(timezone.utc)

    # a mild-but-real drift: strength 0.3 -> +1.5 per tick, threshold is 3.0
    signal = prediction_error_mutation_signals(
        _self_state({"continuity_pressure": 0.3}), min_error=0.3
    )[0]

    pressure = accumulator.apply(current=None, signal=signal, now=now)
    # one weak tick is not enough — governance: surprise must persist
    assert not accumulator.ready_for_proposal(pressure, now=now)

    # the same drift, sustained over further ticks, accumulates past the threshold
    ticks = 1
    while not accumulator.ready_for_proposal(pressure, now=now) and ticks < 10:
        pressure = accumulator.apply(current=pressure, signal=signal, now=now)
        ticks += 1
    assert accumulator.ready_for_proposal(pressure, now=now)
    assert ticks > 1  # required more than a single tick

    proposal = factory.from_pressure(pressure)
    assert proposal is not None
    # routed through the existing governed cognitive lane...
    assert proposal.lane == "cognitive"
    assert proposal.mutation_class == "cognitive_identity_continuity_adjustment"
    # ...and is a DRAFT — never auto-applied
    assert proposal.patch.patch["not_applied_status"] == "draft_only_not_applied"

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.autonomy.deviation_gate import DeviationGate
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1
from orion.self_state.deviation import observe_dimension_deviation
from orion.self_state.policy import load_self_state_policy

REPO = Path(__file__).resolve().parents[1]
POLICY = load_self_state_policy(REPO / "config" / "self_state" / "self_state_policy.v1.yaml")
NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


def _state(scores: dict[str, float]) -> SelfStateV1:
    dims = {
        dim_id: SelfStateDimensionV1(dimension_id=dim_id, score=score, confidence=0.8)
        for dim_id, score in scores.items()
    }
    return SelfStateV1(
        self_state_id="self.state:test",
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        source_attention_frame_id="frame",
        source_attention_generated_at=NOW,
        overall_condition="steady",
        overall_intensity=0.4,
        overall_confidence=0.8,
        dimensions=dims,
    )


def test_policy_dimension_worse_direction_covers_every_real_dimension() -> None:
    # 11 core ALL_DIMENSION_IDS + transport_integrity (conditional 13th) --
    # policy_pressure is gone (Phase 0), so 12 total. Loaded from the real,
    # live config/self_state/self_state_policy.v1.yaml, not a hardcoded
    # Python table (2026-07-12 review: a second, independently-maintained
    # source of the same fact is a drift risk this redesign exists to kill).
    assert set(POLICY.dimension_worse_direction) == {
        "field_intensity",
        "coherence",
        "uncertainty",
        "agency_readiness",
        "resource_pressure",
        "execution_pressure",
        "reasoning_pressure",
        "reliability_pressure",
        "continuity_pressure",
        "introspection_pressure",
        "social_pressure",
        "transport_integrity",
    }


def test_worse_direction_behaviorally_matches_scoring_conventions_for_every_dimension() -> None:
    # 2026-07-12 review finding: a prior version of this test only re-asserted
    # the hardcoded up/down values without independently driving real scores
    # through the gate -- only one dimension was behaviorally verified. This
    # version drives a real rise and a real fall through a fresh gate for
    # EVERY dimension and checks the impulse only fires in the declared
    # worse direction, for all 12.
    for dim_id, worse in POLICY.dimension_worse_direction.items():
        gate = DeviationGate(warmup=3)
        for _ in range(4):
            observe_dimension_deviation(gate, _state({dim_id: 0.50}), POLICY)

        rise = observe_dimension_deviation(gate, _state({dim_id: 0.99}), POLICY)[dim_id]
        gate2 = DeviationGate(warmup=3)
        for _ in range(4):
            observe_dimension_deviation(gate2, _state({dim_id: 0.50}), POLICY)
        fall = observe_dimension_deviation(gate2, _state({dim_id: 0.01}), POLICY)[dim_id]

        if worse == "up":
            assert rise > 0.0, f"{dim_id} (worse=up) should impulse on rise"
            assert fall == 0.0, f"{dim_id} (worse=up) should not impulse on fall"
        else:
            assert fall > 0.0, f"{dim_id} (worse=down) should impulse on fall"
            assert rise == 0.0, f"{dim_id} (worse=down) should not impulse on rise"


def test_observe_dimension_deviation_warms_up_then_fires_on_real_deviation() -> None:
    gate = DeviationGate(warmup=3)
    # Warm up resource_pressure on a stable baseline (~0.20).
    for _ in range(4):
        observe_dimension_deviation(gate, _state({"resource_pressure": 0.20}), POLICY)

    # A sharp rise past the learned baseline should now fire an impulse
    # (resource_pressure is worse=up).
    impulses = observe_dimension_deviation(gate, _state({"resource_pressure": 0.95}), POLICY)
    assert impulses["resource_pressure"] > 0.0


def test_observe_dimension_deviation_no_impulse_within_normal_range() -> None:
    gate = DeviationGate(warmup=3)
    for _ in range(4):
        observe_dimension_deviation(gate, _state({"coherence": 0.80}), POLICY)

    # A tiny wobble within normal noise must not fire.
    impulses = observe_dimension_deviation(gate, _state({"coherence": 0.81}), POLICY)
    assert impulses["coherence"] == 0.0


def test_observe_dimension_deviation_defensively_casts_bad_confidence() -> None:
    # Mirrors orion/autonomy/signal_tension.py's existing defensive cast
    # pattern for this same shared gate (2026-07-12 review finding):
    # DeviationGate.observe()'s own float(confidence) coercion isn't guarded
    # the way its x/score coercion is, so the caller must defend instead.
    # SelfStateDimensionV1.confidence is normally Pydantic-validated, but
    # model_config doesn't set validate_assignment=True, so a plain
    # attribute set bypasses that validation -- exercising the same
    # bypass path a deserialization edge case could hit.
    gate = DeviationGate(warmup=3)
    state = _state({"resource_pressure": 0.5})
    state.dimensions["resource_pressure"].confidence = "not-a-number"  # type: ignore[assignment]
    # Must not raise even with a non-numeric confidence value.
    impulses = observe_dimension_deviation(gate, state, POLICY)
    assert "resource_pressure" in impulses


def test_observe_dimension_deviation_never_raises_on_unknown_dimension() -> None:
    gate = DeviationGate()
    state = _state({"resource_pressure": 0.5})
    # Inject a dimension id the policy's dimension_worse_direction doesn't
    # know about -- must be skipped, not raise.
    state.dimensions["made_up_dimension"] = SelfStateDimensionV1(
        dimension_id="resource_pressure",  # valid literal, arbitrary key name
        score=0.5,
        confidence=0.5,
    )
    impulses = observe_dimension_deviation(gate, state, POLICY)
    assert "made_up_dimension" not in impulses
    assert "resource_pressure" in impulses

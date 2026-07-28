from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.policy import load_attention_policy
from orion.attention.field_attention.selectors import select_system_targets
from orion.field.pressure import RECENT_PERTURBATION_EWMA_MIN_SAMPLES
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")

BASE = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)


def _field(
    *,
    recent_perturbations: list[str],
    zscore: float | None,
    ewma_n: int,
) -> FieldStateV1:
    return FieldStateV1(
        generated_at=BASE,
        tick_id="tick_test",
        recent_perturbations=recent_perturbations,
        recent_perturbation_zscore=zscore,
        recent_perturbation_ewma_n=ewma_n,
    )


def test_no_recent_perturbations_yields_no_target() -> None:
    field = _field(recent_perturbations=[], zscore=5.0, ewma_n=10)
    assert select_system_targets(field, POLICY) == []


def test_no_baseline_yet_yields_no_target() -> None:
    # count > 0 but the EWMA has never observed a second sample -- there is
    # no baseline to be anomalous against (compute_ewma_update's own
    # documented first-observation behavior).
    field = _field(recent_perturbations=["a"], zscore=None, ewma_n=1)
    assert select_system_targets(field, POLICY) == []


def test_below_cold_start_min_samples_yields_no_target() -> None:
    # A numeric zscore already exists, but with too few samples the
    # variance estimate is unreliable (hand-verified: z=1000 on the second
    # observation for a steady ramp) -- must not be trusted yet.
    field = _field(
        recent_perturbations=["a", "b"],
        zscore=50.0,
        ewma_n=RECENT_PERTURBATION_EWMA_MIN_SAMPLES - 1,
    )
    assert select_system_targets(field, POLICY) == []


def test_below_baseline_deviation_is_not_salient() -> None:
    # Quieter than usual isn't "surprising" in the sense this target cares
    # about -- only busier-than-usual bursts should attend here.
    field = _field(
        recent_perturbations=["a"],
        zscore=-2.0,
        ewma_n=RECENT_PERTURBATION_EWMA_MIN_SAMPLES,
    )
    assert select_system_targets(field, POLICY) == []


def test_elevated_deviation_below_min_salience_threshold_yields_no_target() -> None:
    field = _field(
        recent_perturbations=["a"],
        zscore=0.1,  # salience = 0.1/3.0 ~= 0.033, below policy's min_salience 0.10
        ewma_n=RECENT_PERTURBATION_EWMA_MIN_SAMPLES,
    )
    assert select_system_targets(field, POLICY) == []


def test_genuine_burst_produces_a_system_target() -> None:
    field = _field(
        recent_perturbations=[f"label_{i}" for i in range(93)],
        zscore=4.5,  # past the 3.0 saturation point
        ewma_n=RECENT_PERTURBATION_EWMA_MIN_SAMPLES,
    )
    targets = select_system_targets(field, POLICY)
    assert len(targets) == 1
    target = targets[0]
    assert target.target_id == "field:recent_perturbations"
    assert target.target_kind == "system"
    assert target.salience_score == 1.0  # saturates at zscore >= 3.0
    assert "93" in target.reasons[0]

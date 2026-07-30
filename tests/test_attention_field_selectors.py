from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.policy import load_attention_policy
from orion.attention.field_attention.selectors import (
    PREDICTION_ERROR_NATIVE_TARGETS,
    select_capability_targets,
    select_node_targets,
    select_system_targets,
)
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


# --- select_node_targets / select_capability_targets: 2026-07-30 rewrite ---
# (Candidate A precision-weighted salience; hand-weighted compute_salience()
# killed, no fallback.)

_FIELD_FOR_NODE_TESTS = FieldStateV1(
    generated_at=BASE,
    tick_id="tick_node_test",
    node_vectors={
        "node:athena": {"cortex_exec_step_load": 1.0},  # physical host -- no real grounding
    },
)


def test_select_node_targets_scores_only_prediction_error_native_ids() -> None:
    histories = {"node:substrate.execution": [0.05, 0.06, 0.04, 0.05, 0.9]}
    targets = select_node_targets(_FIELD_FOR_NODE_TESTS, POLICY, histories)
    target_ids = {t.target_id for t in targets}
    assert target_ids == {"node:substrate.execution"}
    assert "node:athena" not in target_ids  # never scored, not zero-scored


def test_select_node_targets_excludes_targets_with_no_real_history() -> None:
    targets = select_node_targets(_FIELD_FOR_NODE_TESTS, POLICY, {})
    assert targets == []


def test_select_node_targets_excludes_targets_with_empty_history_list() -> None:
    histories = {"node:substrate.execution": []}
    targets = select_node_targets(_FIELD_FOR_NODE_TESTS, POLICY, histories)
    assert targets == []


def test_select_node_targets_confidence_scales_with_sample_count() -> None:
    few = select_node_targets(
        _FIELD_FOR_NODE_TESTS, POLICY, {"node:substrate.execution": [0.1, 0.2]}
    )[0]
    many = select_node_targets(
        _FIELD_FOR_NODE_TESTS, POLICY, {"node:substrate.execution": [0.1] * 25 + [0.9]}
    )[0]
    assert few.confidence_score < many.confidence_score
    assert many.confidence_score == 1.0  # clamped at QUALIFYING_MIN_ROWS=20


def test_select_node_targets_only_covers_the_confirmed_five_reducers() -> None:
    # Locks the real, live-confirmed mapping (services/orion-substrate-runtime/
    # app/worker.py's _prediction_error_receipt call sites) -- a silent
    # addition/removal here would change which real signals ground live
    # attention without anyone noticing. `node:substrate.transport` is
    # deliberately excluded (code review, 2026-07-30): its write was
    # permanently retired 2026-07-26, superseded by bus_synaptic -- an
    # earlier draft's inclusion of it here was a real bug this test would
    # have caught, and now does.
    assert set(PREDICTION_ERROR_NATIVE_TARGETS.keys()) == {
        "node:substrate.biometrics",
        "node:substrate.execution",
        "node:substrate.chat",
        "node:substrate.route",
        "node:substrate.bus_synaptic",
    }
    assert "node:substrate.transport" not in PREDICTION_ERROR_NATIVE_TARGETS


def test_select_node_targets_multi_target_competition_normalizes_min_to_zero_max_to_one() -> None:
    # Code-review regression test (2026-07-30): the only rewritten test that
    # previously exercised normalize_across_targets() with more than one real
    # competitor -- everything else fed it a single target, which trivially
    # normalizes to 1.0 and never exercises the real min-max path.
    #
    # Deliberately does NOT assert which target_id "should" win by hand-
    # predicting precision_weighted_salience()'s output: that function
    # includes the current reading in its own variance estimate, so a real
    # anomaly partially self-defeats its own salience (a spike inflates the
    # very variance that determines its precision) -- confirmed by hand this
    # session, non-obvious, and not this test's job to re-derive. This test
    # instead checks the actual mechanical guarantee normalize_across_targets
    # makes: whichever of two DIFFERENT real raw scores is lower maps to
    # exactly 0.0, the higher to exactly 1.0.
    histories = {
        "node:substrate.chat": [0.01, 0.01, 0.02, 0.01, 0.02],
        "node:substrate.execution": [0.02, 0.03, 0.02, 0.03, 0.95],
    }
    from orion.attention.field_attention.candidate_precision_weighted import (
        precision_weighted_salience,
    )

    raw = {k: precision_weighted_salience(v).salience for k, v in histories.items()}
    weaker_id = min(raw, key=raw.get)
    stronger_id = max(raw, key=raw.get)
    assert raw[weaker_id] != raw[stronger_id]  # sanity: the fixture must be non-degenerate

    targets = select_node_targets(_FIELD_FOR_NODE_TESTS, POLICY, histories)
    by_id = {t.target_id: t for t in targets}
    assert len(targets) == 2
    # Min-max normalization: the weaker real competitor floors to exactly 0.0,
    # the stronger one ceilings to exactly 1.0 -- by construction, not
    # incidentally. Disclosed design property (code review, 2026-07-30): this
    # means the weakest of N real competitors is *always* classified
    # suppressed by build_attention_frame()'s suppress_below threshold,
    # regardless of its own absolute precision-weighted magnitude -- relative
    # rank, not absolute alarm level, decides suppression when 2+ targets are
    # qualified this tick.
    assert by_id[weaker_id].salience_score == 0.0
    assert by_id[stronger_id].salience_score == 1.0


def test_select_node_targets_min_samples_boundary_at_qualifying_min_rows() -> None:
    just_under = select_node_targets(
        _FIELD_FOR_NODE_TESTS, POLICY, {"node:substrate.execution": [0.1] * 19}
    )[0]
    exactly_at = select_node_targets(
        _FIELD_FOR_NODE_TESTS, POLICY, {"node:substrate.execution": [0.1] * 20}
    )[0]
    over = select_node_targets(
        _FIELD_FOR_NODE_TESTS, POLICY, {"node:substrate.execution": [0.1] * 25}
    )[0]
    assert just_under.confidence_score < 1.0
    assert exactly_at.confidence_score == 1.0
    assert over.confidence_score == 1.0  # clamped, does not exceed 1.0


def test_select_node_targets_surfaces_variance_floored_in_reasons() -> None:
    # A near-perfectly-constant real history -- precision_weighted_salience's
    # own variance-floor instability case. Must be visible in `reasons`, not
    # silently absorbed into a plain-looking salience number.
    near_constant = [0.5] * 25
    targets = select_node_targets(
        _FIELD_FOR_NODE_TESTS, POLICY, {"node:substrate.execution": near_constant}
    )
    assert len(targets) == 1
    joined_reasons = " ".join(targets[0].reasons)
    assert "variance-floor instability" in joined_reasons


def test_select_capability_targets_always_empty() -> None:
    field_with_capabilities = FieldStateV1(
        generated_at=BASE,
        tick_id="tick_cap_test",
        capability_vectors={"capability:orchestration": {"execution_pressure": 1.0}},
    )
    assert select_capability_targets(field_with_capabilities, POLICY) == []

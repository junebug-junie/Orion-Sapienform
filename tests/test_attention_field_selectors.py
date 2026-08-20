from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.candidate_precision_weighted import (
    NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
    NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
    PrecisionEwmaBaseline,
    advance_precision_baseline,
)
from orion.attention.field_attention.policy import load_attention_policy
from orion.attention.field_attention.selectors import (
    PREDICTION_ERROR_NATIVE_TARGETS,
    _current_pressure_proxy,
    select_capability_targets,
    select_host_targets,
    select_node_targets,
    select_system_targets,
)
from orion.field.pressure import RECENT_PERTURBATION_EWMA_MIN_SAMPLES
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1


def _baseline(values: list[float]) -> PrecisionEwmaBaseline:
    """Test helper: build a PrecisionEwmaBaseline the same way the live worker
    does -- folding real values in one at a time via advance_precision_baseline
    -- rather than hand-constructing baseline fields directly. Empty `values`
    yields the cold-start zero baseline (n=0, excluded by select_node_targets)."""
    baseline = PrecisionEwmaBaseline()
    return advance_precision_baseline(
        baseline,
        values,
        alpha=NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
        min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
    )

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
    baselines = {"node:substrate.execution": _baseline([0.05, 0.06, 0.04, 0.05, 0.9])}
    targets = select_node_targets(_FIELD_FOR_NODE_TESTS, POLICY, baselines)
    target_ids = {t.target_id for t in targets}
    assert target_ids == {"node:substrate.execution"}
    assert "node:athena" not in target_ids  # never scored, not zero-scored


def test_select_node_targets_excludes_targets_with_no_real_history() -> None:
    targets = select_node_targets(_FIELD_FOR_NODE_TESTS, POLICY, {})
    assert targets == []


def test_select_node_targets_excludes_targets_with_empty_history_list() -> None:
    baselines = {"node:substrate.execution": _baseline([])}
    targets = select_node_targets(_FIELD_FOR_NODE_TESTS, POLICY, baselines)
    assert targets == []


def test_select_node_targets_confidence_scales_with_sample_count() -> None:
    few = select_node_targets(
        _FIELD_FOR_NODE_TESTS, POLICY, {"node:substrate.execution": _baseline([0.1, 0.2])}
    )[0]
    many = select_node_targets(
        _FIELD_FOR_NODE_TESTS,
        POLICY,
        {"node:substrate.execution": _baseline([0.1] * 25 + [0.9])},
    )[0]
    assert few.confidence_score < many.confidence_score
    assert many.confidence_score == 1.0  # clamped at QUALIFYING_MIN_ROWS=20


def test_select_node_targets_observation_count_survives_a_pruned_window() -> None:
    """2026-07-30 regression test for the live incident (see orion/
    sentience_striving_program/README.md §12): a target's real n_samples must
    reflect its persisted baseline's cumulative observation_count, not shrink
    just because a caller happens to construct the baseline from a small
    per-tick fetch. Simulates two separate ticks (a real historical batch, then
    a later small batch of only-new receipts) folding into the SAME persisted
    baseline -- confidence must reflect the full accumulated count, not just the
    most recent tick's fetch size."""
    baseline = _baseline([0.1] * 19)  # tick 1: a real historical backlog
    baseline = advance_precision_baseline(
        baseline,
        [0.2],  # tick 2: exactly one new real receipt
        alpha=NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
        min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
    )
    targets = select_node_targets(
        _FIELD_FOR_NODE_TESTS, POLICY, {"node:substrate.execution": baseline}
    )
    assert len(targets) == 1
    assert targets[0].confidence_score == 1.0  # 20 real observations total, fully qualified
    assert "n=20" in targets[0].reasons[0]


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
        cross_domain_variance_floor,
        precision_weighted_salience_from_baseline,
    )

    baselines = {k: _baseline(v) for k, v in histories.items()}
    # 2026-08-20: use the same cross-domain floor select_node_targets() itself
    # now applies (see that function's docstring for why the old flat global
    # floor was replaced) -- not the old flat NODE_TARGET_PREDICTION_ERROR_MIN_
    # VARIANCE, which would predict the pre-fix winner and no longer matches
    # what select_node_targets actually computes.
    raw = {
        k: precision_weighted_salience_from_baseline(
            b,
            min_variance=cross_domain_variance_floor(
                baselines, k, min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE
            ),
        ).salience
        for k, b in baselines.items()
    }
    weaker_id = min(raw, key=raw.get)
    stronger_id = max(raw, key=raw.get)
    assert raw[weaker_id] != raw[stronger_id]  # sanity: the fixture must be non-degenerate

    targets = select_node_targets(_FIELD_FOR_NODE_TESTS, POLICY, baselines)
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
        _FIELD_FOR_NODE_TESTS, POLICY, {"node:substrate.execution": _baseline([0.1] * 19)}
    )[0]
    exactly_at = select_node_targets(
        _FIELD_FOR_NODE_TESTS, POLICY, {"node:substrate.execution": _baseline([0.1] * 20)}
    )[0]
    over = select_node_targets(
        _FIELD_FOR_NODE_TESTS, POLICY, {"node:substrate.execution": _baseline([0.1] * 25)}
    )[0]
    assert just_under.confidence_score < 1.0
    assert exactly_at.confidence_score == 1.0
    assert over.confidence_score == 1.0  # clamped, does not exceed 1.0


def test_select_node_targets_surfaces_variance_floored_in_reasons() -> None:
    # A near-perfectly-constant real history -- precision_weighted_salience's
    # own variance-floor instability case. Must be visible in `reasons`, not
    # silently absorbed into a plain-looking salience number.
    near_constant = _baseline([0.5] * 25)
    targets = select_node_targets(
        _FIELD_FOR_NODE_TESTS, POLICY, {"node:substrate.execution": near_constant}
    )
    assert len(targets) == 1
    joined_reasons = " ".join(targets[0].reasons)
    assert "variance-floor instability" in joined_reasons


# --- select_host_targets / select_capability_targets: 2026-07-30 second
# rewrite (Candidate B novelty_scorer for the targets Candidate A can't
# reach -- no longer always []).


def test_current_pressure_proxy_is_the_max_real_channel() -> None:
    assert _current_pressure_proxy({"cpu_pressure": 0.3, "gpu_pressure": 0.8, "disk_pressure": 0.1}) == 0.8


def test_current_pressure_proxy_empty_vector_is_zero() -> None:
    assert _current_pressure_proxy({}) == 0.0


def test_current_pressure_proxy_drops_non_finite_and_non_numeric() -> None:
    assert _current_pressure_proxy({"a": float("nan"), "b": float("inf"), "c": "not_a_number"}) == 0.0
    assert _current_pressure_proxy({"a": float("nan"), "b": 0.4}) == 0.4


def test_current_pressure_proxy_clamps_out_of_range() -> None:
    assert _current_pressure_proxy({"a": 2.5}) == 1.0
    assert _current_pressure_proxy({"a": -1.0}) == 0.0


def test_current_pressure_proxy_inverts_higher_is_better_channels() -> None:
    # Code-review regression test (2026-07-30, Finding 1, CRITICAL):
    # `availability`/`delivery_confidence`/`stream_backlog_health` (node) and
    # `confidence`/`available_capacity` (capability) default to/sit near 1.0
    # when healthy -- without inversion, max() would report ~1.0 for a
    # fully-calm vector purely because of these channels, identical to a
    # genuinely overloaded one.
    calm = {"cpu_pressure": 0.0, "availability": 1.0}
    overloaded = {"cpu_pressure": 0.0, "availability": 0.05}  # host barely reachable
    assert _current_pressure_proxy(calm) == 0.0  # availability=1.0 inverts to 0.0
    assert _current_pressure_proxy(overloaded) == 0.95  # availability=0.05 inverts to 0.95
    assert _current_pressure_proxy(overloaded) > _current_pressure_proxy(calm)


def test_current_pressure_proxy_does_not_invert_ordinary_pressure_channels() -> None:
    # Sanity check that inversion is scoped to the confirmed higher-is-
    # better set only -- an ordinary *_pressure channel at 1.0 must still
    # mean "real severe pressure", not get flipped to 0.0.
    assert _current_pressure_proxy({"cpu_pressure": 1.0}) == 1.0
    assert _current_pressure_proxy({"execution_pressure": 0.9}) == 0.9


def test_current_pressure_proxy_capability_higher_is_better_channels() -> None:
    calm = {"confidence": 1.0, "available_capacity": 1.0, "reasoning_pressure": 0.0}
    degraded = {"confidence": 0.1, "available_capacity": 0.2, "reasoning_pressure": 0.0}
    assert _current_pressure_proxy(calm) == 0.0
    assert _current_pressure_proxy(degraded) == 0.9  # 1 - 0.1 confidence wins the max


def test_select_host_targets_excludes_prediction_error_native_ids() -> None:
    # node:substrate.* targets belong to Candidate A (select_node_targets),
    # never double-scored here even though they're real FieldStateV1.
    # node_vectors keys too.
    field = FieldStateV1(
        generated_at=BASE,
        tick_id="tick_host_test",
        node_vectors={
            "node:athena": {"cpu_pressure": 0.5},
            "node:substrate.execution": {"prediction_error": 0.9},
        },
    )
    targets = select_host_targets(field, POLICY, None)
    target_ids = {t.target_id for t in targets}
    assert target_ids == {"node:athena"}


def test_select_host_targets_first_tick_has_zero_novelty_not_excluded() -> None:
    # No previous_frame -- novelty_for_target's own real contract: no prior
    # to diff against, so novelty is genuinely 0.0. The target still
    # appears (real vector data exists this tick), just with zero novelty --
    # different from Candidate A's "no history at all -> excluded entirely."
    field = FieldStateV1(
        generated_at=BASE,
        tick_id="tick_host_cold",
        node_vectors={"node:athena": {"cpu_pressure": 0.9}},
    )
    targets = select_host_targets(field, POLICY, None)
    assert len(targets) == 1
    assert targets[0].novelty_score == 0.0
    assert targets[0].salience_score == 0.0
    assert targets[0].confidence_score == 0.0  # no real prior context to be confident about


def test_select_host_targets_real_novelty_against_a_prior_frame() -> None:
    field_prev = FieldStateV1(
        generated_at=BASE,
        tick_id="tick_host_prev",
        node_vectors={"node:athena": {"cpu_pressure": 0.1}},
    )
    prev_targets = select_host_targets(field_prev, POLICY, None)
    prev_frame = FieldAttentionFrameV1(
        frame_id="f1",
        generated_at=BASE,
        source_field_tick_id="tick_host_prev",
        source_field_generated_at=BASE,
        attention_policy_id=POLICY.policy_id,
        overall_salience=0.0,
        dominant_targets=prev_targets,
        node_targets=prev_targets,
        capability_targets=[],
        system_targets=[],
        suppressed_targets=[],
        recent_perturbations=[],
        warnings=[],
    )

    field_now = FieldStateV1(
        generated_at=BASE,
        tick_id="tick_host_now",
        node_vectors={"node:athena": {"cpu_pressure": 0.9}},  # real, large jump
    )
    targets = select_host_targets(field_now, POLICY, prev_frame)
    assert len(targets) == 1
    assert targets[0].novelty_score > 0.5  # real, large change vs. the prior tick
    assert targets[0].confidence_score == 1.0  # real prior context existed this time


def test_select_capability_targets_uses_novelty_not_always_empty() -> None:
    field = FieldStateV1(
        generated_at=BASE,
        tick_id="tick_cap_test",
        capability_vectors={"capability:orchestration": {"execution_pressure": 1.0}},
    )
    targets = select_capability_targets(field, POLICY, None)
    assert len(targets) == 1
    assert targets[0].target_id == "capability:orchestration"
    assert targets[0].target_kind == "capability"


def test_select_capability_targets_empty_vectors_yields_no_targets() -> None:
    field = FieldStateV1(generated_at=BASE, tick_id="tick_cap_empty")
    assert select_capability_targets(field, POLICY, None) == []


def test_select_host_targets_new_target_absent_from_existing_prior_frame_has_zero_confidence() -> None:
    # Code-review regression test (2026-07-30, Finding 2, HIGH): a
    # previous_frame existing at all is not the same claim as THIS target
    # having a real entry in it. node:circe is brand-new this tick;
    # node:athena was in the prior frame. Only athena should read
    # confidence_score=1.0.
    field_prev = FieldStateV1(
        generated_at=BASE,
        tick_id="tick_host_prev2",
        node_vectors={"node:athena": {"cpu_pressure": 0.1}},
    )
    prev_targets = select_host_targets(field_prev, POLICY, None)
    prev_frame = FieldAttentionFrameV1(
        frame_id="f2",
        generated_at=BASE,
        source_field_tick_id="tick_host_prev2",
        source_field_generated_at=BASE,
        attention_policy_id=POLICY.policy_id,
        overall_salience=0.0,
        dominant_targets=prev_targets,
        node_targets=prev_targets,
        capability_targets=[],
        system_targets=[],
        suppressed_targets=[],
        recent_perturbations=[],
        warnings=[],
    )

    field_now = FieldStateV1(
        generated_at=BASE,
        tick_id="tick_host_now2",
        node_vectors={
            "node:athena": {"cpu_pressure": 0.2},
            "node:circe": {"cpu_pressure": 0.5},  # new this tick, absent from prev_frame
        },
    )
    targets = select_host_targets(field_now, POLICY, prev_frame)
    by_id = {t.target_id: t for t in targets}
    assert len(targets) == 2
    assert by_id["node:athena"].confidence_score == 1.0  # real prior entry existed
    assert by_id["node:circe"].confidence_score == 0.0  # no real prior entry, despite a real prior_frame existing


def test_select_host_targets_suppressed_prior_entry_still_counts_as_real() -> None:
    # Code-review verification-pass regression test (2026-07-30): a target
    # scored below suppress_below last tick lands ONLY in
    # previous_frame.suppressed_targets, never in .node_targets -- a real
    # prior observation, not an absent one. novelty_for_target() (via
    # prior_salience_for_target()) searches suppressed_targets too and uses
    # that real prior value to compute novelty -- confidence_score must
    # agree that a real prior existed, not just check the two "active"
    # buckets.
    suppressed_prior = FieldAttentionTargetV1(
        target_id="node:circe",
        target_kind="node",
        salience_score=0.02,  # below suppress_below -- landed in suppressed_targets only
        pressure_score=0.02,
        novelty_score=0.0,
        urgency_score=0.0,
        confidence_score=1.0,
        dominant_channels={},
        reasons=["prior tick"],
        evidence_refs=[],
        suggested_observation_mode="ignore",
    )
    prev_frame = FieldAttentionFrameV1(
        frame_id="f3",
        generated_at=BASE,
        source_field_tick_id="tick_host_prev3",
        source_field_generated_at=BASE,
        attention_policy_id=POLICY.policy_id,
        overall_salience=0.0,
        dominant_targets=[],
        node_targets=[],  # NOT here
        capability_targets=[],
        system_targets=[],
        suppressed_targets=[suppressed_prior],  # only here
        recent_perturbations=[],
        warnings=[],
    )

    field_now = FieldStateV1(
        generated_at=BASE,
        tick_id="tick_host_now3",
        node_vectors={"node:circe": {"cpu_pressure": 0.9}},
    )
    targets = select_host_targets(field_now, POLICY, prev_frame)
    assert len(targets) == 1
    # Real novelty is computed against the real suppressed-bucket prior value.
    assert targets[0].novelty_score > 0.5
    # Confidence must agree a real prior existed, even though it lived in
    # suppressed_targets, not node_targets.
    assert targets[0].confidence_score == 1.0


def test_select_host_targets_multi_target_real_hosts() -> None:
    # Code-review regression test (2026-07-30, Finding 4): production always
    # faces multiple real hosts simultaneously (athena/atlas/circe/
    # prometheus/rpc_timeout) -- no prior test exercised more than one.
    field = FieldStateV1(
        generated_at=BASE,
        tick_id="tick_host_multi",
        node_vectors={
            "node:athena": {"cpu_pressure": 0.1},
            "node:atlas": {"cpu_pressure": 0.9},
            "node:circe": {"availability": 0.2},  # inverts to 0.8
        },
    )
    targets = select_host_targets(field, POLICY, None)
    target_ids = {t.target_id for t in targets}
    assert target_ids == {"node:athena", "node:atlas", "node:circe"}
    by_id = {t.target_id: t for t in targets}
    assert by_id["node:atlas"].pressure_score == 0.9
    assert by_id["node:circe"].pressure_score == 0.8


def test_select_capability_targets_multi_target_real_capabilities() -> None:
    field = FieldStateV1(
        generated_at=BASE,
        tick_id="tick_cap_multi",
        capability_vectors={
            "capability:graph": {"pressure": 0.2},
            "capability:memory": {"pressure": 0.7},
            "capability:vision": {"confidence": 0.3},  # inverts to 0.7
        },
    )
    targets = select_capability_targets(field, POLICY, None)
    target_ids = {t.target_id for t in targets}
    assert target_ids == {"capability:graph", "capability:memory", "capability:vision"}
    by_id = {t.target_id: t for t in targets}
    assert by_id["capability:memory"].pressure_score == 0.7
    assert by_id["capability:vision"].pressure_score == 0.7

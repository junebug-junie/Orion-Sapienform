"""Deterministic unit tests for measure_proposal_feedback_correlation.py.

No DB, no network. Everything here is pure: template-key extraction, bucket
stats, cortex completion-rate signals, dominant-dimension resolution (via the
real `ProposalPolicyV1`/`ProposalTemplateV1` schema, not a reimplementation),
and the STOP/PROCEED verdict logic. Same sibling-module-by-file-path loading
pattern as test_measure_precision_weighted_salience_probe.py.

Lives in top-level `tests/`, not `scripts/analysis/tests/` -- `scripts` is in
`pyproject.toml`'s `norecursedirs`, so a bare `pytest` run from repo root
never discovers it there (documented precedent, see
test_measure_precision_weighted_salience_probe.py's own module docstring).
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

from orion.feedback.builder import _observation
from orion.proposals.policy import ProposalPolicyV1, ProposalTemplateV1

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "measure_proposal_feedback_correlation.py"
)
_spec = importlib.util.spec_from_file_location(
    "measure_proposal_feedback_correlation", _MODULE_PATH
)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_proposal_feedback_correlation"] = mod
_spec.loader.exec_module(mod)

UTC = timezone.utc
NOW = datetime(2026, 7, 29, 0, 0, 0, tzinfo=UTC)


# ===========================================================================
# extract_template_key -- pure string parsing, must never raise.
# ===========================================================================


def test_extract_template_key_dispatch_candidate_source_id() -> None:
    source_id = (
        "dispatch:proposal:inspect_execution_pressure:tick_abc123:"
        "attention.frame:tick_abc123:field_attention_policy.v1:"
        "execution_dispatch_policy.v1"
    )
    assert mod.extract_template_key(source_id) == "inspect_execution_pressure"


def test_extract_template_key_cortex_result_source_id() -> None:
    source_id = (
        "result:dispatch:proposal:inspect_node_resource_pressure:tick_xyz:none:"
        "execution_dispatch_policy.v1"
    )
    assert mod.extract_template_key(source_id) == "inspect_node_resource_pressure"


def test_extract_template_key_no_proposal_marker_returns_none() -> None:
    # An id with no "proposal:" marker at all -- the only case this function
    # itself returns None for.
    assert mod.extract_template_key("dispatch:reverie:thought_123:some_policy") is None


def test_extract_template_key_empty_string_returns_none() -> None:
    assert mod.extract_template_key("") is None


def test_extract_template_key_stops_at_first_colon_after_marker() -> None:
    # template_key values are plain config dict keys (never contain ':') --
    # confirms the regex doesn't greedily swallow the rest of the id.
    assert mod.extract_template_key("proposal:watch_reliability:tick_1:rest:more:stuff") == "watch_reliability"


def test_extract_template_key_reverie_source_id_extracts_literal_reverie() -> None:
    # Code review (2026-07-29): orion/reverie/proposal.py builds reverie
    # candidate ids as f"proposal:reverie:{thought.thought_id}" -- structurally
    # IDENTICAL in shape to a real config template's id. extract_template_key()
    # cannot and does not try to distinguish "reverie" from a real template key
    # -- it correctly extracts "reverie" here. Filtering reverie out is
    # is_known_template()'s job, exercised separately below, NOT this
    # function's -- a docstring in an earlier version of this file incorrectly
    # claimed this function itself returns None for reverie ids.
    source_id = "dispatch:proposal:reverie:thought_abc123:execution_dispatch_policy.v1"
    assert mod.extract_template_key(source_id) == "reverie"
    result_source_id = "result:dispatch:proposal:reverie:thought_abc123:execution_dispatch_policy.v1"
    assert mod.extract_template_key(result_source_id) == "reverie"


# ===========================================================================
# is_known_template -- the real filter that excludes reverie (and any other
# unrecognized template_key) from every report table uniformly.
# ===========================================================================


def test_is_known_template_true_for_real_config_key() -> None:
    assert mod.is_known_template("watch_reliability", {"watch_reliability", "inspect_execution_pressure"})


def test_is_known_template_false_for_reverie() -> None:
    assert mod.is_known_template("reverie", {"watch_reliability", "inspect_execution_pressure"}) is False


def test_is_known_template_false_for_none() -> None:
    assert mod.is_known_template(None, {"watch_reliability"}) is False


# ===========================================================================
# population_mean_variance
# ===========================================================================


def test_population_mean_variance_constant_values_zero_variance() -> None:
    mean, var = mod.population_mean_variance([0.55, 0.55, 0.55])
    assert mean == 0.55
    assert var == 0.0


def test_population_mean_variance_empty_list() -> None:
    assert mod.population_mean_variance([]) == (0.0, 0.0)


def test_population_mean_variance_real_spread() -> None:
    mean, var = mod.population_mean_variance([0.0, 1.0])
    assert mean == 0.5
    assert var == 0.25


# ===========================================================================
# build_bucket_stats -- exercises the REAL OutcomeObservationV1 schema.
# ===========================================================================


def _obs(*, kind: str, score: float, source_id: str) -> "object":
    return _observation(
        observation_id=f"obs:{source_id}:{kind}",
        source_kind="dispatch_candidate",
        source_id=source_id,
        outcome_kind=kind,
        score=score,
        confidence=1.0,
        observed_at=NOW,
    )


def test_build_bucket_stats_flags_degenerate_constant_score() -> None:
    obs_list = [
        _obs(kind="blocked", score=0.4, source_id="dispatch:proposal:t1:tick_a:x"),
        _obs(kind="blocked", score=0.4, source_id="dispatch:proposal:t1:tick_b:x"),
    ]
    buckets = {("t1", "dispatch_candidate"): obs_list}
    stats = mod.build_bucket_stats(buckets)
    st = stats[("t1", "dispatch_candidate")]
    assert st.n == 2
    assert st.score_degenerate is True
    assert st.outcome_kind_counts["blocked"] == 2


def test_build_bucket_stats_non_degenerate_when_scores_vary() -> None:
    obs_list = [
        _obs(kind="completed", score=0.85, source_id="result:dispatch:proposal:t2:tick_a:x"),
        _obs(kind="failed", score=0.10, source_id="result:dispatch:proposal:t2:tick_b:x"),
    ]
    buckets = {("t2", "cortex_result"): obs_list}
    stats = mod.build_bucket_stats(buckets)
    st = stats[("t2", "cortex_result")]
    assert st.score_degenerate is False
    assert st.score_stddev > 0.0


# ===========================================================================
# build_cortex_completion_signals
# ===========================================================================


def test_build_cortex_completion_signals_computes_real_rate() -> None:
    obs_list = [
        _obs(kind="completed", score=0.85, source_id="result:dispatch:proposal:t1:tick_a:x")
        for _ in range(9)
    ] + [
        _obs(kind="failed", score=0.10, source_id="result:dispatch:proposal:t1:tick_b:x")
    ]
    buckets = {("t1", "cortex_result"): obs_list}
    signals = mod.build_cortex_completion_signals(buckets, known_templates=["t1", "t2"])
    assert signals["t1"].n_completed == 9
    assert signals["t1"].n_failed == 1
    assert signals["t1"].completion_rate == 0.9
    # t2 never appears in buckets -- must report zero, not KeyError/None-crash.
    assert signals["t2"].n_total == 0
    assert signals["t2"].completion_rate is None


def test_cortex_completion_signal_meets_min_samples_threshold() -> None:
    sig_thin = mod.CortexCompletionSignal(template="t", n_completed=5, n_failed=2, n_other=0)
    sig_ample = mod.CortexCompletionSignal(
        template="t", n_completed=25, n_failed=10, n_other=0
    )
    assert sig_thin.meets_min_samples is False
    assert sig_ample.meets_min_samples is True


# ===========================================================================
# template_dominant_dimension -- real ProposalPolicyV1/ProposalTemplateV1 schema.
# ===========================================================================


def test_template_dominant_dimension_picks_max_weight() -> None:
    policy = ProposalPolicyV1(
        proposal_templates={
            "t1": ProposalTemplateV1(
                kind="inspect",
                target_kind="capability",
                target_id="capability:x",
                proposed_effect="increase_observability",
                required_policy_gate="read_only",
                dimensions={"execution_pressure": 0.3, "resource_pressure": 0.7},
            ),
            "t2": ProposalTemplateV1(
                kind="observe",
                target_kind="capability",
                target_id="capability:y",
                proposed_effect="preserve_stability",
                required_policy_gate="read_only",
                dimensions={},
            ),
        }
    )
    dominant = mod.template_dominant_dimension(policy)
    assert dominant["t1"] == "resource_pressure"
    assert "t2" not in dominant  # no dimensions declared -- not fabricated


# ===========================================================================
# compute_verdict -- STOP vs PROCEED, both reachable branches exercised.
# ===========================================================================


def test_compute_verdict_stops_on_thin_coverage() -> None:
    signals = {
        "t1": mod.CortexCompletionSignal(template="t1", n_completed=100, n_failed=5, n_other=0),
        "t2": mod.CortexCompletionSignal(template="t2", n_completed=0, n_failed=0, n_other=0),
        "t3": mod.CortexCompletionSignal(template="t3", n_completed=0, n_failed=0, n_other=0),
        "t4": mod.CortexCompletionSignal(template="t4", n_completed=0, n_failed=0, n_other=0),
    }
    dominant = {"t1": "execution_pressure", "t2": "resource_pressure", "t3": "field_intensity", "t4": "contract_pressure"}
    verdict = mod.compute_verdict(signals, dominant, total_templates=4)
    assert verdict.proceed_to_calibration is False
    assert verdict.templates_with_sufficient_cortex_signal == 1
    assert verdict.reasons  # non-empty, real reasons recorded


def test_compute_verdict_proceeds_when_coverage_and_dimension_diversity_are_real() -> None:
    signals = {
        "t1": mod.CortexCompletionSignal(template="t1", n_completed=100, n_failed=5, n_other=0),
        "t2": mod.CortexCompletionSignal(template="t2", n_completed=80, n_failed=20, n_other=0),
        "t3": mod.CortexCompletionSignal(template="t3", n_completed=0, n_failed=0, n_other=0),
    }
    dominant = {"t1": "execution_pressure", "t2": "resource_pressure", "t3": "field_intensity"}
    verdict = mod.compute_verdict(signals, dominant, total_templates=3)
    assert verdict.proceed_to_calibration is True
    assert verdict.templates_with_sufficient_cortex_signal == 2
    assert verdict.distinct_dominant_dimensions_with_signal == 2


def test_compute_verdict_dimension_collapse_does_not_block_template_level_proceed() -> None:
    # Two templates both clear the sample floor with real coverage, but both
    # share the same dominant dimension -- per-DIMENSION calibration is
    # indistinguishable from per-TEMPLATE on this data, but that alone must
    # NOT flip proceed_to_calibration to False (code review, 2026-07-29: an
    # earlier version reported this exact scenario as disqualifying in the
    # reason text while proceed_to_calibration silently stayed True with no
    # test ever asserting on the boolean -- fixed by making the reason
    # explicitly non-blocking for per-template calibration, and asserting on
    # proceed_to_calibration directly here so this can't regress silently).
    signals = {
        "t1": mod.CortexCompletionSignal(template="t1", n_completed=100, n_failed=5, n_other=0),
        "t2": mod.CortexCompletionSignal(template="t2", n_completed=80, n_failed=20, n_other=0),
    }
    dominant = {"t1": "execution_pressure", "t2": "execution_pressure"}
    verdict = mod.compute_verdict(signals, dominant, total_templates=2)
    assert verdict.distinct_dominant_dimensions_with_signal == 1
    assert verdict.proceed_to_calibration is True
    assert any("per-DIMENSION calibration cannot be distinguished" in r for r in verdict.reasons)

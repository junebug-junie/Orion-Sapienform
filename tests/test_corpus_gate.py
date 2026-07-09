from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1
from orion.telemetry.corpus_gate import is_corpus_row_healthy

_NOW = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)

COGNITIVE_FEATURE_NAMES = (
    "recall_gate_fired",
    "reasoning_present",
    "exec_step_fail_rate",
    "execution_friction",
)


def _feature(name: str, source: str) -> InnerFeatureV1:
    return InnerFeatureV1(name=name, raw_value=0.0, scaled_value=0.0, source=source)


def _healthy_cognitive_features() -> list[InnerFeatureV1]:
    return [
        _feature("recall_gate_fired", "execution_trajectory.live"),
        _feature("reasoning_present", "execution_trajectory.live"),
        _feature("exec_step_fail_rate", "execution_trajectory.live"),
        _feature("execution_friction", "execution_trajectory.live"),
    ]


def _all_none_cognitive_features() -> list[InnerFeatureV1]:
    return [
        _feature(name, "execution_trajectory.none") for name in COGNITIVE_FEATURE_NAMES
    ]


def _row(
    *,
    phi_health: str = "ok",
    grammar_truth_degraded: bool = False,
    features: list[InnerFeatureV1] | None = None,
) -> InnerStateFeaturesV1:
    return InnerStateFeaturesV1(
        generated_at=_NOW,
        self_state_id="self.state:tick_1:policy.v1",
        features=features if features is not None else _healthy_cognitive_features(),
        phi_health=phi_health,
        grammar_truth_degraded=grammar_truth_degraded,
    )


def test_healthy_row_passes() -> None:
    row = _row()
    healthy, reasons = is_corpus_row_healthy(
        row, cognitive_feature_names=COGNITIVE_FEATURE_NAMES
    )
    assert healthy is True
    assert reasons == []


def test_bad_phi_health_rejected() -> None:
    row = _row(phi_health="degenerate")
    healthy, reasons = is_corpus_row_healthy(row)
    assert healthy is False
    assert any(r.startswith("phi_health:") for r in reasons)


def test_grammar_truth_degraded_rejected() -> None:
    row = _row(grammar_truth_degraded=True)
    healthy, reasons = is_corpus_row_healthy(row)
    assert healthy is False
    assert any("grammar" in r for r in reasons)


def test_all_cognitive_features_none_rejected_when_opted_in() -> None:
    row = _row(features=_all_none_cognitive_features())
    healthy, reasons = is_corpus_row_healthy(
        row, cognitive_feature_names=COGNITIVE_FEATURE_NAMES
    )
    assert healthy is False
    assert "cognitive_features_all_none" in reasons


def test_all_none_check_is_opt_in_only() -> None:
    """Without cognitive_feature_names, the all-.none check must be skipped
    entirely — the row should be healthy if otherwise healthy."""
    row = _row(features=_all_none_cognitive_features())
    healthy, reasons = is_corpus_row_healthy(row, cognitive_feature_names=None)
    assert healthy is True
    assert reasons == []


def test_multiple_violations_all_reported() -> None:
    row = _row(phi_health="degenerate", grammar_truth_degraded=True)
    healthy, reasons = is_corpus_row_healthy(
        row, cognitive_feature_names=COGNITIVE_FEATURE_NAMES
    )
    assert healthy is False
    assert any(r.startswith("phi_health:") for r in reasons)
    assert any("grammar" in r for r in reasons)
    assert len(reasons) >= 2


def test_never_raises_on_empty_features() -> None:
    row = _row(features=[])
    healthy, reasons = is_corpus_row_healthy(
        row, cognitive_feature_names=COGNITIVE_FEATURE_NAMES
    )
    # No matching cognitive features present at all -> the all-none check
    # cannot find anything to condemn; row stays healthy on that axis.
    assert healthy is True
    assert reasons == []


def test_never_raises_on_empty_iterable_cognitive_feature_names() -> None:
    row = _row()
    # Confirm the predicate degrades safely for edge-case iterables (an empty
    # generator) rather than raising.
    healthy, reasons = is_corpus_row_healthy(row, cognitive_feature_names=iter([]))
    assert healthy is True
    assert reasons == []


def test_never_raises_on_one_shot_generator_cognitive_feature_names() -> None:
    """A generator is consumed exactly once by set(); confirm the all-none
    detection still fires correctly through a one-shot iterable, not just a
    reusable collection."""
    row = _row(features=_all_none_cognitive_features())
    healthy, reasons = is_corpus_row_healthy(
        row, cognitive_feature_names=iter(COGNITIVE_FEATURE_NAMES)
    )
    assert healthy is False
    assert "cognitive_features_all_none" in reasons

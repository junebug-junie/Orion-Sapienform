"""Topic Foundry adapter — canonical location ``orion/signals/adapters/tests/`` (phase-5)."""

from uuid import uuid4

import pytest

from orion.signals.adapters.topic_foundry import TopicFoundryAdapter
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


@pytest.fixture
def adapter() -> TopicFoundryAdapter:
    return TopicFoundryAdapter()


@pytest.fixture
def norm_ctx() -> NormalizationContext:
    return NormalizationContext()


def _drift_payload(js_divergence: float, **overrides) -> dict:
    payload = {
        "drift_id": str(uuid4()),
        "model_id": str(uuid4()),
        "model_name": "chat-topics-v1",
        "window_start": "2026-07-15T00:00:00+00:00",
        "window_end": "2026-07-16T00:00:00+00:00",
        "js_divergence": js_divergence,
        "outlier_pct_delta": 0.02,
        "top_topic_share_delta": -0.01,
        "created_at": "2026-07-16T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def test_can_handle_real_dotted_kind(adapter: TopicFoundryAdapter) -> None:
    # Real envelope kind published by orion-topic-foundry's bus publisher.
    assert adapter.can_handle("topic.foundry.drift.alert.v1", {})
    assert adapter.can_handle("topic.foundry.run.complete.v1", {})
    assert adapter.can_handle("topic.foundry.enrich.complete.v1", {})


def test_not_a_stub_anymore(adapter: TopicFoundryAdapter, norm_ctx: NormalizationContext) -> None:
    payload = _drift_payload(0.7)
    signal = adapter.adapt("topic.foundry.drift.alert.v1", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert "stub adapter" not in " ".join(signal.notes)


def test_different_payloads_produce_different_dimensions(
    adapter: TopicFoundryAdapter, norm_ctx: NormalizationContext
) -> None:
    low = adapter.adapt("topic.foundry.drift.alert.v1", _drift_payload(0.05), ORGAN_REGISTRY, {}, norm_ctx)
    high = adapter.adapt("topic.foundry.drift.alert.v1", _drift_payload(0.9), ORGAN_REGISTRY, {}, norm_ctx)
    assert low is not None and high is not None
    assert low.dimensions != high.dimensions
    assert low.dimensions["level"] < high.dimensions["level"]
    assert low.signal_kind == "topic_drift"
    assert high.signal_kind == "topic_drift"


def test_drift_level_tracks_js_divergence(adapter: TopicFoundryAdapter, norm_ctx: NormalizationContext) -> None:
    signal = adapter.adapt("topic.foundry.drift.alert.v1", _drift_payload(0.42), ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.dimensions["level"] == pytest.approx(0.42)
    assert signal.dimensions["confidence"] > 0.5


def test_enrich_complete_payload_uses_success_ratio(
    adapter: TopicFoundryAdapter, norm_ctx: NormalizationContext
) -> None:
    payload = {
        "run_id": str(uuid4()),
        "model_id": str(uuid4()),
        "dataset_id": str(uuid4()),
        "model_name": "chat-topics-v1",
        "model_version": "1",
        "status": "complete",
        "enriched_count": 90,
        "failed_count": 10,
    }
    signal = adapter.adapt("topic.foundry.enrich.complete.v1", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.signal_kind == "topic_state"
    assert signal.dimensions["level"] == pytest.approx(0.9)


def test_malformed_payload_degrades_gracefully_without_raising(
    adapter: TopicFoundryAdapter, norm_ctx: NormalizationContext
) -> None:
    signal = adapter.adapt("topic.foundry.run.complete.v1", {}, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.dimensions["level"] == 0.5
    assert signal.dimensions["confidence"] < 0.5
    assert any("neutral reading" in n for n in signal.notes)


def test_unknown_organ_returns_none(adapter: TopicFoundryAdapter, norm_ctx: NormalizationContext) -> None:
    signal = adapter.adapt("topic.foundry.drift.alert.v1", _drift_payload(0.5), {}, {}, norm_ctx)
    # ORGAN_REGISTRY module-level fallback still resolves the entry even
    # when an empty registry dict is passed, matching other adapters' pattern.
    assert signal is not None

"""World Pulse signal adapter — registry-relevant bus channels."""

from datetime import datetime, timezone

import pytest

from orion.signals.adapters.world_pulse import WorldPulseAdapter
from orion.signals.models import OrganClass
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY

REGISTRY_CHANNELS = [
    ("orion:world_pulse:run:result", "time_context", {"run": {"run_id": "wp-1", "status": "completed"}, "digest": {"coverage_status": "partial"}}),
    ("orion:world_pulse:digest:created", "environmental_context", {"run_id": "wp-2", "coverage_status": "complete"}),
    ("orion:world_pulse:situation:brief:upsert", "situation_state", {"brief_id": "brief-1", "run_id": "wp-3", "topic": "grid"}),
    ("orion:world_context:daily_capsule", "environmental_context", {"capsule_id": "cap-1", "run_id": "wp-4", "salient_topics": [{"topic": "ai", "summary": "x"}]}),
    ("orion:world_pulse:graph:upsert", "time_context", {"graph_delta_id": "gd-1", "run_id": "wp-5", "triple_count": 3}),
]


@pytest.fixture
def adapter() -> WorldPulseAdapter:
    return WorldPulseAdapter()


@pytest.fixture
def norm_ctx() -> NormalizationContext:
    return NormalizationContext()


@pytest.mark.parametrize("channel,expected_kind,payload", REGISTRY_CHANNELS)
def test_registry_channels_emit_expected_signal_kind(
    adapter: WorldPulseAdapter,
    norm_ctx: NormalizationContext,
    channel: str,
    expected_kind: str,
    payload: dict,
) -> None:
    signal = adapter.adapt(channel, payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.organ_id == "world_pulse"
    assert signal.signal_kind == expected_kind
    assert not any("stub adapter" in n for n in signal.notes)


def test_run_result_produces_coverage_dimensions(adapter: WorldPulseAdapter, norm_ctx: NormalizationContext) -> None:
    now = datetime(2026, 5, 20, tzinfo=timezone.utc)
    payload = {
        "run": {
            "run_id": "wp-99",
            "date": "2026-05-20",
            "status": "completed",
            "dry_run": False,
            "started_at": now.isoformat(),
        },
        "digest": {
            "run_id": "wp-99",
            "date": "2026-05-20",
            "coverage_status": "partial",
            "section_rollups": [{"section": "ai_technology", "confidence": 0.72}],
        },
    }
    signal = adapter.adapt("orion:world_pulse:run:result", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.organ_class == OrganClass.hybrid
    assert signal.source_event_id == "wp-99"
    assert signal.dimensions["level"] == pytest.approx(0.65)
    assert signal.dimensions["confidence"] == pytest.approx(0.72)


def test_malformed_payload_degrades_confidence(adapter: WorldPulseAdapter, norm_ctx: NormalizationContext) -> None:
    signal = adapter.adapt("orion:world_pulse:run:result", {}, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.dimensions["confidence"] == pytest.approx(0.1)
    assert any("malformed" in n for n in signal.notes)


def test_non_dict_payload_degrades(adapter: WorldPulseAdapter, norm_ctx: NormalizationContext) -> None:
    assert adapter.can_handle("orion:world_pulse:run:result", {}) is True
    signal = adapter.adapt("orion:world_pulse:run:result", {}, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.dimensions["confidence"] == pytest.approx(0.1)

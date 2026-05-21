"""World Pulse signal adapter — run result and digest channels."""

from datetime import datetime, timezone

import pytest

from orion.signals.adapters.world_pulse import WorldPulseAdapter
from orion.signals.models import OrganClass
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


@pytest.fixture
def adapter() -> WorldPulseAdapter:
    return WorldPulseAdapter()


@pytest.fixture
def norm_ctx() -> NormalizationContext:
    return NormalizationContext()


def test_run_result_produces_non_stub_signal(adapter: WorldPulseAdapter, norm_ctx: NormalizationContext) -> None:
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
    assert signal.organ_id == "world_pulse"
    assert signal.organ_class == OrganClass.hybrid
    assert signal.source_event_id == "wp-99"
    assert signal.dimensions["level"] == pytest.approx(0.65)
    assert signal.dimensions["confidence"] == pytest.approx(0.72)
    assert not any("stub adapter" in n for n in signal.notes)


def test_digest_created_uses_environmental_context_kind(adapter: WorldPulseAdapter, norm_ctx: NormalizationContext) -> None:
    payload = {
        "run_id": "wp-100",
        "coverage_status": "complete",
        "title": "Daily World Pulse",
    }
    signal = adapter.adapt("orion:world_pulse:digest:created", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.signal_kind == "environmental_context"
    assert signal.dimensions["level"] == pytest.approx(0.9)

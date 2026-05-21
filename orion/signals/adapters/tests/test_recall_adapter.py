import json
from pathlib import Path

import pytest

from orion.signals.adapters.biometrics import BiometricsAdapter
from orion.signals.adapters.recall import RecallAdapter
from orion.signals.models import OrganClass
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "recall_exec_result.json"


@pytest.fixture
def adapter() -> RecallAdapter:
    return RecallAdapter()


@pytest.fixture
def norm_ctx() -> NormalizationContext:
    return NormalizationContext()


def test_recall_adapter_exec_result_fixture(adapter: RecallAdapter, norm_ctx: NormalizationContext) -> None:
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    signal = adapter.adapt("recall.reply.v1", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.organ_id == "recall"
    assert signal.organ_class == OrganClass.endogenous
    assert signal.signal_kind == "recall_result"
    assert signal.dimensions["level"] > 0.5
    assert signal.dimensions["confidence"] >= 0.5
    assert signal.source_event_id == "corr-recall-fixture-001"
    assert "stub adapter" not in " ".join(signal.notes)


def test_recall_causal_parents_from_prior(adapter: RecallAdapter, norm_ctx: NormalizationContext) -> None:
    from datetime import datetime, timezone

    from orion.signals.models import OrionSignalV1
    from orion.signals.signal_ids import make_signal_id

    now = datetime.now(timezone.utc)
    auto_sig = OrionSignalV1(
        signal_id=make_signal_id("autonomy", "corr-auto"),
        organ_id="autonomy",
        organ_class=OrganClass.endogenous,
        signal_kind="autonomy_state",
        dimensions={"pressure_coherence": 0.7, "confidence": 0.8},
        source_event_id="corr-auto",
        observed_at=now,
        emitted_at=now,
    )
    prior = {"autonomy": auto_sig}
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    signal = adapter.adapt("recall.reply.v1", payload, ORGAN_REGISTRY, prior, norm_ctx)
    assert signal is not None
    assert auto_sig.signal_id in signal.causal_parents


def test_recall_partial_payload_degrades(adapter: RecallAdapter, norm_ctx: NormalizationContext) -> None:
    signal = adapter.adapt("recall.reply.v1", {"correlation_id": "partial"}, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.signal_kind == "recall_gap"
    assert signal.dimensions["confidence"] < 0.5

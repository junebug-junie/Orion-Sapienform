"""Optional dedupe: suppress adapted emit when passthrough for same source_event_id was recent."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.signals.models import OrganClass, OrionSignalV1


@pytest.mark.asyncio
async def test_suppress_adapted_when_passthrough_same_source_event_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.normalization_state import NormalizationStateRegistry
    from app.processor import SignalProcessor
    from app.settings import settings
    from app.signal_window import SignalWindow

    monkeypatch.setattr(settings, "SUPPRESS_ADAPTED_WHEN_PASSTHROUGH", True, raising=False)
    monkeypatch.setattr(settings, "PASSTHROUGH_DEDUPE_WINDOW_SEC", 60.0, raising=False)

    bus = AsyncMock()
    proc = SignalProcessor(
        bus=bus,
        signal_window=SignalWindow(30.0),
        norm_state=NormalizationStateRegistry(),
        output_channel_prefix="orion:signals",
        passthrough_pattern="orion:signals:*",
        service_ref=ServiceRef(name="orion-signal-gateway", version="0.1.0", node="n"),
    )
    now = datetime.now(timezone.utc)
    src = "evt-shared-1"
    passthrough_sig = OrionSignalV1(
        signal_id="pt1",
        organ_id="biometrics",
        organ_class=OrganClass.exogenous,
        signal_kind="gpu_load",
        dimensions={"level": 0.3},
        source_event_id=src,
        observed_at=now,
        emitted_at=now,
    )
    proc._record_passthrough_dedupe(passthrough_sig)

    adapted = OrionSignalV1(
        signal_id="ad1",
        organ_id="biometrics",
        organ_class=OrganClass.exogenous,
        signal_kind="gpu_load",
        dimensions={"level": 0.9},
        source_event_id=src,
        observed_at=now,
        emitted_at=now,
    )
    assert proc._should_suppress_adapted(adapted) is True

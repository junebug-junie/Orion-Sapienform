"""Multi-emission adapter path (spec §5.1)."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.adapters.result import AdapterResult
from orion.signals.models import OrganClass, OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


class _ListEmitAdapter(OrionSignalAdapter):
    organ_id = "cortex_exec"

    def can_handle(self, channel: str, payload: dict) -> bool:
        return channel == "orion:cognition:trace"

    def adapt(self, channel, payload, registry, prior_signals, norm_ctx) -> AdapterResult:
        now = datetime.now(timezone.utc)
        base = dict(
            organ_class=OrganClass.endogenous,
            dimensions={"success": 1.0},
            causal_parents=[],
            source_event_id="corr-test",
            observed_at=now,
            emitted_at=now,
        )
        return [
            OrionSignalV1(signal_id="run1", organ_id="cortex_exec", signal_kind="cognition_run", **base),
            OrionSignalV1(signal_id="step1", organ_id="graph_cognition", signal_kind="cognition_step", **base),
        ]


@pytest.fixture
def memory_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.mark.asyncio
async def test_gateway_processor_multi_emission(monkeypatch, memory_exporter):
    from app.normalization_state import NormalizationStateRegistry
    from app.processor import SignalProcessor
    from app.signal_window import SignalWindow
    import app.processor as proc_mod

    monkeypatch.setattr(proc_mod, "ADAPTERS", [_ListEmitAdapter()])
    bus = AsyncMock()
    proc = SignalProcessor(
        bus=bus,
        signal_window=SignalWindow(30.0),
        norm_state=NormalizationStateRegistry(),
        output_channel_prefix="orion:signals",
        passthrough_pattern="orion:signals:*",
        service_ref=ServiceRef(name="orion-signal-gateway", version="0.1.0", node="n"),
    )
    CORR = UUID("00000000-0000-4000-8000-000000000001")
    env = BaseEnvelope(
        kind="orion:cognition:trace",
        source=ServiceRef(name="orion-cortex-exec", version="0.1.0", node="n"),
        correlation_id=CORR,
        payload={"verb": "chat_general", "mode": "brain", "steps": []},
    )
    await proc.handle_envelope(env)
    assert bus.publish.await_count == 2
    channels = [c.args[0] for c in bus.publish.await_args_list]
    assert "orion:signals:cortex_exec" in channels
    assert "orion:signals:graph_cognition" in channels

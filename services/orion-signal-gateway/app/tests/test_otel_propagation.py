"""OTEL span emission and parent trace inheritance (spec §5, §3)."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from orion.core.bus.bus_schemas import ServiceRef
from orion.signals.models import OrganClass, OrionSignalV1


@pytest.fixture(scope="session")
def memory_exporter() -> InMemorySpanExporter:
    """One TracerProvider for this module — OTEL forbids replacing the global provider."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


def test_spark_telemetry_kind_not_treated_as_hardened_signal() -> None:
    assert not "spark.signal.v1".startswith("signal.")


def test_gateway_emitted_kind_matches_passthrough_prefix() -> None:
    assert "signal.biometrics.biometrics_state".startswith("signal.")


@pytest.mark.asyncio
async def test_emit_traced_and_parent_trace(memory_exporter: InMemorySpanExporter) -> None:
    from app.normalization_state import NormalizationStateRegistry
    from app.processor import SignalProcessor
    from app.signal_window import SignalWindow

    memory_exporter.clear()
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
    bio = OrionSignalV1(
        signal_id="abc",
        organ_id="biometrics",
        organ_class=OrganClass.exogenous,
        signal_kind="biometrics_state",
        dimensions={"level": 0.5, "confidence": 0.9},
        observed_at=now,
        emitted_at=now,
    )
    await proc._emit_traced(bio, prior={})
    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "signal.biometrics.biometrics_state"
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("organ_id") == "biometrics"

    memory_exporter.clear()
    trace_hex = "a" * 32
    span_hex = "b" * 16
    bio2 = OrionSignalV1(
        signal_id="bio1",
        organ_id="biometrics",
        organ_class=OrganClass.exogenous,
        signal_kind="gpu_load",
        dimensions={"level": 0.8},
        otel_trace_id=trace_hex,
        otel_span_id=span_hex,
        observed_at=now,
        emitted_at=now,
    )
    eq = OrionSignalV1(
        signal_id="eq1",
        organ_id="equilibrium",
        organ_class=OrganClass.hybrid,
        signal_kind="mesh_health",
        dimensions={"level": 0.5},
        observed_at=now,
        emitted_at=now,
    )
    await proc._emit_traced(eq, prior={"biometrics": bio2})
    spans2 = memory_exporter.get_finished_spans()
    assert len(spans2) == 1
    assert int(trace_hex, 16) == spans2[0].context.trace_id

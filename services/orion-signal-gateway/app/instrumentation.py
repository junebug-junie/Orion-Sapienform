"""OpenTelemetry tracer setup for the signal gateway (spec §5 gateway instrumentation)."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class _DropSpanExporter:
    """Swallows finished spans (no backend); span context IDs are still real."""

    def export(self, spans):  # type: ignore[no-untyped-def]
        from opentelemetry.sdk.trace.export import SpanExportResult

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def configure_tracing(
    *,
    service_name: str,
    otlp_endpoint: Optional[str] = None,
    console_export: bool = False,
) -> None:
    """
    Install a TracerProvider. If ``otlp_endpoint`` is set, export spans there; else
    optionally log spans to console; else drop spans silently while preserving IDs.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("OTEL tracing: OTLP gRPC exporter -> %s", otlp_endpoint)
        except Exception as exc:
            logger.warning("OTEL OTLP exporter failed (%s); falling back to drop exporter", exc)
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            provider.add_span_processor(SimpleSpanProcessor(_DropSpanExporter()))
    elif console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("OTEL tracing: console span exporter enabled")
    else:
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider.add_span_processor(SimpleSpanProcessor(_DropSpanExporter()))

    trace.set_tracer_provider(provider)

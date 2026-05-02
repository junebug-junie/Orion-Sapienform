"""Signal processing loop: adapts raw bus events into OrionSignalV1 and emits them."""
import logging
from typing import Optional, Tuple

from opentelemetry import trace
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    TraceFlags,
    format_span_id,
    format_trace_id,
    set_span_in_context,
)

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.signals.adapters import ADAPTERS
from orion.signals.causal_helpers import with_missed_parent_notes
from orion.signals.models import OrganClass, OrionSignalV1
from orion.signals.registry import ORGAN_REGISTRY

from .normalization_state import NormalizationStateRegistry
from .passthrough import PassthroughValidator
from .settings import settings
from .signal_window import SignalWindow

logger = logging.getLogger(__name__)


class SignalProcessor:
    def __init__(
        self,
        *,
        bus: OrionBusAsync,
        signal_window: SignalWindow,
        norm_state: NormalizationStateRegistry,
        output_channel_prefix: str,
        passthrough_pattern: str,
        service_ref: ServiceRef,
    ):
        self._bus = bus
        self._window = signal_window
        self._norm_state = norm_state
        self._output_prefix = output_channel_prefix
        self._passthrough_pattern = passthrough_pattern
        self._source = service_ref
        self._passthrough = PassthroughValidator()

    def _resolve_parent_otel_context(
        self, signal: OrionSignalV1, prior: dict[str, OrionSignalV1]
    ) -> Tuple[Optional[object], Optional[str]]:
        """
        First registry-listed parent with valid otel ids wins (spec §3 OTEL parent context).
        Returns (context for start_as_current_span, parent_span_id_hex for attribute).
        """
        if signal.organ_class == OrganClass.exogenous:
            return None, None
        entry = ORGAN_REGISTRY.get(signal.organ_id)
        if entry is None:
            return None, None
        for organ in entry.causal_parent_organs or []:
            parent = prior.get(organ)
            if parent is None or not parent.otel_trace_id or not parent.otel_span_id:
                continue
            try:
                trace_id_int = int(parent.otel_trace_id, 16)
                span_id_int = int(parent.otel_span_id, 16)
            except ValueError:
                logger.warning("Invalid OTEL hex from parent organ=%s; skipping", organ)
                continue
            pctx = SpanContext(
                trace_id=trace_id_int,
                span_id=span_id_int,
                is_remote=True,
                trace_flags=TraceFlags(0x01),
            )
            return set_span_in_context(NonRecordingSpan(pctx)), parent.otel_span_id
        return None, None

    async def handle_envelope(self, env: BaseEnvelope) -> None:
        """Handle an incoming bus envelope: adapt or passthrough, then emit."""
        payload = env.payload
        if not isinstance(payload, dict):
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump(mode="json")
            else:
                logger.debug(f"Dropping non-dict payload kind={env.kind}")
                return

        if env.kind and env.kind.startswith("signal."):
            if env.source and env.source.name == settings.SERVICE_NAME:
                return
            signal = self._passthrough.validate(payload)
            if signal is not None:
                await self._emit_passthrough(signal)
            return

        prior = self._window.get_all()
        for adapter in ADAPTERS:
            try:
                if not adapter.can_handle(env.kind or "", payload):
                    continue
                norm_ctx = self._norm_state.get(adapter.organ_id)
                signal = adapter.adapt(
                    channel=env.kind or "",
                    payload=payload,
                    registry=ORGAN_REGISTRY,
                    prior_signals=prior,
                    norm_ctx=norm_ctx,
                )
                if signal is not None:
                    signal = with_missed_parent_notes(signal, prior, ORGAN_REGISTRY)
                    await self._emit_traced(signal, prior=prior)
                    break
            except Exception as exc:
                logger.error(f"Adapter {adapter.organ_id} raised: {exc}", exc_info=True)

    async def _emit_passthrough(self, signal: OrionSignalV1) -> None:
        """Self-hardened signal: re-emit without adapting (spec §2); keep existing OTEL fields."""
        await self._publish(signal)

    async def _emit_traced(self, signal: OrionSignalV1, *, prior: dict[str, OrionSignalV1]) -> None:
        """Start a span (spec §5), attach attributes, write OTEL ids onto the signal, then publish."""
        tracer = trace.get_tracer(__name__)
        parent_ctx, parent_span_hex = self._resolve_parent_otel_context(signal, prior)
        span_kwargs: dict = {}
        if parent_ctx is not None:
            span_kwargs["context"] = parent_ctx

        with tracer.start_as_current_span(
            f"signal.{signal.organ_id}.{signal.signal_kind}",
            **span_kwargs,
        ) as span:
            span.set_attribute("organ_id", signal.organ_id)
            span.set_attribute("organ_class", signal.organ_class.value)
            span.set_attribute("signal_kind", signal.signal_kind)
            span.set_attribute("correlation_id", signal.source_event_id or "")
            for k, v in signal.dimensions.items():
                span.set_attribute(f"dim.{k}", float(v))
            sc = span.get_span_context()
            traced = signal.model_copy(
                update={
                    "otel_trace_id": format_trace_id(sc.trace_id),
                    "otel_span_id": format_span_id(sc.span_id),
                    "otel_parent_span_id": parent_span_hex,
                }
            )
        await self._publish(traced)

    async def _publish(self, signal: OrionSignalV1) -> None:
        self._window.put(signal)
        channel = f"{self._output_prefix}:{signal.organ_id}"
        env = BaseEnvelope(
            kind=f"signal.{signal.organ_id}.{signal.signal_kind}",
            source=self._source,
            payload=signal.model_dump(mode="json"),
        )
        try:
            await self._bus.publish(channel, env)
            logger.debug(
                f"Emitted signal organ={signal.organ_id} kind={signal.signal_kind} id={signal.signal_id}"
            )
        except Exception as exc:
            logger.error(f"Failed to emit signal: {exc}")

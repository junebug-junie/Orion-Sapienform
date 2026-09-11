"""Caller-bound reading tools over the existing internal Orion bus trust boundary."""
from __future__ import annotations

from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.reading import (
    DurableReadingReceiptV1, ReadingRequestedV1, ReadingStatusArguments,
    ReadingStatusReceiptV1, ReadingToolBindingV1, ReadingToolRequestV1,
    ReadingToolResultV1, RecommendReadingArguments,
)
from orion.world_pulse_read.events import TOOL_CHANNEL, TOOL_RESULT_PREFIX
from orion.world_pulse_read.urls import normalize_source_url

RECOMMEND_DESCRIPTION = (
    "Asynchronously recommend a public HTTP(S) source for deliberate reading, preservation, "
    "follow-up and integration through Orion's reading pipeline. Use WebFetch/search for "
    "information needed immediately in this turn. Supply why this source matters now. "
    "Acceptance exists only when the response has ok=true and result contains a request_id; "
    "errors or malformed results mean acceptance is unknown. Returns a durable queue receipt, "
    "not an article summary or a promise of future processing. Reading creates source-attributed "
    "candidates, not settled beliefs. Report request_id and status so reading_status can inspect it later."
)
STATUS_DESCRIPTION = "Read durable reading status and source-attributed result by request_id from a previous receipt."


class ReadingTools:
    def __init__(self, bus, binding: ReadingToolBindingV1):
        self.bus = bus
        self.binding = binding

    async def invoke(self, name: str, arguments: dict):
        # Validate before transport: extra provenance fields are rejected, never ignored.
        if name == "recommend_reading":
            args = RecommendReadingArguments.model_validate(arguments)
            url = normalize_source_url(args.url)
            request = ReadingRequestedV1(
                # Stable across retries/restarts within this turn, new on a later turn.
                request_id=deterministic_reading_request_id(
                    self.binding, url=url, why_now=args.why_now
                ),
                url=url, why_now=args.why_now,
                requested_by="juniper" if self.binding.invocation_context == "unified_chat" else "orion",
                invocation_context=self.binding.invocation_context,
                parent_run_id=self.binding.parent_run_id,
                parent_trace_id=self.binding.parent_trace_id,
            )
            command = ReadingToolRequestV1(operation=name, request=request)
        elif name == "reading_status":
            args = ReadingStatusArguments.model_validate(arguments)
            command = ReadingToolRequestV1(operation=name, request_id=args.request_id)
        else:
            raise ValueError("unknown reading tool")
        correlation_id = uuid4()
        reply = f"{TOOL_RESULT_PREFIX}{correlation_id}"
        raw = await self.bus.rpc_request(
            TOOL_CHANNEL,
            BaseEnvelope(
                kind="reading.tool.request.v1", correlation_id=correlation_id,
                reply_to=reply, source=ServiceRef(name="orion-harness-governor"),
                payload=command.model_dump(mode="json"),
            ), reply_channel=reply, timeout_sec=15.0,
        )
        decoded = self.bus.codec.decode(raw.get("data"))
        if not decoded.ok or str(decoded.envelope.correlation_id) != str(correlation_id):
            raise RuntimeError("invalid reading receipt; acceptance is unknown")
        result = ReadingToolResultV1.model_validate(decoded.envelope.payload)
        if not result.ok:
            raise RuntimeError(result.error or "reading tool failed")
        if result.error is not None:
            raise RuntimeError("invalid reading result envelope")
        if name == "recommend_reading":
            try:
                receipt = DurableReadingReceiptV1.model_validate(result.result)
            except ValueError as exc:
                raise RuntimeError(
                    "invalid durable reading receipt; acceptance is unknown"
                ) from exc
            if receipt.request_id != request.request_id:
                raise RuntimeError("invalid durable reading receipt; acceptance is unknown")
        else:
            try:
                receipt = ReadingStatusReceiptV1.model_validate(result.result)
            except ValueError as exc:
                raise RuntimeError("invalid reading status receipt") from exc
            if receipt.request_id != args.request_id:
                raise RuntimeError("invalid reading status receipt")
        # The MCP transcript must retain the explicit acceptance bit. Returning
        # only ``result`` made a merely JSON-shaped payload look authoritative.
        return result.model_dump(mode="json")


def deterministic_reading_request_id(
    binding: ReadingToolBindingV1, *, url: str, why_now: str
) -> UUID:
    """Stable for identical retries in one bound turn; different on later turns."""

    return uuid5(NAMESPACE_URL, f"reading:{binding.parent_run_id}:{url}:{why_now}")

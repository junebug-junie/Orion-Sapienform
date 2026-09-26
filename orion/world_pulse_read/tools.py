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
STATUS_DESCRIPTION = (
    "Read durable reading status by url or request_id (supply exactly one). Use the supplied "
    "link directly; do not ask the user for a request ID when they have a URL. URL lookup is "
    "read-only and returns the latest matching request plus matched_request_count; it does "
    "not enqueue or retry reading. Multiple matches mean earlier attempts may have different "
    "outcomes. Report only the returned status, never infer that queued means never attempted. "
    "While status is 'queued', the response also carries queue_position (1-indexed "
    "place in line) and queue_depth (total pending) -- report those instead of just 'queued' "
    "when asked how long something might take; both are null once the row leaves the queue."
)


def reading_brief_lines() -> list[str]:
    return [
        (
            "Reading MCP is available: recommend_reading queues a public HTTP(S) source for "
            "durable async processing, and reading_status looks up a previously queued source "
            "by URL or request_id. If asked about the status of something already queued for "
            "reading, ToolSearch and call reading_status with the supplied URL directly (or "
            "request_id if provided). Do not ask for an ID when a link is available; do "
            "not guess Postgres table names, grep the repo for the id, or invent a status. A "
            "'queued' result includes queue_position/queue_depth (e.g. '13th of 121') -- use "
            "them, don't just report 'queued' with no sense of scale. Tool discovery is not "
            "a status check: report status only after a successful tool call. URL lookup "
            "selects the latest request; queued does not establish that no earlier attempt ran."
        ),
    ]


def append_reading_mcp_harness_brief(
    parts: list[str],
    *,
    reading_binding: ReadingToolBindingV1 | None = None,
    reading_only: bool = False,
) -> None:
    """Append reading-tool usage lines only when the reading MCP server is
    actually attached this turn.

    Mirrors append_self_index_harness_brief's master-flag gate, but the
    reading server has two more conditions of its own
    (orion/fcc/mcp_config.py: `render_mcp_config`): `reading_only=True` turns
    get an intentionally empty MCP config (built-in WebFetch/WebSearch only,
    unrelated to the async reading-recommendation pipeline despite the name),
    and even with the master flag on, the "orion-reading" server is only
    rendered when a caller actually passed `reading_binding`
    (orion/hub/turn_orchestrator.py's Unified Chat path always does; a future
    caller might not). Checking `reading_binding is not None` directly here,
    rather than assuming every caller of this prefix is Unified Chat, is what
    keeps this brief from lying if that assumption ever stops holding.
    """
    from orion.fcc.github_repo_context import harness_mcp_enabled

    if reading_only or reading_binding is None or not harness_mcp_enabled():
        return
    parts.extend(reading_brief_lines())


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
            command = ReadingToolRequestV1(
                operation=name, request_id=args.request_id,
                url=normalize_source_url(args.url) if args.url is not None else None,
            )
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
            if args.request_id is not None and receipt.request_id != args.request_id:
                raise RuntimeError("invalid reading status receipt")
            if command.url is not None and result.result.get("lookup_url") != command.url:
                raise RuntimeError("invalid reading status receipt")
        # The MCP transcript must retain the explicit acceptance bit. Returning
        # only ``result`` made a merely JSON-shaped payload look authoritative.
        return result.model_dump(mode="json")


def deterministic_reading_request_id(
    binding: ReadingToolBindingV1, *, url: str, why_now: str
) -> UUID:
    """Stable for identical retries in one bound turn; different on later turns."""

    return uuid5(NAMESPACE_URL, f"reading:{binding.parent_run_id}:{url}:{why_now}")

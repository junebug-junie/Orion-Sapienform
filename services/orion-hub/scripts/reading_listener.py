"""Internal bus RPC adapter. Postgres acceptance precedes event and receipt."""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

import httpx

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.introspect.redact import safe_exception_detail
from orion.schemas.introspect import DEFAULT_LIMIT
from orion.schemas.reading import (
    DurableReadingReceiptV1,
    ReadingStatusReceiptV1,
    ReadingToolRequestV1,
    ReadingToolResultV1,
)
from orion.world_pulse_read.events import TOOL_CHANNEL, TOOL_RESULT_PREFIX
from orion.world_pulse_read.documents import DocumentPolicy, DocumentSourceError
from orion.world_pulse_read.introspect import reading_results
from orion.world_pulse_read.queue import enqueue_reading, reading_status
from orion.world_pulse_read.search import (
    HTTP_TIMEOUT_SEC,
    ReadingSearchConfig,
    SearchUnavailableError,
    gated_results,
    index_missing_readings,
    rank_readings,
    verified_rows,
)
from orion.world_pulse_read.urls import normalize_reading_source

logger = logging.getLogger(__name__)

_SAFE_ERROR = "reading_queue_unavailable; acceptance unknown, retry the same request"
_SEARCH_UNAVAILABLE = "reading_search_unavailable; answer unknown"
# The memory pool comes up after listeners start; do not wait a full index interval for it.
_NO_POOL_RETRY_SEC = 15.0
_SCHEMA_SQLSTATES = {"42P01", "42703"}
_CONNECTION_SQLSTATE_PREFIX = "08"


def _failure_category(exc: Exception, *, phase: str) -> str:
    sqlstate = str(getattr(exc, "sqlstate", "") or "")
    constraint = str(getattr(exc, "constraint_name", "") or "")
    if sqlstate in _SCHEMA_SQLSTATES or constraint == "world_pulse_read_seed_kind_check":
        return "schema_incompatible"
    if (
        phase == "pool_acquire"
        or sqlstate.startswith(_CONNECTION_SQLSTATE_PREFIX)
        or isinstance(exc, (ConnectionError, TimeoutError, OSError))
    ):
        return "connection_failure"
    if phase == "enqueue":
        return "enqueue_failure"
    if phase == "reading_search":
        return "reading_search_failure"
    if phase == "reading_result":
        return "reading_result_failure"
    return "status_failure"


class ReadingListener:
    def __init__(
        self, pool_provider, source_ref, search: ReadingSearchConfig | None = None,
        documents: DocumentPolicy | None = None,
    ):
        self.pool_provider = pool_provider
        self.source_ref = source_ref
        self.search = search
        self.documents = documents
        self.task = None
        self.index_task = None
        self.bus = None

    async def handle(self, envelope):
        # Same trusted internal bus boundary as harness RPC; never accept a
        # model-supplied reply subject or expose SQL/credentials through MCP.
        reply = f"{TOOL_RESULT_PREFIX}{envelope.correlation_id}"
        if envelope.reply_to != reply or envelope.kind != "reading.tool.request.v1":
            return
        try:
            command = ReadingToolRequestV1.model_validate(envelope.payload)
        except ValueError as exc:
            response = ReadingToolResultV1(ok=False, error=str(exc))
            await self._publish_response(reply, envelope, response)
            return

        phase = "pool_acquire"
        try:
            pool = self.pool_provider()
            if pool is None:
                logger.warning(
                    "reading_tool_failed correlation_id=%s category=no_pool phase=pool_lookup",
                    envelope.correlation_id,
                )
                response = ReadingToolResultV1(ok=False, error=_SAFE_ERROR)
                await self._publish_response(reply, envelope, response)
                return
            scored = None
            if command.operation == "reading_result" and command.query is not None:
                # Rank before taking a connection: embed + Chroma can take seconds.
                phase = "reading_search"
                if self.search is None or not self.search.enabled:
                    raise SearchUnavailableError("semantic reading search is not configured")
                async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
                    scored = await rank_readings(client, self.search, command.query)
            phase = "pool_acquire"
            async with pool.acquire() as conn:
                if command.operation == "recommend_reading":
                    phase = "enqueue"
                    result = await enqueue_reading(
                        conn, command.request, bus=self.bus, source=self.source_ref,
                        documents=self.documents,
                    )
                    try:
                        receipt = DurableReadingReceiptV1.model_validate(result)
                    except ValueError as exc:
                        raise RuntimeError("enqueue returned a malformed durable receipt") from exc
                    if receipt.request_id != command.request.request_id:
                        raise RuntimeError("enqueue returned a mismatched request_id")
                elif command.operation == "reading_status":
                    phase = "status"
                    if command.url is not None:
                        result = await reading_status(conn, url=command.url)
                    else:
                        result = await reading_status(conn, command.request_id)
                    try:
                        receipt = ReadingStatusReceiptV1.model_validate(result)
                    except ValueError as exc:
                        raise RuntimeError("status returned a malformed receipt") from exc
                    if command.request_id is not None and receipt.request_id != command.request_id:
                        raise RuntimeError("status returned a mismatched request_id")
                    if command.url is not None:
                        if result.get("lookup_url") != normalize_reading_source(command.url):
                            raise RuntimeError("status returned a mismatched URL")
                else:
                    if scored is not None:
                        phase = "reading_search"
                        mode = "query"
                        introspection = await gated_results(
                            conn, scored, limit=command.limit or DEFAULT_LIMIT, since=command.since,
                        )
                    else:
                        phase = "reading_result"
                        mode = "lookup" if (command.request_id or command.url) else "recent"
                        introspection = await reading_results(
                            conn,
                            request_id=command.request_id,
                            url=command.url,
                            limit=command.limit or DEFAULT_LIMIT,
                            since=command.since,
                        )
                    result = introspection.model_dump(mode="json")
                    logger.info(
                        "introspect op=reading_result corr=%s items=%d total=%s mode=%s",
                        envelope.correlation_id,
                        len(introspection.items),
                        introspection.total_available,
                        mode,
                    )
            response = ReadingToolResultV1(ok=True, result=result)
        except Exception as exc:
            logger.warning(
                "reading_tool_failed correlation_id=%s category=%s phase=%s "
                "exc_type=%s sqlstate=%s detail=%s",
                envelope.correlation_id,
                _failure_category(exc, phase=phase),
                phase,
                type(exc).__name__,
                str(getattr(exc, "sqlstate", "") or "none"),
                safe_exception_detail(exc, limit=1000),
            )
            # Do not leak DSNs, SQL or arbitrary exception text into model context.
            if isinstance(exc, SearchUnavailableError):
                error = _SEARCH_UNAVAILABLE
            elif isinstance(exc, DocumentSourceError):
                # A fixed policy code (documents.py), never file content or paths.
                error = str(exc)
            else:
                error = _SAFE_ERROR
            response = ReadingToolResultV1(ok=False, error=error)
        await self._publish_response(reply, envelope, response)

    async def _publish_response(self, reply, envelope, response):
        await self.bus.publish(reply, BaseEnvelope(
            kind="reading.tool.result.v1", correlation_id=envelope.correlation_id,
            source=self.source_ref, payload=response.model_dump(mode="json"),
        ))

    async def start(self, bus):
        self.bus = bus
        self.task = asyncio.create_task(self._run(), name="hub-reading-tools")
        if self.search is not None and self.search.enabled:
            self.index_task = asyncio.create_task(self._index_loop(), name="hub-reading-search-index")

    async def stop(self):
        for attr in ("task", "index_task"):
            task = getattr(self, attr)
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                setattr(self, attr, None)

    async def index_once(self):
        pool = self.pool_provider()
        if pool is None:
            logger.info("reading_search_index skipped reason=no_pool")
            return None
        # Release the connection before embedding; a pass can take seconds.
        async with pool.acquire() as conn:
            rows = await verified_rows(conn)
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
            result = await index_missing_readings(
                rows, self.search, client=client, bus=self.bus, source=self.source_ref,
            )
        logger.info("reading_search_index indexed=%d pending=%d", result.indexed, result.pending)
        return result

    async def _index_loop(self):
        while True:
            delay = self.search.index_interval_sec
            try:
                if await self.index_once() is None:
                    delay = min(delay, _NO_POOL_RETRY_SEC)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "reading_search_index_failed exc_type=%s detail=%s",
                    type(exc).__name__, safe_exception_detail(exc, limit=1000),
                )
            await asyncio.sleep(delay)

    async def _run(self):
        while True:
            try:
                async with self.bus.subscribe(TOOL_CHANNEL) as pubsub:
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                await self.handle(decoded.envelope)
                        except Exception:
                            logger.warning("reading_rpc_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("reading_listener_reconnecting", exc_info=True)
                await asyncio.sleep(1)

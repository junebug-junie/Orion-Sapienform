"""orion-dream's introspect responder: answers the `dreams` tool over the bus.

Trust: replies only when reply_to is exactly orion:introspect:result:<corr> and
the kind matches; never to a model-supplied subject. Every connection is a
read-only transaction. Errors become ok=false ("answer unknown"), never [].
A search that finds nothing is only reported empty when the index is known to
hold every dream recorded so far; otherwise "no match" could mean "not indexed".
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
from sqlalchemy import text

from app.dream_search import index_missing, rank
from app.introspect_dreams import by_ids, index_rows, one, recent
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.introspect.redact import safe_exception_detail
from orion.introspect.semantic_index import HTTP_TIMEOUT_SEC, SearchConfig, SearchUnavailableError
from orion.introspect.transport import DREAM_REQUEST_CHANNEL, REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.schemas.introspect import DreamsArguments, IntrospectRequestV1, IntrospectResultV1

logger = logging.getLogger("orion-dream.introspect")

QUERY_UNAVAILABLE = "dreams_unavailable; answer unknown"
SEARCH_UNAVAILABLE = "dream_search_unavailable; answer unknown"
_INVALID_CAP = 400


def _failed(now: datetime, error: str) -> IntrospectResultV1:
    return IntrospectResultV1(ok=False, operation="dreams", as_of=now, error=error)


class DreamIntrospectListener:
    def __init__(
        self, *, bus_url: str, engine_provider: Callable[[], Any], source: Any,
        search: SearchConfig | None, bus_factory: Callable[[str], Any] = OrionBusAsync,
    ):
        self.bus_url = bus_url
        self.engine_provider = engine_provider
        self.source = source
        self.search = search
        self.bus_factory = bus_factory
        self.bus: Any = None
        self._ready = asyncio.Event()
        self.task: asyncio.Task | None = None
        self.index_task: asyncio.Task | None = None
        # Start time of the last index pass that left nothing pending: every
        # dream recorded before it is searchable. None until one succeeds.
        self.index_complete_as_of: datetime | None = None

    def _read(self, fn: Callable[[Any], IntrospectResultV1]) -> Any:
        with self.engine_provider().connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            return fn(conn)

    async def handle(self, envelope: BaseEnvelope) -> None:
        reply = f"{RESULT_PREFIX}{envelope.correlation_id}"
        if envelope.reply_to != reply or envelope.kind != REQUEST_KIND:
            return
        now = datetime.now(timezone.utc)
        try:
            request = IntrospectRequestV1.model_validate(envelope.payload)
            args = DreamsArguments.model_validate(request.args)
        except ValueError as exc:
            await self._publish(reply, envelope, _failed(now, f"invalid dreams request: {str(exc)[:_INVALID_CAP]}"))
            return
        mode = "one" if args.dream_id else "search" if args.query else "recent"
        try:
            if args.dream_id is not None:
                result = await asyncio.to_thread(self._read, lambda c: one(c, args.dream_id, now=now))
            elif args.query is not None:
                if self.search is None or not self.search.enabled:
                    raise SearchUnavailableError("dream search is not configured")
                async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
                    scored = await rank(client, self.search, args.query, kind=args.kind, since=args.since)
                if not scored:
                    result = IntrospectResultV1(ok=True, operation="dreams", as_of=now, total_available=0)
                else:
                    result = await asyncio.to_thread(self._read, lambda c: by_ids(
                        c, scored, kind=args.kind, since=args.since, limit=args.limit, now=now,
                    ))
                if not result.items and await asyncio.to_thread(self._read, lambda c: self._unindexed(
                    c, kind=args.kind, since=args.since, now=now,
                )):
                    raise SearchUnavailableError("dream index behind the record; empty search is not proof")
            else:
                result = await asyncio.to_thread(self._read, lambda c: recent(
                    c, kind=args.kind, since=args.since, limit=args.limit, now=now,
                ))
        except Exception as exc:
            search_failed = isinstance(exc, SearchUnavailableError)
            logger.warning(
                "introspect_failed op=dreams corr=%s mode=%s category=%s exc_type=%s detail=%s",
                envelope.correlation_id, mode,
                "dream_search_failure" if search_failed else "dream_query_failure",
                type(exc).__name__, safe_exception_detail(exc),
            )
            result = _failed(now, SEARCH_UNAVAILABLE if search_failed else QUERY_UNAVAILABLE)
        else:
            logger.info(
                "introspect op=dreams corr=%s mode=%s items=%d total=%s",
                envelope.correlation_id, mode, len(result.items), result.total_available,
            )
        await self._publish(reply, envelope, result)

    async def _publish(self, reply: str, envelope: BaseEnvelope, result: IntrospectResultV1) -> None:
        await self.bus.publish(reply, BaseEnvelope(
            kind=RESULT_KIND, correlation_id=envelope.correlation_id,
            source=self.source, payload=result.model_dump(mode="json"),
        ))

    def _unindexed(self, conn: Any, *, kind: Any, since: datetime | None, now: datetime) -> bool:
        """True when a dream matching the filters may be missing from the index."""
        as_of = self.index_complete_as_of
        if as_of is None:
            return True
        window = as_of if since is None else max(since, as_of)
        return bool(recent(conn, kind=kind, since=window, limit=1, now=now).total_available)

    async def index_once(self):
        started = datetime.now(timezone.utc)
        pairs = await asyncio.to_thread(self._read, index_rows)
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
            result = await index_missing(pairs, self.search, client=client, bus=self.bus, source=self.source)
        if result.pending == 0:
            self.index_complete_as_of = started
        logger.info("dream_search_index indexed=%d pending=%d", result.indexed, result.pending)
        return result

    async def _index_loop(self) -> None:
        while True:
            await self._ready.wait()
            try:
                await self.index_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("dream_search_index_failed exc_type=%s detail=%s",
                               type(exc).__name__, safe_exception_detail(exc))
            await asyncio.sleep(self.search.index_interval_sec)

    async def _run(self) -> None:
        while True:
            bus = self.bus_factory(self.bus_url)
            try:
                await bus.connect()
                self.bus = bus
                self._ready.set()
                async with bus.subscribe(DREAM_REQUEST_CHANNEL) as pubsub:
                    logger.info("dream_introspect_listening channel=%s", DREAM_REQUEST_CHANNEL)
                    async for raw in bus.iter_messages(pubsub):
                        try:
                            decoded = bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                await self.handle(decoded.envelope)
                        except Exception:
                            logger.warning("dream_introspect_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("dream_introspect_reconnecting", exc_info=True)
            finally:
                self._ready.clear()
                with suppress(Exception):
                    await bus.close()
            await asyncio.sleep(1)

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name="dream-introspect")
        if self.search is not None and self.search.enabled:
            self.index_task = asyncio.create_task(self._index_loop(), name="dream-search-index")

    async def stop(self) -> None:
        for attr in ("task", "index_task"):
            task = getattr(self, attr)
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                setattr(self, attr, None)


def build_listener() -> DreamIntrospectListener:
    from sqlalchemy import create_engine

    from app.settings import settings
    from orion.core.bus.bus_schemas import ServiceRef

    engine = create_engine(settings.POSTGRES_URI, pool_pre_ping=True)
    return DreamIntrospectListener(
        bus_url=settings.ORION_BUS_URL,
        engine_provider=lambda: engine,
        source=ServiceRef(name="orion-dream", version=settings.SERVICE_VERSION, node=settings.NODE_NAME),
        search=SearchConfig(
            chroma_url=settings.DREAM_SEARCH_CHROMA_URL,
            embed_url=settings.DREAM_SEARCH_EMBED_URL,
            collection=settings.DREAM_SEARCH_COLLECTION,
            min_similarity=settings.DREAM_SEARCH_MIN_SIMILARITY,
            index_interval_sec=settings.DREAM_SEARCH_INDEX_INTERVAL_SEC,
            index_batch=settings.DREAM_SEARCH_INDEX_BATCH,
        ),
    )

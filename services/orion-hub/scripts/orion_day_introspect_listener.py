"""Hub's introspect responder for the `orion_day` tool, plus its search index loop.

Orion rereads an Orion's Day letter they wrote: the outline, a note paragraph with its
claim check, a carry item with its citations resolved, one day section's records, or
parts by meaning. Design: docs/superpowers/specs/2026-10-11-orion-day-letter-reread-design.md.

Same trust and honesty rules as the curiosity responder (curiosity_introspect_listener.py):
replies only when reply_to is exactly orion:introspect:result:<corr> and the kind matches;
every Postgres read runs in a READ ONLY transaction; errors become ok=false ("answer
unknown"), never []. A letter or part that does not exist is a tool error that says
"not found", never items=[] (an empty list would read as "the letter said nothing").
A letter that exists with an empty section is ok=true, items=[].

Search: each numbered note paragraph and carry item is embedded once and indexed under its
ref ("2026-10-09 ¶3") via semantic_index.py; every hit is re-read from orion_day_letter.
An empty search is only reported empty once the index is known to hold every letter.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import date, datetime, timezone
from typing import Any, Callable, Optional

import httpx

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.introspect.redact import safe_exception_detail
from orion.introspect.semantic_index import (
    HTTP_TIMEOUT_SEC,
    IndexPass,
    SearchConfig,
    SearchUnavailableError,
    confirmed_complete_as_of,
    embed,
    index_docs,
    nearest,
)
from orion.introspect.transport import ORION_DAY_REQUEST_CHANNEL, REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.orion_day.letter_parts import LetterPart, find_part, parse_ref, section_records, split_carry, split_note
from orion.orion_day.store import (
    LETTER_CREATED_SINCE_SQL,
    fetch_latest_letter,
    fetch_letter,
    fetch_letter_texts,
)
from orion.schemas.introspect import IntrospectRequestV1, IntrospectResultV1, OrionDayArguments
from orion.schemas.orion_day import OrionDayLetterV1

from scripts.curiosity_introspect_listener import ReadOnlyPool
from scripts.orion_day_introspect import index_docs_from_rows, outline_items, part_item, section_item

logger = logging.getLogger("orion-hub.orion_day_introspect")

OPERATION = "orion_day"
QUERY_UNAVAILABLE = "orion_day_unavailable; answer unknown"
SEARCH_UNAVAILABLE = "orion_day_search_unavailable; answer unknown"
# Letter parts share the curiosity search's Chroma / embedder / floor settings; only the
# collection differs. No env key of its own (it names an index, not a behaviour).
SEARCH_COLLECTION = "orion_day_letter_parts"
_INVALID_CAP = 400
_NO_POOL_RETRY_SEC = 15.0
CANDIDATES = 10
_DOC_PREFIX = "orion-day-search"
_HASH_KEYS = ("letter_date", "part", "occurred_ts")


class NotFoundError(LookupError):
    """The letter or part asked for does not exist. The message says so plainly."""


def _failed(now: datetime, error: str) -> IntrospectResultV1:
    return IntrospectResultV1(ok=False, operation=OPERATION, as_of=now, error=error)


def _ok(now: datetime, items: list, total: int) -> IntrospectResultV1:
    return IntrospectResultV1(ok=True, operation=OPERATION, as_of=now, total_available=total, items=items)


def search_filter(letter_date: Optional[date], part: str) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if letter_date is not None:
        clauses.append({"letter_date": letter_date.isoformat()})
    if part in ("note", "carry_forward"):
        clauses.append({"part": part})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _numbered(parts: list[LetterPart], kind: str) -> list[LetterPart]:
    return [p for p in parts if p.kind == kind]


class OrionDayIntrospectListener:
    def __init__(self, *, pool_provider: Callable[[], Any], source_ref: Any, search: SearchConfig | None):
        self.pool_provider = pool_provider
        self.source_ref = source_ref
        self.search = search
        self.client_factory: Callable[[], httpx.AsyncClient] = lambda: httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC)
        self.bus: Any = None
        self.task: asyncio.Task | None = None
        self.index_task: asyncio.Task | None = None
        # Every letter created before this is searchable (semantic_index.confirmed_complete_as_of).
        self.index_complete_as_of: datetime | None = None

    def _pool(self) -> ReadOnlyPool:
        pool = self.pool_provider()
        if pool is None:
            raise RuntimeError("no memory pool")
        return ReadOnlyPool(pool)

    async def handle(self, envelope: BaseEnvelope) -> None:
        reply = f"{RESULT_PREFIX}{envelope.correlation_id}"
        if envelope.reply_to != reply or envelope.kind != REQUEST_KIND:
            return
        now = datetime.now(timezone.utc)
        try:
            request = IntrospectRequestV1.model_validate(envelope.payload)
            if request.operation != OPERATION:
                raise ValueError(f"operation {request.operation!r} is not answered here")
            args = OrionDayArguments.model_validate(request.args)
        except ValueError as exc:
            await self._publish(reply, envelope, _failed(now, f"invalid orion_day request: {str(exc)[:_INVALID_CAP]}"))
            return
        if args.query is not None:
            mode = "search"
        elif args.part in ("note", "carry_forward"):
            mode = "one" if args.index is not None else "parts"
        else:
            mode = args.part  # list | section
        try:
            pool = self._pool()
            if mode == "search":
                result = await self._search(pool, args, now)
            else:
                result = await self._read(pool, args, mode, now)
        except NotFoundError as exc:
            logger.info(
                "orion_day_introspect_not_found corr=%s part=%s mode=%s detail=%s",
                envelope.correlation_id, args.part, mode, str(exc)[:_INVALID_CAP],
            )
            result = _failed(now, str(exc)[:_INVALID_CAP])
        except Exception as exc:
            search_failed = isinstance(exc, SearchUnavailableError)
            logger.warning(
                "introspect_failed op=orion_day corr=%s mode=%s category=%s exc_type=%s detail=%s",
                envelope.correlation_id, mode,
                "orion_day_search_failure" if search_failed else "orion_day_query_failure",
                type(exc).__name__, safe_exception_detail(exc),
            )
            result = _failed(now, SEARCH_UNAVAILABLE if search_failed else QUERY_UNAVAILABLE)
        else:
            logger.info(
                "orion_day_introspect_answered corr=%s part=%s mode=%s items=%d total=%s refs=%s",
                envelope.correlation_id, args.part, mode, len(result.items), result.total_available,
                ",".join(i.id for i in result.items),
            )
        await self._publish(reply, envelope, result)

    # --- modes -------------------------------------------------------------

    @staticmethod
    async def _letter(conn: Any, letter_date: Optional[date]) -> OrionDayLetterV1:
        if letter_date is None:
            letter = await fetch_latest_letter(conn)
            if letter is None:
                raise NotFoundError("unknown letter: no Orion's Day letter has been written yet (not found)")
            return letter
        letter = await fetch_letter(conn, letter_date)
        if letter is None:
            raise NotFoundError(
                f"unknown letter_date {letter_date.isoformat()}: no Orion's Day letter for that date (not found)"
            )
        return letter

    async def _read(self, pool: ReadOnlyPool, args: OrionDayArguments, mode: str, now: datetime) -> IntrospectResultV1:
        async with pool.acquire() as conn:
            letter = await self._letter(conn, args.letter_date)
        date_s = letter.letter_date.isoformat()
        note, carry = split_note(letter.note_md), split_carry(letter.carry_forward_md)
        if mode == "list":
            items = outline_items(letter, note, carry)
            return _ok(now, items, len(items))
        if mode == "section":
            records = section_records(letter.material, args.section)
            shown = records[: args.limit]
            items = [section_item(letter, args.section, ref, rec, n_items=len(shown)) for ref, rec in shown]
            return _ok(now, items, len(records))
        kind = "paragraph" if args.part == "note" else "carry"
        parts = _numbered(note if kind == "paragraph" else carry, kind)
        if mode == "one":
            part = find_part(parts, kind, args.index)
            if part is None:
                label = "note paragraphs" if kind == "paragraph" else "carry items"
                mark = f"¶{args.index}" if kind == "paragraph" else f"carry {args.index}"
                raise NotFoundError(
                    f"unknown part {date_s} {mark}: that letter has {len(parts)} numbered {label} (not found)"
                )
            return _ok(now, [part_item(letter, part, full=True)], 1)
        shown = parts[: args.limit]
        return _ok(now, [part_item(letter, p, full=False, n_items=len(shown)) for p in shown], len(parts))

    async def _search(self, pool: ReadOnlyPool, args: OrionDayArguments, now: datetime) -> IntrospectResultV1:
        letters: dict[str, Optional[OrionDayLetterV1]] = {}
        if args.letter_date is not None:
            # A search narrowed to a letter that does not exist is "not found", not "no match",
            # and that is known before the embedder or index is asked anything.
            async with pool.acquire() as conn:
                letters[args.letter_date.isoformat()] = await self._letter(conn, args.letter_date)
        if self.search is None or not self.search.enabled:
            raise SearchUnavailableError("orion_day search is not configured")
        async with self.client_factory() as client:
            vector, _ = await embed(client, self.search, args.query, doc_prefix=_DOC_PREFIX)
            hits = await nearest(client, self.search, vector, CANDIDATES,
                                 where=search_filter(args.letter_date, args.part))
        scored = [h for h in hits if h[1] >= self.search.min_similarity]
        found: list[tuple[OrionDayLetterV1, LetterPart, float]] = []
        dropped = 0
        async with pool.acquire() as conn:
            for ref, score in scored:
                parsed = parse_ref(ref)
                if parsed is None:
                    dropped += 1
                    continue
                day, kind, index = parsed
                if day not in letters:
                    letters[day] = await fetch_letter(conn, date.fromisoformat(day))
                letter = letters[day]
                if letter is None:
                    dropped += 1  # indexed, but the letter row is gone
                    continue
                parts = split_note(letter.note_md) if kind == "paragraph" else split_carry(letter.carry_forward_md)
                part = find_part(parts, kind, index)
                if part is None:
                    dropped += 1  # indexed under a number the stored letter no longer has
                    continue
                found.append((letter, part, score))
        if not found and dropped:
            raise RuntimeError(f"{dropped} search hit(s) could not be read back")
        narrowed = letters.get(args.letter_date.isoformat()) if args.letter_date is not None else None
        if not found and await self._unindexed(pool, narrowed):
            raise SearchUnavailableError("orion_day index behind the record; empty search is not proof")
        shown = found[: args.limit]
        items = [
            part_item(letter, part, full=False, n_items=len(shown), extra={"similarity": round(float(score), 3)})
            for letter, part, score in shown
        ]
        return _ok(now, items, len(found))

    async def _unindexed(self, pool: ReadOnlyPool, letter: Optional[OrionDayLetterV1] = None) -> bool:
        """True when a letter the search could match may be missing from the index. A search
        narrowed to one letter asks only about that letter."""
        as_of = self.index_complete_as_of
        if as_of is None:
            return True
        if letter is not None:
            created = letter.created_at if letter.created_at.tzinfo else letter.created_at.replace(tzinfo=timezone.utc)
            return created >= as_of
        async with pool.acquire() as conn:
            return bool(await conn.fetchval(LETTER_CREATED_SINCE_SQL, as_of))

    async def _publish(self, reply: str, envelope: BaseEnvelope, result: IntrospectResultV1) -> None:
        await self.bus.publish(reply, BaseEnvelope(
            kind=RESULT_KIND, correlation_id=envelope.correlation_id,
            source=self.source_ref, payload=result.model_dump(mode="json"),
        ))

    # --- index -------------------------------------------------------------

    async def index_once(self) -> Optional[IndexPass]:
        if self.pool_provider() is None:
            logger.info("orion_day_search_index skipped reason=no_pool")
            return None
        started = datetime.now(timezone.utc)
        async with self._pool().acquire() as conn:
            rows = await fetch_letter_texts(conn)
        async with self.client_factory() as client:
            result = await index_docs(
                index_docs_from_rows(rows), self.search, client=client, bus=self.bus,
                source=self.source_ref, doc_prefix=_DOC_PREFIX, hash_keys=_HASH_KEYS,
            )
        confirmed = confirmed_complete_as_of(result, started)
        if confirmed is not None:
            self.index_complete_as_of = confirmed
        logger.info("orion_day_search_index indexed=%d pending=%d", result.indexed, result.pending)
        return result

    async def _index_loop(self) -> None:
        while True:
            delay = self.search.index_interval_sec
            try:
                if await self.index_once() is None:
                    delay = min(delay, _NO_POOL_RETRY_SEC)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("orion_day_search_index_failed exc_type=%s detail=%s",
                               type(exc).__name__, safe_exception_detail(exc))
            await asyncio.sleep(delay)

    # --- lifecycle ---------------------------------------------------------

    async def _run(self) -> None:
        while True:
            try:
                async with self.bus.subscribe(ORION_DAY_REQUEST_CHANNEL) as pubsub:
                    logger.info("orion_day_introspect_listening channel=%s", ORION_DAY_REQUEST_CHANNEL)
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                await self.handle(decoded.envelope)
                        except Exception:
                            logger.warning("orion_day_introspect_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("orion_day_introspect_reconnecting", exc_info=True)
                await asyncio.sleep(1)

    async def start(self, bus: Any) -> None:
        self.bus = bus
        self.task = asyncio.create_task(self._run(), name="hub-orion-day-introspect")
        if self.search is not None and self.search.enabled:
            self.index_task = asyncio.create_task(self._index_loop(), name="hub-orion-day-search-index")

    async def stop(self) -> None:
        for attr in ("task", "index_task"):
            task = getattr(self, attr)
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                setattr(self, attr, None)

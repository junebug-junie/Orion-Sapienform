"""Hub's introspect responder for the `curiosity` tool, plus its search index loop.

Trust: replies only when reply_to is exactly orion:introspect:result:<corr> and
the kind matches; never to a model-supplied subject. Every Postgres read runs
in a READ ONLY transaction (`ReadOnlyPool`), including the run join's own
reads. Errors become ok=false ("answer unknown"), never []. A search that finds
nothing is only reported empty when the index is known to hold every write-up
recorded so far; otherwise "no match" could mean "not indexed" (the dreams
`index_complete_as_of` rule).

Runs come from `curiosity_run_store.read_run_payload` / `read_runs_payload`
-- the Curiosity tab's own join -- so the tool and the tab cannot disagree
about what a run was.
"""
from __future__ import annotations

import asyncio
import logging
import math
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.curiosity.atlas import valid_run_id
from orion.curiosity.run_story import _line_for, _obj
from orion.introspect.redact import safe_exception_detail
from orion.introspect.semantic_index import (
    HTTP_TIMEOUT_SEC,
    IndexPass,
    SearchConfig,
    SearchUnavailableError,
    embed,
    index_docs,
    nearest,
)
from orion.introspect.transport import CURIOSITY_REQUEST_CHANNEL, REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.schemas.introspect import CuriosityArguments, IntrospectRequestV1, IntrospectResultV1

from scripts import curiosity_run_store as run_store
from scripts.curiosity_introspect import index_text, occurred_at, run_item, self_question_item

logger = logging.getLogger("orion-hub.curiosity_introspect")

QUERY_UNAVAILABLE = "curiosity_unavailable; answer unknown"
SEARCH_UNAVAILABLE = "curiosity_search_unavailable; answer unknown"
_INVALID_CAP = 400
# The memory pool comes up after listeners start; do not wait a full index interval for it.
_NO_POOL_RETRY_SEC = 15.0
# Above-floor hits re-read through the run join per query (~0.1 s each live).
CANDIDATES = 10
_DOC_PREFIX = "curiosity-search"
_HASH_KEYS = ("occurred_ts", "line")
_RUNNING = "running"
# Recent mode re-reads at most limit * this many listed runs to fill `limit`.
_REREAD_FACTOR = 3

OUTCOMES_SQL = (
    "SELECT run_id, turn_ok, n_tested, n_moved, n_formed, unknown_reason "
    "FROM curiosity_run_outcomes WHERE run_id = ANY($1::text[])"
)
SELF_QUESTIONS_SQL = (
    "SELECT question_id, text, family, pinned, ask_count, last_asked_at, created_at, "
    "count(*) OVER () AS total FROM curiosity_self_questions "
    "WHERE status = 'open' AND ($1::timestamptz IS NULL OR created_at >= $1) "
    "ORDER BY created_at DESC, question_id DESC LIMIT $2"
)
# One write-up per run (the newest journal, the one the run story shows), with
# the cheap Postgres signals `run_story._line_for` reads, so the index can
# carry each run's line and a `line` filter runs before the top-N cut.
INDEX_ROWS_SQL = (
    "SELECT DISTINCT ON (j.source_ref) j.source_ref, j.title, j.body, j.created_at, "
    "a.request::text AS request, "
    "(SELECT s.detail::text FROM substrate_durable_run_state s "
    " WHERE s.run_id = substr(j.source_ref, 11) AND s.status = 'completed' "
    " ORDER BY s.created_at DESC LIMIT 1) AS detail, "
    "EXISTS (SELECT 1 FROM self_sense_eval_log e WHERE e.run_id = substr(j.source_ref, 11)) AS self_sense "
    "FROM journal_entries j LEFT JOIN durable_admission_runs a ON a.run_id = substr(j.source_ref, 11) "
    "WHERE j.source_ref LIKE 'curiosity:%' AND j.body IS NOT NULL "
    "ORDER BY j.source_ref, j.created_at DESC"
)
UNINDEXED_SQL = (
    "SELECT EXISTS (SELECT 1 FROM journal_entries "
    "WHERE source_ref LIKE 'curiosity:%' AND created_at >= $1)"
)


class GraphUnreadError(RuntimeError):
    """A run is absent from Postgres and the graph could not be read: unknown, not absent."""


class ReadOnlyPool:
    """Wraps an asyncpg pool so every acquired connection sits in a READ ONLY transaction."""

    def __init__(self, pool: Any):
        self._pool = pool

    @asynccontextmanager
    async def acquire(self):
        async with self._pool.acquire() as conn:
            async with conn.transaction(readonly=True):
                yield conn


def _failed(now: datetime, error: str) -> IntrospectResultV1:
    return IntrospectResultV1(ok=False, operation="curiosity", as_of=now, error=error)


def _ok(now: datetime, items: list, total: int) -> IntrospectResultV1:
    return IntrospectResultV1(ok=True, operation="curiosity", as_of=now, total_available=total, items=items)


def _epoch(value: datetime) -> float:
    return (value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)).timestamp()


def _days_since(since: Optional[datetime], now: datetime) -> int:
    if since is None:
        return run_store.WINDOW_DAYS_MAX
    return run_store.clamp_days(math.ceil((now - since).total_seconds() / 86400.0))


def _graph_extra(stores: Any) -> dict[str, Any]:
    """Hop/finding/revision counts come from the graph; say so when it was not read."""
    graph = (stores or {}).get("graph") if isinstance(stores, dict) else None
    return {} if graph == "ok" else {"graph_read": False}


def index_line(row: dict[str, Any]) -> str:
    """The run's line from the Postgres half of the run join's own rule.

    Calls `run_story._line_for` on a minimal slot rather than restating the
    rule; the graph-only fallback (prior revisions) is not available here, and
    the search re-checks the story's line after every re-read anyway.
    """
    slot = {
        "admission": {"request": row.get("request")} if row.get("request") else None,
        "self_sense": [True] if row.get("self_sense") else [],
        "lifecycle": [],
        "journals": [{"title": row.get("title")}],
        "revisions": [],
    }
    return _line_for(slot, _obj(row.get("detail")), {})[0]


def index_docs_from_rows(rows: list[dict[str, Any]]) -> list[tuple[str, str, dict[str, Any]]]:
    docs = []
    for row in rows:
        ref = str(row.get("source_ref") or "")
        run_id = valid_run_id(ref.split(":", 1)[1]) if ref.startswith("curiosity:") else None
        text = index_text(row.get("body"))
        created = row.get("created_at")
        if not run_id or not text or not isinstance(created, datetime):
            continue
        docs.append((run_id, text, {
            "occurred_at": created.isoformat(), "occurred_ts": _epoch(created), "line": index_line(row),
        }))
    return docs


def search_filter(since: Optional[datetime], line: Optional[str]) -> dict[str, Any] | None:
    """Chroma where-clause, so `since` and `line` apply before the top-N cut.

    `since` is on the write-up's own clock, which is never earlier than the
    run's start, so it only removes runs the re-read would drop too.
    """
    clauses: list[dict[str, Any]] = []
    if line is not None:
        clauses.append({"line": line})
    if since is not None:
        clauses.append({"occurred_ts": {"$gte": _epoch(since)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


async def rank(
    client: httpx.AsyncClient, cfg: SearchConfig, query: str,
    *, since: Optional[datetime] = None, line: Optional[str] = None,
) -> list[tuple[str, float]]:
    """Embed the query once; (run_id, similarity) at or above the floor, best first."""
    vector, _ = await embed(client, cfg, query, doc_prefix=_DOC_PREFIX)
    hits = await nearest(client, cfg, vector, CANDIDATES, where=search_filter(since, line))
    return [s for s in hits if s[1] >= cfg.min_similarity]


class CuriosityIntrospectListener:
    def __init__(
        self, *, pool_provider: Callable[[], Any], reader_provider: Callable[[], Any],
        source_ref: Any, search: SearchConfig | None,
    ):
        self.pool_provider = pool_provider
        self.reader_provider = reader_provider
        self.source_ref = source_ref
        self.search = search
        self.client_factory: Callable[[], httpx.AsyncClient] = lambda: httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC)
        self.bus: Any = None
        self.task: asyncio.Task | None = None
        self.index_task: asyncio.Task | None = None
        # Start time of the last index pass that left nothing pending: every
        # write-up recorded before it is searchable. None until one succeeds.
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
            if request.operation != "curiosity":
                raise ValueError(f"operation {request.operation!r} is not answered here")
            args = CuriosityArguments.model_validate(request.args)
        except ValueError as exc:
            await self._publish(reply, envelope, _failed(now, f"invalid curiosity request: {str(exc)[:_INVALID_CAP]}"))
            return
        if args.kind == "self_question":
            mode = "self_question"
        else:
            mode = "one" if args.run_id else "search" if args.query else "recent"
        try:
            pool = self._pool()
            if mode == "self_question":
                result = await self._self_questions(pool, args, now)
            elif mode == "one":
                result = await self._one(pool, args.run_id, now)
            elif mode == "search":
                result = await self._search(pool, args, now)
            else:
                result = await self._recent(pool, args, now)
        except Exception as exc:
            search_failed = isinstance(exc, SearchUnavailableError)
            logger.warning(
                "introspect_failed op=curiosity corr=%s mode=%s category=%s exc_type=%s detail=%s",
                envelope.correlation_id, mode,
                "curiosity_search_failure" if search_failed else "curiosity_query_failure",
                type(exc).__name__, safe_exception_detail(exc),
            )
            result = _failed(now, SEARCH_UNAVAILABLE if search_failed else QUERY_UNAVAILABLE)
        else:
            logger.info(
                "introspect op=curiosity corr=%s mode=%s items=%d total=%s",
                envelope.correlation_id, mode, len(result.items), result.total_available,
            )
        await self._publish(reply, envelope, result)

    # --- modes -------------------------------------------------------------

    async def _story(self, pool: ReadOnlyPool, run_id: str) -> Optional[dict[str, Any]]:
        """One run through the tab's join. None = no store knows it; raises when unknown.

        "Not found" is only trusted when the graph was read too: runs from
        before the admission path exist only there.
        """
        payload = await run_store.read_run_payload(pool=pool, reader=self.reader_provider(), run_id=run_id)
        # Postgres holds the admission rows and the write-up; without it a
        # "not found" or a story with no write-up would be a guess.
        stores = payload.get("stores") or {}
        if not payload.get("available") or stores.get("postgres") != "ok":
            raise RuntimeError(f"run store unavailable: {payload.get('reason') or stores}")
        if payload.get("found"):
            return payload
        if stores.get("graph") != "ok":
            raise GraphUnreadError(f"run not in postgres and graph unread: {stores.get('graph')}")
        return None

    async def _outcomes(self, pool: ReadOnlyPool, run_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not run_ids:
            return {}
        async with pool.acquire() as conn:
            rows = await conn.fetch(OUTCOMES_SQL, run_ids)
        return {str(r["run_id"]): dict(r) for r in rows}

    async def _items(
        self, pool: ReadOnlyPool, stories: list[dict[str, Any]], *, full: bool,
    ) -> list:
        outcomes = await self._outcomes(pool, [str(s["run"]["run_id"]) for s in stories])
        items = []
        for story in stories:
            rid = str(story["run"]["run_id"])
            extra = _graph_extra(story.get("stores"))
            if "similarity" in story:
                extra["similarity"] = round(float(story["similarity"]), 3)
            item = run_item(story, outcomes.get(rid), full=full, extra=extra)
            if item is not None:
                items.append(item)
        return items

    async def _one(self, pool: ReadOnlyPool, run_id: str, now: datetime) -> IntrospectResultV1:
        story = await self._story(pool, run_id)
        if story is None:
            return _ok(now, [], 0)
        items = await self._items(pool, [story], full=True)
        if not items:
            raise RuntimeError("run found but carries no clock")
        return _ok(now, items, 1)

    async def _recent(self, pool: ReadOnlyPool, args: CuriosityArguments, now: datetime) -> IntrospectResultV1:
        payload = await run_store.read_runs_payload(
            pool=pool, reader=self.reader_provider(), days=_days_since(args.since, now),
            line=args.line or "all", now=now,
        )
        if not payload.get("available") or (payload.get("stores") or {}).get("postgres") != "ok":
            raise RuntimeError(f"run store unavailable: {payload.get('reason') or payload.get('stores')}")
        runs = []
        for run in payload.get("runs") or []:
            if run.get("status") == _RUNNING:
                continue
            when = occurred_at(run)
            if args.since is not None and when is not None and when < args.since:
                continue
            runs.append(run)
        # `runs` is newest first with clockless runs last; a clockless run can
        # still be returned on its write-up's clock after the re-read.
        stories = []
        for run in runs[: args.limit * _REREAD_FACTOR]:
            if len(stories) >= args.limit:
                break
            story = await self._story_or_skip(pool, str(run["run_id"]))
            if story is not None:
                stories.append(story)
        items = [
            i for i in await self._items(pool, stories, full=False)
            if args.since is None or i.occurred_at >= args.since
        ]
        items.sort(key=lambda i: i.occurred_at, reverse=True)
        return _ok(now, items, max(len(runs), len(items)))

    async def _story_or_skip(self, pool: ReadOnlyPool, run_id: str) -> Optional[dict[str, Any]]:
        """Recent mode: the window just listed this run, so a vanished story is skipped."""
        try:
            return await self._story(pool, run_id)
        except GraphUnreadError:
            return None

    async def _search(self, pool: ReadOnlyPool, args: CuriosityArguments, now: datetime) -> IntrospectResultV1:
        if self.search is None or not self.search.enabled:
            raise SearchUnavailableError("curiosity search is not configured")
        async with self.client_factory() as client:
            scored = await rank(client, self.search, args.query, since=args.since, line=args.line)
        matched, dropped = [], 0
        for run_id, score in scored:
            story = await self._story(pool, run_id)
            if story is None:
                dropped += 1  # indexed, but no store knows the run any more
                continue
            if args.line is not None and story["run"].get("line") != args.line:
                continue
            matched.append({**story, "similarity": score})
        built = await self._items(pool, matched, full=False)
        dropped += len(matched) - len(built)  # found but carrying no clock at all
        items = [i for i in built if args.since is None or i.occurred_at >= args.since]
        if not items and dropped:
            # A hit lost for a reason other than the filters is not evidence of no match.
            raise RuntimeError(f"{dropped} search hit(s) could not be read back")
        if not items and await self._unindexed(pool, args.since):
            raise SearchUnavailableError("curiosity index behind the record; empty search is not proof")
        return _ok(now, items[: args.limit], len(items))

    async def _self_questions(self, pool: ReadOnlyPool, args: CuriosityArguments, now: datetime) -> IntrospectResultV1:
        async with pool.acquire() as conn:
            rows = [dict(r) for r in await conn.fetch(SELF_QUESTIONS_SQL, args.since, args.limit)]
        total = int(rows[0]["total"]) if rows else 0
        return _ok(now, [self_question_item(r) for r in rows], total)

    async def _unindexed(self, pool: ReadOnlyPool, since: Optional[datetime]) -> bool:
        """True when a write-up matching the filters may be missing from the index."""
        as_of = self.index_complete_as_of
        if as_of is None:
            return True
        window = as_of if since is None else max(since, as_of)
        async with pool.acquire() as conn:
            return bool(await conn.fetchval(UNINDEXED_SQL, window))

    async def _publish(self, reply: str, envelope: BaseEnvelope, result: IntrospectResultV1) -> None:
        await self.bus.publish(reply, BaseEnvelope(
            kind=RESULT_KIND, correlation_id=envelope.correlation_id,
            source=self.source_ref, payload=result.model_dump(mode="json"),
        ))

    # --- index -------------------------------------------------------------

    async def index_once(self) -> Optional[IndexPass]:
        if self.pool_provider() is None:
            logger.info("curiosity_search_index skipped reason=no_pool")
            return None
        started = datetime.now(timezone.utc)
        # Release the connection before embedding; a pass can take seconds.
        async with self._pool().acquire() as conn:
            rows = [dict(r) for r in await conn.fetch(INDEX_ROWS_SQL)]
        async with self.client_factory() as client:
            result = await index_docs(
                index_docs_from_rows(rows), self.search, client=client, bus=self.bus,
                source=self.source_ref, doc_prefix=_DOC_PREFIX, hash_keys=_HASH_KEYS,
            )
        if result.pending == 0:
            self.index_complete_as_of = started
        logger.info("curiosity_search_index indexed=%d pending=%d", result.indexed, result.pending)
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
                logger.warning("curiosity_search_index_failed exc_type=%s detail=%s",
                               type(exc).__name__, safe_exception_detail(exc))
            await asyncio.sleep(delay)

    # --- lifecycle ---------------------------------------------------------

    async def _run(self) -> None:
        while True:
            try:
                async with self.bus.subscribe(CURIOSITY_REQUEST_CHANNEL) as pubsub:
                    logger.info("curiosity_introspect_listening channel=%s", CURIOSITY_REQUEST_CHANNEL)
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                await self.handle(decoded.envelope)
                        except Exception:
                            logger.warning("curiosity_introspect_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("curiosity_introspect_reconnecting", exc_info=True)
                await asyncio.sleep(1)

    async def start(self, bus: Any) -> None:
        self.bus = bus
        self.task = asyncio.create_task(self._run(), name="hub-curiosity-introspect")
        if self.search is not None and self.search.enabled:
            self.index_task = asyncio.create_task(self._index_loop(), name="hub-curiosity-search-index")

    async def stop(self) -> None:
        for attr in ("task", "index_task"):
            task = getattr(self, attr)
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                setattr(self, attr, None)

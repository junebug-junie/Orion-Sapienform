"""Single writer for Orion's personality graph (spec rev 2, bottom-up).

A plain periodic task (no LangGraph): all state lives in the ``personality_*`` tables and their
per-source cursors, so a restart resumes exactly. One tick, every ``PERSONALITY_TICK_SEC`` and
once on start:

1. load non-retired nodes and the cursors;
2. first tick ever: write a start cursor (= now) for every source (no backfill) and seed nodes
   from the latest version of each curiosity self-inquiry concept (all-or-nothing: a failed embed
   leaves the ``seed`` marker unwritten, so the whole seed is retried next tick);
3. read every source after its cursor, up to now minus the read lag, capped per source;
4. drop evidence ids already in ``personality_evidence`` (a re-read curiosity run is not
   re-weighted), embed the rest (bounded concurrency); a failed embed stops THAT source at the
   previous row, so the failed row is read again next tick;
5. ``fold`` (fades, then folds) or, with no evidence, ``apply_fade`` so fade and retirement still
   persist;
6. commit evidence, nodes, history and cursors in one transaction.

A failed tick logs ``personality_tick_failed`` with the exception type and writes nothing.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional, Sequence

from orion.personality.graph import Evidence, FoldResult, Node, apply_fade, fold, seed_nodes
from orion.personality.situation import COMPANY_CLOSE_GRACE, situation_at

from app.personality_sources import (
    READ_LAG_SEC,
    SEED_CURSOR,
    SOURCE_NAMES,
    Candidate,
    Cursor,
    SourceItem,
)
from app.personality_store import TickWrite, node_key

logger = logging.getLogger("orion-durable-runs.personality_driver")

EMBED_CONCURRENCY = 4
EMBED_TIMEOUT_SEC = 15.0
# A row whose embed fails this many ticks in a row (or that the embedder rejects with a 4xx) is
# consumed as skipped evidence, so one bad row cannot hold its lane dark forever.
EMBED_MAX_FAILURES = 3
REJECTED: list[float] = []  # vector placeholder: fold records it as skipped ("bad_embedding")

# (doc_id, text) -> unit vector; raises on failure.
EmbedFn = Callable[[str, str], Awaitable[list[float]]]


def http_embedder(url: str, *, timeout_sec: float = EMBED_TIMEOUT_SEC) -> tuple[EmbedFn, Callable[[], Awaitable[None]]]:
    """POST ``EmbeddingGenerateV1`` to vector-host's ``/embedding`` (compute only, nothing stored)."""
    import httpx

    from orion.schemas.vector.schemas import EmbeddingGenerateV1, EmbeddingResultV1

    client = httpx.AsyncClient(timeout=timeout_sec)

    async def embed(doc_id: str, text: str) -> list[float]:
        req = EmbeddingGenerateV1(doc_id=doc_id, text=text)
        resp = await client.post(url, json=req.model_dump(mode="json"))
        resp.raise_for_status()
        result = EmbeddingResultV1.model_validate(resp.json())
        if not result.embedding:
            raise ValueError("embedder returned no vector")
        return list(result.embedding)

    return embed, client.aclose


def _rejected(exc: BaseException) -> bool:
    """The embedder refused this text (HTTP 4xx): retrying the same row cannot help."""
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    return isinstance(code, int) and 400 <= code < 500


async def embed_all(embed: EmbedFn, cands: Sequence[Candidate], *,
                    concurrency: int = EMBED_CONCURRENCY) -> dict[str, Optional[list[float]]]:
    """evidence_id -> vector; None when that embed failed (retry later); ``REJECTED`` on a 4xx."""
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(c: Candidate) -> tuple[str, Optional[list[float]]]:
        async with sem:
            try:
                return c.evidence_id, await embed(f"personality-{c.evidence_id}", c.text)
            except Exception as exc:  # noqa: BLE001 - one failed embed holds its source, nothing more
                logger.warning("personality_embed_failed evidence_id=%s error=%s", c.evidence_id, type(exc).__name__)
                return c.evidence_id, (REJECTED if _rejected(exc) else None)

    return dict(await asyncio.gather(*(one(c) for c in cands)))


def to_evidence(c: Candidate, vec: list[float]) -> Evidence:
    meta = dict(c.meta)
    if not vec:
        meta["embed_rejected"] = True
    return Evidence(evidence_id=c.evidence_id, lane=c.lane, source_ref=c.source_ref, occurred_at=c.occurred_at,
                    excerpt=c.excerpt, embedding=vec, meta=meta)


def accept(items: Sequence[SourceItem], existing: set[str], vectors: dict[str, Optional[list[float]]],
           ) -> tuple[list[Evidence], Optional[Cursor], bool]:
    """Walk one source's items in order. Returns (new evidence, cursor to persist or None when no
    item was consumed, held) -- ``held`` means an embed failed and the source stopped before it."""
    out: list[Evidence] = []
    last: Optional[SourceItem] = None
    for item in items:
        new = [c for c in item.candidates if c.evidence_id not in existing]
        if any(vectors.get(c.evidence_id) is None for c in new):
            return out, (Cursor(last.at, last.ref) if last else None), True
        out.extend(to_evidence(c, vectors[c.evidence_id]) for c in new)
        last = item
    return out, (Cursor(last.at, last.ref) if last else None), False


def merge(seeded: Optional[FoldResult], folded: FoldResult, input_ids: set[str]) -> FoldResult:
    """Seed result + fold result, deduped (the fold's copy of a node wins)."""
    nodes: dict[str, Node] = {}
    history: dict[tuple[str, datetime], Any] = {}
    evidence = []
    for res in (seeded, folded):
        if res is None:
            continue
        for n in res.nodes:
            nodes[n.disposition_id] = n
        for h in res.history:
            history[(h.disposition_id, h.at)] = h
        evidence.extend(res.evidence_rows)
    return FoldResult(nodes=list(nodes.values()), evidence_rows=evidence, history=list(history.values()))


@dataclass
class TickStats:
    ticks: int = 0
    failures: int = 0
    last_tick_at: Optional[str] = None
    last_error: Optional[str] = None
    evidence_last_tick: int = 0
    skipped_last_tick: int = 0
    rows_last_tick: dict = field(default_factory=dict)
    held_sources: list = field(default_factory=list)
    nodes_by_status: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    seeded: bool = False


CHRONICLE_MAX_LAG = timedelta(minutes=30)


class PersonalityDriver:
    def __init__(self, *, store: Any, reader: Any, embed: EmbedFn, now: Callable[[], datetime],
                 tick_sec: float = 600.0, read_lag_sec: float = READ_LAG_SEC,
                 close_embedder: Optional[Callable[[], Awaitable[None]]] = None) -> None:
        self._store = store
        self._reader = reader
        self._embed = embed
        self._now = now
        self._tick_sec = tick_sec
        self._lag = timedelta(seconds=read_lag_sec)
        self._close_embedder = close_embedder
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._embed_failures: dict[str, int] = {}  # evidence_id -> consecutive failed ticks
        self.stats = TickStats()

    async def start(self, stop: asyncio.Event) -> None:
        self._task = asyncio.create_task(self._loop(stop))

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if self._close_embedder is not None:
            try:
                await self._close_embedder()
            except Exception:  # noqa: BLE001
                pass

    async def _loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._tick_sec)
            except asyncio.TimeoutError:
                pass

    async def run_once(self) -> Optional[dict]:
        """One tick; never raises (the service must not crash on a bad tick)."""
        try:
            return await self.tick()
        except Exception as exc:  # noqa: BLE001
            self.stats.failures += 1
            self.stats.last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            logger.warning("personality_tick_failed error_type=%s error=%s", type(exc).__name__, str(exc)[:300])
            return None

    async def tick(self) -> dict:
        async with self._lock:
            return await self._tick()

    async def _with_situation(self, evidence: list[Evidence], warnings: list[str]) -> list[Evidence]:
        """Stamp ``meta.situation`` (arousal, day phase, company at the row's own time) onto new
        evidence. Recorded only (CAPS if-then data for later situation subnodes); fail-open: if the
        Temporal Self tables cannot be read the evidence still folds, without a situation."""
        if not evidence:
            return evidence
        try:
            times = [e.occurred_at for e in evidence]
            inputs = await self._store.load_situation_inputs(min(times), max(times))
            return [replace(e, meta={**dict(e.meta or {}), "situation": situation_at(e.occurred_at, inputs)})
                    for e in evidence]
        except Exception as exc:  # noqa: BLE001 - the situation is an annotation, never a gate
            warnings.append(f"situation_unavailable:{type(exc).__name__}")
            return evidence

    async def _tick(self) -> dict:
        now = self._now()
        hi = now - self._lag
        # Read no newer than the Temporal Self chronicle has folded, so each row's situation
        # (company arcs close ~5 min late) is final when stamped. A chronicle stalled past
        # CHRONICLE_MAX_LAG does not stall personality: rows past its watermark get company=None.
        try:
            watermark = await self._store.chronicle_watermark()
        except Exception:  # noqa: BLE001 - the situation is an annotation, never a gate
            watermark = None
        chronicle_fresh = watermark is not None and now - watermark <= CHRONICLE_MAX_LAG
        if chronicle_fresh:
            hi = min(hi, watermark - COMPANY_CLOSE_GRACE)
        warnings: list[str] = []
        if not chronicle_fresh:
            warnings.append("situation_chronicle_lagging:company_unknown_past_watermark")
        nodes = await self._store.load_nodes()
        cursors = await self._store.load_cursors()
        write_cursors: dict[str, Cursor] = {}

        # First boot (per source): start at now, no backfill. Not read this tick.
        for name in SOURCE_NAMES:
            if name not in cursors:
                write_cursors[name] = Cursor(now, "")
        readable = {k: v for k, v in cursors.items() if k in SOURCE_NAMES}

        seeded: Optional[FoldResult] = None
        seed_ids: set[str] = set()
        if SEED_CURSOR not in cursors:
            # Bounded by the ongoing self-inquiry source's start: a version written after boot is
            # that source's row, never also a seed (a retried seed would otherwise fold it twice).
            start = cursors.get("self_inquiry_seed")
            seed_items = await self._reader.read_seeds(start.at if start is not None else now)
            seed_cands = [c for it in seed_items for c in it.candidates]
            seed_ids = {c.evidence_id for c in seed_cands}
            vecs = await embed_all(self._embed, seed_cands)
            if seed_cands and all(vecs.get(c.evidence_id) is not None for c in seed_cands):
                # A rejected (4xx) seed text is dropped, not retried forever.
                evs = [to_evidence(c, vecs[c.evidence_id]) for c in seed_cands if vecs[c.evidence_id]]
                seeded = await asyncio.to_thread(seed_nodes, evs, now=now)
                write_cursors[SEED_CURSOR] = Cursor(now, "")
            elif not seed_cands:
                write_cursors[SEED_CURSOR] = Cursor(now, "")
                warnings.append("seed:no_self_inquiry_rows")
            else:
                warnings.append("seed:embed_failed")

        by_source = await self._reader.read(readable, hi) if readable else {}
        all_cands = [c for items in by_source.values() for it in items for c in it.candidates]
        existing = await self._store.existing_evidence_ids(c.evidence_id for c in all_cands)
        if seeded is not None:
            existing |= seed_ids  # belt and braces: a seed row is never folded twice in one tick
        vectors = await embed_all(self._embed, [c for c in all_cands if c.evidence_id not in existing])
        for eid, vec in vectors.items():
            if vec is None:
                n = self._embed_failures.get(eid, 0) + 1
                self._embed_failures[eid] = n
                if n >= EMBED_MAX_FAILURES:
                    vectors[eid] = REJECTED
                    warnings.append(f"embed_gave_up:{eid}"[:200])
            else:
                self._embed_failures.pop(eid, None)
        for eid, vec in vectors.items():
            if vec is not None and not vec:
                self._embed_failures.pop(eid, None)

        evidence: list[Evidence] = []
        held: list[str] = []
        rows: dict[str, int] = {}
        for name, items in by_source.items():
            rows[name] = len(items)
            ev, cur, was_held = accept(items, existing, vectors)
            evidence.extend(ev)
            if cur is not None:
                write_cursors[name] = cur
            if was_held:
                held.append(name)
                warnings.append(f"embed_failed:{name}")

        evidence = await self._with_situation(evidence, warnings)

        input_ids = {n.disposition_id for n in nodes} | {n.disposition_id for n in (seeded.nodes if seeded else [])}
        base = list(nodes) + (list(seeded.nodes) if seeded else [])
        # Pure and O(nodes x evidence x dim): off the event loop, which also runs admission and RPCs.
        if evidence:
            folded = await asyncio.to_thread(fold, base, evidence, now=now)
        else:
            folded = await asyncio.to_thread(apply_fade, base, now=now)
        result = merge(seeded, folded, input_ids)
        loaded = {n.disposition_id: node_key(n) for n in nodes}

        await self._store.commit_tick(TickWrite(now=now, evidence=result.evidence_rows, nodes=result.nodes,
                                                loaded=loaded, history=result.history, cursors=write_cursors))

        try:
            status = await self._store.status_counts()
        except Exception:  # noqa: BLE001 - the tick is committed; counts are only for /health
            status = self.stats.nodes_by_status
        s = self.stats
        s.ticks += 1
        s.last_tick_at = now.isoformat()
        s.last_error = None
        s.evidence_last_tick = sum(1 for e in result.evidence_rows if e.disposition_id is not None)
        s.skipped_last_tick = sum(1 for e in result.evidence_rows if e.disposition_id is None)
        s.rows_last_tick = rows
        s.held_sources = held
        s.nodes_by_status = status
        s.warnings = warnings
        s.seeded = s.seeded or seeded is not None or SEED_CURSOR in cursors
        summary = {"evidence": s.evidence_last_tick, "skipped": s.skipped_last_tick, "rows": rows, "held": held,
                   "nodes_written": len(result.nodes), "history": len(result.history), "warnings": warnings}
        logger.info("personality_tick %s", summary)
        return summary

    def health(self) -> dict:
        s = self.stats
        return {"ticks": s.ticks, "failures": s.failures, "last_tick_at": s.last_tick_at, "last_error": s.last_error,
                "evidence_last_tick": s.evidence_last_tick, "skipped_last_tick": s.skipped_last_tick,
                "rows_last_tick": s.rows_last_tick, "held_sources": s.held_sources,
                "nodes_by_status": s.nodes_by_status, "warnings": s.warnings, "seeded": s.seeded,
                "tick_sec": self._tick_sec}

    def frame_warnings(self, now: datetime) -> list[str]:
        """Warnings carried into the frame: staleness/failure FIRST (they must survive the 20 cap),
        then the last tick's own warnings."""
        s = self.stats
        out: list[str] = []
        if s.last_error:
            out.append(f"last_tick_failed:{s.last_error.split(':', 1)[0]}")
        if s.last_tick_at is None:
            out.append("no_tick_committed_since_start")
        else:
            age = (now - datetime.fromisoformat(s.last_tick_at)).total_seconds()
            if age > 3 * self._tick_sec:
                out.append(f"stale:{int(age)}s_since_last_tick")
        out.extend(s.warnings)
        return out[:20]

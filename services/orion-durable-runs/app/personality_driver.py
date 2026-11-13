"""Single writer for Orion's personality graph (spec rev 4: choices only).

A plain periodic task (no LangGraph): all state lives in the ``personality_*`` tables and their
cursors, so a restart resumes exactly. One tick, every ``PERSONALITY_TICK_SEC`` and once on start:

1. curiosity sync: read curiosity runs after the ``curiosity_sync`` cursor (from the epoch the first
   time: the one-time backfill), turn each into its ``PersonalityChoiceV1`` rows and insert them into
   ``personality_choice`` (insert-only) with the cursor, in one transaction;
2. load the non-retired choice nodes and read ``personality_choice`` rows after the ``choice``
   cursor (bus producers and the curiosity sync alike), up to now minus the read lag, capped;
3. drop evidence ids already in ``personality_evidence``; embed every distinct option text (bounded
   concurrency, cached across ticks); a failed embed stops the read at the previous row, so the
   failed row is read again next tick, for as long as the embedder is down (choices are sparse and
   are the graph's only evidence: an outage never consumes one). Only an HTTP 4xx (the embedder
   refused the text) consumes the choice as skipped evidence;
4. each choice -> one ``Evidence``: direction = unit(embed(chosen) - mean(embed(declined))), weight
   = revealed-preference weight, label = "<chosen gist> over <declined gist>";
5. ``fold`` (fades, then folds) or, with no evidence, ``apply_fade``;
6. self-naming (``PERSONALITY_NAMING_APPLY_ENABLED``): apply Orion's answers to the folded nodes;
7. commit evidence, nodes, history and cursors in one transaction.

``offer_choice`` is the bus intake (``orion:personality:choice``): it validates and inserts one
choice. It does not fold; the next tick does.

A failed tick logs ``personality_tick_failed`` with the exception type and writes nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional, Sequence

from orion.personality.graph import (
    EVIDENCE_PREFIX,
    Evidence,
    FoldResult,
    NamingApplyResult,
    NamingRow,
    Node,
    apply_fade,
    apply_namings,
    choice_direction,
    decline_text,
    evidence_weight,
    fold,
    pair_label,
)
from orion.personality.naming import grounding
from orion.personality.situation import COMPANY_CLOSE_GRACE, situation_at
from orion.schemas.personality import PersonalityChoiceV1

from app.personality_sources import (
    CHOICE_CURSOR,
    CURIOSITY_CURSOR,
    READ_LAG_SEC,
    Cursor,
    choice_from_row,
    curiosity_run_choices,
    utc,
)
from app.personality_store import NAMING_CURSOR, TickWrite, node_key

logger = logging.getLogger("orion-durable-runs.personality_driver")

EMBED_CONCURRENCY = 4
EMBED_TIMEOUT_SEC = 15.0
# Self-naming grounding only: a NAME whose grounding embed fails this many ticks is applied ungrounded.
# Choices never give up on a transient failure (see the module docstring).
EMBED_MAX_FAILURES = 3
# A bus choice dated further ahead than this is refused (a future time would never fade).
MAX_FUTURE_SKEW_SEC = 300.0
# Lanes durable-runs writes itself; a bus producer may not write them.
SELF_WRITTEN_LANES = frozenset({"curiosity"})
EMBED_CACHE_MAX = 4000  # option text -> vector, across ticks (curiosity priors repeat across runs)
REJECTED: list[float] = []  # vector placeholder: the embedder refused this text

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
    """The embedder refused this text (HTTP 4xx): retrying the same text cannot help."""
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    return isinstance(code, int) and 400 <= code < 500


def option_texts(c: PersonalityChoiceV1) -> tuple[str, list[str]]:
    """(chosen-side text, declined texts in shown order). ``declined_all`` uses the lane's decline
    phrase as the chosen side: a decline of every option."""
    chosen = decline_text(c.lane) if c.declined_all or c.chosen is None else c.chosen.text
    declined = sorted(c.alternatives, key=lambda o: (o.market_rank is None, o.market_rank or 0))
    return chosen, [o.text for o in declined]


def all_texts(c: PersonalityChoiceV1) -> list[str]:
    chosen, declined = option_texts(c)
    return [chosen, *declined]


def choice_evidence(c: PersonalityChoiceV1, vectors: dict[str, Optional[list[float]]]) -> Evidence:
    """One choice -> one ``Evidence`` (a missing/rejected vector gives an empty direction, which
    ``fold`` records as skipped ``bad_direction``)."""
    chosen, declined = option_texts(c)
    vc = vectors.get(chosen)
    vds = [vectors.get(t) for t in declined]
    direction = None
    if vc and all(vds):
        direction = choice_direction(vc, vds)  # type: ignore[arg-type]
    meta = {
        "choice_id": c.choice_id, "chosen_market_rank": c.chosen_market_rank, "declined_all": c.declined_all,
        "care_promoted": c.care_promoted, "picks_in_decision": c.picks_in_decision,
        "alternatives": len(c.alternatives), "act_id": c.act_id, "correlation_id": c.correlation_id,
    }
    if c.situation:
        meta["choice_situation"] = dict(c.situation)
    return Evidence(
        evidence_id=f"{EVIDENCE_PREFIX}{c.choice_id}", lane=c.lane, source_ref=c.act_id or c.choice_id,
        occurred_at=utc(c.occurred_at), excerpt=pair_label(chosen, declined), embedding=direction or [],
        weight=evidence_weight(c.chosen_market_rank, declined_all=c.declined_all, care_promoted=c.care_promoted,
                               picks_in_decision=c.picks_in_decision),
        meta=meta, text="\n".join([chosen, *declined]),
    )


def merge(folded: FoldResult, named: Optional[NamingApplyResult] = None) -> FoldResult:
    """Fold + naming results, deduped (the naming step's copy of a node wins)."""
    nodes: dict[str, Node] = {}
    history: dict[tuple[str, datetime], Any] = {}
    evidence = []
    for res in (folded, named):
        if res is None:
            continue
        for n in res.nodes:
            nodes[n.disposition_id] = n
        for h in res.history:
            history[(h.disposition_id, h.at)] = h
        evidence.extend(getattr(res, "evidence_rows", ()))
    return FoldResult(nodes=list(nodes.values()), evidence_rows=evidence, history=list(history.values()))


@dataclass
class TickStats:
    ticks: int = 0
    failures: int = 0
    last_tick_at: Optional[str] = None
    last_error: Optional[str] = None
    evidence_last_tick: int = 0
    skipped_last_tick: int = 0
    choices_read_last_tick: int = 0
    curiosity_synced_last_tick: int = 0  # new personality_choice rows from the curiosity sync
    bus_choices_inserted: int = 0  # since start
    bus_choices_invalid: int = 0  # since start
    held: bool = False
    nodes_by_status: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    namings_last_tick: list = field(default_factory=list)  # [{naming, verdict, apply_status, ...}]


CHRONICLE_MAX_LAG = timedelta(minutes=30)


class PersonalityDriver:
    def __init__(self, *, store: Any, reader: Any, embed: EmbedFn, now: Callable[[], datetime],
                 tick_sec: float = 600.0, read_lag_sec: float = READ_LAG_SEC,
                 close_embedder: Optional[Callable[[], Awaitable[None]]] = None,
                 naming_enabled: bool = True) -> None:
        self._store = store
        self._reader = reader
        self._embed = embed
        self._now = now
        self._tick_sec = tick_sec
        self._lag = timedelta(seconds=read_lag_sec)
        self._close_embedder = close_embedder
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._embed_failures: dict[str, int] = {}  # choice_id / naming_id -> consecutive failed ticks
        self._vec_cache: "OrderedDict[str, list[float]]" = OrderedDict()
        self.naming_enabled = naming_enabled
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

    # --- bus intake -------------------------------------------------------------------------------

    async def offer_choice(self, payload: Any) -> str:
        """One ``orion:personality:choice`` payload: validate, insert (no-op when already stored).
        Returns ``inserted`` | ``duplicate`` | ``invalid``. Raises only on a DB failure."""
        try:
            choice = PersonalityChoiceV1.model_validate(payload)
            if choice.lane in SELF_WRITTEN_LANES:
                raise ValueError(f"lane {choice.lane!r} is written by durable-runs' own sync, not the bus")
            if (choice.occurred_at - self._now()).total_seconds() > MAX_FUTURE_SKEW_SEC:
                raise ValueError("occurred_at is in the future")
        except Exception as exc:  # noqa: BLE001 - a bad payload is logged and dropped, never retried
            self.stats.bus_choices_invalid += 1
            logger.warning("personality_choice_invalid error=%s", str(exc)[:300])
            return "invalid"
        new = await self._store.insert_choices([choice], written_by="bus")
        if new:
            self.stats.bus_choices_inserted += 1
        logger.info("personality_choice_received choice_id=%s lane=%s rank=%s declined_all=%s new=%s",
                    choice.choice_id, choice.lane, choice.chosen_market_rank, choice.declined_all, bool(new))
        return "inserted" if new else "duplicate"

    # --- tick -------------------------------------------------------------------------------------

    async def _with_situation(self, evidence: list[Evidence], warnings: list[str]) -> list[Evidence]:
        """Stamp ``meta.situation`` (arousal, day phase, company at the choice's own time). Recorded
        only; fail-open: if the Temporal Self tables cannot be read the evidence still folds."""
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

    async def _sync_curiosity(self, cursors: dict, hi: datetime) -> int:
        """Curiosity runs -> personality_choice rows, with the cursor, in one transaction."""
        rows = await self._reader.read_curiosity(cursors.get(CURIOSITY_CURSOR), hi)
        if not rows:
            return 0
        choices = [c for r in rows for c in curiosity_run_choices(r)]
        last = rows[-1]
        cur = Cursor(utc(last["completed_at"]), str(last["run_id"]))
        new = await self._store.insert_choices(choices, written_by="curiosity_sync", cursor=(CURIOSITY_CURSOR, cur))
        cursors[CURIOSITY_CURSOR] = cur
        return new

    async def _embed_texts(self, texts: Sequence[str]) -> dict[str, Optional[list[float]]]:
        """text -> vector; None when the embed failed (retry later); ``REJECTED`` on a 4xx."""
        out: dict[str, Optional[list[float]]] = {}
        todo = []
        for t in dict.fromkeys(texts):
            if t in self._vec_cache:
                self._vec_cache.move_to_end(t)
                out[t] = self._vec_cache[t]
            else:
                todo.append(t)
        sem = asyncio.Semaphore(EMBED_CONCURRENCY)

        async def one(t: str) -> tuple[str, Optional[list[float]]]:
            async with sem:
                doc = "personality-opt-" + hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]
                try:
                    return t, await self._embed(doc, t)
                except Exception as exc:  # noqa: BLE001 - one failed embed holds its choice, nothing more
                    logger.warning("personality_embed_failed error=%s", type(exc).__name__)
                    return t, (REJECTED if _rejected(exc) else None)

        for t, vec in await asyncio.gather(*(one(t) for t in todo)):
            out[t] = vec
            if vec:
                self._vec_cache[t] = vec
                while len(self._vec_cache) > EMBED_CACHE_MAX:
                    self._vec_cache.popitem(last=False)
        return out

    async def _tick(self) -> dict:
        now = self._now()
        hi = now - self._lag
        # Read no newer than the Temporal Self chronicle has folded, so each choice's situation is
        # final when stamped. A stalled chronicle does not stall personality.
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
        cursors = await self._store.load_cursors()
        synced = await self._sync_curiosity(cursors, now - self._lag)
        nodes = await self._store.load_nodes()
        write_cursors: dict[str, Cursor] = {}

        rows = await self._reader.read_choices(cursors.get(CHOICE_CURSOR), hi)
        parsed = [(r, choice_from_row(r)) for r in rows]
        ids = [f"{EVIDENCE_PREFIX}{r['choice_id']}" for r in rows]
        existing = await self._store.existing_evidence_ids(ids)
        fresh = [c for (r, c) in parsed if c is not None and f"{EVIDENCE_PREFIX}{c.choice_id}" not in existing]
        texts = [t for c in fresh for t in all_texts(c)]
        vectors = await self._embed_texts(texts)

        evidence: list[Evidence] = []
        held = False
        last: Optional[dict] = None
        for r, c in parsed:
            eid = f"{EVIDENCE_PREFIX}{r['choice_id']}"
            if c is None:
                # Stored row no longer validates: consumed as skipped evidence, never folded.
                evidence.append(Evidence(evidence_id=eid, lane="invalid", source_ref=str(r["choice_id"]),
                                         occurred_at=utc(r["occurred_at"]) or now, excerpt="invalid stored choice",
                                         embedding=[], meta={"choice_id": r["choice_id"]}))
            elif eid not in existing:
                if any(vectors.get(t) is None for t in all_texts(c)):
                    # Transient (embedder down or slow): hold here; never consume a real choice for it.
                    held = True
                    warnings.append(f"embed_failed:choice:{c.choice_id}"[:200])
                    break
                evidence.append(choice_evidence(c, vectors))
            last = r
        if last is not None:
            write_cursors[CHOICE_CURSOR] = Cursor(utc(last["created_at"]), str(last["choice_id"]))

        evidence = await self._with_situation(evidence, warnings)
        # Pure and O(nodes x evidence x dim): off the event loop, which also runs admission and RPCs.
        if evidence:
            folded = await asyncio.to_thread(fold, nodes, evidence, now=now)
        else:
            folded = await asyncio.to_thread(apply_fade, nodes, now=now)
        named: Optional[NamingApplyResult] = None
        if self.naming_enabled:
            current = {n.disposition_id: n for n in nodes} | {n.disposition_id: n for n in folded.nodes}
            named = await self._apply_namings(list(current.values()), cursors, write_cursors, now, warnings)
        result = merge(folded, named)
        loaded = {n.disposition_id: node_key(n) for n in nodes}

        await self._store.commit_tick(TickWrite(now=now, evidence=result.evidence_rows, nodes=result.nodes,
                                                loaded=loaded, history=result.history, cursors=write_cursors))
        # Logged only once committed: the line is the acceptance evidence that a verdict landed.
        applied = []
        for o in (named.outcomes if named is not None else []):
            margin = f"{o.grounding_margin:.4f}" if o.grounding_margin is not None else "none"
            logger.info("personality_naming_applied naming=%s verdict=%s grounding_rank=%s margin=%s apply_status=%s",
                        o.naming_id, o.verdict, o.grounding_rank if o.grounding_rank is not None else "none",
                        margin, o.apply_status)
            applied.append({"naming": o.naming_id, "disposition_id": o.disposition_id, "verdict": o.verdict,
                            "apply_status": o.apply_status, "grounding_rank": o.grounding_rank,
                            "grounding_margin": o.grounding_margin})
        self.stats.namings_last_tick = applied

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
        s.choices_read_last_tick = len(rows)
        s.curiosity_synced_last_tick = synced
        s.held = held
        s.nodes_by_status = status
        s.warnings = warnings
        summary = {"evidence": s.evidence_last_tick, "skipped": s.skipped_last_tick, "choices_read": len(rows),
                   "curiosity_synced": synced, "held": held, "namings": len(applied),
                   "nodes_written": len(result.nodes), "history": len(result.history), "warnings": warnings}
        logger.info("personality_tick %s", summary)
        return summary

    async def _apply_namings(self, current: list[Node], cursors: dict, write_cursors: dict, now: datetime,
                             warnings: list[str]) -> Optional[NamingApplyResult]:
        """Orion's answers after the ``naming`` cursor, applied to the folded nodes (same tick, same
        transaction). A NAME's grounding embed that fails holds the cursor at the row before it (the
        row is re-read next tick) until ``EMBED_MAX_FAILURES``; then it is applied ungrounded."""
        rows = await self._reader.read_namings(cursors.get(NAMING_CURSOR), now - self._lag)
        if not rows:
            return None
        names = [r for r in rows if r.get("verdict") == "name" and r.get("hub_check") == "accepted"]
        vecs: dict[str, Optional[list[float]]] = {}
        sem = asyncio.Semaphore(EMBED_CONCURRENCY)

        async def one(r: dict) -> None:
            async with sem:
                text = f"{(r.get('name') or '').strip()} {(r.get('sentence') or '').strip()}".strip()
                try:
                    vecs[r["naming_id"]] = await self._embed(f"personality-naming-{r['naming_id']}", text)
                except Exception as exc:  # noqa: BLE001 - grounding is recorded, never a gate
                    logger.warning("personality_naming_embed_failed naming=%s error=%s",
                                   r["naming_id"], type(exc).__name__)
                    vecs[r["naming_id"]] = REJECTED if _rejected(exc) else None

        await asyncio.gather(*(one(r) for r in names))
        live = [n for n in current if n.status != "retired"]
        take: list[dict] = []
        ground: dict[str, tuple[int, float]] = {}
        for r in rows:
            nid = r["naming_id"]
            if nid in vecs:
                vec = vecs[nid]
                if vec is None:
                    fails = self._embed_failures.get(nid, 0) + 1
                    self._embed_failures[nid] = fails
                    if fails < EMBED_MAX_FAILURES:
                        warnings.append("naming_grounding_embed_failed")
                        break
                    warnings.append(f"naming_grounding_gave_up:{nid}"[:200])
                self._embed_failures.pop(nid, None)
                g = grounding(vec, r["disposition_id"], live) if vec else None
                if g is not None:
                    ground[nid] = g
            take.append(r)
        if not take:
            return None
        namings = [NamingRow(naming_id=r["naming_id"], disposition_id=r["disposition_id"], verdict=r["verdict"],
                             hub_check=r["hub_check"], written_at=r["written_at"], name=r.get("name"),
                             sentence=r.get("sentence"), merge_with=r.get("merge_with")) for r in take]
        res = apply_namings(current, namings, ground, now=now)
        last = take[-1]
        write_cursors[NAMING_CURSOR] = Cursor(last["created_at"], last["naming_id"])
        return res

    def health(self) -> dict:
        s = self.stats
        return {"ticks": s.ticks, "failures": s.failures, "last_tick_at": s.last_tick_at, "last_error": s.last_error,
                "evidence_last_tick": s.evidence_last_tick, "skipped_last_tick": s.skipped_last_tick,
                "choices_read_last_tick": s.choices_read_last_tick,
                "curiosity_synced_last_tick": s.curiosity_synced_last_tick,
                "bus_choices_inserted": s.bus_choices_inserted, "bus_choices_invalid": s.bus_choices_invalid,
                "held": s.held, "nodes_by_status": s.nodes_by_status, "warnings": s.warnings,
                "tick_sec": self._tick_sec, "naming_enabled": self.naming_enabled,
                "namings_last_tick": s.namings_last_tick}

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

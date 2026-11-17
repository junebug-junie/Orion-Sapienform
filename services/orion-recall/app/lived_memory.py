"""Lived-memory recall for the Juniper chat prompt (issue #3533, "Recall rev 2" step 1).

Finds the few lived memories (``orion/memory/lived_memory.py``: episode_memory, stance
crystallizations, person facts) that fit this chat turn by meaning, and renders at most
``max_lines`` of them into DURABLE MEMORY. Zero lines is a valid answer.

Pipeline, once per chat turn (PCR continuity phase):

1. **Candidates, two ways.** Meaning: embed the turn (orion-vector-host ``/embedding``,
   bge-large) and take the nearest docs in the ``orion_lived_memory`` Chroma collection.
   Exact words: BM25 over the same live records, so names and rare words are not lost; a query
   word that is a name (capitalized mid-sentence, or a known person's name) counts
   ``NAME_WEIGHT`` times.
2. **Gate.** Every candidate must be in the current live snapshot (Postgres, not Chroma): a
   superseded, rejected, expired or removed record is never served, whatever Chroma still holds.
3. **Rerank.** A local cross-encoder (orion-vector-host ``/rerank``) scores each (turn, memory)
   pair. ``fit = sigmoid(logit)``.
4. **Quality bar.** Records with ``fit < min_fit`` are dropped. The bar is tuned by replay
   (``evals/run_lived_memory_replay_eval.py``).
5. **Rank** (Generative Agents, Park et al. 2023): ``score = fit + W_RECENCY * recency +
   W_IMPORTANCE * importance + W_CORRECTION * is_correction``. Fit dominates; recency and
   importance only order memories that already fit.
6. **Re-check and render.** The chosen ids are re-read from Postgres with the serving gate
   (a correction written since the snapshot removes what it corrects), then rendered one line
   each with kind, record id and date.

Indexing: ``index_loop`` re-embeds new or changed records every ``index_interval_sec`` through
the existing semantic-index plumbing (``orion/introspect/semantic_index.py``: vector-host embeds,
orion-vector-writer stores).
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import httpx

from orion.introspect.semantic_index import (
    IndexPass,
    SearchConfig,
    SearchUnavailableError,
    delete_ids,
    embed,
    index_docs,
    indexed_ids,
    nearest,
)
from orion.memory.lived_memory import (
    KINDS,
    KIND_PERSON_FACT,
    LivedRecord,
    fetch_live_records,
    index_doc,
    split_doc_id,
)

logger = logging.getLogger("orion-recall.lived_memory")

SOURCE = "lived_memory"
HEADER = "[Lived memories that fit this turn -- kind, record id, date]"

DENSE_TOP_K = 16
LEXICAL_TOP_K = 10
MAX_RERANK = 24
NAME_WEIGHT = 2.5
RECENCY_HALF_LIFE_DAYS = 30.0
W_RECENCY = 0.10
W_IMPORTANCE = 0.05
W_CORRECTION = 0.05
LINE_MAX_CHARS = 320
QUERY_MAX_CHARS = 1000
# What the cross-encoder reads. Cost grows with pair length (24 long pairs: ~1.3 s on CPU);
# the opening of a message and of a memory carry what it is about.
RERANK_QUERY_CHARS = 600
RERANK_DOC_CHARS = 400
SNAPSHOT_TTL_SEC = 60.0
# A short message still searches when it names something specific: a word found in at most this
# many lived records ("Biscuit", "pottery") is a reference, not a pleasantry.
RARE_WORD_MAX_DF = 3

# Standard English function words (BM25 stopping), not a topic list.
_STOPWORDS = frozenset(
    """a an the and or but if then so because as of at by for from in into on onto to with without
    about over under after before again also just only very really too not no yes ok okay
    i me my mine myself we us our ours you your yours he him his she her hers it its they them their
    this that these those there here what which who whom whose when where why how
    is am are was were be been being do does did doing have has had having will would shall should
    can could may might must let lets get got gonna wanna yeah haha lol oh hey hi hello thanks thank
    all any some each every more most other such own same than both few much many
    up down out off now still even ever like well one two""".split()
)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’]*")
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+|\n\s*)$")


def tokenize(text: str) -> List[str]:
    out = []
    for raw in _TOKEN.findall(text or ""):
        tok = raw.lower().replace("’", "'")
        if tok.endswith("'s"):
            tok = tok[:-2]
        tok = tok.replace("'", "")
        if len(tok) < 2 or tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


def name_tokens(query: str, known_names: Iterable[str] = ()) -> set[str]:
    """Query words that are names: capitalized and not the first word of a sentence, or a word of
    a known person's display name."""
    names: set[str] = set()
    for m in _TOKEN.finditer(query or ""):
        word = m.group(0)
        if not word[:1].isupper():
            continue
        if _SENTENCE_START.search(query[: m.start()]):
            continue
        toks = tokenize(word)
        if toks:
            names.add(toks[0])
    q = set(tokenize(query))
    for name in known_names:
        for tok in tokenize(name):
            if tok in q:
                names.add(tok)
    return names


class BM25:
    """Okapi BM25 over a small in-memory corpus (k1=1.2, b=0.75)."""

    def __init__(self, docs: Sequence[str], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(self.tfs)
        self.df = df
        self.idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens: Sequence[str], weights: Dict[str, float] | None = None) -> List[float]:
        weights = weights or {}
        qset = list(dict.fromkeys(query_tokens))
        out = []
        for tf, length in zip(self.tfs, self.lens):
            s = 0.0
            for t in qset:
                f = tf.get(t)
                if not f:
                    continue
                norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1.0)))
                s += weights.get(t, 1.0) * self.idf.get(t, 0.0) * norm
            out.append(s)
        return out


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def recency(occurred_at: datetime | None, now: datetime) -> float:
    if occurred_at is None:
        return 0.0
    age_days = max(0.0, (now - occurred_at).total_seconds() / 86400.0)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


@dataclass
class Scored:
    rec: LivedRecord
    fit: float
    score: float
    dense: float | None = None
    lexical: float | None = None


@dataclass
class LivedRecall:
    lines: List[str] = field(default_factory=list)
    chosen: List[Scored] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)
    # Every reranked candidate with its raw logit, before the bar (evals sweep the bar on this).
    reranked: List[tuple[LivedRecord, float]] = field(default_factory=list)

    @property
    def rendered(self) -> str:
        if not self.lines:
            return ""
        return HEADER + "\n" + "\n".join(self.lines)


def render_line(rec: LivedRecord) -> str:
    text = rec.text if len(rec.text) <= LINE_MAX_CHARS else rec.text[: LINE_MAX_CHARS - 1].rstrip() + "…"
    label = rec.kind if rec.kind == rec.purpose or rec.kind == KIND_PERSON_FACT else f"{rec.kind}/{rec.purpose}"
    return f"- ({label} · {rec.doc_id} · {rec.date}) {text}"


def rank(candidates: Sequence[tuple[LivedRecord, float]], *, min_fit: float, max_lines: int,
         now: datetime) -> List[Scored]:
    """(record, rerank logit) -> the records that clear the bar, best first, at most max_lines."""
    kept: List[Scored] = []
    for rec, logit in candidates:
        if not math.isfinite(float(logit)):
            continue
        fit = sigmoid(float(logit))
        if fit < min_fit:
            continue
        score = (fit + W_RECENCY * recency(rec.occurred_at, now) + W_IMPORTANCE * rec.importance
                 + (W_CORRECTION if rec.is_correction else 0.0))
        kept.append(Scored(rec=rec, fit=fit, score=score))
    kept.sort(key=lambda s: (-s.score, s.rec.doc_id))
    return kept[: max(0, int(max_lines))]


Reranker = Callable[[str, List[str]], Any]  # async (query, docs) -> list[float]


@dataclass(frozen=True)
class LivedMemoryConfig:
    search: SearchConfig
    rerank_url: str
    min_fit: float
    max_lines: int
    timeout_sec: float
    # A message with fewer content words than this ("hi Orion, how are you?") has nothing to
    # search with: greetings matched greeting-shaped stance messages by form, not content.
    # Unless it names something rare in the store (RARE_WORD_MAX_DF).
    min_query_tokens: int = 0


class LivedMemoryRecall:
    """Owns the live snapshot and runs one retrieval per call. ``pool`` is an asyncpg pool."""

    def __init__(self, pool: Any, cfg: LivedMemoryConfig, *, reranker: Optional[Reranker] = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.pool = pool
        self.cfg = cfg
        self._reranker = reranker
        self._clock = clock
        self._snapshot: List[LivedRecord] = []
        self._by_doc: Dict[str, LivedRecord] = {}
        self._bm25: BM25 | None = None
        self._names: List[str] = []
        self._loaded_at = 0.0
        self._lock = asyncio.Lock()

    async def snapshot(self, *, force: bool = False) -> List[LivedRecord]:
        async with self._lock:
            if force or not self._snapshot or time.monotonic() - self._loaded_at > SNAPSHOT_TTL_SEC:
                recs = await fetch_live_records(self.pool)
                self._snapshot = recs
                self._by_doc = {r.doc_id: r for r in recs}
                self._bm25 = BM25([r.text for r in recs])
                self._names = sorted({r.text.split(",", 1)[0] for r in recs if r.kind == KIND_PERSON_FACT})
                self._loaded_at = time.monotonic()
        return self._snapshot

    async def _rerank_http(self, client: httpx.AsyncClient, query: str, docs: List[str]) -> List[float]:
        resp = await client.post(self.cfg.rerank_url, json={"query": query, "documents": docs})
        resp.raise_for_status()
        scores = resp.json().get("scores")
        if not isinstance(scores, list) or len(scores) != len(docs):
            raise SearchUnavailableError("reranker returned a malformed reply")
        return [float(s) for s in scores]

    def has_rare_word(self, tokens: Iterable[str]) -> bool:
        df = self._bm25.df if self._bm25 is not None else {}
        return any(0 < df.get(t, 0) <= RARE_WORD_MAX_DF for t in tokens)

    def lexical(self, query: str, k: int = LEXICAL_TOP_K, *,
                allowed: Callable[[LivedRecord], bool] | None = None) -> List[tuple[str, float]]:
        if self._bm25 is None:
            return []
        qtoks = tokenize(query)
        if not qtoks:
            return []
        weights = {t: NAME_WEIGHT for t in name_tokens(query, self._names)}
        scores = self._bm25.scores(qtoks, weights)
        ranked = sorted(
            ((self._snapshot[i].doc_id, s) for i, s in enumerate(scores)
             if s > 0 and (allowed is None or allowed(self._snapshot[i]))),
            key=lambda x: -x[1],
        )
        return ranked[:k]

    async def recall(self, query: str, *, client: httpx.AsyncClient | None = None,
                     as_of: datetime | None = None) -> LivedRecall:
        """One retrieval. ``as_of`` (replay evals only) hides records Orion stored after that time,
        so a replayed message cannot be answered by the memory distilled from it."""
        started = time.perf_counter()
        timings: Dict[str, int] = {}
        out = LivedRecall(debug={"timings_ms": timings})
        query = " ".join(str(query or "").split())[:QUERY_MAX_CHARS]
        if not query:
            out.debug["skipped"] = "empty_query"
            return out
        own_client = client is None
        client = client or httpx.AsyncClient(timeout=self.cfg.timeout_sec)
        try:
            t = time.perf_counter()
            await self.snapshot()
            timings["snapshot"] = int((time.perf_counter() - t) * 1000)
            qtoks = set(tokenize(query))
            if len(qtoks) < self.cfg.min_query_tokens and not self.has_rare_word(qtoks):
                out.debug["skipped"] = "few_content_words"
                out.debug["content_tokens"] = len(qtoks)
                return out

            def allowed(rec: LivedRecord) -> bool:
                return as_of is None or rec.created_at is None or rec.created_at < as_of

            t = time.perf_counter()
            dense: List[tuple[str, float]] = []
            try:
                vector, _model = await embed(client, self.cfg.search, query, doc_prefix="lived-memory-query")
                # Replay hides later records, so ask for more neighbours to keep DENSE_TOP_K eligible.
                dense = await nearest(client, self.cfg.search, vector, DENSE_TOP_K if as_of is None else DENSE_TOP_K * 4)
            except SearchUnavailableError as exc:
                out.debug["dense_error"] = str(exc)
            timings["dense"] = int((time.perf_counter() - t) * 1000)
            lexical = self.lexical(query, allowed=allowed)

            pool: Dict[str, Dict[str, float]] = {}
            kept_dense = 0
            for did, sim in dense:
                rec = self._by_doc.get(did)
                if rec is not None and allowed(rec) and kept_dense < DENSE_TOP_K:
                    pool.setdefault(did, {})["dense"] = sim
                    kept_dense += 1
            for did, s in lexical:
                pool.setdefault(did, {})["lexical"] = s
            out.debug["candidates"] = {"dense": len(dense), "lexical": len(lexical), "live_pool": len(pool),
                                       "dense_not_live": sum(1 for d, _ in dense if d not in self._by_doc)}
            if not pool:
                return out
            ordered = sorted(pool, key=lambda d: (-pool[d].get("dense", 0.0), -pool[d].get("lexical", 0.0)))[:MAX_RERANK]
            recs = [self._by_doc[d] for d in ordered]

            t = time.perf_counter()
            docs = [r.text[:RERANK_DOC_CHARS] for r in recs]
            rq = query[:RERANK_QUERY_CHARS]
            if self._reranker is not None:
                logits = list(await self._reranker(rq, docs))
            else:
                logits = await self._rerank_http(client, rq, docs)
            timings["rerank"] = int((time.perf_counter() - t) * 1000)
            out.reranked = list(zip(recs, [float(x) for x in logits]))

            chosen = rank(list(zip(recs, logits)), min_fit=self.cfg.min_fit, max_lines=self.cfg.max_lines,
                          now=self._clock())
            out.debug["reranked"] = [
                {"id": r.doc_id, "fit": round(sigmoid(l), 4), "dense": round(pool[r.doc_id].get("dense", 0.0), 4),
                 "lexical": round(pool[r.doc_id].get("lexical", 0.0), 3)}
                for r, l in sorted(zip(recs, logits), key=lambda x: -x[1])[:8]
            ]
            if not chosen:
                return out

            t = time.perf_counter()
            ids_by_kind: Dict[str, List[str]] = {}
            for s in chosen:
                ids_by_kind.setdefault(s.rec.kind, []).append(s.rec.record_id)
            still_live = {r.doc_id for r in await fetch_live_records(self.pool, kinds=list(ids_by_kind), ids_by_kind=ids_by_kind)}
            timings["recheck"] = int((time.perf_counter() - t) * 1000)
            dropped = [s.rec.doc_id for s in chosen if s.rec.doc_id not in still_live]
            if dropped:
                out.debug["recheck_dropped"] = dropped
            chosen = [s for s in chosen if s.rec.doc_id in still_live]
            for s in chosen:
                d = pool.get(s.rec.doc_id, {})
                s.dense, s.lexical = d.get("dense"), d.get("lexical")
            out.chosen = chosen
            out.lines = [render_line(s.rec) for s in chosen]
            return out
        except (SearchUnavailableError, httpx.HTTPError, ValueError) as exc:
            out.debug["error"] = f"{type(exc).__name__}: {exc}"[:300]
            return out
        finally:
            timings["total"] = int((time.perf_counter() - started) * 1000)
            if own_client:
                await client.aclose()


@dataclass(frozen=True)
class LivedIndexPass:
    indexed: int
    pending: int
    removed: int = 0


async def run_index_pass(pool: Any, cfg: SearchConfig, *, client: httpx.AsyncClient, bus: Any, source: Any,
                         batch: int) -> LivedIndexPass:
    """Make the index mirror what may be served: embed and upsert live records Chroma lacks or
    holds stale text for (up to ``batch``), and delete docs whose record is no longer servable
    (superseded, rejected, expired, removed). Postgres keeps every row; only the index entry, and
    the private text copied into it, goes. Serving re-gates every hit anyway."""
    recs = await fetch_live_records(pool)
    docs = [index_doc(r) for r in recs]
    result: IndexPass = await index_docs(docs, cfg, client=client, bus=bus, source=source,
                                         doc_prefix="lived-memory", batch=batch, hash_keys=("occurred_ts",))
    removed = 0
    if recs:  # an empty live read is more likely an outage than a world with no memories
        live = {d[0] for d in docs}
        stale = [i for i in await indexed_ids(client, cfg) if i not in live]
        removed = await delete_ids(client, cfg, stale)
    return LivedIndexPass(indexed=result.indexed, pending=result.pending, removed=removed)


async def index_loop(pool: Any, cfg: SearchConfig, *, bus: Any, source: Any, interval_sec: float,
                     batch: int, stop: asyncio.Event | None = None) -> None:
    """Keep ``orion_lived_memory`` current. A pass with work left goes again after a short pause, so
    a backlog (first boot, a backfill of new memories) drains in minutes, not one batch per interval."""
    stop = stop or asyncio.Event()
    while not stop.is_set():
        delay = interval_sec
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                result = await run_index_pass(pool, cfg, client=client, bus=bus, source=source, batch=batch)
            if result.indexed or result.pending or result.removed:
                logger.info("lived_memory_index_pass indexed=%s pending=%s removed=%s collection=%s",
                            result.indexed, result.pending, result.removed, cfg.collection)
            if result.pending:
                delay = 5.0
        except Exception as exc:  # noqa: BLE001 -- the loop must outlive a down embedder or DB
            logger.warning("lived_memory_index_pass_failed error=%s", f"{type(exc).__name__}: {exc}"[:300])
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass


def items_for(chosen: Sequence[Scored]) -> List[Dict[str, Any]]:
    """MemoryItemV1 kwargs for the bundle (ids land in recall_telemetry.selected_ids: the read log)."""
    out = []
    for s in chosen:
        out.append({
            "id": s.rec.doc_id,
            "source": SOURCE,
            "source_ref": s.rec.kind,
            "score": round(s.fit, 4),
            "ts": s.rec.occurred_at.timestamp() if s.rec.occurred_at else None,
            "snippet": s.rec.text[:LINE_MAX_CHARS],
            "tags": [SOURCE, s.rec.kind, s.rec.purpose],
        })
    return out


__all__ = [
    "BM25", "HEADER", "KINDS", "LivedMemoryConfig", "LivedMemoryRecall", "LivedRecall", "SOURCE",
    "index_loop", "items_for", "name_tokens", "rank", "recency", "render_line", "run_index_pass",
    "sigmoid", "split_doc_id", "tokenize",
]

"""Lived-memory recall for the Juniper chat prompt (issues #3533 and #3534, "Recall rev 2" steps 1-2).

Finds the few lived memories (``orion/memory/lived_memory.py``: episode_memory, stance
crystallizations, person facts) that fit this chat turn by meaning, and renders at most
``max_lines`` of them into DURABLE MEMORY. Zero lines is a valid answer.

Pipeline, once per chat turn (PCR continuity phase):

0. **Query rewrite** (step 2, ``app/lived_rewrite.py``). A small local model reads the message
   and the last turns of the conversation and writes up to 3 short searches that name what the
   message points at ("okay, I beat it!" -> the game). Bounded by ``rewrite_timeout_sec``; on
   timeout or error the raw message is used alone.

   **Short-message gate** (issue #3613). A message below ``min_query_tokens`` content words is
   searched only when it carries signal: it names a known entity (``app/lived_entities.py``:
   referents and people from Orion's own stores, gate ``short_entity``), or the rewrite, asked
   the short-message question (``SHORT_PROMPT``: does a good reply need something from before
   this conversation?), writes searches (gate ``short_rewrite``). A message with no content word
   at all ("ok", "thanks!") is never searched and never rewritten (``short_no_content``); a
   rewrite that answers NONE, fails or times out means no search (``short_none``). Every call
   records its gate in the recall trace (``gate``).
1. **Candidates, two ways, per search** (the message plus each rewrite). Meaning: embed it
   (orion-vector-host ``/embedding``, bge-large) and take the nearest docs in the
   ``orion_lived_memory`` Chroma collection. Exact words: BM25 over the same live records, so
   names and rare words are not lost; a query word that is a name (capitalized mid-sentence, or
   a known person's name) counts ``NAME_WEIGHT`` times. All lists merge by reciprocal-rank fusion.
2. **Gate.** Every candidate must be in the current live snapshot (Postgres, not Chroma): a
   superseded, rejected or removed record is never served, whatever Chroma still holds. An
   episode memory past its stated end date stays in the snapshot, labelled with the day it ended
   (``EPISODE_ENDED_SQL``): the trip ended, the memory of it did not. An expired follow-up is gone.
3. **Link-following** (step 2, ``app/lived_links.py``). Memories that share a referent with the
   top candidates, or with a name in the message, join as extra candidates (two-hop personalized
   PageRank over ``episode_memory_referent``). They only ever reach the prompt through the rerank.
4. **Rerank.** A local cross-encoder (orion-vector-host ``/rerank``) scores each (turn, memory)
   pair; the turn side is the message followed by the rewritten searches. ``fit = sigmoid(logit)``.
   Candidates go in chunks of ``RERANK_CHUNK``, best-retrieved first; a deadline that stops the
   rerank keeps the chunks already scored (``rerank_partial`` in the trace, issue #3662).
5. **Quality bar.** Records with ``fit < min_fit`` are dropped. The bar is tuned by replay
   (``evals/run_lived_memory_replay_eval.py``).
6. **Rank** (Generative Agents, Park et al. 2023): ``score = logit + W_RECENCY * recency +
   W_IMPORTANCE * importance + W_CORRECTION * is_correction`` (logit units). Fit dominates;
   recency and importance only order memories that already fit.
7. **Re-check and render.** The chosen ids are re-read from Postgres with the serving gate
   (a correction written since the snapshot removes what it corrects), then rendered one line
   each with kind, record id and date.

**Degraded mode** (issue #3573). When the reranker is down, errors or runs past the deadline
before scoring a single chunk, the exact-word hits are ranked by BM25 alone (``lexical_fallback``) and the block says it ran
degraded. When the embedder is down, exact-word candidates are still reranked, and the block
says it ran degraded. Recall never silently skips: a degraded search that finds nothing still
puts a one-line note in the prompt. ``LivedRecall.backend`` is ``hybrid`` or ``degraded``.

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
from itertools import zip_longest
from dataclasses import dataclass, field, replace
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
    fetch_valid_during,
    index_doc,
    split_doc_id,
)

from .lived_entities import EntityIndex, load_entities
from .lived_links import SEED_DOCS, ReferentGraph, load_graph
from .lived_rewrite import Rewriter, fetch_context
from .lived_time import DEFAULT_TZ, WINDOW_PAD, TimeWindow, local_zone, parse_time_window

logger = logging.getLogger("orion-recall.lived_memory")

SOURCE = "lived_memory"
HEADER = "[Lived memories that fit this turn -- kind, record id, date]"
# Issue #3573: when the reranker or embedder is down, recall still runs on exact words and says
# so, so Orion does not read a short or empty list as "I have no memory of that".
DEGRADED_HEADER = ("[Lived memories found by exact words only -- memory search ran degraded this turn, "
                   "so these may fit loosely and a missing memory is not proof there is none -- "
                   "kind, record id, date]")
DEGRADED_EMPTY_NOTE = ("[Lived memory search ran degraded this turn (exact words only) and found no "
                       "match. That is not proof there is no memory: look again before saying you do "
                       "not remember.]")
DEGRADED_EMPTY_NOTE_RERANKED = ("[Lived memory search ran degraded this turn (meaning search unavailable) "
                                "and nothing it found fit. That is not proof there is no memory: look "
                                "again before saying you do not remember.]")

DENSE_TOP_K = 16
LEXICAL_TOP_K = 10
MAX_RERANK = 24
# Step 2 ceiling: step 1's pool (at most MAX_RERANK) plus rewrite hits and link neighbours.
MAX_RERANK_STEP2 = 32
# ``orion.dig know``: at most this many hits (strong first, then weak) per lookup.
KNOW_MAX_HITS = 8
KNOW_RECHECK_TIMEOUT_SEC = 1.5
# Step 2 (issue #3534): each rewritten search adds its own, smaller candidate lists; all lists
# are merged by reciprocal-rank fusion before the rerank.
REWRITE_DENSE_TOP_K = 8
REWRITE_LEXICAL_TOP_K = 6
RRF_K = 60.0
# Messages longer than this carry their own content words; the rewrite would add latency, not
# signal (and world-pulse-sized texts do not belong on the fast lane).
REWRITE_MAX_MESSAGE_CHARS = 1500
# Time kept back from the rewrite for the rerank, and from the rerank for the final re-check.
RERANK_RESERVE_SEC = 1.5
FINISH_RESERVE_SEC = 0.25
# Exact-word fallback (issue #3573): a record needs this many shared content words (or a shared
# name) to be served without a fit score, and at most DEGRADED_MAX_LINES go in.
DEGRADED_MIN_SHARED = 2
DEGRADED_MAX_LINES = 3
NAME_WEIGHT = 2.5
RECENCY_HALF_LIFE_DAYS = 30.0
# Ranking weights, in reranker-logit units. Cross-encoder logits for a real fit sit around -6..+2
# and their sigmoid is tiny (0.002..0.9), so blending in probability space let recency outrank
# fit (replay 2026-11-17). In logit units fit stays dominant: a month of age costs 0.5.
W_RECENCY = 1.0
W_IMPORTANCE = 0.5
W_CORRECTION = 0.5
LINE_MAX_CHARS = 320
QUERY_MAX_CHARS = 1000
# What the cross-encoder reads. Cost grows with pair length (24 long pairs: ~1.3 s on CPU);
# the opening of a message and of a memory carry what it is about.
RERANK_QUERY_CHARS = 600
RERANK_DOC_CHARS = 400
# Candidates per /rerank call (issue #3662). On the shared P4, a 27-candidate call for a long
# message took 2-4.4 s and, past the deadline, all of it was lost. Chunks keep what finished.
RERANK_CHUNK = 8
SNAPSHOT_TTL_SEC = 60.0
# Time windows (issue #3674). When the message names a time, candidates whose occurred_at falls in
# the window (padded by a day) go into the pool first, before it is cut: the message's meaning
# search read WINDOW_DENSE_DEPTH deep and filtered to the window here, and an exact-word search over
# the in-window records. (A Chroma ``where`` filter on ``occurred_ts`` was measured at ~1 s a query
# against ~20 ms for a plain 256-deep query, 2026-11-25.) At most OUT_OF_WINDOW_MAX candidates from outside the window stay in the pool,
# and at most OUT_OF_WINDOW_LINES of them reach the prompt when an in-window memory clears the bar.
# A window with nothing in it changes nothing: recall is never emptied by a time word.
WINDOW_DENSE_TOP_K = 12
WINDOW_DENSE_DEPTH = 256
WINDOW_LEXICAL_TOP_K = 8
OUT_OF_WINDOW_MAX = 4
OUT_OF_WINDOW_LINES = 2
# The best OUT_OF_WINDOW_FIRST outside candidates go into the first rerank chunk, right after the
# best in-window ones: a deadline that stops the rerank early still scores the message's own top
# hits, so a wrong window cannot leave only in-window noise (#3662 chunking).
OUT_OF_WINDOW_FIRST = 2
W_WINDOW = 1.0  # logit units, like W_RECENCY: orders memories that already fit
# Past-state questions also read facts an update has retired since (the as-of reader); at most
# this many join the rerank, chosen by shared rare words. The read runs alongside the meaning
# search, bounded by AS_OF_TIMEOUT_SEC and by the turn's deadline.
AS_OF_TOP_K = 4
AS_OF_TIMEOUT_SEC = 0.8
AS_OF_READ_LIMIT = 500
# The referent graph changes only when the distiller writes; it is refreshed in the background.
GRAPH_TTL_SEC = 600.0

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
    link: bool = False  # reached by link-following, not by the searches themselves
    degraded: bool = False  # chosen by exact words alone; the reranker could not score it


@dataclass
class LivedRecall:
    lines: List[str] = field(default_factory=list)
    chosen: List[Scored] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)
    # Every reranked candidate with its raw logit, before the bar (evals sweep the bar on this).
    reranked: List[tuple[LivedRecord, float]] = field(default_factory=list)
    # "hybrid" (meaning + exact words, reranked) or "degraded" (issue #3573); "" = no search ran.
    backend: str = ""
    degraded_reasons: List[str] = field(default_factory=list)
    lexical_hits: int = 0
    quiet_empty: bool = False  # degraded, nothing found, and nothing to note (small talk)
    # Every record that reached the rerank, in rerank order, and the best exact-word score of each
    # search hit (``orion.dig know`` shows the below-bar ones that share words as weak matches).
    candidates: List[LivedRecord] = field(default_factory=list)
    lexical_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return self.backend == "degraded"

    @property
    def rendered(self) -> str:
        if self.lines:
            return (DEGRADED_HEADER if self.degraded else HEADER) + "\n" + "\n".join(self.lines)
        if not self.degraded or self.quiet_empty:
            return ""
        reranked = not any(r.startswith("rerank") or r == "timeout" for r in self.degraded_reasons)
        return DEGRADED_EMPTY_NOTE_RERANKED if reranked else DEGRADED_EMPTY_NOTE


# Issue #3674: a fact an update has retired since, served for a past-tense question.
AS_OF_LABEL = "true then, not now"


def render_line(rec: LivedRecord, tz: str = DEFAULT_TZ) -> str:
    text = rec.text if len(rec.text) <= LINE_MAX_CHARS else rec.text[: LINE_MAX_CHARS - 1].rstrip() + "…"
    label = rec.kind if rec.kind == rec.purpose or rec.kind == KIND_PERSON_FACT else f"{rec.kind}/{rec.purpose}"
    when = rec.date
    if rec.invalid_at is not None:
        when += f" · {AS_OF_LABEL}: changed {rec.invalid_at.astimezone(local_zone(tz)).date().isoformat()}"
    elif rec.ended_at is not None:
        when += f" · {AS_OF_LABEL}: ended {rec.ended_at.astimezone(local_zone(tz)).date().isoformat()}"
    return f"- ({label} · {rec.doc_id} · {when}) {text}"


def rank(candidates: Sequence[tuple[LivedRecord, float]], *, min_fit: float, max_lines: int,
         now: datetime, in_window: Callable[[LivedRecord], bool] | None = None) -> List[Scored]:
    """(record, rerank logit) -> the records that clear the bar, best first, at most max_lines.

    ``in_window`` (issue #3674): records from the time the message names get ``W_WINDOW``, and when
    any of them clears the bar, at most ``OUT_OF_WINDOW_LINES`` records from outside it are kept."""
    kept: List[Scored] = []
    for rec, logit in candidates:
        if not math.isfinite(float(logit)):
            continue
        fit = sigmoid(float(logit))
        if fit < min_fit:
            continue
        score = (float(logit) + W_RECENCY * recency(rec.occurred_at, now) + W_IMPORTANCE * rec.importance
                 + (W_CORRECTION if rec.is_correction else 0.0)
                 + (W_WINDOW if in_window is not None and in_window(rec) else 0.0))
        kept.append(Scored(rec=rec, fit=fit, score=score))
    kept.sort(key=lambda s: (-s.score, s.rec.doc_id))
    if in_window is not None and any(in_window(s.rec) for s in kept):
        capped: List[Scored] = []
        outside = 0
        for s in kept:
            if not in_window(s.rec):
                if outside >= OUT_OF_WINDOW_LINES:
                    continue
                outside += 1
            capped.append(s)
        kept = capped
    return kept[: max(0, int(max_lines))]


Reranker = Callable[[str, List[str]], Any]  # async (query, docs) -> list[float]

BACKEND_HYBRID = "hybrid"
BACKEND_DEGRADED = "degraded"


def rerank_query(raw: str, rewrites: Sequence[str]) -> str:
    """What the cross-encoder scores each memory against: the message, then the rewritten
    searches. An implicit message ("okay, I beat it!") fits nothing on its own words; the
    searches carry what it points at."""
    if not rewrites:
        return raw[:RERANK_QUERY_CHARS]
    suffix = " | " + " ; ".join(rewrites)
    head = raw[: max(160, RERANK_QUERY_CHARS - len(suffix))]
    return (head + suffix)[: RERANK_QUERY_CHARS + 200]


def rrf(lists: Sequence[Sequence[tuple[str, float]]], k: float = RRF_K) -> Dict[str, float]:
    """Reciprocal-rank fusion (Cormack et al. 2009) of ranked id lists."""
    out: Dict[str, float] = {}
    for lst in lists:
        for i, (did, _s) in enumerate(lst):
            out[did] = out.get(did, 0.0) + 1.0 / (k + i + 1)
    return out


def lexical_fallback(recs: Sequence[LivedRecord], lexical_scores: Dict[str, float], query_tokens: set[str],
                     names: set[str], *, max_lines: int) -> List[Scored]:
    """Exact-word ranking when the reranker cannot score (issue #3573). Only records the
    exact-word search found, and only when they share at least two content words with the
    turn or a name: one shared common word is too weak to stand in for a fit score."""
    kept: List[Scored] = []
    for rec in recs:
        s = lexical_scores.get(rec.doc_id)
        if not s:
            continue
        shared = query_tokens & set(tokenize(rec.text))
        if len(shared) < DEGRADED_MIN_SHARED and not (shared & names):
            continue
        kept.append(Scored(rec=rec, fit=0.0, score=float(s), lexical=float(s), degraded=True))
    kept.sort(key=lambda x: (-x.score, x.rec.doc_id))
    return kept[: max(0, min(int(max_lines), DEGRADED_MAX_LINES))]


@dataclass(frozen=True)
class LivedMemoryConfig:
    search: SearchConfig
    rerank_url: str
    min_fit: float
    max_lines: int
    timeout_sec: float
    # A message with fewer content words than this ("hi Orion, how are you?") is searched only
    # through the short-message gate (issue #3613): it names a known entity, or the short-message
    # rewrite judges that a reply needs a memory. Blind bypasses were tried and let small talk in:
    # any rare word (13% -> 30-48% of small-talk messages injecting, replay 2026-11-17), and any
    # rewrite of the general prompt (35%). Greetings matched greeting-shaped stance messages by
    # form, not content.
    min_query_tokens: int = 0
    # Issue #3613 rollback switch: False = the plain floor (a short message is never searched).
    short_gate: bool = True
    # Step 2 (issue #3534). rewrite_timeout_sec bounds the pre-turn rewrite; past it the raw
    # message is used. link_top_k is how many referent neighbours join the rerank (0 = off).
    rewrite_timeout_sec: float = 2.5
    link_top_k: int = 2
    # Issue #3674: read time words ("last week", "in October", a date) into a window, and favour
    # memories from it. False = recall ignores time words, as before.
    time_window: bool = True
    timezone: str = DEFAULT_TZ


class LivedMemoryRecall:
    """Owns the live snapshot and runs one retrieval per call. ``pool`` is an asyncpg pool."""

    def __init__(self, pool: Any, cfg: LivedMemoryConfig, *, reranker: Optional[Reranker] = None,
                 rewriter: Optional[Rewriter] = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.pool = pool
        self.cfg = cfg
        self._reranker = reranker
        self._rewriter = rewriter
        self._clock = clock
        self._snapshot: List[LivedRecord] = []
        self._by_doc: Dict[str, LivedRecord] = {}
        self._bm25: BM25 | None = None
        self._names: List[str] = []
        self._entities: EntityIndex | None = None
        self._graph: ReferentGraph | None = None
        self._graph_loaded_at = 0.0
        self._graph_task: asyncio.Future | None = None
        self._loaded_at = 0.0
        self._lock = asyncio.Lock()

    async def snapshot(self, *, force: bool = False) -> List[LivedRecord]:
        async with self._lock:
            if force or not self._snapshot or time.monotonic() - self._loaded_at > SNAPSHOT_TTL_SEC:
                recs = await fetch_live_records(self.pool, include_ended=True)
                self._snapshot = recs
                self._by_doc = {r.doc_id: r for r in recs}
                self._bm25 = BM25([r.text for r in recs])
                self._names = sorted({r.text.split(",", 1)[0] for r in recs if r.kind == KIND_PERSON_FACT})
                try:
                    self._entities = await load_entities(self.pool)
                except Exception as exc:  # noqa: BLE001 -- keep the last good index; none means the rewrite decides
                    logger.warning("lived_memory_entities_unavailable error=%s kept_previous=%s",
                                   type(exc).__name__, self._entities is not None)
                self._loaded_at = time.monotonic()
        if self.cfg.link_top_k > 0:
            if force or self._graph_loaded_at == 0.0:
                await self._load_graph()  # once, inline; later refreshes run off the turn path
            elif (time.monotonic() - self._graph_loaded_at > GRAPH_TTL_SEC
                  and (self._graph_task is None or self._graph_task.done())):
                self._graph_task = asyncio.ensure_future(self._load_graph())
        return self._snapshot

    async def _load_graph(self) -> None:
        try:
            self._graph = await load_graph(self.pool)
        except Exception as exc:  # noqa: BLE001 -- links are an extra, never a blocker
            logger.warning("lived_memory_links_unavailable error=%s", type(exc).__name__)
            self._graph = None
        self._graph_loaded_at = time.monotonic()

    async def _rerank_http(self, client: httpx.AsyncClient, query: str, docs: List[str],
                           timeout: float | None = None) -> List[float]:
        kw = {"timeout": timeout} if timeout is not None else {}
        resp = await client.post(self.cfg.rerank_url, json={"query": query, "documents": docs}, **kw)
        resp.raise_for_status()
        scores = resp.json().get("scores")
        if not isinstance(scores, list) or len(scores) != len(docs):
            raise SearchUnavailableError("reranker returned a malformed reply")
        return [float(s) for s in scores]

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

    async def _dense(self, client: httpx.AsyncClient, query: str, k: int, *, replay: bool) -> List[tuple[str, float]]:
        vector, _model = await embed(client, self.cfg.search, query, doc_prefix="lived-memory-query")
        # Replay hides later records, so ask for more neighbours to keep k eligible.
        return await nearest(client, self.cfg.search, vector, k if not replay else k * 4)

    async def _embed_query(self, client: httpx.AsyncClient, query: str) -> List[float]:
        vector, _model = await embed(client, self.cfg.search, query, doc_prefix="lived-memory-query")
        return vector

    async def _dense_vec(self, client: httpx.AsyncClient, vector: "asyncio.Future", k: int, *,
                         replay: bool) -> List[tuple[str, float]]:
        """Nearest docs for an already-started query embedding (one embed serves the message's plain
        and time-window searches, issue #3674)."""
        vec = await asyncio.shield(vector)
        return await nearest(client, self.cfg.search, vec, k if not replay else k * 4)

    async def _rewrite(self, query: str, *, session_id: str | None, exclude_ids: Sequence[str],
                       as_of: datetime | None, short: bool = False) -> List[str]:
        context = await fetch_context(self.pool, session_id, before=as_of or self._clock(), exclude_text=query,
                                      exclude_ids=exclude_ids)
        assert self._rewriter is not None
        if short:
            return list(await self._rewriter(query, context, short=True))
        return list(await self._rewriter(query, context))

    def entity_mentions(self, text: str) -> set[str]:
        """Known entities named in ``text`` (empty when the entity index is not loaded)."""
        return self._entities.mentions(text) if self._entities is not None else set()

    def entity_name_tokens(self, text: str) -> set[str]:
        """Words of the known entity names in ``text``: names for the exact-word fallback, so a
        short message naming one ("vesper again?") can be served when the reranker is down."""
        if self._entities is None:
            return set()
        return {t for alias in self._entities.matched_names(text) for t in tokenize(alias)}

    async def recall(self, query: str, *, client: httpx.AsyncClient | None = None,
                     as_of: datetime | None = None, session_id: str | None = None,
                     exclude_ids: Sequence[str] = (), timeout_sec: float | None = None,
                     min_query_tokens: int | None = None, rewrite: bool = True,
                     now: datetime | None = None) -> LivedRecall:
        """One retrieval. ``session_id`` lets the rewrite read the last turns of this
        conversation. ``as_of`` (replay evals only) hides records Orion stored after that time,
        so a replayed message cannot be answered by the memory distilled from it.

        Bounded by ``timeout_sec`` (default ``cfg.timeout_sec``) end to end: the rewrite gets
        ``rewrite_timeout_sec``, the meaning search what is left minus ``RERANK_RESERVE_SEC``, and
        the rerank the rest. A reranker or embedder that is down or too slow drops to exact-word
        ranking (backend ``degraded``), never to nothing (issue #3573).

        ``min_query_tokens`` and ``rewrite`` are for a deliberate lookup (``know``): its words are
        already a search, so it skips the chat-turn floor and the pre-turn rewrite.

        ``now`` is when the message was asked (default ``as_of``, else the clock): time words
        ("last week", "in October") resolve against it, and recency is measured from it (#3674)."""
        started = time.perf_counter()
        deadline = started + float(timeout_sec if timeout_sec is not None else self.cfg.timeout_sec)
        timings: Dict[str, int] = {}
        out = LivedRecall(debug={"timings_ms": timings})
        query = " ".join(str(query or "").split())
        long_message = len(query) > REWRITE_MAX_MESSAGE_CHARS
        query = query[:QUERY_MAX_CHARS]
        if not query:
            out.debug["skipped"] = "empty_query"
            return out
        own_client = client is None
        client = client or httpx.AsyncClient(timeout=self.cfg.timeout_sec)
        pending: List[asyncio.Future] = []
        degraded: List[str] = []
        rewrite_info: Dict[str, Any] = {"outcome": "off"}
        out.debug["rewrite"] = rewrite_info
        try:
            qtoks = set(tokenize(query))
            floor = self.cfg.min_query_tokens if min_query_tokens is None else int(min_query_tokens)
            # Issue #3613: why this call did or did not search (persisted in the recall trace).
            gate = GATE_WORDS
            if len(qtoks) < floor:
                out.debug["content_tokens"] = len(qtoks)
                if not qtoks:
                    gate = GATE_SHORT_NO_CONTENT  # "ok", "thanks!", "haha yeah": nothing to search with
                elif not self.cfg.short_gate:
                    gate = GATE_SHORT_OFF         # the plain floor (switched back off)
                else:
                    if self._entities is None:
                        # First call only: later calls match against the index already loaded
                        # (at most SNAPSHOT_TTL_SEC old) and refresh it below, overlapping the rewrite.
                        t = time.perf_counter()
                        await self.snapshot()
                        timings["snapshot"] = int((time.perf_counter() - t) * 1000)
                    named = self.entity_mentions(query)
                    if named:
                        gate = GATE_SHORT_ENTITY
                        out.debug["entities"] = len(named)
                    elif rewrite and self._rewriter is not None and not long_message:
                        # (A message too long for the rewrite with this few content words is a
                        # pasted id or number blob: nothing to ask about.)
                        gate = GATE_SHORT_REWRITE  # decided below, once the rewrite answers
                    else:
                        gate = GATE_SHORT_NONE
                out.debug["gate"] = gate
                if gate in (GATE_SHORT_NO_CONTENT, GATE_SHORT_NONE, GATE_SHORT_OFF):
                    out.debug["skipped"] = "few_content_words"
                    return out
            else:
                out.debug["gate"] = gate
            short_rewrite = gate == GATE_SHORT_REWRITE
            ref_now = now or as_of or self._clock()
            # Issue #3674: time words become a window, read by a deterministic parser, so the window
            # never waits on (or times out with) the model. Asking the fast-lane rewrite for a time
            # line was measured and dropped: it stopped searching or ignored the line (PR report).
            window: TimeWindow | None = (parse_time_window(query, ref_now, self.cfg.timezone)
                                         if self.cfg.time_window else None)
            # The rewrite starts first: it does not need the snapshot, and it is the slow part.
            rewrite_task: asyncio.Future | None = None
            rewrite_started = time.perf_counter()
            if not rewrite:
                pass  # rewrite_info stays "off"
            elif short_rewrite:
                rewrite_task = asyncio.ensure_future(
                    self._rewrite(query, session_id=session_id, exclude_ids=exclude_ids, as_of=as_of, short=True))
                pending.append(rewrite_task)
            elif self._rewriter is not None and not long_message:
                rewrite_task = asyncio.ensure_future(
                    self._rewrite(query, session_id=session_id, exclude_ids=exclude_ids, as_of=as_of))
                pending.append(rewrite_task)
            elif self._rewriter is not None:
                rewrite_info["outcome"] = "skipped_long_message"
            rewrite_result: List[List[str]] = []

            async def rewrites() -> List[str]:
                if rewrite_result:
                    return rewrite_result[0]
                qs: List[str] = []
                if rewrite_task is not None:
                    budget = min(float(self.cfg.rewrite_timeout_sec),
                                 deadline - time.perf_counter() - RERANK_RESERVE_SEC)
                    try:
                        qs = list(await asyncio.wait_for(asyncio.shield(rewrite_task), timeout=max(0.001, budget)))
                        rewrite_info["outcome"] = "ok" if qs else "none"
                    except asyncio.TimeoutError:
                        rewrite_task.cancel()
                        rewrite_info["outcome"] = "timeout"
                    except Exception as exc:  # noqa: BLE001 -- no rewrite means the raw message, never a failure
                        rewrite_info["outcome"] = f"error:{type(exc).__name__}"
                    timings["rewrite"] = int((time.perf_counter() - rewrite_started) * 1000)
                # The searches are Orion's own query text: the recall trace the spec asks for.
                rewrite_info["queries"] = list(qs)
                rewrite_result.append(qs)
                return qs

            t = time.perf_counter()
            await self.snapshot()
            timings["snapshot"] = int((time.perf_counter() - t) * 1000)

            def allowed(rec: LivedRecord) -> bool:
                return as_of is None or rec.created_at is None or rec.created_at < as_of

            replay = as_of is not None
            if short_rewrite:
                # The short-message judgment decides whether to search at all, so nothing else
                # (not even the embedder call) runs until it answers. NONE, a failure or a
                # timeout: no search.
                queries = await rewrites()
                if not queries:
                    out.debug["gate"] = GATE_SHORT_NONE
                    out.debug["skipped"] = "few_content_words"
                    return out
                # How many known entities the searches name (the trace keeps the count; #3613 (b)).
                out.debug["rewrite_entities"] = len(self.entity_mentions(" ; ".join(queries)))
            raw_vec = asyncio.ensure_future(self._embed_query(client, query))
            pending.append(raw_vec)
            raw_dense = asyncio.ensure_future(self._dense_vec(client, raw_vec, DENSE_TOP_K, replay=replay))
            pending.append(raw_dense)
            window_dense: asyncio.Future | None = None
            if window is not None:  # search the window alongside the plain search
                window_dense = asyncio.ensure_future(self._dense_vec(client, raw_vec, WINDOW_DENSE_DEPTH,
                                                                     replay=replay))
                pending.append(window_dense)
            queries = await rewrites()
            as_of_task: asyncio.Future | None = None
            if window is not None and window.past_state:  # runs alongside the meaning searches
                as_of_task = asyncio.ensure_future(self._as_of_read(window))
                pending.append(as_of_task)
            t = time.perf_counter()  # dense time past the rewrite wait (the raw search overlaps it)
            dense_tasks = [raw_dense] + [
                asyncio.ensure_future(self._dense(client, q, REWRITE_DENSE_TOP_K, replay=replay)) for q in queries]
            pending.extend(dense_tasks[1:])
            dense_budget = deadline - time.perf_counter() - RERANK_RESERVE_SEC
            wait_on = dense_tasks + ([window_dense] if window_dense is not None else [])
            done, not_done = await asyncio.wait(wait_on, timeout=max(0.001, dense_budget))
            for task in not_done:
                task.cancel()
            dense_lists: List[List[tuple[str, float]]] = []
            dense_failed = 0
            for i, task in enumerate(dense_tasks):
                res: Any = []
                if task in not_done:
                    dense_failed += 1
                    out.debug.setdefault("dense_error", "timeout")
                elif task.exception() is not None:
                    exc = task.exception()
                    dense_failed += 1
                    out.debug.setdefault("dense_error", f"{type(exc).__name__}: {exc}"[:200])
                else:
                    res = task.result()
                k = DENSE_TOP_K if i == 0 else REWRITE_DENSE_TOP_K
                kept = []
                for did, sim in res:
                    rec = self._by_doc.get(did)
                    if rec is not None and allowed(rec) and len(kept) < k:
                        kept.append((did, sim))
                dense_lists.append(kept)
                if i == 0:
                    out.debug["dense_not_live"] = sum(1 for d, _ in res if d not in self._by_doc)
            timings["dense"] = int((time.perf_counter() - t) * 1000)
            if dense_failed == len(dense_tasks):
                degraded.append("embedder_timeout" if not_done else "embedder_unavailable")

            # Issue #3674: the window's own searches (meaning, read deep and filtered here; exact
            # words, over the in-window records), and for a past-tense question the facts retired since.
            in_window: Callable[[LivedRecord], bool] | None = None
            window_lists: List[List[tuple[str, float]]] = []
            as_of_recs: Dict[str, LivedRecord] = {}
            win_info: Dict[str, Any] = {}
            if window is not None:
                win = window
                win_info.update(win.trace(self.cfg.timezone))
                out.debug["window"] = win_info

                def _in_win(rec: LivedRecord) -> bool:
                    return rec.doc_id in as_of_recs or win.contains(rec.occurred_at)

                in_window = _in_win

                wd: List[tuple[str, float]] = []
                if window_dense is not None and window_dense in done and window_dense.exception() is None:
                    for did, sim in window_dense.result():
                        rec = self._by_doc.get(did)
                        if rec is not None and allowed(rec) and in_window(rec) and len(wd) < WINDOW_DENSE_TOP_K:
                            wd.append((did, sim))
                elif window_dense is not None:
                    exc = window_dense.exception() if window_dense in done else None
                    win_info["dense_error"] = type(exc).__name__ if exc is not None else "timeout"

                def allowed_in(rec: LivedRecord) -> bool:
                    return allowed(rec) and win.contains(rec.occurred_at)

                window_lists = [wd, self.lexical(query, WINDOW_LEXICAL_TOP_K, allowed=allowed_in)]
                window_lists += [self.lexical(q, REWRITE_LEXICAL_TOP_K, allowed=allowed_in) for q in queries]
                if as_of_task is not None:
                    budget = min(AS_OF_TIMEOUT_SEC, deadline - time.perf_counter() - RERANK_RESERVE_SEC)
                    as_of_recs = await self._as_of_candidates(as_of_task, max(0.001, budget), query, queries,
                                                              allowed, win_info, as_of=as_of)

            lexical_lists = [self.lexical(query, allowed=allowed)]
            lexical_lists += [self.lexical(q, REWRITE_LEXICAL_TOP_K, allowed=allowed) for q in queries]
            lexical_best: Dict[str, float] = {}
            for lst in lexical_lists + window_lists[1:]:
                for did, s in lst:
                    lexical_best[did] = max(lexical_best.get(did, 0.0), s)
            dense_best: Dict[str, float] = {}
            for lst in dense_lists + window_lists[:1]:
                for did, s in lst:
                    dense_best[did] = max(dense_best.get(did, 0.0), s)

            # Step 1's pool comes first and whole (the message's own meaning hits, then its exact-
            # word hits): the rewrite and the links only add candidates, never push one out. The
            # rewrite's hits follow in reciprocal-rank-fusion order (Cormack et al. 2009).
            fused = rrf(dense_lists + lexical_lists)
            base = list(dict.fromkeys([d for d, _ in dense_lists[0]] + [d for d, _ in lexical_lists[0]]))
            base = sorted(base, key=lambda d: (-dense_best.get(d, 0.0), -lexical_best.get(d, 0.0)))[:MAX_RERANK]
            # Rerank order (#3662): the message's own meaning and exact-word lists take turns, so the
            # first chunk holds the top hits of both. An exact-word-only hit (a name, a rare word)
            # has no meaning score and would otherwise sit in the last chunk, past a short deadline.
            # Link seeds still come from ``ordered`` (meaning first), so only the chunk order changes.
            kept_base = set(base)
            turns = [d for pair in zip_longest([d for d, _ in dense_lists[0]], [d for d, _ in lexical_lists[0]])
                     for d in pair if d is not None and d in kept_base]
            rerank_base = list(dict.fromkeys(turns + base))
            extra = sorted((d for d in fused if d not in set(base)), key=lambda d: (-fused[d], d))
            ordered = base + extra
            by_doc = self._by_doc
            if as_of_recs:
                by_doc = dict(self._by_doc)
                by_doc.update(as_of_recs)
            if in_window is not None:
                # Issue #3674: the window applies before the pool is cut. In-window candidates come
                # first (the window's own meaning and exact-word hits taking turns, then the retired
                # facts, then every other in-window hit by fused rank); then at most OUT_OF_WINDOW_MAX
                # of the best outside the window. Nothing in the window: the pool stays as it was.
                fused_all = rrf(dense_lists + lexical_lists + window_lists)
                win_turns = [d for pair in zip_longest([d for d, _ in window_lists[0]], [d for d, _ in window_lists[1]])
                             for d in pair if d is not None]
                rest_in = sorted((d for d in fused_all if in_window(by_doc[d])), key=lambda d: (-fused_all[d], d))
                in_ids = list(dict.fromkeys(win_turns + list(as_of_recs) + rest_in))
                win_info["in_window"] = len(in_ids)
                win_info["as_of_rows"] = len(as_of_recs)
                if in_ids:
                    out_ids = [d for d in dict.fromkeys(rerank_base + extra) if d not in set(in_ids)]
                    win_info["out_of_window_kept"] = min(len(out_ids), OUT_OF_WINDOW_MAX)
                    out_ids = out_ids[:OUT_OF_WINDOW_MAX]
                    in_ids = in_ids[: max(0, MAX_RERANK_STEP2 - len(out_ids) - int(self.cfg.link_top_k))]
                    first = RERANK_CHUNK - OUT_OF_WINDOW_FIRST
                    base = (in_ids[:first] + out_ids[:OUT_OF_WINDOW_FIRST] + in_ids[first:]
                            + out_ids[OUT_OF_WINDOW_FIRST:])
                    extra = []
                    rerank_base = list(base)
                    ordered = list(base)
                    fused = fused_all
                else:
                    win_info["empty"] = True
                    in_window = None  # nothing to favour: ranking stays as without a window

            # Link-following: neighbours of the top candidates and of names in the message or
            # in the rewritten searches.
            links: List[tuple[str, float]] = []
            link_k = max(0, min(int(self.cfg.link_top_k), MAX_RERANK // 2))
            if self._graph is not None and link_k > 0 and len(self._graph):
                t = time.perf_counter()
                seed_refs = set(self._graph.mentions(query))
                for q in queries:
                    seed_refs |= self._graph.mentions(q)
                links = self._graph.spread(
                    ordered[:SEED_DOCS], seed_refs, exclude=set(ordered[:MAX_RERANK_STEP2]),
                    allowed=lambda d: d in self._by_doc and allowed(self._by_doc[d]), top_k=link_k)
                timings["links"] = int((time.perf_counter() - t) * 1000)
                out.debug["links"] = {"seed_names": len(seed_refs), "added": len(links)}
            link_ids = {d for d, _ in links}
            out.debug["candidates"] = {"dense": sum(len(x) for x in dense_lists),
                                       "lexical": sum(len(x) for x in lexical_lists),
                                       "live_pool": len(fused), "links": len(links),
                                       "dense_not_live": out.debug.pop("dense_not_live", 0)}
            out.lexical_hits = len(lexical_best)
            out.lexical_scores = dict(lexical_best)

            def finish_backend() -> None:
                out.backend = BACKEND_DEGRADED if degraded else BACKEND_HYBRID
                out.debug["backend"] = out.backend
                if degraded:
                    out.debug["degraded"] = list(degraded)
                    out.degraded_reasons = list(degraded)
                    # Small talk the rewrite called NONE, with no word in common with any memory,
                    # gets no note: there is nothing it could be denying.
                    out.quiet_empty = rewrite_info.get("outcome") == "none" and not lexical_best

            if not ordered and not links:
                finish_backend()
                return out
            cand_ids = rerank_base + extra[: max(0, MAX_RERANK_STEP2 - len(base) - len(links))] + [d for d, _ in links]
            recs = [by_doc[d] for d in cand_ids]
            out.candidates = list(recs)
            # Every id that reached the rerank, not only the injected ones (#3628): "the target was a
            # candidate but fell below the bar" is then visible in recall_telemetry.lived_trace.
            out.debug["candidate_ids"] = [r.doc_id for r in recs]
            names = name_tokens(query, self._names) | self.entity_name_tokens(query)
            all_tokens = set(qtoks)
            for q in queries:
                all_tokens |= set(tokenize(q))
                names |= name_tokens(q, self._names)

            t = time.perf_counter()
            docs = [r.text[:RERANK_DOC_CHARS] for r in recs]
            rq = rerank_query(query, queries)
            logits: List[float | None] = [None] * len(docs)
            budget = deadline - time.perf_counter() - FINISH_RESERVE_SEC

            async def score(idx: List[int]) -> None:
                part = [docs[i] for i in idx]
                if self._reranker is not None:
                    got = list(await self._reranker(rq, part))
                else:
                    got = await self._rerank_http(client, rq, part,
                                                  timeout=max(0.001, deadline - time.perf_counter() - FINISH_RESERVE_SEC))
                if len(got) != len(part):
                    raise SearchUnavailableError("reranker returned a malformed reply")
                for i, x in zip(idx, got):
                    logits[i] = float(x)

            # Issue #3662: the pool goes in chunks of RERANK_CHUNK, best-retrieved first, one call
            # at a time (the vector host refuses a third request in flight). When the deadline or an
            # error stops it, the chunks already scored are kept: the top candidates still get a
            # fit score, and only a call that scored nothing falls back to exact words.
            async def all_chunks() -> None:
                for start in range(0, len(docs), RERANK_CHUNK):
                    await score(list(range(start, min(start + RERANK_CHUNK, len(docs)))))

            stopped: str | None = None
            try:
                if budget <= 0:
                    raise asyncio.TimeoutError
                await asyncio.wait_for(all_chunks(), timeout=budget)
            except asyncio.TimeoutError:
                stopped = "timeout"
            except Exception as exc:  # noqa: BLE001 -- a down reranker degrades, never skips (#3573)
                stopped = "error"
                out.debug["rerank_error"] = f"{type(exc).__name__}: {exc}"[:200]
            timings["rerank"] = int((time.perf_counter() - t) * 1000)
            scored_idx = [i for i, x in enumerate(logits) if x is not None]
            if stopped and not scored_idx:
                degraded.append("rerank_timeout" if stopped == "timeout" else "rerank_unavailable")
            elif stopped:
                out.debug["rerank_partial"] = {"scored": len(scored_idx), "of": len(docs), "reason": stopped}

            if scored_idx:
                # Unscored candidates (a partial rerank) are left out, like link neighbours below:
                # nothing reaches the prompt without a fit score unless the whole rerank failed.
                out.reranked = [(recs[i], float(logits[i])) for i in scored_idx]
                chosen = rank(out.reranked, min_fit=self.cfg.min_fit, max_lines=self.cfg.max_lines,
                              now=ref_now, in_window=in_window)
                out.debug["reranked"] = [
                    {"id": r.doc_id, "fit": round(sigmoid(l), 4), "dense": round(dense_best.get(r.doc_id, 0.0), 4),
                     "lexical": round(lexical_best.get(r.doc_id, 0.0), 3), "link": r.doc_id in link_ids}
                    for r, l in sorted(out.reranked, key=lambda x: -x[1])[:8]
                ]
                if not chosen and stopped:
                    # The scored chunks hold nothing that fits and the rest was never scored, so
                    # this was not a full search: serve the unscored exact-word hits, degraded, and
                    # say so. Never a silent "below the bar" (#3663: a part-search is no denial).
                    degraded.append("rerank_partial")
                    chosen = lexical_fallback([recs[i] for i, x in enumerate(logits)
                                               if x is None and recs[i].doc_id not in link_ids],
                                              lexical_best, all_tokens, names, max_lines=self.cfg.max_lines)
            else:
                # Never inject a link neighbour without a rerank: exact-word hits only.
                chosen = lexical_fallback([r for r in recs if r.doc_id not in link_ids], lexical_best,
                                          all_tokens, names, max_lines=self.cfg.max_lines)
            finish_backend()
            if not chosen:
                return out

            chosen = await self._recheck(chosen, out, timings, window=window)
            for s in chosen:
                if as_of is not None and s.rec.ended_at is not None and s.rec.ended_at >= as_of:
                    s.rec = replace(s.rec, ended_at=None)  # a replay: it had not ended yet then
                s.dense, s.lexical = dense_best.get(s.rec.doc_id), lexical_best.get(s.rec.doc_id)
                s.link = s.rec.doc_id in link_ids
            out.chosen = chosen
            out.lines = [render_line(s.rec, self.cfg.timezone) for s in chosen]
            return out
        except (SearchUnavailableError, httpx.HTTPError, ValueError) as exc:
            out.debug["error"] = f"{type(exc).__name__}: {exc}"[:300]
            return out
        finally:
            for task in pending:
                if not task.done():
                    task.cancel()
            if pending:
                # Collect every child (cancelled or failed) so none logs "exception never retrieved".
                await asyncio.gather(*pending, return_exceptions=True)
            timings["total"] = int((time.perf_counter() - started) * 1000)
            if own_client:
                await client.aclose()

    async def _as_of_read(self, window: TimeWindow) -> List[LivedRecord]:
        return await fetch_valid_during(self.pool, window.start - WINDOW_PAD, window.end + WINDOW_PAD,
                                        retired_only=True, limit=AS_OF_READ_LIMIT)

    async def _as_of_candidates(self, task: "asyncio.Future", budget: float, query: str, queries: Sequence[str],
                                allowed: Callable[[LivedRecord], bool], info: Dict[str, Any], *,
                                as_of: datetime | None = None) -> Dict[str, LivedRecord]:
        """Issue #3674: facts true during the window that an update has retired since (the as-of
        reader, ``fetch_valid_during``, started alongside the meaning search). They are not in the
        live snapshot or the index, so they are ranked here by the rare words they share with the
        turn (BM25 idf), at most AS_OF_TOP_K. An extra, never a blocker: a failed or late read adds
        nothing. In a replay (``as_of``), a row retired after then was still current then: skipped."""
        try:
            rows = await asyncio.wait_for(asyncio.shield(task), timeout=budget)
        except Exception as exc:  # noqa: BLE001 -- the past-state read is an extra
            info["as_of_error"] = type(exc).__name__ or "timeout"
            return {}
        idf = self._bm25.idf if self._bm25 is not None else {}
        toks = set(tokenize(query))
        for q in queries:
            toks |= set(tokenize(q))
        scored = []
        for rec in rows:
            if rec.doc_id in self._by_doc or rec.invalid_at is None or not allowed(rec):
                continue
            if as_of is not None and rec.invalid_at >= as_of:
                continue
            s = sum(idf.get(t, 1.0) for t in toks & set(tokenize(rec.text)))
            if s > 0:
                scored.append((s, rec))
        scored.sort(key=lambda x: (-x[0], x[1].doc_id))
        info["as_of_read"] = len(rows)
        return {rec.doc_id: rec for _s, rec in scored[:AS_OF_TOP_K]}

    async def _recheck(self, chosen: List[Scored], out: LivedRecall, timings: Dict[str, int],
                       window: TimeWindow | None = None) -> List[Scored]:
        """Re-read the chosen ids with the serving gate: a correction written since the snapshot
        removes what it corrects. A fact served as true then (#3674) is re-read with the as-of
        reader over the same window instead."""
        t = time.perf_counter()
        ids_by_kind: Dict[str, List[str]] = {}
        as_of_ids: List[str] = []
        for s in chosen:
            if s.rec.invalid_at is not None:
                as_of_ids.append(s.rec.record_id)
            else:
                ids_by_kind.setdefault(s.rec.kind, []).append(s.rec.record_id)
        live_now = {r.doc_id: r for r in await fetch_live_records(self.pool, kinds=list(ids_by_kind),
                                                                   ids_by_kind=ids_by_kind,
                                                                   include_ended=True)} if ids_by_kind else {}
        still_live = set(live_now)
        reread: Dict[str, LivedRecord] = {}
        if as_of_ids and window is not None:
            try:
                rows = await asyncio.wait_for(fetch_valid_during(
                    self.pool, window.start - WINDOW_PAD, window.end + WINDOW_PAD, ids=as_of_ids),
                    timeout=AS_OF_TIMEOUT_SEC)
            except Exception as exc:  # noqa: BLE001 -- an unverified row is never served
                out.debug["as_of_recheck_error"] = type(exc).__name__ or "timeout"
                rows = []
            reread = {r.doc_id: r for r in rows}
            still_live |= set(reread)
        timings["recheck"] = int((time.perf_counter() - t) * 1000)
        dropped = [s.rec.doc_id for s in chosen if s.rec.doc_id not in still_live]
        if dropped:
            out.debug["recheck_dropped"] = dropped
        kept = [s for s in chosen if s.rec.doc_id in still_live]
        for s in kept:
            fresh = reread.get(s.rec.doc_id)
            if fresh is not None and fresh.invalid_at is None:
                s.rec = fresh  # its retirement was undone since: it is true now, so no "true then" label
            now_rec = live_now.get(s.rec.doc_id)
            if now_rec is not None and now_rec.ended_at != s.rec.ended_at:
                s.rec = now_rec  # its end date passed since the snapshot: label it
        return kept

    async def degraded_now(self, query: str, *, reason: str, timeout_sec: float = 0.3) -> LivedRecall:
        """Exact-word recall from the in-memory snapshot, for when the full recall ran out of time
        (issue #3573). No embedder, no reranker, no rewrite; only the re-check touches Postgres,
        bounded by ``timeout_sec``. If the re-check cannot finish, nothing is served but the
        degraded note still is."""
        started = time.perf_counter()
        timings: Dict[str, int] = {}
        out = LivedRecall(debug={"timings_ms": timings, "backend": BACKEND_DEGRADED, "degraded": [reason]},
                          backend=BACKEND_DEGRADED, degraded_reasons=[reason])
        query = " ".join(str(query or "").split())[:QUERY_MAX_CHARS]
        qtoks = set(tokenize(query))
        # Same gate as recall(), minus the rewrite (no time left for it): the trace says which.
        if len(qtoks) >= self.cfg.min_query_tokens:
            gate = GATE_WORDS
        elif not qtoks:
            gate = GATE_SHORT_NO_CONTENT
        elif not self.cfg.short_gate:
            gate = GATE_SHORT_OFF
        elif self.entity_mentions(query):
            gate = GATE_SHORT_ENTITY
        else:
            gate = GATE_SHORT_NONE
        out.debug["gate"] = gate
        if not query or self._bm25 is None or gate not in (GATE_WORDS, GATE_SHORT_ENTITY):
            out.backend = ""  # nothing a search could have run on: no note
            out.debug = {"timings_ms": timings, "skipped": "few_content_words_or_no_snapshot", "gate": gate}
            return out
        lexical_best = dict(self.lexical(query))
        out.lexical_hits = len(lexical_best)
        recs = [self._by_doc[d] for d in lexical_best if d in self._by_doc]
        chosen = lexical_fallback(recs, lexical_best, qtoks,
                                  name_tokens(query, self._names) | self.entity_name_tokens(query),
                                  max_lines=self.cfg.max_lines)
        if chosen:
            try:
                chosen = await asyncio.wait_for(self._recheck(chosen, out, timings), timeout=timeout_sec)
            except Exception as exc:  # noqa: BLE001 -- unverified rows are never served
                out.debug["recheck_error"] = type(exc).__name__
                chosen = []
        out.chosen = chosen
        out.lines = [render_line(s.rec) for s in chosen]
        timings["total"] = int((time.perf_counter() - started) * 1000)
        return out

    async def know(self, words: str, *, max_hits: int = KNOW_MAX_HITS,
                   timeout_sec: float | None = None) -> "KnowResult":
        """A deliberate lookup (``python -P -m orion.dig know <words>``, issue #3535): the same
        retrieval a chat turn gets, minus the chat-turn floor and the rewrite (the words already
        are the search). Strong hits cleared recall's bar (the first ``max_lines`` of them are what
        a chat turn would inject). Weak hits are below the bar but share at least one word with the search, so a partial match is shown, never hidden
        behind "nothing found". Every hit passes the serving gate (re-checked against Postgres)."""
        max_hits = max(1, int(max_hits))
        out = await self.recall(words, min_query_tokens=1, rewrite=False, timeout_sec=timeout_sec)
        strong = list(out.chosen)[:max_hits]
        taken = {s.rec.doc_id for s in strong}
        lex = out.lexical_scores
        more_strong: List[Scored] = []  # above the bar, past the chat turn's line cap
        pool: List[Scored] = []
        if out.reranked:
            above = rank(out.reranked, min_fit=self.cfg.min_fit, max_lines=max_hits, now=self._clock())
            more_strong = [s for s in above if s.rec.doc_id not in taken]
            taken |= {s.rec.doc_id for s in above}
            for rec, logit in sorted(out.reranked, key=lambda x: (-x[1], x[0].doc_id)):
                if rec.doc_id in taken or not lex.get(rec.doc_id) or not math.isfinite(float(logit)):
                    continue
                pool.append(Scored(rec=rec, fit=sigmoid(float(logit)), score=float(logit),
                                   lexical=lex.get(rec.doc_id)))
            if out.debug.get("rerank_partial"):  # #3662: candidates the deadline left unscored
                scored = {rec.doc_id for rec, _l in out.reranked}
                for rec in sorted(out.candidates, key=lambda r: (-lex.get(r.doc_id, 0.0), r.doc_id)):
                    if rec.doc_id in taken or rec.doc_id in scored or not lex.get(rec.doc_id):
                        continue
                    pool.append(Scored(rec=rec, fit=0.0, score=lex[rec.doc_id], lexical=lex[rec.doc_id],
                                       degraded=True))
        else:  # no rerank ran (degraded): exact-word hits the fallback did not choose
            for rec in sorted(out.candidates, key=lambda r: (-lex.get(r.doc_id, 0.0), r.doc_id)):
                if rec.doc_id in taken or not lex.get(rec.doc_id):
                    continue
                pool.append(Scored(rec=rec, fit=0.0, score=lex[rec.doc_id], lexical=lex[rec.doc_id],
                                   degraded=True))
        more_strong = more_strong[: max_hits - len(strong)]
        weak = pool[: max(0, max_hits - len(strong) - len(more_strong))]
        extra = more_strong + weak
        unverified = 0
        if extra:
            timings = out.debug.setdefault("timings_ms", {})
            try:
                checked = await asyncio.wait_for(self._recheck(extra, out, timings), timeout=KNOW_RECHECK_TIMEOUT_SEC)
                live = {s.rec.doc_id for s in checked}
            except Exception as exc:  # noqa: BLE001 -- an unverified row is never shown, but is counted
                out.debug["know_recheck_error"] = type(exc).__name__ or "error"
                live = set()
                unverified = len(extra)
            more_strong = [s for s in more_strong if s.rec.doc_id in live]
            weak = [s for s in weak if s.rec.doc_id in live]
        strong = strong + more_strong
        status = status_of(out.debug, len(strong), backend=out.backend)
        if unverified and not strong and not weak:
            status = STATUS_UNVERIFIED  # matches existed but could not be checked: never "below_bar"
        return KnowResult(strong=strong, weak=weak, recall=out, status=status, unverified=unverified)


# Why a lived-recall call injected what it did (#3628), persisted as recall_telemetry.lived_status.
# NULL in that column means lived recall did not run on that call (not a chat-turn continuity call).
STATUS_OK = "ok"                    # full search ran and at least one memory cleared the bar
STATUS_BELOW_BAR = "below_bar"      # full search ran; nothing cleared the bar (or nothing matched)
STATUS_SKIPPED_SHORT = "skipped_short"  # too few content words to search with
STATUS_DEGRADED = "degraded"        # exact words only (embedder/reranker down or slow); count says if it served
STATUS_TIMEOUT = "timeout"          # ran out of time and the exact-word fallback found nothing
STATUS_ERROR = "error"              # failed, and nothing was served
STATUS_OFF = "off"                  # the reader is switched off or not started
# `know` only: matches were found but the serving-gate re-check failed, so none could be shown.
STATUS_UNVERIFIED = "unverified"
# Issue #3613: why a chat-turn call did or did not search, persisted as ``lived_trace.gate``.
# lived_status keeps the outcome (ok, below_bar, skipped_short, degraded, ...); the gate says how a
# short message got there. A short message that searched reads e.g. status ok + gate short_entity.
GATE_WORDS = "words"                      # at or above the content-word floor: always searched
GATE_SHORT_ENTITY = "short_entity"        # short, but names a known entity: searched
GATE_SHORT_REWRITE = "short_rewrite"      # short; the short-message rewrite wrote searches: searched
GATE_SHORT_NONE = "short_none"            # short; no entity, rewrite said NONE (or failed): not searched
GATE_SHORT_NO_CONTENT = "short_no_content"  # no content word at all: not searched, not rewritten
GATE_SHORT_OFF = "short_off"              # short, and the gate is switched off (plain floor): not searched
GATES = (GATE_WORDS, GATE_SHORT_ENTITY, GATE_SHORT_REWRITE, GATE_SHORT_NONE, GATE_SHORT_NO_CONTENT,
         GATE_SHORT_OFF)

STATUSES = (STATUS_OK, STATUS_BELOW_BAR, STATUS_SKIPPED_SHORT, STATUS_DEGRADED, STATUS_TIMEOUT,
            STATUS_ERROR, STATUS_OFF, STATUS_UNVERIFIED)


def status_of(debug: Dict[str, Any] | None, served: int, *, backend: str | None = None) -> str:
    """One status per lived-recall call, from its debug dict and how many memories it served.

    ``debug`` is a ``LivedRecall.debug`` (or the worker's copy of it, which adds ``outer`` when
    the worker's own wait timed out or raised). Pure: the same input always gives the same status."""
    d = debug or {}
    outer = str(d.get("outer") or "")
    skipped = str(d.get("skipped") or "")
    if skipped == "disabled":
        return STATUS_OFF
    if outer == "timeout" or skipped == "timeout":
        return STATUS_DEGRADED if served else STATUS_TIMEOUT
    if outer.startswith("error") or skipped.startswith("error"):
        return STATUS_DEGRADED if served else STATUS_ERROR
    if skipped in ("few_content_words", "empty_query"):
        return STATUS_SKIPPED_SHORT
    if d.get("error") and not served:
        return STATUS_ERROR
    if (backend or d.get("backend")) == BACKEND_DEGRADED:
        return STATUS_DEGRADED
    return STATUS_OK if served else STATUS_BELOW_BAR


TRACE_MAX_IDS = 32


def trace_of(debug: Dict[str, Any] | None) -> Dict[str, Any]:
    """The persisted recall trace (#3628): rewritten queries, every id that reached the rerank, the
    injected ids, and the backend. Ids and Orion's own search text only, never memory text."""
    d = debug or {}
    rewrite = d.get("rewrite") or {}
    out: Dict[str, Any] = {
        "backend": d.get("backend") or None,
        "queries": [str(q)[:200] for q in (rewrite.get("queries") or [])][:5],
        "rewrite": rewrite.get("outcome"),
        "candidate_ids": [str(x) for x in (d.get("candidate_ids") or [])][:TRACE_MAX_IDS],
        "injected_ids": [str(x.get("id")) for x in (d.get("injected") or []) if isinstance(x, dict)][:TRACE_MAX_IDS],
    }
    if d.get("gate"):  # #3613: why this chat-turn call did or did not search
        out["gate"] = str(d["gate"])
    if d.get("rewrite_entities") is not None:  # known entities the short-message searches named
        out["rewrite_entities"] = int(d["rewrite_entities"])
    if d.get("weak_ids"):
        out["weak_ids"] = [str(x) for x in d["weak_ids"]][:TRACE_MAX_IDS]
    if d.get("recheck_dropped"):  # the serving gate removed these after the snapshot
        out["recheck_dropped"] = [str(x) for x in d["recheck_dropped"]][:TRACE_MAX_IDS]
    if d.get("rerank_partial"):  # #3662: the deadline stopped the rerank after some chunks
        out["rerank_partial"] = dict(d["rerank_partial"])
    if d.get("window"):  # #3674: the time window recall read (local dates and shape, no message words)
        out["window"] = {k: v for k, v in dict(d["window"]).items()
                         if k in ("start", "end", "kind", "past_state", "in_window", "as_of_rows",
                                  "out_of_window_kept", "empty", "dense_error", "as_of_error")}
    for key in ("degraded", "skipped", "outer", "error", "know_recheck_error"):
        if d.get(key):
            out[key] = d[key] if key == "degraded" else str(d[key])[:200]
    return out


@dataclass
class KnowResult:
    strong: List[Scored]
    weak: List[Scored]
    recall: LivedRecall
    status: str
    unverified: int = 0  # hits found but not shown: the serving-gate re-check failed


@dataclass(frozen=True)
class LivedIndexPass:
    indexed: int
    pending: int
    removed: int = 0


async def run_index_pass(pool: Any, cfg: SearchConfig, *, client: httpx.AsyncClient, bus: Any, source: Any,
                         batch: int) -> LivedIndexPass:
    """Make the index mirror what may be served: embed and upsert live records Chroma lacks or
    holds stale text for (up to ``batch``), and delete docs whose record is no longer servable
    (superseded, rejected, removed, an expired follow-up). Postgres keeps every row; only the index
    entry, and the private text copied into it, goes. Serving re-gates every hit anyway. A memory
    past its stated end date is still served (labelled), so it stays indexed."""
    recs = await fetch_live_records(pool, include_ended=True)
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
            "tags": [SOURCE, s.rec.kind, s.rec.purpose] + (["degraded"] if s.degraded else [])
                    + (["link"] if s.link else []) + (["ended"] if s.rec.ended_at is not None else [])
                    + (["as_of"] if s.rec.invalid_at is not None else []),
        })
    return out


__all__ = [
    "BACKEND_DEGRADED", "BACKEND_HYBRID", "BM25", "DEGRADED_EMPTY_NOTE", "DEGRADED_EMPTY_NOTE_RERANKED",
    "GATES", "KNOW_MAX_HITS", "KnowResult", "STATUSES",
    "DEGRADED_HEADER", "HEADER", "KINDS", "LivedMemoryConfig", "LivedMemoryRecall", "LivedRecall", "SOURCE",
    "AS_OF_LABEL", "index_loop", "items_for", "lexical_fallback", "name_tokens", "rank", "recency", "render_line",
    "rerank_query", "rrf", "run_index_pass",
    "sigmoid", "split_doc_id", "status_of", "tokenize", "trace_of",
]

"""Mesh-edge leak check (issue #3409).

Core pages and lived memories are free inside Orion's mesh (chat, Prime, curiosity, ...). They
must not be pasted into public output: GitHub writes made with Orion's account (issues, PRs,
comments) and email to anyone other than Juniper. This module is the one deterministic check
every such writer calls before it sends.

What is guarded (loaded from Postgres, read-only):

- every sentence of every ``page:juniper``, ``page:origin``, ``page:lately`` and ``page:anchor`` version in
  ``self_concept_history`` (the core pages, old versions included);
- every sentence of every lived memory recall can inject into the chat prompt, and of the ones it
  no longer may (superseded, rejected, expired, removed): episode_memory, stance crystallizations,
  person facts. Retired text is still private.

A sentence counts only when it has at least ``MIN_WORDS`` words: "she/her" or "7" in a public PR
is not a leak of anything. Two matches run on the same text (issue #3554):

- **Verbatim.** A whole guarded sentence, after normalization (case, quotes, punctuation and
  whitespace folded), so shell quoting or a Markdown wrap does not hide a quoted sentence.
- **Shingle.** Any run of ``SHINGLE_WORDS`` (default 8) consecutive normalized words shared with
  one guarded sentence. This catches near-verbatim edits: a word swapped at one end, a clause
  cut, a sentence spliced into a longer one. Articles (a/an/the) and which/that are folded to one
  word each first; nothing is stemmed. It does not catch real rewording (follow-up #3699;
  the embedding idea from #3554). ``ORION_MESH_EDGE_SHINGLE_WORDS`` sets the run length; 0 turns it
  off. The threshold was tuned on made-up paraphrases and on real public PR and issue bodies
  (``orion/memory/evals/run_mesh_edge_shingle_eval.py``).

``MeshEdgeGuard.redact`` blanks each sentence holding a match instead of refusing, for a writer with no
author in the loop to rewrite (the Claude consult send, issue #3541).

Failure mode is closed: a writer that cannot load the guarded text must refuse, not send
(``load_guard`` raises ``MeshEdgeUnavailable``).
"""
from __future__ import annotations

import html
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

MIN_WORDS = 6
CACHE_TTL_SEC = 300.0
# Shingle run length (issue #3554), tuned by orion/memory/evals/run_mesh_edge_shingle_eval.py on
# 2026-11-18: on 866 real public PR/issue bodies and comments, runs of 7-12 words add no refusal
# beyond the verbatim check, 6 adds 5 and 5 adds 28. 8 keeps one word of margin over 7. Shipped ON
# (0 = off; a value below 4 is raised to 4).
SHINGLE_WORDS = 8
SHINGLE_WORDS_ENV_KEY = "ORION_MESH_EDGE_SHINGLE_WORDS"
SHINGLE_WORDS_FLOOR = 4
REDACTION_MARKER = "[withheld: private memory]"
# Light function-word folding: interchangeable articles and relative pronouns count as one word.
# No stemming, no stop-word removal: every word still holds its place in the run.
_FUNCTION_FOLD = {"an": "a", "the": "a", "which": "that"}

# Every version of every core page: an older version is still private once replaced.
CORE_PAGES_SQL = (
    "SELECT concept_id, content FROM self_concept_history "
    "WHERE concept_id IN ('page:juniper', 'page:origin', 'page:lately', 'page:anchor')"
)
# Privacy, not serving: retired, superseded, rejected and removed rows stay guarded too.
# Bare table names on purpose (issue #3567): the governor reads as orion_readonly, whose search_path
# is (orion_scope, public), so memory_crystallizations resolves to the AI-Town-filtered view
# (scripts/sql/scope_orion_readonly_off_aitown.sql). AI Town stances are therefore not guarded there.
GUARD_MEMORY_SQL = (
    "SELECT 'episode_memory:' || memory_id::text AS source, statement AS text FROM episode_memory "
    "UNION ALL SELECT 'stance:' || crystallization_id::text, summary FROM memory_crystallizations "
    "WHERE kind = 'stance' "
    "UNION ALL SELECT 'person_fact:' || f.fact_id, coalesce(p.display_name, 'someone') || ', ' || "
    "replace(f.category, '_', ' ') || ': ' || f.value_text "
    "FROM person_fact f LEFT JOIN person p ON p.person_id = f.person_id"
)

# DSN env keys tried in order. All already exist in the services that write at the mesh edge
# (orion-hub: POSTGRES_URI; orion-harness-governor: SUBSTRATE_FELT_STATE_DATABASE_URL; orion-notify:
# POSTGRES_URI). ORION_MESH_EDGE_PG_DSN overrides them.
DSN_ENV_KEYS = ("ORION_MESH_EDGE_PG_DSN", "POSTGRES_URI", "DATABASE_URL", "SUBSTRATE_FELT_STATE_DATABASE_URL",
                "RECALL_PG_DSN")

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": " ", "–": " "})
_NON_WORD = re.compile(r"[^a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CITE = re.compile(r"\s*\[[^\]]*(?<![\w-])(?:person_fact|episode_memory|stance|self_concept_history|chat_history_log):[^\]]*\]")


class MeshEdgeLeak(RuntimeError):
    """Raised by ``check_outbound`` when text quotes a guarded sentence verbatim."""

    def __init__(self, surface: str, hits: Sequence["LeakHit"]) -> None:
        self.surface = surface
        self.hits = list(hits)
        super().__init__(
            f"mesh_edge_refused surface={surface} guarded_sentences={len(self.hits)} "
            f"sources={sorted({h.source for h in self.hits})}"
        )


class MeshEdgeUnavailable(RuntimeError):
    """The guarded text could not be loaded; an edge writer must refuse rather than send."""


_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
# Invisible characters that would otherwise split one word in two ("qu<ZWSP>iet").
_INVISIBLE = re.compile("[\u200b-\u200d\u2060\ufeff\u00ad]")
_CONTROL_ESCAPE = re.compile(r"\\[nrt]")


def normalize(text: str) -> str:
    """Lowercase, fold quotes/punctuation/whitespace: 'She’s  "here".' -> 'shes here'.

    Undoes the encodings a body passes through on its way out: HTML entities (``&#39;``), JSON
    escapes (``\\u2019``, ``\\n``) and shell quoting (``'doesn'\\''t'``), so an apostrophe
    inside quoting still folds to the same word."""
    raw = html.unescape(str(text or ""))
    raw = _UNICODE_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), raw)
    raw = _CONTROL_ESCAPE.sub(" ", raw)
    raw = raw.replace("'\\''", "'").replace("\\", "")
    raw = _INVISIBLE.sub("", raw)
    folded = raw.translate(_QUOTES).lower().replace("'", "")
    return " ".join(_NON_WORD.sub(" ", folded).split())


def sentences(text: str) -> List[str]:
    cleaned = _CITE.sub("", str(text or ""))
    return [s.strip() for s in _SENTENCE_SPLIT.split(cleaned) if s and s.strip()]


def fold_words(norm: str) -> List[str]:
    """Words of an already-normalized string, with the light function-word folding applied."""
    return [_FUNCTION_FOLD.get(w, w) for w in norm.split()]


def shingle_words(env: Optional[dict] = None) -> int:
    """The configured run length: ``SHINGLE_WORDS`` unless the env key overrides it. 0 = off.
    An unreadable value keeps the default (the stricter side), never turns the check off."""
    env = os.environ if env is None else env
    raw = str(env.get(SHINGLE_WORDS_ENV_KEY) or "").strip()
    if not raw:
        return SHINGLE_WORDS
    try:
        n = int(raw)
    except ValueError:
        return SHINGLE_WORDS
    return 0 if n <= 0 else max(n, SHINGLE_WORDS_FLOOR)


@dataclass(frozen=True)
class LeakHit:
    source: str   # "page:juniper" | "page:origin" | "page:lately" | "page:anchor" | lived-memory doc id
    words: int    # verbatim: length of the matched sentence; shingle: outgoing words covered
    kind: str = "verbatim"  # "verbatim" | "shingle" (the text itself is never logged)


_TOKEN = re.compile(r"\S+")
_SENTENCE_END = re.compile(r"[.!?][\"')\]*_`’”]*$")
# A URL, file path, code identifier or module name inside a guarded sentence is a public pointer,
# not private prose: "https://github.com/<owner>/<repo>/pull/12" alone normalizes to 8+ words.
# Such tokens split a guarded sentence into separate runs for shingling (verbatim matching still
# uses the whole sentence). Outgoing text is never stripped, so nothing can hide inside a URL.
# Ordinary prose like "she/her", "and/or" or "24/7" is NOT a pointer (one slash, no extension).
_POINTER_TOKEN = re.compile(r"://|^www\.|/[^/\s]*/|\\|`|\w_\w|[A-Za-z0-9]\.[A-Za-z]{2,}")


def prose_segments(sentence: str) -> List[List[str]]:
    """Folded word runs of one guarded sentence, split at pointer tokens (URLs, paths, code)."""
    if not _POINTER_TOKEN.search(sentence):
        return [fold_words(normalize(sentence))]
    segments: List[List[str]] = [[]]
    for tok in _TOKEN.findall(sentence):
        if _POINTER_TOKEN.search(tok):
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].extend(fold_words(normalize(tok)))
    return [seg for seg in segments if seg]


@dataclass
class MeshEdgeGuard:
    """Normalized guarded sentences -> where each came from, plus their word shingles."""

    phrases: dict[str, str] = field(default_factory=dict)
    loaded_at: float = 0.0
    shingle_n: int = 0
    shingles: dict[str, str] = field(default_factory=dict)
    source_rows: int = 0      # rows read (every page version, every retired memory): grows with time
    source_sentences: int = 0  # guarded-length sentences before dedup; ``phrases`` is after

    @classmethod
    def from_texts(cls, items: Iterable[tuple[str, str]], *, min_words: int = MIN_WORDS,
                   shingle_n: Optional[int] = None) -> "MeshEdgeGuard":
        n = shingle_words() if shingle_n is None else max(0, int(shingle_n))
        phrases: dict[str, str] = {}
        shingles: dict[str, str] = {}
        rows = raw = 0
        for source, text in items:
            rows += 1
            for sent in sentences(text):
                norm = normalize(sent)
                words = norm.split()
                if len(words) >= min_words:
                    raw += 1
                    # Keyed by the normalized sentence: a sentence repeated across page versions or
                    # rows is stored once, so the guard grows with new sentences, not new versions.
                    phrases.setdefault(" ".join(fold_words(norm)), source)
                if n and len(words) >= n:
                    for segment in prose_segments(sent):
                        for i in range(len(segment) - n + 1):
                            shingles.setdefault(" ".join(segment[i:i + n]), source)
        return cls(phrases=phrases, loaded_at=time.monotonic(), shingle_n=n, shingles=shingles,
                   source_rows=rows, source_sentences=raw)

    def _shingle_positions(self, folded: Sequence[str]) -> List[tuple[int, str]]:
        n = self.shingle_n
        if not n or not self.shingles or len(folded) < n:
            return []
        out: List[tuple[int, str]] = []
        for i in range(len(folded) - n + 1):
            src = self.shingles.get(" ".join(folded[i:i + n]))
            if src is not None:
                out.append((i, src))
        return out

    def find(self, text: str) -> List[LeakHit]:
        folded = fold_words(normalize(text))
        hay = " " + " ".join(folded) + " "
        hits = [LeakHit(source=src, words=len(p.split())) for p, src in self.phrases.items() if f" {p} " in hay]
        seen = {h.source for h in hits}
        covered: dict[str, set] = {}
        for i, src in self._shingle_positions(folded):
            if src not in seen:
                covered.setdefault(src, set()).update(range(i, i + self.shingle_n))
        hits.extend(LeakHit(source=src, words=len(idx), kind="shingle") for src, idx in covered.items())
        return hits

    def check(self, text: str, *, surface: str) -> None:
        hits = self.find(text)
        if hits:
            raise MeshEdgeLeak(surface, hits)

    def redact(self, text: str, *, marker: str = REDACTION_MARKER) -> tuple[str, List[LeakHit]]:
        """``text`` with every outgoing sentence that holds a verbatim or shingle match replaced by
        ``marker`` (one marker per run of withheld sentences), and the hits found in the original.

        The whole sentence goes, not just the matched words: a near-miss leaves neighbouring
        guarded words in clear, and the writers that redact have no author waiting to rewrite, so
        over-redaction is cheap. The caller must ``find`` again on the result and refuse if
        anything is left: normalization across token boundaries can, rarely, match where the
        per-token mapping cannot place the span."""
        hits = self.find(text)
        if not hits:
            return text, []
        tokens = list(_TOKEN.finditer(text))
        owner: List[int] = []   # normalized word index -> token index
        norm_words: List[str] = []
        sentence_of: List[int] = []  # token index -> outgoing sentence number
        sid = 0
        for t_idx, m in enumerate(tokens):
            gap = text[tokens[t_idx - 1].end():m.start()] if t_idx else ""
            if t_idx and ("\n" in gap or _SENTENCE_END.search(tokens[t_idx - 1].group(0))):
                sid += 1
            sentence_of.append(sid)
            for w in fold_words(normalize(m.group(0))):
                norm_words.append(w)
                owner.append(t_idx)
        withheld: set = set()
        n = self.shingle_n
        for i, _src in self._shingle_positions(norm_words):
            withheld.update(owner[j] for j in range(i, i + n))
        joined = " ".join(norm_words)
        hay = " " + joined + " "
        # Char offset in ``joined`` -> word index, for placing verbatim sentences.
        starts = {}
        pos = 0
        for idx, w in enumerate(norm_words):
            starts[pos] = idx
            pos += len(w) + 1
        for phrase in self.phrases:
            needle = f" {phrase} "
            at = hay.find(needle)
            while at != -1:
                first = starts.get(at)  # hay has one leading space, so ``at`` is the offset in joined
                if first is not None:
                    withheld.update(owner[j] for j in range(first, first + len(phrase.split())))
                at = hay.find(needle, at + 1)
        if not withheld:
            return text, hits
        hit_sentences = {sentence_of[t] for t in withheld}
        withheld = {t for t in range(len(tokens)) if sentence_of[t] in hit_sentences}
        out: List[str] = []
        cursor = 0
        in_run = False
        for t_idx, m in enumerate(tokens):
            if t_idx in withheld:
                if not in_run:
                    out.append(text[cursor:m.start()])
                    out.append(marker)
                    in_run = True
                cursor = m.end()
            else:
                out.append(text[cursor:m.start()])
                out.append(m.group(0))
                in_run = False
                cursor = m.end()
        out.append(text[cursor:])
        return "".join(out), hits


def resolve_dsn(env: Optional[dict] = None) -> str:
    env = os.environ if env is None else env
    for key in DSN_ENV_KEYS:
        value = str(env.get(key) or "").strip()
        if value:
            return value
    return ""


def guard_texts_from_rows(page_rows: Iterable[Any], lived_records: Iterable[Any]) -> List[tuple[str, str]]:
    items: List[tuple[str, str]] = []
    for row in page_rows:
        items.append((str(row["concept_id"]), str(row["content"] or "")))
    for rec in lived_records:
        items.append((rec.doc_id, rec.text))
    return items


def _load_sync(dsn: str) -> MeshEdgeGuard:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore

    conn = psycopg2.connect(dsn, connect_timeout=5)
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(CORE_PAGES_SQL)
            items = [(str(r["concept_id"]), str(r["content"] or "")) for r in cur.fetchall()]
            cur.execute(GUARD_MEMORY_SQL)
            items.extend((str(r["source"]), str(r["text"] or "")) for r in cur.fetchall())
    finally:
        conn.close()
    return MeshEdgeGuard.from_texts(items)


_cache_lock = threading.Lock()
_cache: dict[str, MeshEdgeGuard] = {}


def load_guard(dsn: Optional[str] = None, *, ttl_sec: float = CACHE_TTL_SEC) -> MeshEdgeGuard:
    """The guard, cached for ``ttl_sec`` per DSN. Raises MeshEdgeUnavailable when it cannot load."""
    dsn = dsn if dsn is not None else resolve_dsn()
    if not dsn:
        raise MeshEdgeUnavailable("no database DSN for the mesh-edge check")
    with _cache_lock:
        cached = _cache.get(dsn)
        if cached is not None and time.monotonic() - cached.loaded_at < ttl_sec:
            return cached
    started = time.monotonic()
    try:
        guard = _load_sync(dsn)
    except Exception as exc:  # noqa: BLE001
        logger.warning("mesh_edge_guard_load_failed err=%s load_ms=%d", type(exc).__name__,
                       (time.monotonic() - started) * 1000)
        raise MeshEdgeUnavailable(f"mesh-edge guard load failed: {type(exc).__name__}") from exc
    # Size and cost of every (re)load, so growth of the guarded set is visible (thread cf5d4218097a).
    logger.info("mesh_edge_guard_loaded rows=%d sentences=%d phrases=%d shingles=%d load_ms=%d",
                guard.source_rows, guard.source_sentences, len(guard.phrases), len(guard.shingles),
                (time.monotonic() - started) * 1000)
    with _cache_lock:
        _cache[dsn] = guard
    return guard


def check_outbound(text: str, *, surface: str, guard: Optional[MeshEdgeGuard] = None) -> None:
    """Refuse (raise) when ``text`` quotes a core-page or lived-memory sentence verbatim or shares a
    shingle run with one, or when the guard cannot be loaded. ``surface`` names the writer."""
    (guard or load_guard()).check(text, surface=surface)


__all__ = [
    "CORE_PAGES_SQL", "LeakHit", "MIN_WORDS", "MeshEdgeGuard", "MeshEdgeLeak", "MeshEdgeUnavailable",
    "REDACTION_MARKER", "SHINGLE_WORDS", "SHINGLE_WORDS_ENV_KEY", "check_outbound", "fold_words",
    "guard_texts_from_rows", "load_guard", "normalize", "prose_segments", "resolve_dsn", "sentences",
    "shingle_words",
]

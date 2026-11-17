"""Mesh-edge leak check (issue #3409).

Core pages and lived memories are free inside Orion's mesh (chat, Prime, curiosity, ...). They
must not be pasted into public output: GitHub writes made with Orion's account (issues, PRs,
comments) and email to anyone other than Juniper. This module is the one deterministic check
every such writer calls before it sends.

What is guarded (loaded from Postgres, read-only):

- every sentence of every ``page:juniper`` and ``page:anchor`` version in
  ``self_concept_history`` (the core pages, old versions included);
- every sentence of every lived memory recall can inject into the chat prompt, and of the ones it
  no longer may (superseded, rejected, expired, removed): episode_memory, stance crystallizations,
  person facts. Retired text is still private.

A sentence counts only when it has at least ``MIN_WORDS`` words: "she/her" or "7" in a public PR
is not a leak of anything. Matching is verbatim after normalization (case, quotes, punctuation
and whitespace folded), so shell quoting or a Markdown wrap does not hide a quoted sentence.

Failure mode is closed: a writer that cannot load the guarded text must refuse, not send
(``load_guard`` raises ``MeshEdgeUnavailable``).
"""
from __future__ import annotations

import html
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

MIN_WORDS = 6
CACHE_TTL_SEC = 300.0

# Every version of every core page: an older version is still private once replaced.
CORE_PAGES_SQL = (
    "SELECT concept_id, content FROM self_concept_history "
    "WHERE concept_id IN ('page:juniper', 'page:anchor')"
)
# Privacy, not serving: retired, superseded, rejected and removed rows stay guarded too.
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
_CITE = re.compile(r"\s*\[[^\]]*(?:person_fact|episode_memory)[^\]]*\]")


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
    folded = raw.translate(_QUOTES).lower().replace("'", "")
    return " ".join(_NON_WORD.sub(" ", folded).split())


def sentences(text: str) -> List[str]:
    cleaned = _CITE.sub("", str(text or ""))
    return [s.strip() for s in _SENTENCE_SPLIT.split(cleaned) if s and s.strip()]


@dataclass(frozen=True)
class LeakHit:
    source: str   # "page:juniper" | "page:anchor" | lived-memory doc id
    words: int    # length of the matched sentence (the sentence itself is never logged)


@dataclass
class MeshEdgeGuard:
    """Normalized guarded sentences -> where each came from."""

    phrases: dict[str, str] = field(default_factory=dict)
    loaded_at: float = 0.0

    @classmethod
    def from_texts(cls, items: Iterable[tuple[str, str]], *, min_words: int = MIN_WORDS) -> "MeshEdgeGuard":
        phrases: dict[str, str] = {}
        for source, text in items:
            for sent in sentences(text):
                norm = normalize(sent)
                if len(norm.split()) >= min_words:
                    phrases.setdefault(norm, source)
        return cls(phrases=phrases, loaded_at=time.monotonic())

    def find(self, text: str) -> List[LeakHit]:
        hay = " " + normalize(text) + " "
        return [LeakHit(source=src, words=len(p.split())) for p, src in self.phrases.items() if f" {p} " in hay]

    def check(self, text: str, *, surface: str) -> None:
        hits = self.find(text)
        if hits:
            raise MeshEdgeLeak(surface, hits)


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
    try:
        guard = _load_sync(dsn)
    except Exception as exc:  # noqa: BLE001
        raise MeshEdgeUnavailable(f"mesh-edge guard load failed: {type(exc).__name__}") from exc
    with _cache_lock:
        _cache[dsn] = guard
    return guard


def check_outbound(text: str, *, surface: str, guard: Optional[MeshEdgeGuard] = None) -> None:
    """Refuse (raise) when ``text`` quotes a core-page or lived-memory sentence verbatim, or when
    the guard cannot be loaded. ``surface`` names the writer for the log line."""
    (guard or load_guard()).check(text, surface=surface)


__all__ = [
    "CORE_PAGES_SQL", "LeakHit", "MIN_WORDS", "MeshEdgeGuard", "MeshEdgeLeak", "MeshEdgeUnavailable",
    "check_outbound", "guard_texts_from_rows", "load_guard", "normalize", "resolve_dsn", "sentences",
]

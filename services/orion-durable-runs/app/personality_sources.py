"""The personality reducer's reads: Orion's own output per lane, by per-source keyset cursors.

Spec: docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md (rev 2).
Writer: ``app.personality_driver``. Pure graph: ``orion.personality.graph``.

Each source is read in ``(occurred_at, ref)`` order strictly after its stored cursor and strictly
before ``hi`` (now minus the read lag), at most ``PER_TICK_CAP`` rows per tick, so a backlog drains
over several ticks. A source with no cursor row starts at the reducer's FIRST boot (no backfill:
rev 2 starts the graph from today); the driver persists that start cursor on its first tick.

One source row becomes one ``SourceItem`` with zero or more ``Candidate`` evidence. Zero is a row
read and consumed (nothing Orion said in it, or a curiosity run where nothing was tested). Evidence
ids are ``<lane>:<source pk>`` (``curiosity_choice:<run_id>:<prior_id>`` for picks), so a row read
twice (``curiosity_run_outcomes.completed_at`` is reset on every outcome upsert) is the same id
twice; the driver drops ids already in ``personality_evidence`` before folding.

Live rates (read-only, 7 days to 2026-10-16): dream_hypothesis 56; journal_entries 1,583 (metacog
1,164, world_pulse 222, self_study 105, embodiment 52, manual 15, scheduler 8, orion_day 8,
self_reflection 5, notify 4; world_pulse and orion_day are mirrors and are not read, see
``JOURNAL_MIRROR_SOURCE_KINDS``); substrate_reverie_thought 3,686 (0 empty interpretations);
world_pulse_read_seed stage2 done 105 (``summary`` present on 105/105, ``article_summary`` 88);
orion_day_letter 8; chat_history_log 30 with a prompt, 28 without (outreach); curiosity runs
with an offer row 50; self_concept_history curiosity_self_inquiry 149 rows over 21 concepts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

from orion.personality.choices import PRIOR_REF_PREFIX, curiosity_choice_from_rows

STATEMENT_TIMEOUT = "30s"  # a stalled read fails the tick (retried next tick), never hangs it
READ_LAG_SEC = 120.0       # rows newer than now - lag wait for the next tick (late writers)
PER_TICK_CAP = 200         # rows per source per tick
EMBED_TEXT_MAX = 2000
EXCERPT_MAX = 160
SEED_CURSOR = "seed"       # one-shot marker: the first-boot self-inquiry seed was written
# Journal source kinds that are copies of a lane already read at its source, so counting them would
# fake cross-lane recurrence (the promotion rule needs independent lanes). Live 2026-10-16, 7 days:
# 106 of 225 ``world_pulse`` journal bodies equal a reading's ``stage2_result_json.summary`` and the
# rest carry ``source_ref`` ``world_pulse_read:<id>`` (the same reading runs); 17 of 17 ``orion_day``
# entries (30 days) open with their day letter's text.
JOURNAL_MIRROR_SOURCE_KINDS = ("world_pulse", "orion_day")


@dataclass(frozen=True)
class Cursor:
    at: datetime
    ref: str = ""


@dataclass(frozen=True)
class Candidate:
    """Evidence before embedding. ``text`` is what gets embedded; ``excerpt`` is the label."""

    evidence_id: str
    lane: str
    source_ref: str
    occurred_at: datetime
    text: str
    excerpt: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SourceItem:
    at: datetime
    ref: str
    candidates: tuple[Candidate, ...] = ()


# --- text helpers (pure) ------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
_DATE_SUFFIX = re.compile(r"_\d{8}$")
_HEADING_LINE = re.compile(r"^\s*(#{1,6}\s+.*|\*\*[^*]+\*\*|__[^_]+__)\s*$")


def utc(value: Any) -> Optional[datetime]:
    """Aware UTC. Naive values are UTC by contract (``chat_history_log.created_at``)."""
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def clean(text: Any) -> str:
    return text.strip() if isinstance(text, str) else ""


def embed_text(*parts: Any) -> str:
    joined = "\n\n".join(p for p in (clean(x) for x in parts) if p)
    return joined[:EMBED_TEXT_MAX]


def excerpt(text: Any, max_chars: int = EXCERPT_MAX) -> str:
    """First sentence-ish, whitespace-collapsed, at most ``max_chars``. A leading markdown
    heading or bold title line ("**Note for 2026-10-15**") is dropped when body text follows."""
    raw = clean(text)
    if not raw:
        return ""
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    while len(lines) > 1 and _HEADING_LINE.match(lines[0]):
        lines = lines[1:]
    flat = _WS.sub(" ", " ".join(lines)).replace("**", "").replace("__", "").lstrip("#> ").strip()
    if not flat:
        return ""
    # First sentence end at or after 20 chars, so "Dr. X" or "1." does not cut the sentence.
    for m in _SENTENCE_END.finditer(flat):
        if m.end() >= 20:
            flat = flat[: m.end()]
            break
    if len(flat) <= max_chars:
        return flat
    cut = flat[: max_chars - 1]
    space = cut.rfind(" ")
    if space >= max_chars // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "…"


def readable_prior(ref: Any) -> str:
    """``curiosity_prior:self:reading_link_can_cross_region_boundary_20261015`` ->
    ``reading link can cross region boundary``."""
    s = clean(ref)
    if s.startswith(PRIOR_REF_PREFIX):
        s = s[len(PRIOR_REF_PREFIX):]
    if s.startswith("self:"):
        s = s[len("self:"):]
    s = _DATE_SUFFIX.sub("", s)
    return _WS.sub(" ", s.replace("_", " ")).strip()


# --- row -> item builders (pure) ----------------------------------------------------------------


def _one(lane: str, ref: str, at: datetime, text: str, ex: str, meta: Optional[dict] = None,
         source_ref: Optional[str] = None) -> SourceItem:
    if not text or not ex:
        return SourceItem(at, ref)
    return SourceItem(at, ref, (Candidate(f"{lane}:{ref}", lane, source_ref or ref, at, text, ex, meta or {}),))


def dream_item(r: Mapping[str, Any]) -> SourceItem:
    ref, at = str(r["hypothesis_id"]), utc(r["created_at"])
    return _one("dream", ref, at, embed_text(r.get("claim"), r.get("why")),
                excerpt(r.get("claim")) or excerpt(r.get("why")), {"arm": r.get("arm")},
                source_ref=f"dream_hypothesis:{ref}")


def journal_item(r: Mapping[str, Any]) -> SourceItem:
    ref, at = str(r["entry_id"]), utc(r["created_at"])
    return _one("journal", ref, at, embed_text(r.get("title"), r.get("body")),
                excerpt(r.get("body")) or excerpt(r.get("title")),
                {"source_kind": r.get("source_kind"), "mode": r.get("mode")},
                source_ref=f"journal_entries:{ref}")


def reverie_item(r: Mapping[str, Any]) -> SourceItem:
    ref, at = str(r["thought_id"]), utc(r["created_at"])
    text = clean(r.get("interpretation"))
    return _one("reverie", ref, at, embed_text(text), excerpt(text), source_ref=f"substrate_reverie_thought:{ref}")


def reading_item(r: Mapping[str, Any]) -> SourceItem:
    """Orion's own read of the article (``stage2_result_json.summary``), not the article's."""
    ref, at = str(r["seed_id"]), utc(r["stage2_completed_at"])
    return _one("reading", ref, at, embed_text(r.get("title"), r.get("summary")), excerpt(r.get("summary")),
                {"title": clean(r.get("title"))[:200]}, source_ref=f"world_pulse_read_seed:{ref}")


def day_letter_item(r: Mapping[str, Any]) -> SourceItem:
    ref, at = str(r["letter_date"]), utc(r["created_at"])
    return _one("day_letter", ref, at, embed_text(r.get("note_md")), excerpt(r.get("note_md")),
                source_ref=f"orion_day_letter:{ref}")


def chat_item(lane: str) -> Callable[[Mapping[str, Any]], SourceItem]:
    def build(r: Mapping[str, Any]) -> SourceItem:
        ref, at = str(r["id"]), utc(r["created_at"])
        return _one(lane, ref, at, embed_text(r.get("response")), excerpt(r.get("response")),
                    {"correlation_id": r.get("correlation_id"), "session_id": r.get("session_id")},
                    source_ref=f"chat_history_log:{ref}")
    return build


def self_inquiry_item(r: Mapping[str, Any]) -> SourceItem:
    ref, at = str(r["entry_id"]), utc(r["created_at"])
    return _one("self_inquiry_seed", ref, at, embed_text(r.get("content")), excerpt(r.get("content")),
                {"concept_id": r.get("concept_id"), "version": r.get("version")},
                source_ref=f"self_concept_history:{ref}")


_OFFER_KEYS = ("run_id", "offered", "stale_offered", "material_ids", "turn_started_at", "decided_at", "arm")
_OUTCOME_KEYS = ("run_id", "per_prior", "completed_at", "turn_ok", "realized_nats")


def curiosity_item(r: Mapping[str, Any]) -> SourceItem:
    """One run -> one evidence per chosen (tested) prior: the prior's slug made readable."""
    run_id, at = str(r["run_id"]), utc(r["completed_at"])
    choice = curiosity_choice_from_rows({k: r.get(k) for k in _OFFER_KEYS}, {k: r.get(k) for k in _OUTCOME_KEYS})
    if choice is None:
        return SourceItem(at, run_id)
    out: list[Candidate] = []
    for opt in choice.chosen:
        prior_id = opt.ref[len(PRIOR_REF_PREFIX):] if opt.ref.startswith(PRIOR_REF_PREFIX) else opt.ref
        text = readable_prior(opt.ref)
        if not text:
            continue
        out.append(Candidate(
            evidence_id=f"curiosity_choice:{run_id}:{prior_id}", lane="curiosity_choice",
            source_ref=choice.source_ref, occurred_at=utc(choice.occurred_at), text=text[:EMBED_TEXT_MAX],
            excerpt=excerpt(text),
            meta={"choice_id": choice.choice_id, "chosen_best_rank": choice.chosen_best_rank, "rank": opt.rank,
                  "prior_id": prior_id}))
    return SourceItem(at, run_id, tuple(out))


# --- SQL ----------------------------------------------------------------------------------------
# Keyset: (t, ref) > (cursor.at, cursor.ref) AND t < hi, written so the time index applies.

def _keyset(t: str, ref: str) -> str:
    return (f"{t} >= %(at)s AND ({t} > %(at)s OR {ref} > %(ref)s) AND {t} < %(hi)s "
            f"ORDER BY {t}, {ref} LIMIT %(limit)s")


# chat_history_log.created_at is timestamp WITHOUT time zone holding UTC.
def _keyset_naive_utc(t: str, ref: str) -> str:
    at, hi = "(%(at)s::timestamptz AT TIME ZONE 'UTC')", "(%(hi)s::timestamptz AT TIME ZONE 'UTC')"
    return f"{t} >= {at} AND ({t} > {at} OR {ref} > %(ref)s) AND {t} < {hi} ORDER BY {t}, {ref} LIMIT %(limit)s"


@dataclass(frozen=True)
class SourceSpec:
    name: str   # personality_cursor.source
    sql: str
    build: Callable[[Mapping[str, Any]], SourceItem]


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("dream", "SELECT hypothesis_id, created_at, arm, claim, why FROM dream_hypothesis WHERE "
               "btrim(coalesce(claim, '')) <> '' AND " + _keyset("created_at", "hypothesis_id"), dream_item),
    SourceSpec("journal", "SELECT entry_id, created_at, title, body, source_kind, mode FROM journal_entries WHERE "
               "btrim(coalesce(body, '')) <> '' AND coalesce(source_kind, '') <> ALL(%(mirror_kinds)s) AND "
               + _keyset("created_at", "entry_id"), journal_item),
    SourceSpec("reverie", "SELECT thought_id, created_at, interpretation FROM substrate_reverie_thought WHERE "
               "btrim(coalesce(interpretation, '')) <> '' AND " + _keyset("created_at", "thought_id"), reverie_item),
    SourceSpec("reading", "SELECT seed_id, stage2_completed_at, title, stage2_result_json->>'summary' AS summary "
               "FROM world_pulse_read_seed WHERE stage2_status = 'done' "
               "AND btrim(coalesce(stage2_result_json->>'summary', '')) <> '' AND "
               + _keyset("stage2_completed_at", "seed_id"), reading_item),
    SourceSpec("day_letter", "SELECT letter_date::text AS letter_date, created_at, note_md FROM orion_day_letter WHERE "
               "btrim(coalesce(note_md, '')) <> '' AND " + _keyset("created_at", "letter_date::text"), day_letter_item),
    SourceSpec("chat", "SELECT id, created_at, correlation_id, session_id, response FROM chat_history_log WHERE "
               "btrim(coalesce(prompt, '')) <> '' AND btrim(coalesce(response, '')) <> '' AND "
               + _keyset_naive_utc("created_at", "id"), chat_item("chat")),
    SourceSpec("outreach", "SELECT id, created_at, correlation_id, session_id, response FROM chat_history_log WHERE "
               "btrim(coalesce(prompt, '')) = '' AND btrim(coalesce(response, '')) <> '' AND "
               + _keyset_naive_utc("created_at", "id"), chat_item("outreach")),
    SourceSpec("curiosity_choice", "SELECT d.run_id, d.offered, d.stale_offered, d.material_ids, d.turn_started_at, "
               "d.decided_at, d.arm, o.per_prior, o.completed_at, o.turn_ok, o.realized_nats "
               "FROM curiosity_run_outcomes o JOIN curiosity_offer_decisions d USING (run_id) WHERE "
               + _keyset("o.completed_at", "o.run_id"), curiosity_item),
    SourceSpec("self_inquiry_seed", "SELECT entry_id, created_at, concept_id, version, content FROM self_concept_history "
               "WHERE produced_by = 'curiosity_self_inquiry' AND btrim(coalesce(content, '')) <> '' AND "
               + _keyset("created_at", "entry_id"), self_inquiry_item),
)
SOURCE_NAMES = tuple(s.name for s in SOURCES)

# First-boot seed: the latest version of each self-inquiry concept (spec rev 2, "Seeds").
SEED_SQL = """
SELECT DISTINCT ON (concept_id) entry_id, created_at, concept_id, version, content
FROM self_concept_history
WHERE produced_by = 'curiosity_self_inquiry' AND btrim(coalesce(content, '')) <> '' AND created_at <= %(now)s
ORDER BY concept_id, version DESC, created_at DESC
"""


async def _fetch(conn: Any, sql: str, params: dict) -> list[dict]:
    cur = await conn.execute(sql, params)
    return list(await cur.fetchall())


class PersonalitySourceReader:
    """Reads every source in one read-only transaction on the service's psycopg pool (dict rows)."""

    def __init__(self, pool: Any, *, limit: int = PER_TICK_CAP) -> None:
        self._pool = pool
        self._limit = limit

    async def read(self, cursors: Mapping[str, Cursor], hi: datetime) -> dict[str, list[SourceItem]]:
        """Items per source strictly after its cursor and before ``hi``. Sources without a cursor
        are not read (the driver starts them at first boot)."""
        out: dict[str, list[SourceItem]] = {}
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET TRANSACTION READ ONLY")
                await conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                for spec in SOURCES:
                    cur = cursors.get(spec.name)
                    if cur is None or cur.at >= hi:
                        continue
                    rows = await _fetch(conn, spec.sql, {"at": cur.at, "ref": cur.ref, "hi": hi, "limit": self._limit,
                                                         "mirror_kinds": list(JOURNAL_MIRROR_SOURCE_KINDS)})
                    out[spec.name] = [spec.build(r) for r in rows]
        return out

    async def read_seeds(self, now: datetime) -> list[SourceItem]:
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET TRANSACTION READ ONLY")
                await conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                rows = await _fetch(conn, SEED_SQL, {"now": now})
        return [self_inquiry_item(r) for r in rows]

#!/usr/bin/env python3
"""Weekly-cadence batch job: turn Orion's own real, persisted drive-activation
history into ONE narrative `reflection`-kind crystallization.

Context: `DriveAuditV1` (`orion/core/schemas/drives.py`) is computed live on every
DriveEngine tick and persisted, append-only, to the Postgres `drive_audits` table
(written by orion-sql-writer) as a timestamped row -- a genuine historical
time-series, unlike the "latest value only" stores this repo already has for the
same drive signal (see `orion/spark/concept_induction/store.py`'s
`LocalProfileStore` and `orion.autonomy.state_store`'s single-row-per-subject
UPSERT). (Historical note: the original source was the RDF drives graph, which
stopped receiving writes on 2026-06-19 and was removed as a read path entirely --
Postgres `drive_audits` is the only source, with no graph fallback.)

This script is the first real reader of that history for self-modeling purposes.

Architecture (event -> schema -> trace -> reducer -> projection -> eval/LLM phrasing
-> crystallization -- per AGENTS.md's mandated path, NOT "vague theory -> LLM invents
a pattern"):

    1. FETCH   -- read real DriveAuditV1-derived rows from the Postgres
                  `drive_audits` table (`fetch_drive_history_events`), over the
                  same DSN this script already uses for crystallization writes.
    2. REDUCE  -- `reduce_drive_history()`: a PURE, deterministic, unit-tested Python
                  function (no I/O, no LLM) that aggregates the raw per-tick events
                  into `DriveHistoryAggregationV1` -- dominant-drive counts/shares,
                  per-day breakdown, active-drive frequency, mean pressures, and a
                  sufficiency verdict. This is the same bar as
                  `orion/spark/concept_induction/drive_tension.py`: synthetic-input,
                  known-output tests, zero LLM involvement.
    3. GUARDRAIL -- lives INSIDE the reducer (`agg.sufficient` /
                  `agg.insufficiency_reason`), not bolted on after: it is a
                  reduction-time decision ("not enough real data to claim a
                  pattern"), not a separate pre-check. See MIN_EVENTS /
                  MIN_DISTINCT_DAYS below.
    4. PHRASE  -- `build_fact_sheet()` renders the aggregation into a small, numbered
                  list of already-computed, already-verified fact strings. The LLM
                  (`_call_llm_narrative`, bus RPC per
                  `orion/memory/crystallization/concept_relation.py`'s established
                  pattern) receives ONLY this fact sheet -- never raw per-tick
                  rows. Its job shrinks to "phrase these facts as one narrative
                  sentence and cite which facts you used" -- it cannot invent a
                  pattern that was not already established deterministically, because
                  the pattern (or lack of one) was computed before the LLM ever saw
                  anything.
    5. VALIDATE -- `parse_and_validate_narrative()`: every cited fact number must be
                  real, and the fact's own literal tokens (real dates/percentages/
                  drive names/timestamps) must appear verbatim in the LLM's narrative
                  text. A citation that does not literally quote real data is
                  rejected -- this is the mandatory grounding defense the task brief
                  calls out as the single biggest risk in this build.
    6. WRITE   -- one `reflection`-kind `MemoryCrystallizationV1` via
                  `apply_salience()` + `insert_crystallization()`, mirroring
                  `scripts/concept_relation_digest.py`'s governance/evidence shape
                  exactly. Evidence refs cite BOTH the reducer's aggregation object
                  AND the individual raw DriveAuditV1 artifacts the LLM actually
                  cited, so a human reviewer can verify the narrative against what
                  was really computed, not just that some event IDs exist.

Not a service loop -- a standalone script, run on demand (see
services/orion-memory-consolidation/README.md for why this one is NOT cron'd, unlike
concept_relation_digest.py).

Usage:
    POSTGRES_URI=postgresql://user:pass@host:port/db python scripts/drive_history_reflection_synthesis.py
    python scripts/drive_history_reflection_synthesis.py --postgres-uri postgresql://... --since-days 14
    python scripts/drive_history_reflection_synthesis.py --json
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/drive_history_reflection_synthesis.py` puts scripts/ on
# sys.path[0], which shadows stdlib `platform` via scripts/platform/ (imported
# transitively by `uuid`, `asyncio`, etc.) -- same issue documented in
# scripts/check_inner_state_registry.py / scripts/concept_relation_digest.py. This
# fix MUST run before any stdlib import that might transitively pull in `platform`,
# so it comes immediately after the only two imports (`sys`, `pathlib.Path`) needed
# to perform it.
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

logger = logging.getLogger("orion.scripts.drive_history_reflection_synthesis")

DEFAULT_SINCE_DAYS = 30
DEFAULT_SUBJECT = "orion"
# Hard cap on raw events fetched from Postgres -- DriveEngine ticks can run several
# times a minute (live volume observed: tens of thousands of audits in a 30-day
# window), so an uncapped fetch would transfer an unbounded result set. This caps
# the *fetch*, not the reduction: reduce_drive_history() aggregates over exactly
# whatever was fetched and reports that honestly (see DriveHistoryAggregationV1.
# total_events). ORDER BY COALESCE(observed_at, created_at) DESC means a capped
# fetch is the most recent `max_events` real ticks in the window, not a fabricated
# or estimated sample.
DEFAULT_MAX_EVENTS = 500

# Guardrail thresholds (module-level constants, visible + testable, mirrors
# scripts/analysis/measure_autonomy_gate.py's threshold posture). Both must clear
# for reduce_drive_history() to consider the window "sufficient":
#   MIN_EVENTS: below this, a handful of ticks cannot distinguish a real
#     recurring tendency from a momentary blip -- 5 is the smallest count where
#     "most often" (a majority/plurality claim) is even a meaningful statement.
#   MIN_DISTINCT_DAYS: DriveEngine ticks frequently within a single session
#     (observed: multiple ticks per minute), so event COUNT alone can be
#     satisfied entirely within one sitting. Requiring >=2 distinct calendar
#     days is what actually distinguishes a "long-horizon pattern" (the whole
#     point of this script) from an artifact of one continuous session.
MIN_EVENTS = 5
MIN_DISTINCT_DAYS = 2
# Minimum number of real facts the LLM must cite (by number) for its narrative to
# be considered grounded rather than generic filler.
MIN_CITED_FACTS = 2


# ===========================================================================
# Fetch-layer types (I/O boundary representation of a real DriveAuditV1 tick)
# ===========================================================================


@dataclass(frozen=True)
class DriveAuditEvent:
    """One real, already-persisted DriveAuditV1 tick, as read back from the
    Postgres `drive_audits` table. `created_at` carries the event time,
    i.e. COALESCE(observed_at, created_at) per the table contract."""

    artifact_id: str
    created_at: datetime
    dominant_drive: str | None = None
    summary: str | None = None
    active_drives: tuple[str, ...] = ()
    drive_pressures: Mapping[str, float] = field(default_factory=dict)


# ===========================================================================
# REDUCER -- pure, deterministic, no I/O, no LLM. Mirrors
# orion/spark/concept_induction/drive_tension.py's contract exactly: typed
# dataclass output, synthetic-input/known-output unit tests.
# ===========================================================================


@dataclass(frozen=True)
class DriveHistoryAggregationV1:
    """Deterministic aggregation over a real, already-fetched set of
    DriveAuditEvent ticks. Nothing in this object is inferred or guessed --
    every field is computed directly from `events` by reduce_drive_history()."""

    subject: str
    since: datetime
    until: datetime | None
    total_events: int
    distinct_days: int
    dominant_drive_counts: Mapping[str, int]
    dominant_drive_share: Mapping[str, float]
    top_dominant_drive: str | None
    top_dominant_drive_count: int
    top_dominant_drive_share: float
    day_counts: Mapping[str, int]
    day_dominant_drive: Mapping[str, str]
    active_drive_frequency: Mapping[str, int]
    mean_pressure_by_drive: Mapping[str, float]
    cited_event_ids: tuple[str, ...]
    sufficient: bool
    insufficiency_reason: str | None


def reduce_drive_history(
    events: Sequence[DriveAuditEvent],
    *,
    subject: str,
    since: datetime,
    min_events: int = MIN_EVENTS,
    min_distinct_days: int = MIN_DISTINCT_DAYS,
    max_cited_events: int = 8,
) -> DriveHistoryAggregationV1:
    """Pure reduction over a real, already-fetched event list. Never raises on
    empty input (degrades to a `sufficient=False` aggregation with zeroed
    counts) -- callers must check `.sufficient` before doing anything with the
    rest of the fields. This is the ONLY place the "insufficient history"
    guardrail decision is made; nothing downstream re-derives it.
    """
    ordered = sorted(events, key=lambda e: e.created_at)
    total_events = len(ordered)

    day_counts: dict[str, int] = {}
    dominant_drive_counts: dict[str, int] = {}
    day_dominant_raw: dict[str, dict[str, int]] = {}
    active_drive_frequency: dict[str, int] = {}
    pressure_sums: dict[str, float] = {}
    pressure_counts: dict[str, int] = {}

    for ev in ordered:
        day = ev.created_at.astimezone(timezone.utc).date().isoformat()
        day_counts[day] = day_counts.get(day, 0) + 1
        if ev.dominant_drive:
            dominant_drive_counts[ev.dominant_drive] = dominant_drive_counts.get(ev.dominant_drive, 0) + 1
            day_dominant_raw.setdefault(day, {})
            day_dominant_raw[day][ev.dominant_drive] = day_dominant_raw[day].get(ev.dominant_drive, 0) + 1
        for drive in ev.active_drives:
            active_drive_frequency[drive] = active_drive_frequency.get(drive, 0) + 1
        for drive, pressure in ev.drive_pressures.items():
            try:
                p = float(pressure)
            except (TypeError, ValueError):
                continue
            pressure_sums[drive] = pressure_sums.get(drive, 0.0) + p
            pressure_counts[drive] = pressure_counts.get(drive, 0) + 1

    distinct_days = len(day_counts)

    dominant_drive_share = (
        {drive: count / total_events for drive, count in dominant_drive_counts.items()} if total_events else {}
    )

    top_dominant_drive: str | None = None
    top_dominant_drive_count = 0
    if dominant_drive_counts:
        # Deterministic tie-break: highest count first, then alphabetically first
        # name -- two runs over the same data must always agree.
        top_dominant_drive = sorted(dominant_drive_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        top_dominant_drive_count = dominant_drive_counts[top_dominant_drive]
    top_dominant_drive_share = (
        top_dominant_drive_count / total_events if total_events and top_dominant_drive else 0.0
    )

    day_dominant_drive = {
        day: sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for day, counts in day_dominant_raw.items()
    }

    mean_pressure_by_drive = {drive: pressure_sums[drive] / pressure_counts[drive] for drive in pressure_sums}

    # Deterministic, evenly-spaced sample across the time-ordered event list --
    # always includes the earliest and latest event (so the fact sheet can
    # ground a real time span, not just a cluster) -- used both for LLM
    # citation and as crystallization evidence refs.
    cited_event_ids: tuple[str, ...] = ()
    if ordered:
        k = min(max_cited_events, total_events)
        if k <= 1:
            indices = [0]
        else:
            indices = sorted({round(i * (total_events - 1) / (k - 1)) for i in range(k)})
        cited_event_ids = tuple(ordered[i].artifact_id for i in indices)

    sufficient = total_events >= min_events and distinct_days >= min_distinct_days
    insufficiency_reason: str | None = None
    if not sufficient:
        reasons = []
        if total_events < min_events:
            reasons.append(f"only {total_events} event(s) found (need >= {min_events})")
        if distinct_days < min_distinct_days:
            reasons.append(f"events span only {distinct_days} distinct day(s) (need >= {min_distinct_days})")
        insufficiency_reason = "; ".join(reasons)

    until = ordered[-1].created_at if ordered else None

    return DriveHistoryAggregationV1(
        subject=subject,
        since=since,
        until=until,
        total_events=total_events,
        distinct_days=distinct_days,
        dominant_drive_counts=dominant_drive_counts,
        dominant_drive_share=dominant_drive_share,
        top_dominant_drive=top_dominant_drive,
        top_dominant_drive_count=top_dominant_drive_count,
        top_dominant_drive_share=top_dominant_drive_share,
        day_counts=day_counts,
        day_dominant_drive=day_dominant_drive,
        active_drive_frequency=active_drive_frequency,
        mean_pressure_by_drive=mean_pressure_by_drive,
        cited_event_ids=cited_event_ids,
        sufficient=sufficient,
        insufficiency_reason=insufficiency_reason,
    )


# ===========================================================================
# Fact sheet -- deterministic rendering of the aggregation into short,
# citable, already-verified fact strings. This is the ONLY thing the LLM
# sees; it never sees raw per-tick rows.
# ===========================================================================


@dataclass(frozen=True)
class GroundedFact:
    index: int
    text: str
    # Literal substrings that must ALL appear verbatim (case-insensitively) in
    # the narrative for a citation of this fact to count as grounded (real
    # dates/percentages/drive names/naturalized timestamps -- never
    # LLM-generated). Requiring every token, not just one, matters: a fact's
    # low-entropy count token (e.g. "5") or a single correct drive name is not
    # enough on its own to prove the LLM reproduced the fact's actual claim
    # rather than coincidentally overlapping one substring.
    tokens: tuple[str, ...]
    # Set only for individual-event facts; lets evidence-building look up the
    # real DriveAuditEvent a citation refers back to, and lets
    # parse_and_validate_narrative require at least one real per-tick
    # citation (not just aggregate-level facts).
    artifact_id: str | None = None


# Per-tick `summary` text (currently always a deterministic template from
# orion/spark/concept_induction/audit.py::build_drive_audit -- never
# freeform/model-generated) rides along in individual-event facts as
# descriptive context, NOT as a grounding token requiring narrative
# reproduction. Truncated defensively so a future non-deterministic summary
# producer cannot blow up prompt size or smuggle unbounded content through.
_SUMMARY_TRUNC_CHARS = 200

# Bounds how many per-day facts the fact sheet can contain, independent of
# `--since-days` -- unlike every other stage of this pipeline (fetch is
# capped by --max-events, citation is capped by max_cited_events), a wide
# --since-days window with daily activity would otherwise inflate the
# prompt/cost without limit. Keeps the most recent days (most relevant to a
# "recent tendency" narrative).
MAX_DAY_FACTS = 14


def _naturalized_ts(dt: datetime) -> str:
    """Human-reproducible timestamp token (no seconds/timezone-offset
    punctuation) -- a full tz-aware ISO-8601 string with seconds and a UTC
    offset (e.g. `2026-06-14T09:00:00+00:00`) is not something a short
    reflective-voice sentence naturally reproduces verbatim, which made the
    individual-event citation path practically unreachable in practice."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def build_fact_sheet(
    agg: DriveHistoryAggregationV1, events_by_id: Mapping[str, DriveAuditEvent]
) -> list[GroundedFact]:
    """Pure function: renders an already-computed DriveHistoryAggregationV1 into
    a small numbered list of fact strings. No randomness, no LLM, fully
    deterministic given the same aggregation."""
    facts: list[GroundedFact] = []
    idx = 1

    if agg.top_dominant_drive:
        pct = f"{agg.top_dominant_drive_share * 100:.0f}%"
        since_day = agg.since.astimezone(timezone.utc).date().isoformat()
        until_day = (agg.until or agg.since).astimezone(timezone.utc).date().isoformat()
        facts.append(
            GroundedFact(
                index=idx,
                text=(
                    f"'{agg.top_dominant_drive}' was the dominant drive in {agg.top_dominant_drive_count} of "
                    f"{agg.total_events} audited ticks ({pct}) across {agg.distinct_days} distinct day(s) "
                    f"between {since_day} and {until_day}."
                ),
                tokens=(agg.top_dominant_drive, pct, str(agg.top_dominant_drive_count)),
            )
        )
        idx += 1

    # Most recent MAX_DAY_FACTS days only -- see MAX_DAY_FACTS docstring.
    for day in sorted(agg.day_dominant_drive)[-MAX_DAY_FACTS:]:
        drive = agg.day_dominant_drive[day]
        count = agg.day_counts.get(day, 0)
        facts.append(
            GroundedFact(
                index=idx,
                text=f"On {day}, the dominant drive was '{drive}' ({count} tick(s) that day).",
                tokens=(day, drive),
            )
        )
        idx += 1

    for eid in agg.cited_event_ids:
        ev = events_by_id.get(eid)
        if ev is None:
            continue
        ts = _naturalized_ts(ev.created_at)
        summary = (ev.summary or "(no summary)").strip()
        if len(summary) > _SUMMARY_TRUNC_CHARS:
            summary = summary[: _SUMMARY_TRUNC_CHARS - 3] + "..."
        dominant = ev.dominant_drive or "none"
        facts.append(
            GroundedFact(
                index=idx,
                text=(f"Audit {ev.artifact_id} at {ts} UTC: dominant_drive={dominant}, summary={summary}."),
                tokens=(ts, dominant),
                artifact_id=ev.artifact_id,
            )
        )
        idx += 1

    return facts


def _build_narrative_prompt(agg: DriveHistoryAggregationV1, facts: list[GroundedFact]) -> str:
    facts_block = "\n".join(f"Fact {f.index}: {f.text}" for f in facts)
    event_fact_numbers = [f.index for f in facts if f.artifact_id]
    instructions = (
        "You are given a set of ALREADY-COMPUTED, verified facts about Orion's own drive-activation "
        "history over a real time window (queried directly from Orion's persisted drive-audit history; "
        "every fact below was computed deterministically, not inferred or estimated by you). Do not "
        "invent any fact, number, date, or drive name that is not explicitly listed below.\n\n"
        "Write ONE short narrative observation (2-4 sentences) in Orion's own reflective voice, about "
        "the long-horizon pattern these facts reveal about Orion's own drive activity.\n\n"
        "You MUST cite at least 2 of the facts above by number, including AT LEAST ONE of these specific "
        f"individual-audit facts: {event_fact_numbers}. For every fact you cite, ALL of its literal "
        "date(s)/timestamp(s), percentage(s), count(s), and drive name(s) MUST appear verbatim in your "
        "narrative text -- partially reproducing a fact (e.g. only its drive name, without its exact "
        "percentage or count) does not count as citing it. Reproduce dates/timestamps in EXACTLY the "
        "same format shown in the fact (e.g. write '2026-06-14', not 'June 14'). Reproduce every "
        "percentage EXACTLY as the digits+percent-sign shown in the fact -- e.g. if a fact says '100%', "
        "your sentence must contain the literal text '100%'; do NOT substitute words like 'all', "
        "'every', 'always', or 'entirely' in place of the percentage, even when they are accurate in "
        "meaning. The same applies to counts and drive names: write them exactly as shown, never "
        "paraphrased, summarized, or rounded differently. IMPORTANT: some facts contain MORE THAN ONE "
        "number (e.g. both a raw count like '500' AND a percentage like '100%' in the same fact) -- you "
        "MUST include EVERY one of those numbers in your sentence, even if one seems to make another "
        "redundant. Do not drop the raw count just because you already stated the percentage, and do "
        "not drop the percentage just because you already stated the count -- a fact with two numbers "
        "requires both numbers to appear, not just whichever one reads more naturally.\n\n"
        "Respond with ONLY a single JSON object, no prose, no markdown fences, shaped exactly like:\n"
        '{"narrative": "<your 2-4 sentence observation>", "cited_fact_numbers": [<int>, ...]}\n'
    )
    return f"FACTS:\n{facts_block}\n\n{instructions}"


# ===========================================================================
# Narrative validation -- the mandatory anti-hallucination guardrail. Pure,
# no I/O; fully unit-testable with synthetic LLM output strings.
# ===========================================================================


class NarrativeUngrounded(Exception):
    """Raised when the LLM's reply is malformed, cites nothing real, or cites
    a fact without literally quoting any of that fact's real tokens. Callers
    must treat this the same as an LLM call failure: no crystallization is
    written."""


@dataclass(frozen=True)
class GroundedNarrative:
    narrative: str
    cited_fact_numbers: tuple[int, ...]


def parse_and_validate_narrative(
    raw_content: str, facts: list[GroundedFact], *, min_cited: int = MIN_CITED_FACTS
) -> GroundedNarrative:
    from orion.core.llm_json import parse_json_object

    try:
        obj = parse_json_object(raw_content)
    except Exception as exc:
        raise NarrativeUngrounded(f"LLM reply was not valid JSON: {exc}") from exc

    narrative = str(obj.get("narrative") or "").strip()
    cited_raw = obj.get("cited_fact_numbers")
    if not narrative:
        raise NarrativeUngrounded("LLM reply had an empty narrative")
    if not isinstance(cited_raw, list) or not cited_raw:
        raise NarrativeUngrounded("LLM reply did not cite any fact numbers")

    facts_by_index = {f.index: f for f in facts}
    cited_numbers: list[int] = []
    for raw in cited_raw:
        try:
            n = int(raw)
        except (TypeError, ValueError):
            continue
        if n in facts_by_index and n not in cited_numbers:
            cited_numbers.append(n)

    if len(cited_numbers) < min_cited:
        raise NarrativeUngrounded(f"LLM cited only {len(cited_numbers)} valid fact(s), need >= {min_cited}")

    narrative_lower = narrative.lower()
    for n in cited_numbers:
        fact = facts_by_index[n]
        # ALL of a fact's tokens must match, not just one: a fact carries
        # several independent claims (e.g. drive name + percentage + count),
        # and matching only the drive name (or, worse, only a low-entropy
        # digit like a small count) would let a narrative "cite" a fact
        # without actually reproducing its specific numbers -- defeating the
        # point of a verbatim-token guardrail. Case-insensitive: an LLM
        # naturally capitalizing a drive name at a sentence start (e.g.
        # "Continuity has...") is still a genuine, faithful citation and
        # must not be rejected over case alone -- dates/percentages/counts
        # are digit strings unaffected by lowercasing, so this does not
        # weaken the numeric half of the guarantee.
        if not all(token and token.lower() in narrative_lower for token in fact.tokens):
            raise NarrativeUngrounded(
                f"cited fact {n} but not all of its real tokens {fact.tokens!r} appear verbatim in the narrative"
            )

    # At least one citation must be a real, individual DriveAuditV1 artifact
    # (not just an aggregate-level fact) -- this is what lets
    # _build_reflection_crystallization() attach genuine per-tick evidence
    # refs, not just a description of the aggregation. Aggregate-only
    # citations are still real and grounded, but the task's evidence
    # contract requires the queried event set itself to be traceable, not
    # only a summary of it.
    if not any(facts_by_index[n].artifact_id for n in cited_numbers):
        raise NarrativeUngrounded(
            "no cited fact referenced an individual audited tick (all citations were aggregate-level); "
            "at least one citation must be a real per-tick DriveAuditV1 artifact"
        )

    return GroundedNarrative(narrative=narrative, cited_fact_numbers=tuple(cited_numbers))


# ===========================================================================
# Postgres fetch layer -- row->event conversion is pure (unit-testable
# without a live database); `fetch_drive_history_events` is the only I/O
# function here.
# ===========================================================================


# Local copy of scripts/analysis/measure_autonomy_gate.py::is_undefined_table_error
# -- scripts in this repo are standalone (deliberately not importable from each
# other; same script-to-script sharing precedent as extract_final_text), so a
# small verbatim copy with this pointer is the established pattern.
def is_undefined_table_error(exc: BaseException) -> bool:
    """True iff ``exc`` is Postgres "relation does not exist" (SQLSTATE 42P01).

    Checked structurally (``pgcode`` attribute / exception class name) so the
    caller can distinguish "the drive_audits table hasn't been created yet by
    the writer track" from any other query failure WITHOUT importing psycopg2
    at module scope. Message sniffing is a last-resort fallback for driver
    objects that carry neither. Pure + unit-testable.
    """
    if getattr(exc, "pgcode", None) == "42P01":
        return True
    if type(exc).__name__ == "UndefinedTable":
        return True
    msg = str(exc).lower()
    return "relation" in msg and "does not exist" in msg


# Event time = COALESCE(observed_at, created_at) per the drive_audits table
# contract (expression-indexed DESC, so the ORDER BY + window filter below are
# both index-served). The `subject` predicate preserves the previous fetch's
# per-subject scoping (--subject / DriveAuditV1 subject key): without it, rows
# from any other subject would be silently aggregated under this run's subject
# label, which would fabricate attribution.
_DRIVE_AUDIT_EVENTS_SQL = """
SELECT artifact_id,
       COALESCE(observed_at, created_at) AS created_at,
       dominant_drive,
       summary,
       drive_pressures,
       active_drives
FROM drive_audits
WHERE subject = %s
  AND COALESCE(observed_at, created_at) >= %s
ORDER BY COALESCE(observed_at, created_at) DESC
LIMIT %s
""".strip()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _coerce_event_time(value: Any) -> datetime | None:
    """Event-time column -> aware datetime. psycopg2 hands back tz-aware
    datetimes for timestamptz; a naive datetime or ISO string is coerced
    defensively (assume UTC), anything else degrades to None (row skipped)."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        return _parse_dt(value)
    return None


def _coerce_drive_pressures(value: Any) -> dict[str, float]:
    """JSONB `drive_pressures` column -> dict[str, float]. Degrades to {} on
    None or a non-dict value; individual entries whose value is not a real,
    finite, non-negative number (or whose key is not a non-empty string) are
    skipped. Never raises."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if isinstance(raw, bool):
            continue
        try:
            p = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(p) or p < 0.0:
            continue
        out[key] = p
    return out


def _coerce_active_drives(value: Any) -> tuple[str, ...]:
    """JSONB `active_drives` column -> tuple[str, ...]. Degrades to () on None
    or a non-list value; non-string/empty entries are skipped; duplicates are
    collapsed (order preserved) so active_drive_frequency counts each drive at
    most once per tick. Never raises."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(v for v in value if isinstance(v, str) and v.strip()))


def drive_audit_row_to_event(row: Any) -> DriveAuditEvent | None:
    """Pure conversion of one `drive_audits` row (column order of
    _DRIVE_AUDIT_EVENTS_SQL) into a DriveAuditEvent. Returns None (row
    skipped) when the identity fields (artifact_id / event time) are unusable;
    the JSONB detail columns degrade to empty rather than dropping the whole
    row -- a tick with a garbled pressure map is still a real tick."""
    if not isinstance(row, (list, tuple)) or len(row) != 6:
        return None
    artifact_id_raw, created_raw, dominant_raw, summary_raw, pressures_raw, actives_raw = row
    artifact_id = artifact_id_raw.strip() if isinstance(artifact_id_raw, str) else ""
    created_at = _coerce_event_time(created_raw)
    if not artifact_id or created_at is None:
        return None
    dominant = dominant_raw.strip() if isinstance(dominant_raw, str) and dominant_raw.strip() else None
    summary = summary_raw.strip() if isinstance(summary_raw, str) and summary_raw.strip() else None
    return DriveAuditEvent(
        artifact_id=artifact_id,
        created_at=created_at,
        dominant_drive=dominant,
        summary=summary,
        active_drives=_coerce_active_drives(actives_raw),
        drive_pressures=_coerce_drive_pressures(pressures_raw),
    )


def parse_drive_audit_rows(rows: Iterable[Any]) -> list[DriveAuditEvent]:
    events: list[DriveAuditEvent] = []
    for row in rows:
        ev = drive_audit_row_to_event(row)
        if ev is not None:
            events.append(ev)
    return events


def fetch_drive_history_events(
    postgres_uri: str,
    *,
    since: datetime,
    subject: str,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> list[DriveAuditEvent]:
    """I/O function: reads the drive-audit time-series from the Postgres
    `drive_audits` table (written by orion-sql-writer) over the SAME DSN this
    script already uses for crystallization writes -- one DSN concept, no
    second config surface.

    Degrade posture (mirrors scripts/analysis/measure_autonomy_gate.py::
    fetch_drive_stats_postgres): a missing table (SQLSTATE 42P01 -- the writer
    track not deployed yet), a failed connection, an unavailable driver, or
    any query failure returns an EMPTY event list with a clear log line --
    never raises, and never falls back to any other source. An empty result
    flows into reduce_drive_history(), which reports it honestly as
    insufficient history rather than fabricating a pattern.

    Unlike the removed graph read path, every returned event is FULLY
    populated: `drive_pressures`/`active_drives` come straight off the row's
    JSONB columns, so downstream `mean_pressure_by_drive` /
    `active_drive_frequency` are genuine full-window statistics.
    """
    try:
        import psycopg2  # lazy import so the module imports cleanly for tests
    except Exception:
        logger.error("drive_audits fetch unavailable: psycopg2 not importable; returning empty event list")
        return []

    try:
        conn = psycopg2.connect(postgres_uri)
    except Exception as exc:
        logger.error("drive_audits fetch failed: could not connect to postgres (%s); returning empty event list", exc)
        return []

    try:
        with conn.cursor() as cur:
            cur.execute(_DRIVE_AUDIT_EVENTS_SQL, (subject, since, int(max_events)))
            rows = cur.fetchall()
    except Exception as exc:
        if is_undefined_table_error(exc):
            logger.warning(
                "drive_audits table does not exist yet (sql-writer track not deployed?); "
                "returning empty event list"
            )
        else:
            logger.error("drive_audits query failed (%s); returning empty event list", exc)
        return []
    finally:
        try:
            conn.close()
        except Exception:  # pragma: no cover - close failure is not actionable
            pass

    return parse_drive_audit_rows(rows)


# ===========================================================================
# LLM call -- bus RPC, mirrors orion/memory/crystallization/concept_relation.py
# ::resolve_concept_relation() exactly (same envelope/channel/decode shape).
# Unlike that function, this one does NOT degrade silently on failure -- a
# failed or ungrounded LLM call must abort the run, not write a fabricated
# crystallization.
# ===========================================================================


async def _call_llm_narrative(
    bus: Any,
    *,
    prompt: str,
    route: str,
    request_channel: str,
    timeout_sec: float,
    service_name: str,
) -> str:
    from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef

    corr_id = str(uuid4())
    reply_channel = f"orion:exec:result:LLMGatewayService:{corr_id}"
    payload = ChatRequestPayload(
        messages=[LLMMessage(role="user", content=prompt)],
        route=route,
        options={
            "return_logprobs": False,
            "max_tokens": 400,
            "llm_route": route,
            "purpose": "reflect",
            "skip_spark_candidate_publish": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    env = BaseEnvelope(
        kind="llm.chat.request",
        source=ServiceRef(name=service_name, version="0.1.0", node=os.getenv("NODE_NAME", "athena")),
        correlation_id=corr_id,
        reply_to=reply_channel,
        payload=payload.model_dump(mode="json"),
    )
    msg = await bus.rpc_request(request_channel, env, reply_channel=reply_channel, timeout_sec=timeout_sec)
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(f"LLM gateway RPC failed: {decoded.error}")
    payload_out = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
    content = str(payload_out.get("content") or payload_out.get("text") or "")
    return content


# ===========================================================================
# Crystallization builder -- mirrors
# scripts/concept_relation_digest.py::_build_reflection_crystallization()
# exactly (same governance shape, same apply_salience()+insert_crystallization()
# pipeline).
# ===========================================================================


def _build_reflection_crystallization(
    agg: DriveHistoryAggregationV1,
    narrative: GroundedNarrative,
    facts: list[GroundedFact],
    events_by_id: Mapping[str, DriveAuditEvent],
):
    from orion.memory.crystallization.salience import apply_salience
    from orion.memory.crystallization.schemas import (
        CrystallizationEvidenceRefV1,
        CrystallizationGovernanceV1,
        MemoryCrystallizationV1,
        _utc_now,
        new_crystallization_id,
    )

    now = _utc_now()
    subject_line = f"Drive-activation history reflection: {agg.subject}"

    facts_by_index = {f.index: f for f in facts}
    cited_facts = [facts_by_index[n] for n in narrative.cited_fact_numbers if n in facts_by_index]

    evidence: list[CrystallizationEvidenceRefV1] = [
        # Evidence ref #1: the reducer's own aggregation, so a human reviewer
        # can verify the narrative actually matches what was computed -- not
        # just that some raw event IDs exist.
        CrystallizationEvidenceRefV1(
            source_kind="rdf_memory_graph",
            source_id=f"drive_history_aggregation:{agg.subject}:{agg.since.date().isoformat()}",
            strength=1.0,
            note=(
                f"aggregation: top_dominant_drive={agg.top_dominant_drive} "
                f"count={agg.top_dominant_drive_count}/{agg.total_events} "
                f"distinct_days={agg.distinct_days} cited_fact_numbers={list(narrative.cited_fact_numbers)}"
            ),
        )
    ]

    # Evidence refs #2..N: the real underlying DriveAuditV1 artifacts the LLM
    # actually cited, each traceable back to the Postgres drive_audits table
    # by artifact_id. (source_kind stays "rdf_memory_graph" for dedup
    # continuity with previously synthesized reflections -- the same-day dedup
    # guard in _run_once matches on this exact source_kind.)
    for fact in cited_facts:
        if not fact.artifact_id:
            continue
        ev = events_by_id.get(fact.artifact_id)
        if ev is None:
            continue
        note = f"raw_event: ts={ev.created_at.isoformat()} dominant_drive={ev.dominant_drive}"
        # Pressures/active-drives come straight off the drive_audits row's
        # JSONB columns (populated for every fetched event whose columns held
        # valid data) -- included when present so this evidence note carries
        # the per-drive detail a human reviewer would want, not just
        # timestamp+dominant.
        if ev.drive_pressures:
            note += f" pressures={dict(sorted(ev.drive_pressures.items()))}"
        if ev.active_drives:
            note += f" active_drives={list(ev.active_drives)}"
        evidence.append(
            CrystallizationEvidenceRefV1(
                source_kind="rdf_memory_graph",
                source_id=ev.artifact_id,
                strength=0.8,
                note=note,
            )
        )

    crystallization = MemoryCrystallizationV1(
        crystallization_id=new_crystallization_id(),
        kind="reflection",
        subject=subject_line,
        summary=narrative.narrative,
        status="active",
        scope=["project:orion", f"subject:{agg.subject}"],
        evidence=evidence,
        governance=CrystallizationGovernanceV1(
            proposed_by="system:drive_history_reflection_synthesis",
            approved_by="system:drive_history_reflection_synthesis",
            approval_mode="auto_policy",
            validation_status="valid",
            requires_manual_review=False,
            sensitivity="private",
            created_from_policy="drive_history_reflection_synthesis",
        ),
        created_at=now,
        updated_at=now,
    )
    return apply_salience(crystallization)


class _SingleConnPool:
    """Thin shim so repository.py::insert_crystallization() (which expects an
    asyncpg.Pool and calls pool.acquire()) can run on a single already-open
    connection. Mirrors scripts/concept_relation_digest.py's identical shim."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def acquire(self) -> "_SingleConnPool":
        return self

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


# ===========================================================================
# Orchestration
# ===========================================================================


@dataclass
class RunResult:
    status: str
    subject: str
    since: datetime
    max_events: int
    aggregation: DriveHistoryAggregationV1 | None = None
    narrative: str | None = None
    cited_fact_numbers: tuple[int, ...] = ()
    crystallization_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        agg = self.aggregation
        return {
            "status": self.status,
            "subject": self.subject,
            "since": self.since.isoformat(),
            "max_events": self.max_events,
            "aggregation": (
                {
                    "total_events": agg.total_events,
                    "distinct_days": agg.distinct_days,
                    "top_dominant_drive": agg.top_dominant_drive,
                    "top_dominant_drive_count": agg.top_dominant_drive_count,
                    "top_dominant_drive_share": agg.top_dominant_drive_share,
                    "dominant_drive_counts": dict(agg.dominant_drive_counts),
                    "day_counts": dict(agg.day_counts),
                    "sufficient": agg.sufficient,
                    "insufficiency_reason": agg.insufficiency_reason,
                }
                if agg is not None
                else None
            ),
            "narrative": self.narrative,
            "cited_fact_numbers": list(self.cited_fact_numbers),
            "crystallization_id": self.crystallization_id,
            "error": self.error,
        }


async def _run_once(
    *,
    bus: Any,
    postgres_uri: str,
    subject: str,
    since: datetime,
    max_events: int = DEFAULT_MAX_EVENTS,
    llm_route: str = "metacog",
    llm_request_channel: str = "orion:exec:request:LLMGatewayService",
    llm_timeout_sec: float = 30.0,
    fetch_events: Any = None,
) -> RunResult:
    """Core orchestration, fully injectable (fetch_events / bus / postgres_uri)
    for testing without a live Postgres/Redis. `bus` needs only
    `.rpc_request()` + `.codec.decode()`; `fetch_events` (defaults to the real
    `fetch_drive_history_events`) needs only the same call signature."""
    result = RunResult(status="unknown", subject=subject, since=since, max_events=max_events)

    # fetch_drive_history_events is a blocking psycopg2 call -- run it off the
    # event loop so it cannot stall bus/asyncio machinery. It degrades to an
    # empty list on any DB failure (see its docstring), which the reducer then
    # reports honestly as insufficient history -- there is no separate
    # "query failed" status and no fallback source.
    fetcher = fetch_events if fetch_events is not None else fetch_drive_history_events
    events = await asyncio.to_thread(
        fetcher,
        postgres_uri,
        since=since,
        subject=subject,
        max_events=max_events,
    )

    agg = reduce_drive_history(events, subject=subject, since=since)
    result.aggregation = agg

    if not agg.sufficient:
        result.status = "insufficient_history"
        result.error = agg.insufficiency_reason
        return result

    # Same-day dedup guard: skip cleanly (no LLM call, no write) if a
    # reflection for this exact (subject, calendar day of `since`) was
    # already synthesized -- the most common accidental-duplicate case for a
    # manual/on-demand tool is simply re-running it the same day. This does
    # NOT solve cross-window redundancy in general (a rolling --since-days
    # window run on two different days will still both pass this check and
    # can each produce a real, independently-grounded reflection covering
    # overlapping history) -- that broader problem is intentionally left to
    # human review, per this script's manual/on-demand design (see README).
    dedup_source_id = f"drive_history_aggregation:{subject}:{since.date().isoformat()}"
    try:
        import asyncpg

        dedup_conn = await asyncpg.connect(postgres_uri)
        try:
            existing_id = await dedup_conn.fetchval(
                "SELECT crystallization_id FROM memory_crystallization_sources "
                "WHERE source_kind = $1 AND source_id = $2 LIMIT 1",
                "rdf_memory_graph",
                dedup_source_id,
            )
        finally:
            await dedup_conn.close()
    except Exception as exc:
        # Dedup is a best-effort convenience, not a correctness guarantee --
        # a failed check must not block an otherwise-valid synthesis run.
        logger.warning("drive_history_reflection_synthesis: dedup check failed (continuing): %s", exc)
        existing_id = None

    if existing_id:
        result.status = "already_synthesized_today"
        result.crystallization_id = str(existing_id)
        result.error = (
            f"a reflection for subject={subject!r} since={since.date().isoformat()} already exists "
            f"(crystallization_id={existing_id}); skipping duplicate synthesis"
        )
        return result

    # Events arrive fully populated from the single Postgres query
    # (drive_pressures/active_drives come off the row's JSONB columns), so
    # there is no separate detail-enrichment pass and no re-reduce: the
    # aggregation computed above already reflects real full-window data.
    events_by_id = {ev.artifact_id: ev for ev in events}

    facts = build_fact_sheet(agg, events_by_id)
    prompt = _build_narrative_prompt(agg, facts)

    try:
        raw_content = await _call_llm_narrative(
            bus,
            prompt=prompt,
            route=llm_route,
            request_channel=llm_request_channel,
            timeout_sec=llm_timeout_sec,
            service_name="drive-history-reflection-synthesis",
        )
    except Exception as exc:
        result.status = "llm_call_failed"
        result.error = str(exc)
        return result

    try:
        narrative = parse_and_validate_narrative(raw_content, facts)
    except NarrativeUngrounded as exc:
        result.status = "llm_output_ungrounded"
        result.error = str(exc)
        return result

    result.narrative = narrative.narrative
    result.cited_fact_numbers = narrative.cited_fact_numbers

    crystallization = _build_reflection_crystallization(agg, narrative, facts, events_by_id)

    import asyncpg

    from orion.memory.crystallization.repository import insert_crystallization

    conn = await asyncpg.connect(postgres_uri)
    try:
        cid = await insert_crystallization(_SingleConnPool(conn), crystallization)
    except Exception as exc:
        result.status = "insert_failed"
        result.error = str(exc)
        return result
    finally:
        await conn.close()

    result.status = "created"
    result.crystallization_id = cid
    return result


def _print_report(result: RunResult) -> None:
    print(
        f"drive_history_reflection_synthesis: subject={result.subject!r} "
        f"since={result.since.isoformat()} status={result.status}"
    )
    agg = result.aggregation
    if agg is not None:
        print(
            f"  history: {agg.total_events} event(s) fetched, {agg.distinct_days} distinct day(s) "
            f"(min required: {MIN_EVENTS} events / {MIN_DISTINCT_DAYS} days)"
        )
        if agg.top_dominant_drive:
            print(
                f"  top dominant drive: {agg.top_dominant_drive} "
                f"({agg.top_dominant_drive_count}/{agg.total_events}, "
                f"{agg.top_dominant_drive_share * 100:.0f}%)"
            )

    if result.status == "insufficient_history":
        print(f"  refusing to synthesize: {result.error}")
    elif result.status == "already_synthesized_today":
        print(f"  skipping duplicate: {result.error}")
    elif result.status == "llm_call_failed":
        print(f"  LLM call failed: {result.error}")
    elif result.status == "llm_output_ungrounded":
        print(f"  LLM output rejected as ungrounded: {result.error}")
    elif result.status == "insert_failed":
        print(f"  crystallization insert failed: {result.error}")
    elif result.status == "created":
        print(f"  cited facts: {list(result.cited_fact_numbers)}")
        print(f"  crystallization created: {result.crystallization_id}")
        print("  summary:")
        print(f"    {result.narrative}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--postgres-uri",
        default=os.getenv("POSTGRES_URI", ""),
        help="Postgres DSN. Defaults to $POSTGRES_URI (e.g. services/orion-hub/.env).",
    )
    parser.add_argument("--subject", default=DEFAULT_SUBJECT, help="DriveAuditV1 subjectKey to read (default: orion).")
    parser.add_argument(
        "--since-days",
        type=int,
        default=DEFAULT_SINCE_DAYS,
        help=f"How many days of drive-audit history to consider (default: {DEFAULT_SINCE_DAYS}). "
        "The system may not have this much retained history yet -- real coverage found is always "
        "reported explicitly, never assumed.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_MAX_EVENTS,
        help=f"Cap on raw ticks fetched from the Postgres drive_audits table "
        f"(default: {DEFAULT_MAX_EVENTS}, most recent first).",
    )
    parser.add_argument("--redis", default=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"), help="Orion bus URL.")
    parser.add_argument(
        "--llm-request-channel",
        default=os.getenv("CHANNEL_LLM_INTAKE", "orion:exec:request:LLMGatewayService"),
    )
    parser.add_argument("--llm-route", default=os.getenv("TURN_CHANGE_CLASSIFY_ROUTE", "metacog"))
    parser.add_argument("--llm-timeout-sec", type=float, default=30.0)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    args = parser.parse_args(argv)

    if not args.postgres_uri.strip():
        print(
            "drive_history_reflection_synthesis: no --postgres-uri given and $POSTGRES_URI is unset. "
            "Check services/orion-hub/.env for POSTGRES_URI.",
            file=sys.stderr,
        )
        return 2

    from orion.core.bus.async_service import OrionBusAsync

    since = datetime.now(timezone.utc) - timedelta(days=args.since_days)

    async def _main_async() -> RunResult:
        bus = OrionBusAsync(args.redis)
        await bus.connect()
        try:
            return await _run_once(
                bus=bus,
                postgres_uri=args.postgres_uri,
                subject=args.subject,
                since=since,
                max_events=args.max_events,
                llm_route=args.llm_route,
                llm_request_channel=args.llm_request_channel,
                llm_timeout_sec=args.llm_timeout_sec,
            )
        finally:
            await bus.close()

    try:
        result = asyncio.run(_main_async())
    except Exception as exc:
        print(f"drive_history_reflection_synthesis: run failed -- {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), default=str))
    else:
        _print_report(result)

    return 0 if result.status in ("created", "insufficient_history", "already_synthesized_today") else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Weekly-cadence batch job: turn Orion's own real, persisted drive-activation
history into ONE narrative `reflection`-kind crystallization.

Context: `DriveAuditV1` (`orion/core/schemas/drives.py`) is computed live on every
DriveEngine tick and written, append-only, by `services/orion-rdf-writer/app/
autonomy.py::_handle_drive_audit()` to the Fuseki `drives` graph
(`http://conjourney.net/graph/autonomy/drives`) as a timestamped `orion:DriveAudit` --
a genuine historical time-series, unlike the "latest value only" stores this repo
already has for the same drive signal (see `orion/spark/concept_induction/store.py`'s
`LocalProfileStore` and `orion.autonomy.state_store`'s single-row-per-subject UPSERT).

This script is the first real reader of that history for self-modeling purposes.

Architecture (event -> schema -> trace -> reducer -> projection -> eval/LLM phrasing
-> crystallization -- per AGENTS.md's mandated path, NOT "vague theory -> LLM invents
a pattern"):

    1. FETCH   -- read real DriveAuditV1-derived triples from Fuseki via
                  orion.graph.sparql_client.SparqlQueryClient (`fetch_drive_history_events`).
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
                  pattern) receives ONLY this fact sheet -- never raw per-tick SPARQL
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
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

logger = logging.getLogger("orion.scripts.drive_history_reflection_synthesis")

ORION_NS = "http://conjourney.net/orion#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"
AUTONOMY_DRIVES_GRAPH = "http://conjourney.net/graph/autonomy/drives"

DEFAULT_SINCE_DAYS = 30
DEFAULT_SUBJECT = "orion"
# Hard cap on raw events fetched from Fuseki -- DriveEngine ticks can run several
# times a minute (live volume observed: tens of thousands of audits in a 30-day
# window), so an uncapped fetch would transfer an unbounded result set. This caps
# the *fetch*, not the reduction: reduce_drive_history() aggregates over exactly
# whatever was fetched and reports that honestly (see DriveHistoryAggregationV1.
# total_events). ORDER BY DESC(created_at) means a capped fetch is the most recent
# `max_events` real ticks in the window, not a fabricated or estimated sample.
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
    """One real, already-persisted DriveAuditV1 tick, as read back from Fuseki.

    `artifact_uri` is the RDF subject IRI (used only to join the detail query);
    it is optional so synthetic events built directly in tests do not need one.
    """

    artifact_id: str
    created_at: datetime
    dominant_drive: str | None = None
    summary: str | None = None
    active_drives: tuple[str, ...] = ()
    drive_pressures: Mapping[str, float] = field(default_factory=dict)
    artifact_uri: str | None = None


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
# sees; it never sees raw per-tick SPARQL rows.
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
        "history over a real time window (queried directly from Orion's persisted drive-audit graph; "
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
# SPARQL fetch layer -- query builders are pure (unit-testable without a
# live endpoint); `fetch_drive_history_events` is the only I/O function here.
# ===========================================================================


# _escape_sparql / _literal below are intentionally NOT imported from
# orion.autonomy.repository (which has the same two tiny helpers) --
# importing that module was measured at ~2.6s just to load its transitive
# dependency chain, which is a real, disproportionate cost for a script whose
# whole point is a light, occasional batch job. This repeats a pattern
# already independently duplicated 4+ times elsewhere in the repo (no single
# canonical orion.graph-level home exists to import from without the same
# cost); a shared low-weight `orion.graph.sparql_bindings` module would be
# the real fix, but is out of scope for this patch.
def _escape_sparql(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _is_safe_sparql_iri(uri: str) -> bool:
    """Defensive validation before interpolating a URI raw into a SPARQL
    `VALUES ?x { <uri> ... }` clause (unlike string literals, IRIREFs cannot
    be escaped -- illegal characters must be rejected instead). These URIs
    are always ones this same process just read back from our own list
    query's `?artifact` bindings (never user input), so this is defense in
    depth against a malformed/legacy artifact URI breaking the query, not a
    response to any known live issue."""
    if not uri or "<" in uri or ">" in uri or '"' in uri or "{" in uri or "}" in uri:
        return False
    return not any(ch.isspace() or ord(ch) < 0x20 for ch in uri)


def build_drive_audit_event_list_sparql(
    *, since: datetime, subject: str, graph_uri: str = AUTONOMY_DRIVES_GRAPH, limit: int
) -> str:
    """Single-valued fields only (artifact_id/created_at/dominant_drive/summary) --
    no multi-valued predicate (`orion:hasDriveAssessment`, `orion:highlightsActiveDrive`)
    is joined here, so this query cannot fan out and LIMIT means what it says.
    `orion:timestamp` is multi-valued per audit (up to several per artifact per
    scripts/analysis/measure_autonomy_gate.py's documented live finding); the inner
    subquery aggregates with MIN(?ts) per artifact so each artifact contributes
    exactly one row to the outer query, using only the timestamp(s) that actually
    fall in-window.
    """
    since_iso = since.astimezone(timezone.utc).isoformat()
    return f"""PREFIX orion: <{ORION_NS}>
PREFIX xsd: <{XSD_NS}>
SELECT ?artifact ?artifact_id ?created_at ?dominant_drive ?summary WHERE {{
  GRAPH <{graph_uri}> {{
    {{
      SELECT ?artifact ?artifact_id (MIN(?ts) AS ?created_at) WHERE {{
        ?artifact a orion:DriveAudit ;
          orion:subjectKey "{_escape_sparql(subject)}" ;
          orion:artifactId ?artifact_id ;
          orion:timestamp ?ts .
        FILTER(?ts >= "{since_iso}"^^xsd:dateTime)
      }}
      GROUP BY ?artifact ?artifact_id
    }}
    OPTIONAL {{ ?artifact orion:dominantDriveName ?dominant_drive . }}
    OPTIONAL {{ ?artifact orion:auditSummary ?summary . }}
  }}
}}
ORDER BY DESC(?created_at)
LIMIT {int(limit)}
""".strip()


def build_drive_audit_detail_sparql(*, artifact_uris: Sequence[str], graph_uri: str = AUTONOMY_DRIVES_GRAPH) -> str:
    """Per-drive pressure + active-drive detail for a bounded, explicit set of
    artifact URIs (the events the list query already selected) -- fan-out from
    the multi-valued `hasDriveAssessment` / `highlightsActiveDrive` predicates
    is safe here because it is scoped to exactly this small VALUES set, not the
    whole window. Malformed URIs (see _is_safe_sparql_iri) are silently
    dropped rather than raising -- a single bad artifact URI should not abort
    detail enrichment for every other artifact in the set."""
    safe_uris = [uri for uri in artifact_uris if _is_safe_sparql_iri(uri)]
    values = " ".join(f"<{uri}>" for uri in safe_uris)
    return f"""PREFIX orion: <{ORION_NS}>
PREFIX rdfs: <{RDFS_NS}>
SELECT ?artifact ?drive_name ?drive_pressure ?active_drive WHERE {{
  VALUES ?artifact {{ {values} }}
  GRAPH <{graph_uri}> {{
    OPTIONAL {{
      ?artifact orion:hasDriveAssessment ?assessment .
      ?assessment orion:driveDimension ?drive_ref ;
        orion:drivePressure ?drive_pressure .
      OPTIONAL {{ FILTER(isIRI(?drive_ref)) ?drive_ref rdfs:label ?drive_name_label . }}
      BIND(COALESCE(?drive_name_label, STR(?drive_ref)) AS ?drive_name)
    }}
    OPTIONAL {{
      ?artifact orion:highlightsActiveDrive ?active_drive_ref .
      OPTIONAL {{ FILTER(isIRI(?active_drive_ref)) ?active_drive_ref rdfs:label ?active_drive_label . }}
      BIND(COALESCE(?active_drive_label, STR(?active_drive_ref)) AS ?active_drive)
    }}
  }}
}}
""".strip()


def _literal(binding: Mapping[str, Mapping[str, str]], key: str) -> str | None:
    val = binding.get(key, {}).get("value")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


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


def parse_event_list_bindings(bindings: Iterable[dict]) -> list[DriveAuditEvent]:
    events: list[DriveAuditEvent] = []
    for row in bindings:
        if not isinstance(row, dict):
            continue
        artifact_uri = _literal(row, "artifact")
        artifact_id = _literal(row, "artifact_id")
        created_at = _parse_dt(_literal(row, "created_at"))
        if not artifact_id or created_at is None:
            continue
        events.append(
            DriveAuditEvent(
                artifact_id=artifact_id,
                created_at=created_at,
                dominant_drive=_literal(row, "dominant_drive"),
                summary=_literal(row, "summary"),
                artifact_uri=artifact_uri,
            )
        )
    return events


def merge_event_details(events: list[DriveAuditEvent], detail_bindings: Iterable[dict]) -> list[DriveAuditEvent]:
    pressures: dict[str, dict[str, float]] = {}
    actives: dict[str, set[str]] = {}
    for row in detail_bindings:
        if not isinstance(row, dict):
            continue
        artifact_uri = _literal(row, "artifact")
        if not artifact_uri:
            continue
        drive_name = _literal(row, "drive_name")
        drive_pressure = _literal(row, "drive_pressure")
        if drive_name and drive_pressure is not None:
            try:
                pressures.setdefault(artifact_uri, {})[drive_name] = float(drive_pressure)
            except ValueError:
                pass
        active_drive = _literal(row, "active_drive")
        if active_drive:
            actives.setdefault(artifact_uri, set()).add(active_drive)

    merged: list[DriveAuditEvent] = []
    for ev in events:
        key = ev.artifact_uri
        merged.append(
            replace(
                ev,
                drive_pressures=pressures.get(key, {}) if key else {},
                active_drives=tuple(sorted(actives.get(key, set()))) if key else (),
            )
        )
    return merged


def fetch_drive_history_events(
    client: Any,
    *,
    since: datetime,
    subject: str,
    graph_uri: str = AUTONOMY_DRIVES_GRAPH,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> list[DriveAuditEvent]:
    """I/O function: `client` needs only a `.select(sparql: str) -> list[dict]`
    method (the shape of `orion.graph.sparql_client.SparqlQueryClient`, and
    trivially fakeable in tests). Any exception from `client.select()`
    propagates to the caller -- this function does not swallow SPARQL errors,
    unlike the LLM call below (which is documented to degrade); a failed graph
    read must abort the run cleanly, not silently report "no history".

    Returns list-level fields only (artifact_id/created_at/dominant_drive/
    summary) -- `drive_pressures`/`active_drives` are left at their empty
    defaults here. Measured live against real Fuseki: a detail join
    (`orion:hasDriveAssessment`/`orion:highlightsActiveDrive`, both
    multi-valued) scoped to `max_events` (up to `DEFAULT_MAX_EVENTS=500`)
    artifacts did not return within several minutes -- the VALUES+double-
    OPTIONAL combinatorics do not scale to hundreds of artifacts. Per-drive
    detail is fetched separately, ONLY for the bounded citable sample
    (`DriveHistoryAggregationV1.cited_event_ids`, capped at 8) via
    `fetch_event_details()` below, which is fast at that scale. This is a
    deliberate, disclosed scope reduction: `mean_pressure_by_drive` /
    `active_drive_frequency` on the resulting aggregation therefore reflect
    only whichever events actually got enriched, not the full window --
    callers must not read them as full-window statistics.
    """
    list_sparql = build_drive_audit_event_list_sparql(since=since, subject=subject, graph_uri=graph_uri, limit=max_events)
    bindings = client.select(list_sparql)
    return parse_event_list_bindings(bindings)


def fetch_event_details(
    client: Any,
    events: Sequence[DriveAuditEvent],
    *,
    graph_uri: str = AUTONOMY_DRIVES_GRAPH,
) -> list[DriveAuditEvent]:
    """Enrich a SMALL, already-selected set of events (real live measurement:
    safe at citable-sample scale, i.e. <=8-20 artifacts -- see
    fetch_drive_history_events's docstring for why this is not run over the
    full fetched window). Any exception from `client.select()` propagates;
    callers that consider detail enrichment non-critical (current caller,
    `_run_once`, does -- pressures/active-drives are not referenced by the
    narrative prompt today, only carried for evidence completeness) should
    catch it themselves rather than relying on this function to degrade."""
    uris = [ev.artifact_uri for ev in events if ev.artifact_uri and _is_safe_sparql_iri(ev.artifact_uri)]
    if not uris:
        return list(events)

    detail_sparql = build_drive_audit_detail_sparql(artifact_uris=uris, graph_uri=graph_uri)
    detail_bindings = client.select(detail_sparql)
    return merge_event_details(list(events), detail_bindings)


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
    # actually cited, each traceable back to the Fuseki drives graph by
    # artifact_id.
    for fact in cited_facts:
        if not fact.artifact_id:
            continue
        ev = events_by_id.get(fact.artifact_id)
        if ev is None:
            continue
        note = f"raw_event: ts={ev.created_at.isoformat()} dominant_drive={ev.dominant_drive}"
        # Pressures/active-drives are only present when detail enrichment
        # succeeded (best-effort, see fetch_event_details) -- included when
        # available so this evidence note is actually useful for the "per-drive
        # detail" a human reviewer would want, not just timestamp+dominant.
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
    sparql_client: Any,
    bus: Any,
    postgres_uri: str,
    subject: str,
    since: datetime,
    max_events: int = DEFAULT_MAX_EVENTS,
    llm_route: str = "metacog",
    llm_request_channel: str = "orion:exec:request:LLMGatewayService",
    llm_timeout_sec: float = 30.0,
    graph_uri: str = AUTONOMY_DRIVES_GRAPH,
    detail_sparql_client: Any | None = None,
) -> RunResult:
    """Core orchestration, fully injectable (sparql_client / bus / postgres_uri)
    for testing without a live Fuseki/Redis/Postgres. `sparql_client` needs only
    `.select()`; `bus` needs only `.rpc_request()` + `.codec.decode()`."""
    result = RunResult(status="unknown", subject=subject, since=since, max_events=max_events)

    try:
        # fetch_drive_history_events is a blocking `requests`-based call (real
        # windows have been observed to take 20+ seconds against Fuseki) --
        # run it off the event loop so it cannot stall bus/asyncio machinery.
        events = await asyncio.to_thread(
            fetch_drive_history_events,
            sparql_client,
            since=since,
            subject=subject,
            graph_uri=graph_uri,
            max_events=max_events,
        )
    except Exception as exc:
        result.status = "sparql_query_failed"
        result.error = str(exc)
        return result

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

    events_by_id = {ev.artifact_id: ev for ev in events}

    # Per-drive pressure/active-drive detail is fetched ONLY for the bounded
    # citable sample (see fetch_drive_history_events's docstring for the real
    # live-measured performance reason this is not done for the full fetched
    # window). Not fatal on failure -- the narrative prompt does not reference
    # raw pressures/active-drives today, only timestamp/dominant_drive/summary,
    # so a detail-enrichment failure degrades to plainer evidence notes rather
    # than aborting an otherwise-valid, already-reduced run.
    #
    # Live measurement finding: the detail join (VALUES + double OPTIONAL
    # through orion:hasDriveAssessment / orion:highlightsActiveDrive) did not
    # return within 60s against a live Fuseki instance with ~437k persisted
    # DriveAudit artifacts (~2.6M assessment triples) EVEN at citable-sample
    # scale (8 artifacts) -- the slowness is not proportional to the VALUES
    # set size, so it is very likely Jena/Fuseki failing to push the VALUES
    # restriction down before scanning DriveAssessment triples. A dedicated,
    # short `detail_sparql_client` (default well under `sparql_client`'s
    # timeout) is used here so this best-effort enrichment fails fast and
    # degrades rather than stalling an otherwise-successful run -- especially
    # important because `orion.graph.fuseki_http.call_with_fuseki_retry`
    # retries transient failures up to 4x with backoff, which would otherwise
    # multiply a single slow query into several minutes of blocking.
    cited_events = [events_by_id[eid] for eid in agg.cited_event_ids if eid in events_by_id]
    if cited_events:
        try:
            enriched = await asyncio.to_thread(
                fetch_event_details, detail_sparql_client or sparql_client, cited_events, graph_uri=graph_uri
            )
            for ev in enriched:
                events_by_id[ev.artifact_id] = ev
            # Re-reduce over the now-enriched events so mean_pressure_by_drive /
            # active_drive_frequency on the reported aggregation reflect real
            # data instead of staying unconditionally empty (reduce_drive_history
            # is pure and deterministic over cited_event_ids/day/dominant-drive
            # counts, so re-running it here changes nothing except the two
            # pressure/active fields, which now see the enriched subset).
            agg = reduce_drive_history(list(events_by_id.values()), subject=subject, since=since)
            result.aggregation = agg
        except Exception as exc:
            logger.warning("drive_history_reflection_synthesis: event-detail enrichment failed: %s", exc)

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
    elif result.status == "sparql_query_failed":
        print(f"  SPARQL query failed: {result.error}")
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
        help=f"Cap on raw ticks fetched from Fuseki (default: {DEFAULT_MAX_EVENTS}, most recent first).",
    )
    parser.add_argument(
        "--query-url",
        default="",
        help="Explicit Fuseki SPARQL query URL. Defaults to autonomy-graph resolution "
        "(AUTONOMY_GRAPH_QUERY_URL / RDF_STORE_QUERY_URL / RDF_STORE_BASE_URL derivation).",
    )
    parser.add_argument(
        "--sparql-timeout-sec",
        type=float,
        default=60.0,
        help="Live measurement: a 30-day/500-event list query took ~24s against real "
        "Fuseki -- default gives headroom over that.",
    )
    parser.add_argument(
        "--detail-timeout-sec",
        type=float,
        default=10.0,
        help="Timeout for the small, best-effort per-drive detail enrichment query "
        "(see fetch_event_details). Deliberately short and separate from "
        "--sparql-timeout-sec: live measurement found this specific join did not "
        "return within 60s even at citable-sample scale (~8 artifacts), and "
        "orion.graph.fuseki_http.call_with_fuseki_retry retries transient failures "
        "up to 4x -- a short timeout here keeps a slow/stuck detail join from "
        "blocking an otherwise-successful run for minutes.",
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

    from orion.graph.backend_config import resolve_autonomy_read_query_url, resolve_rdf_store_auth
    from orion.graph.sparql_client import SparqlQueryClient
    from orion.core.bus.async_service import OrionBusAsync

    query_url = args.query_url.strip() or resolve_autonomy_read_query_url()[0]
    if not query_url:
        print(
            "drive_history_reflection_synthesis: no SPARQL query endpoint resolved. "
            "Set --query-url, AUTONOMY_GRAPH_QUERY_URL, RDF_STORE_QUERY_URL, or RDF_STORE_BASE_URL.",
            file=sys.stderr,
        )
        return 2

    user, password = resolve_rdf_store_auth()
    sparql_client = SparqlQueryClient(query_url, timeout_sec=args.sparql_timeout_sec, user=user, password=password)
    detail_sparql_client = SparqlQueryClient(
        query_url, timeout_sec=args.detail_timeout_sec, user=user, password=password
    )

    since = datetime.now(timezone.utc) - timedelta(days=args.since_days)

    async def _main_async() -> RunResult:
        bus = OrionBusAsync(args.redis)
        await bus.connect()
        try:
            return await _run_once(
                sparql_client=sparql_client,
                detail_sparql_client=detail_sparql_client,
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

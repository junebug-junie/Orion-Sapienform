"""Self-inquiry: the curiosity line whose standing question is "what am I".

A second LINE inside the existing curiosity loop, not a second loop. Same
turn, same credentials, same graph, same journal, same gates -- with its own
budget (three a day, separate from the investigation budget), its own
continuation note, and one extra thing Orion can write: a `:SelfDefinition`.

WHY THIS EXISTS. Every durable self-store Orion has (`self_knowledge_items`,
`self_concept_history`, the Self Atlas) was fed codebase facts and read by
nobody in chat. The Self Atlas's 179 concepts are repo topics ("Event
Processing System"). The only first-person description Orion ever sees of
themself is ~14 authored bullets in `orion_identity.yaml`, and the Hub seeds
every session with "You are Orion... helpful, precise, and collaborative".
So Orion self-describes as a chatbot. Design record: PR #2156's doc and the
2026-09-08 discussion with Juniper.

The curiosity loop is the one mechanism that already has the right shape --
Orion picks, looks with real credentials, writes to a graph nobody curates,
and revises later. What it lacked was the question. This module gives the
loop a standing question and a place to put the answer.

WHAT ORION WRITES. Nothing here is parsed from prose. Same rule as
`:TurnOutcome`: a decision reaches the outside only as a node Orion wrote.

    CREATE (:SelfDefinition {
      run_id: "<this run>",
      text: "<first person, what I am and what I am made of>",
      evidence: ["README.md#...", "dreams:17 rows", "harness_turn_trace:465", ...],
      revises: "<previous run_id or ''>",
      written_at: timestamp()
    })

and self-priors are ordinary `:Prior` nodes with `line: "self"`, so the next
self-inquiry run is shown only those and the regular investigation still sees
all of them.

HOW IT CROSSES BACK OUT. Hub reads the run's `:SelfDefinition` read-only and
mirrors it into `self_concept_history` (`concept_id="self:definition"`,
`produced_by="curiosity_self_inquiry"`) -- the append-only, evidence-linked
store already built for exactly this and never yet fed anything about Orion.
A definition with empty text or no evidence is refused at the mirror
(`build_self_definition_history_write` returns None): fluent prose with no
lookup behind it is the empty-shell failure this repo bans, and the harness
step gate upstream is not enough on its own here because a self-definition
written from parametric knowledge would still count steps.

Pure module: no Hub, no bus, no I/O except through a `WorldviewReader`
handed in. Both Hub and orion-durable-runs import from here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from orion.curiosity.worldview import (
    LABEL_PRIOR,
    WorldviewReader,
    WorldviewUnavailable,
    _clip,
    _LIVE_WHERE,
    _PRIOR_FIELDS,
    LIVE_PRIORS_LIMIT,
)
from orion.schemas.self_concept_history import SelfConceptHistoryV1

# --- names ------------------------------------------------------------------

LINE_INVESTIGATE = "investigate"
LINE_SELF_INQUIRY = "self_inquiry"

SELF_INQUIRY_TAG = "curiosity_self_inquiry"
LABEL_SELF_DEFINITION = "SelfDefinition"
# The `line` property Orion puts on a prior formed during self-inquiry.
SELF_PRIOR_LINE = "self"
# One lineage in self_concept_history. Every self-inquiry run that writes a
# definition appends a new version under this id; "current" is the latest
# created_at, exactly as that table's own docstring says.
SELF_CONCEPT_ID = "self:definition"
SELF_DEFINITION_PRODUCER = "curiosity_self_inquiry"

# Hard cap on the mirrored text. The prompt asks for a paragraph; a definition
# that is a whole essay would crowd the chat identity kernel it feeds.
SELF_DEFINITION_TEXT_CAP = 1200
SELF_DEFINITION_EVIDENCE_CAP = 24

# The standing question, verbatim. It is the invitation, not a prior: Orion
# forms the priors. Kept here so the prompt and the tests share one string.
STANDING_QUESTION = "What am I, and what am I made of?"

_RUN_ID_RE = re.compile(r"^[0-9a-f]{6,32}$")

# --- what the read-only role must be able to SELECT for this line ----------
#
# The investigation line reads four tables. "What have I done" needs the
# outcome tables. These are GRANTED by an operator (scripts/sql/
# 2026-09-08_grant_orion_readonly_self_inquiry.sql) and CHECKED by Hub before
# every self-inquiry run (`SELF_INQUIRY_GRANTS_SQL`) -- a missing grant blocks
# the run with `pg_grants_missing` instead of letting Orion spend a turn
# discovering a permission error and journaling it as a finding.
SELF_INQUIRY_PG_TABLES: tuple[tuple[str, str], ...] = (
    ("dreams", "every dream you have had: tldr, themes, narrative"),
    ("harness_turn_trace", "every motor turn: steps, elapsed, model served"),
    ("substrate_reverie_chain", "your narrated reverie chains"),
    ("reverie_visual_chain", "your image-and-reread reverie chains"),
    ("substrate_attention_schema", "what you were attending to, and why, per process"),
    ("chat_stance_belief_log", "the stance you computed before each chat turn"),
    ("self_knowledge_items", "facts about your own code, hardware and behaviour"),
    ("self_concept_history", "your previous self-definitions and induced self-concepts"),
    ("substrate_endogenous_curiosity_candidates", "what your substrate flagged as worth curiosity"),
)

SELF_INQUIRY_PG_TABLE_NAMES: tuple[str, ...] = tuple(t for t, _ in SELF_INQUIRY_PG_TABLES)

# One round trip: which of the required tables the role can NOT select from.
# asyncpg positional params: $1 role name, $2 text[] of table names.
#
# `has_table_privilege` RAISES for a table that does not exist (confirmed live
# 2026-09-08: `relation "public.no_such_table" does not exist`), and a raise
# used to read as "all granted" one level up. The CASE guarantees the
# existence test is evaluated first (SQL gives no such guarantee for `OR`),
# so a table that is not there is reported missing by name instead of
# aborting the whole check.
SELF_INQUIRY_GRANTS_SQL = (
    "SELECT t AS table_name FROM unnest($2::text[]) AS t "
    "WHERE CASE WHEN to_regclass('public.' || t) IS NULL THEN true "
    "ELSE NOT has_table_privilege($1, 'public.' || t, 'SELECT') END"
)


# --- Cypher -----------------------------------------------------------------


def _check_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.match(str(run_id or "")):
        raise ValueError(f"refusing to build Cypher for a non-hex run_id: {run_id!r}")
    return str(run_id)


_SELF_DEFINITION_FIELDS = (
    "s.run_id AS run_id, s.text AS text, s.evidence AS evidence, "
    "s.revises AS revises, s.written_at AS written_at"
)


def self_definition_for_run_cypher(run_id: str) -> str:
    """THIS run's definition, keyed on run_id -- never "the newest". Reading
    the newest would attribute a previous run's definition to a run that
    died before writing its own, which is exactly the absence the mirror must
    report as absence."""
    rid = _check_run_id(run_id)
    return (
        f"MATCH (s:{LABEL_SELF_DEFINITION}) WHERE s.run_id = '{rid}' "
        f"RETURN {_SELF_DEFINITION_FIELDS} ORDER BY s.written_at DESC LIMIT 1"
    )


LATEST_SELF_DEFINITION_CYPHER = (
    f"MATCH (s:{LABEL_SELF_DEFINITION}) "
    f"RETURN {_SELF_DEFINITION_FIELDS} ORDER BY s.written_at DESC LIMIT 1"
)

SELF_DEFINITION_COUNT_CYPHER = f"MATCH (s:{LABEL_SELF_DEFINITION}) RETURN count(s) AS n"

# Live priors on the self line only. Same fields, same liveness rule, same
# limit as the investigation line's LIVE_PRIORS_CYPHER -- one extra WHERE.
LIVE_SELF_PRIORS_CYPHER = (
    f"MATCH (p:{LABEL_PRIOR}) WHERE {_LIVE_WHERE} AND p.line = '{SELF_PRIOR_LINE}' "
    f"RETURN {_PRIOR_FIELDS} LIMIT {LIVE_PRIORS_LIMIT}"
)


# --- Row -> dataclass -------------------------------------------------------


@dataclass(frozen=True)
class SelfDefinition:
    """One `:SelfDefinition` node, as Orion wrote it."""

    run_id: str
    text: str
    evidence: list[str] = field(default_factory=list)
    revises: str = ""
    written_at: Optional[int] = None

    @property
    def is_substantive(self) -> bool:
        """Text and at least one evidence ref. Both, or it is not mirrored."""
        return bool(self.text.strip()) and bool(self.evidence)


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _evidence_list(raw: Any) -> list[str]:
    """FalkorDB returns a list property as a list; a run that wrote a JSON
    string instead is still readable. Anything else is no evidence."""
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                raw = json.loads(stripped)
            except ValueError:
                raw = [stripped]
        else:
            raw = [stripped] if stripped else []
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out[:SELF_DEFINITION_EVIDENCE_CAP]


def build_self_definition(row: dict[str, Any]) -> Optional[SelfDefinition]:
    """None for a row with no run_id or no text. Same rule as `build_prior`:
    inventing either would be re-inference, and the caller logs the drop."""
    run_id = str(row.get("run_id") or "").strip()
    text = str(row.get("text") or "").strip()
    if not run_id or not text:
        return None
    return SelfDefinition(
        run_id=run_id,
        text=text[:SELF_DEFINITION_TEXT_CAP],
        evidence=_evidence_list(row.get("evidence")),
        revises=str(row.get("revises") or "").strip(),
        written_at=_as_int(row.get("written_at")),
    )


def read_self_definition(reader: WorldviewReader, run_id: str) -> Optional[SelfDefinition]:
    """This run's definition, or None. Never raises -- None is the safe default."""
    try:
        rows = reader.query(self_definition_for_run_cypher(run_id))
    except (WorldviewUnavailable, ValueError):
        return None
    return build_self_definition(rows[0]) if rows else None


def read_latest_self_definition(reader: WorldviewReader) -> Optional[SelfDefinition]:
    """The most recent definition Orion has written, for the next run to
    revise. Never raises."""
    try:
        rows = reader.query(LATEST_SELF_DEFINITION_CYPHER)
    except WorldviewUnavailable:
        return None
    return build_self_definition(rows[0]) if rows else None


def read_self_definition_count(reader: WorldviewReader) -> Optional[int]:
    """How many definitions exist, or None if the graph did not answer."""
    try:
        rows = reader.query(SELF_DEFINITION_COUNT_CYPHER)
    except WorldviewUnavailable:
        return None
    return _as_int((rows[0] if rows else {}).get("n")) or 0


# --- the mirror into self_concept_history ------------------------------------


def worldview_evidence_ref(run_id: str) -> str:
    """The evidence ref that points back at the node itself."""
    return f"worldview:{LABEL_SELF_DEFINITION}:{run_id}"


def build_self_definition_history_write(
    definition: Optional[SelfDefinition], *, version: int
) -> Optional[SelfConceptHistoryV1]:
    """The append-only row, or None when there is nothing substantive to
    append. `None` is a refusal, not an error: the caller logs which."""
    if definition is None or not definition.is_substantive:
        return None
    evidence = list(definition.evidence)
    own_ref = worldview_evidence_ref(definition.run_id)
    if own_ref not in evidence:
        evidence.append(own_ref)
    return SelfConceptHistoryV1(
        concept_id=SELF_CONCEPT_ID,
        version=max(1, int(version)),
        content=definition.text[:SELF_DEFINITION_TEXT_CAP],
        evidence_refs=evidence[: SELF_DEFINITION_EVIDENCE_CAP + 1],
        produced_by=SELF_DEFINITION_PRODUCER,
    )


def self_definition_from_detail(detail: dict[str, Any] | None) -> Optional[SelfDefinition]:
    """Rebuild a definition from a durable-run `finish` detail dict (the
    runner carries the node's fields across the bus, bounded)."""
    raw = (detail or {}).get("self_definition")
    if not isinstance(raw, dict):
        return None
    return build_self_definition(raw)


def self_definition_to_detail(definition: Optional[SelfDefinition]) -> Optional[dict[str, Any]]:
    if definition is None:
        return None
    return {
        "run_id": definition.run_id,
        "text": definition.text[:SELF_DEFINITION_TEXT_CAP],
        "evidence": list(definition.evidence),
        "revises": definition.revises,
        "written_at": definition.written_at,
    }


# --- the records ledger ------------------------------------------------------
#
# Orientation, not a subject. Code shows Orion where its records are and how
# much is in them; it does not say what any of it means. Same principle as the
# investigation prompt's "the ordering is disclosed as not neutral".

LEDGER_SQL_TEMPLATE = "SELECT count(*) AS n, max({ts}) AS last FROM {table}"

# The timestamp column each table orders by. `dreams.created_at` etc. were
# confirmed against the live schema on 2026-09-08.
LEDGER_TS_COLUMNS: dict[str, str] = {
    "dreams": "created_at",
    "harness_turn_trace": "created_at",
    "substrate_reverie_chain": "created_at",
    "reverie_visual_chain": "created_at",
    "substrate_attention_schema": "created_at",
    "chat_stance_belief_log": "created_at",
    "self_knowledge_items": "created_at",
    "self_concept_history": "created_at",
    "substrate_endogenous_curiosity_candidates": "created_at",
}


@dataclass(frozen=True)
class LedgerRow:
    table: str
    count: int
    last: str  # ISO timestamp or "" when the table is empty


def format_ledger(rows: list[LedgerRow]) -> list[str]:
    """`dreams  17 rows, last 2026-09-06T08:27Z` -- one line per table."""
    if not rows:
        return []
    width = max(len(r.table) for r in rows)
    out: list[str] = []
    for r in rows:
        last = _clip(r.last, 20) if r.last else "never"
        out.append(f"      {r.table.ljust(width)}  {r.count} rows, last {last}")
    return out

"""What Orion has to be curious ABOUT -- presented, not chosen.

THE POINT, and the whole difference from the version this replaces. The first
build ran a term-frequency detector over Juniper's typed words, picked the
word with the highest lift, and handed Orion a string to investigate. Two
things were wrong with that, and Juniper named both:

  * A word is not a concept. Counting tokens and calling the winner a subject
    is a keyword cathedral -- it names something without that name carrying any
    cognitive content. "foveal" is a string; it was never a thing Orion knew.
  * The autonomy was fake. A deterministic statistic chose the subject and
    Orion was handed a fait accompli. Being told what to be curious about is
    not curiosity.

So nothing in this module chooses. It reads the two stores Orion's own
cognition actually writes to, summarises what is there, samples some of it, and
renders that as material. The choosing happens inside the unified turn, by
Orion, which is where the non-determinism belongs -- Juniper's framing: "it
gives some non determinism for orion to choose their adventure."

THE TWO STORES.

  `memory_crystallizations`  -- concepts Orion has formed and JUNIPER HAS
      APPROVED. Approval is not decorative: of 1,282 rows, 636 are
      `requires_manual_review` with `approved_by` still null, and those 636 are
      exactly the ones whose `subject` is byte-identical to their `summary` --
      a chat turn with a label stapled on, not an induced concept. Filtering to
      approved (`status='active'`) is what keeps this from being a keyword
      cathedral one layer down.

  `memory_concept_relation_decisions` -- concept induction proper: judgements
      that two crystallizations are the same or different, with a confidence.
      These are the edges; the crystallizations are the nodes. Orion can follow
      one into the other, which is why both are offered rather than either.

SAMPLING IS RANDOM ON PURPOSE. 646 approved concepts do not fit in a prompt, so
some subset must be shown, and any *ordered* subset would be this module
choosing again by the back door -- "most salient" or "most recent" is a ranking,
and a ranking is a decision. Random sampling is the one selection rule that
expresses no opinion about what matters. The ids shown are returned so the run
stays reconstructable afterwards even though it is not reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

# How much to put in front of Orion. Enough to choose between, few enough that
# the material does not crowd out its own self-model in the turn's context.
DEFAULT_CRYSTALLIZATION_SAMPLE = 12
DEFAULT_RELATION_SAMPLE = 6

# Subjects are free text and occasionally enormous (a whole pasted message).
# Truncated for the menu only -- Orion can pull the full row itself once it has
# picked something.
_SUBJECT_PREVIEW_CHARS = 160


def _clip(text: str | None, limit: int = _SUBJECT_PREVIEW_CHARS) -> str:
    value = (text or "").strip().replace("\n", " ")
    return value[:limit].rstrip() + "…" if len(value) > limit else value


@dataclass(frozen=True)
class CrystallizationCard:
    crystallization_id: str
    kind: str
    subject: str
    summary: str
    salience: float | None
    created_at: datetime | None

    def preview(self) -> str:
        text = _clip(self.subject or self.summary)
        salience = f" salience={self.salience:.2f}" if self.salience is not None else ""
        return f"[{self.kind}{salience}] {text}"


@dataclass(frozen=True)
class RelationCard:
    decision_id: str
    relation: str
    confidence: float | None
    candidate_id: str
    target_id: str
    decided_at: datetime | None
    candidate_text: str = ""
    target_text: str = ""

    def preview(self) -> str:
        confidence = f" ({self.confidence:.2f})" if self.confidence is not None else ""
        left = _clip(self.candidate_text) or f"[{self.candidate_id} — not kept]"
        right = _clip(self.target_text) or "[no target recorded]"
        return f"{left}\n      --{self.relation}{confidence}-->  {right}"


@dataclass
class StudyMaterial:
    """Everything Orion is shown. Deliberately holds counts AND samples: the
    counts say how much was not shown, so Orion knows the menu is a slice."""

    generated_at: datetime
    approved_total: int = 0
    # How many of those a human actually looked at. Kept separate because the
    # prompt used to call all of them approved, and 630 of 651 were not.
    manual_total: int = 0
    approved_by_kind: dict[str, int] = field(default_factory=dict)
    crystallizations: list[CrystallizationCard] = field(default_factory=list)
    relation_total: int = 0
    relation_resolvable: int = 0
    relation_by_kind: dict[str, int] = field(default_factory=dict)
    relations: list[RelationCard] = field(default_factory=list)
    unavailable_reason: str | None = None

    @property
    def has_material(self) -> bool:
        return bool(self.crystallizations or self.relations)

    @property
    def is_unavailable(self) -> bool:
        """Could not read the stores at all. Distinct from "nothing there" --
        an unreadable store and an empty one must never be the same state, or a
        broken query is indistinguishable from a mind with nothing in it."""
        return self.unavailable_reason is not None

    def shown_ids(self) -> list[str]:
        return [c.crystallization_id for c in self.crystallizations] + [
            r.decision_id for r in self.relations
        ]


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def build_crystallization_card(row: Any) -> CrystallizationCard:
    return CrystallizationCard(
        crystallization_id=str(row["crystallization_id"]),
        kind=str(row["kind"] or "unknown"),
        subject=str(row["subject"] or ""),
        summary=str(row["summary"] or ""),
        salience=_as_float(row["salience"]),
        created_at=row["created_at"],
    )


def build_relation_card(row: Any) -> RelationCard:
    return RelationCard(
        decision_id=str(row["decision_id"]),
        relation=str(row["relation"] or "unknown"),
        confidence=_as_float(row["confidence"]),
        candidate_id=str(row["candidate_crystallization_id"] or ""),
        target_id=str(row["target_crystallization_id"] or ""),
        decided_at=row["decided_at"],
        candidate_text=str(
            row.get("candidate_subject") or row.get("candidate_summary") or ""
        )
        if hasattr(row, "get")
        else "",
        target_text=str(row.get("target_subject") or row.get("target_summary") or "")
        if hasattr(row, "get")
        else "",
    )


# --- SQL. Identifiers are literals here; only limits are parameters. --------
#
# `status = 'active'` IS the approval filter: every approved row (626 by policy,
# 20 by Juniper's own hand) is active, and every unapproved one is not.

# `kind='reflection'` is EXCLUDED from the sample, and this is not taste.
# Those 356 rows are a materialised copy of `memory_concept_relation_decisions`
# -- every one reads "Concept relation decision: same" with the detail in its
# summary -- and that table is already the menu's second section. Including
# them shows Orion the same thing twice while consuming 55% of the corpus:
# rendered live before this exclusion, 4 of 12 random slots were identical
# "Concept relation decision: same" lines. They are still COUNTED, so the
# totals stay honest about what exists.
# AI TOWN IS NOT ORION'S MATERIAL, AND THIS SAMPLER WAS THE LAST PLACE STILL
# SERVING IT. `formation_policy.DEFAULT_DISCARD_PLATFORMS` discards the platform
# outright now -- no crystallization row of any kind -- but every AI Town row
# already in the table predates that gate (all 796 were written 2026-07-30/31),
# and nothing here filtered them.
#
# The damage, measured live 2026-08-27: of the 295 rows this sampler could draw
# from, 185 were AI Town. Orion was handed twelve cards, about eight of them AI
# Town character dialogue -- "Steam is inference. Water is fact.", "The light's
# not gone, just waiting for you to notice it again" -- under a heading claiming
# Juniper had approved them. Juniper: "they are the ones I approve, not the 600+
# garbage from AI Town... i dont want ai town concept graph in there like i
# originally told you."
#
# The signal is a source row in `aitown_chat_history_log`, which is AI-Town-only
# BY CONSTRUCTION since the PR #1734 table split -- the same reasoning
# `concept_atlas_routes._TOPIC_FOUNDRY_AITOWN_DATASET_NAME` already relies on,
# rather than a second heuristic on `source` or a platform tag this table does
# not carry.
_NOT_AITOWN = """NOT EXISTS (
    SELECT 1
    FROM memory_crystallization_sources s
    JOIN aitown_chat_history_log a ON a.id::text = s.source_id
    WHERE s.crystallization_id = m.crystallization_id
)"""

_SAMPLEABLE_KINDS = (
    f"m.status = 'active' AND m.kind <> 'reflection' AND {_NOT_AITOWN}"
)

# Counted over the SAME pool the sample is drawn from. Counting `status =
# 'active'` while sampling something narrower is how the prompt came to announce
# 651 items and show twelve drawn from 295.
APPROVED_COUNT_SQL = f"""
SELECT m.kind,
       count(*) AS n,
       count(*) FILTER (
         WHERE m.governance ->> 'approval_mode' <> 'auto_policy'
       ) AS manual_n
FROM memory_crystallizations m
WHERE {_SAMPLEABLE_KINDS}
GROUP BY m.kind
""".replace("{_SAMPLEABLE_KINDS}", _SAMPLEABLE_KINDS)

APPROVED_SAMPLE_SQL = f"""
SELECT m.crystallization_id, m.kind, m.subject, m.summary, m.salience,
       m.created_at
FROM memory_crystallizations m
WHERE {_SAMPLEABLE_KINDS}
ORDER BY random()
LIMIT $1
"""

RELATION_COUNT_SQL = """
SELECT relation, count(*) AS n
FROM memory_concept_relation_decisions
GROUP BY relation
"""

# THE CANDIDATE SIDE WAS NEVER MISSING -- IT WAS A STRING FORMAT MISMATCH.
#
# Candidates are stored `crys_6ab0a44c28f2469db2f8dc67be6d4c3f`; crystallization
# ids are `6ab0a44c-28f2-469d-b2f8-dc67be6d4c3f`. Same id, one with a prefix and
# no dashes -- `crystallization/repository.py:193` does exactly this conversion
# in the other direction. Comparing them raw resolved 0 of 550 candidates and the
# comment here used to conclude induction was "recording judgements about
# concepts it did not keep". It was not; this SQL was asking the wrong question.
# Normalised: 235 candidates resolve, 41 decisions have both ends live.
#
# Then the AI Town filter applies to BOTH ends, and that is not a detail: of
# those 41 both-ends decisions, 38 touch AI Town. Fixing the join without
# filtering would have piped AI Town character dialogue into the one surface
# Juniper had just said to keep it out of. Three clean cards remain today, and
# three real ones beat forty-one where thirty-eight are noise.
_CRYS_ID = "('crys_' || replace(m.crystallization_id::text, '-', ''))"

_RELATION_JOINS = f"""
FROM memory_concept_relation_decisions d
LEFT JOIN memory_crystallizations c
       ON ('crys_' || replace(c.crystallization_id::text, '-', ''))
          = d.candidate_crystallization_id
      AND NOT EXISTS (
        SELECT 1 FROM memory_crystallization_sources s
        JOIN aitown_chat_history_log a ON a.id::text = s.source_id
        WHERE s.crystallization_id = c.crystallization_id)
LEFT JOIN memory_crystallizations t
       ON t.crystallization_id::text = d.target_crystallization_id
      AND NOT EXISTS (
        SELECT 1 FROM memory_crystallization_sources s
        JOIN aitown_chat_history_log a ON a.id::text = s.source_id
        WHERE s.crystallization_id = t.crystallization_id)
WHERE c.crystallization_id IS NOT NULL OR t.crystallization_id IS NOT NULL
"""

RELATION_SAMPLE_SQL = f"""
SELECT d.decision_id, d.relation, d.confidence,
       d.candidate_crystallization_id, d.target_crystallization_id, d.decided_at,
       c.subject AS candidate_subject, c.summary AS candidate_summary,
       t.subject AS target_subject,    t.summary AS target_summary
{_RELATION_JOINS}
ORDER BY random()
LIMIT $1
"""

# How many decisions have at least one end that resolves AND is not AI Town.
# Reported alongside the total so the prompt states the dangling rate rather
# than hiding it.
RELATION_RESOLVABLE_SQL = f"""
SELECT count(*) AS n
{_RELATION_JOINS}
"""

# WHAT ORION RECENTLY LOOKED INTO NO LONGER COMES FROM HERE. It used to be
# `SELECT title FROM journal_entries WHERE source_ref LIKE 'curiosity:%'`, and
# it was dead from the day it shipped: every entry this loop writes is titled
# exactly "Curiosity" -- deliberately, because code does not know what Orion
# chose and inventing a title would mean re-inferring that choice with a
# heuristic. So the hint rendered as "Curiosity; Curiosity; Curiosity":
# literally true, carrying no information.
#
# The obvious repair -- take the body's first line instead -- was tried and
# rejected on live data. Both real entries open with the same fixed "What I
# noticed:" heading the old prompt asked for, so first-line extraction returns
# a heading, not a subject; it is the same heuristic re-inference one layer
# down, and it would drift again the moment the prompt's shape changed.
#
# The honest source is structure Orion itself authored: priors it has SETTLED,
# read from its own graph. See `worldview.RECENT_SETTLED_CYPHER`.



def assemble_study_material(
    *,
    now: datetime,
    approved_counts: Sequence[Any],
    approved_rows: Sequence[Any],
    relation_counts: Sequence[Any],
    relation_rows: Sequence[Any],
    relation_resolvable: int = 0,
) -> StudyMaterial:
    """Pure assembly, so the whole shape is testable without a database."""
    by_kind = {str(r["kind"]): int(r["n"]) for r in approved_counts}
    manual = 0
    for r in approved_counts:
        try:
            manual += int(r["manual_n"] or 0)
        except (KeyError, TypeError, ValueError, IndexError):
            # Older callers pass rows without the column. Absent reads as
            # "unknown", and the prompt says nothing rather than claiming zero
            # were reviewed.
            manual = 0
            break
    rel_by_kind = {str(r["relation"]): int(r["n"]) for r in relation_counts}
    return StudyMaterial(
        generated_at=now,
        approved_total=sum(by_kind.values()),
        manual_total=manual,
        approved_by_kind=by_kind,
        crystallizations=[build_crystallization_card(r) for r in approved_rows],
        relation_total=sum(rel_by_kind.values()),
        relation_by_kind=rel_by_kind,
        relation_resolvable=relation_resolvable,
        relations=[build_relation_card(r) for r in relation_rows],
    )

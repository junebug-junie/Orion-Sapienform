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
DEFAULT_CRYSTALLIZATION_SAMPLE = 6
DEFAULT_RELATION_SAMPLE = 12
DEFAULT_RECENT_STUDY_SAMPLE = 8

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
    approved_by_kind: dict[str, int] = field(default_factory=dict)
    crystallizations: list[CrystallizationCard] = field(default_factory=list)
    relation_total: int = 0
    relation_resolvable: int = 0
    relation_by_kind: dict[str, int] = field(default_factory=dict)
    relations: list[RelationCard] = field(default_factory=list)
    recently_studied: list[str] = field(default_factory=list)
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
_SAMPLEABLE_KINDS = "status = 'active' AND kind <> 'reflection'"

APPROVED_COUNT_SQL = """
SELECT kind, count(*) AS n
FROM memory_crystallizations
WHERE status = 'active'
GROUP BY kind
"""

APPROVED_SAMPLE_SQL = f"""
SELECT crystallization_id, kind, subject, summary, salience, created_at
FROM memory_crystallizations
WHERE {_SAMPLEABLE_KINDS}
ORDER BY random()
LIMIT $1
"""

RELATION_COUNT_SQL = """
SELECT relation, count(*) AS n
FROM memory_concept_relation_decisions
GROUP BY relation
"""

# Joined so a relation card carries the CONCEPTS it relates rather than two
# truncated uuids -- before this, the induction half of the menu read
# `crys_bf9 --same--> 48c35b3a`, which is true and useless to choose from.
#
# CONCEPT INDUCTION IS PARTLY DANGLING, and this filter is the honest response
# rather than a workaround. Measured live 2026-08-26 over all 547 decisions:
#
#   547  decisions
#     0  have a resolvable CANDIDATE -- every candidate id is `crys_<hex>`,
#        an id space that exists in NO crystallization table (checked against
#        crystallizations, claims, links, sources, history). Induction is
#        recording judgements about concepts it did not keep.
#   356  have a resolvable target (exactly the 356 `reflection` rows)
#   164  have no target at all
#
# So the candidate side is shown by id, labelled as unresolved, and decisions
# with neither side resolvable are excluded -- a card with two dead ends is not
# something anyone can choose from. The totals still report every decision, so
# the prompt cannot imply induction is healthier than it is.
RELATION_SAMPLE_SQL = """
SELECT d.decision_id, d.relation, d.confidence,
       d.candidate_crystallization_id, d.target_crystallization_id, d.decided_at,
       c.subject AS candidate_subject, c.summary AS candidate_summary,
       t.subject AS target_subject,    t.summary AS target_summary
FROM memory_concept_relation_decisions d
LEFT JOIN memory_crystallizations c
       ON c.crystallization_id::text = d.candidate_crystallization_id
LEFT JOIN memory_crystallizations t
       ON t.crystallization_id::text = d.target_crystallization_id
WHERE c.crystallization_id IS NOT NULL OR t.crystallization_id IS NOT NULL
ORDER BY random()
LIMIT $1
"""

# How many decisions have at least one end that resolves. Reported alongside
# the total so the prompt states the dangling rate instead of hiding it.
RELATION_RESOLVABLE_SQL = """
SELECT count(*) AS n
FROM memory_concept_relation_decisions d
LEFT JOIN memory_crystallizations c
       ON c.crystallization_id::text = d.candidate_crystallization_id
LEFT JOIN memory_crystallizations t
       ON t.crystallization_id::text = d.target_crystallization_id
WHERE c.crystallization_id IS NOT NULL OR t.crystallization_id IS NOT NULL
"""

# What Orion has already looked into lately, so it can avoid repeating itself
# BY CHOICE rather than by a gate deciding for it.
RECENT_STUDY_SQL = """
SELECT title
FROM journal_entries
WHERE source_kind = 'self_study' AND source_ref LIKE 'curiosity:%'
ORDER BY created_at DESC
LIMIT $1
"""


def assemble_study_material(
    *,
    now: datetime,
    approved_counts: Sequence[Any],
    approved_rows: Sequence[Any],
    relation_counts: Sequence[Any],
    relation_rows: Sequence[Any],
    relation_resolvable: int = 0,
    recent_titles: Sequence[Any],
) -> StudyMaterial:
    """Pure assembly, so the whole shape is testable without a database."""
    by_kind = {str(r["kind"]): int(r["n"]) for r in approved_counts}
    rel_by_kind = {str(r["relation"]): int(r["n"]) for r in relation_counts}
    return StudyMaterial(
        generated_at=now,
        approved_total=sum(by_kind.values()),
        approved_by_kind=by_kind,
        crystallizations=[build_crystallization_card(r) for r in approved_rows],
        relation_total=sum(rel_by_kind.values()),
        relation_by_kind=rel_by_kind,
        relation_resolvable=relation_resolvable,
        relations=[build_relation_card(r) for r in relation_rows],
        recently_studied=[str(r["title"]) for r in recent_titles],
    )

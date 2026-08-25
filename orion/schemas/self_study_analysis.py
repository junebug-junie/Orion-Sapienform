"""Typed result contract for `skills.self_study.analyze.v1`.

ONE action shape, FOUR inputs. The verb runs the same window-contrast analysis
over a different already-stored telemetry source each time and, when (and only
when) a disclosed notability rule fires, writes ONE self-study journal entry.

Why a single shape rather than four verbs: the four sources differ only in
which table is read and which columns are summarised. The rules that decide
"is there anything here worth Orion writing down" are shared
(`services/orion-cortex-exec/app/self_study_analysis.py::evaluate_rules`), so
four adapters would be four copies of one analysis with four names -- the exact
"names the world but changes nothing" shape CLAUDE.md section 0A bans.

The live dispatch route pins NO source. The verb picks whichever input has gone
longest without being ANALYSED (`select_least_recently_analysed`, keyed on a
per-source run mark rather than on journal writes -- see that function for the
measured reason). A route's `skill_args.source` exists so an operator, or a
test, can deliberately narrow the action to one lens.

NOT a new metric. Nothing here is wired into field pressure, proposal scoring,
or any cognition model -- every number below is a read-only summary of rows
that already exist, rendered into a journal entry. `status` is the load-
bearing field: `skipped_not_notable` is a real, common, correct outcome and
writes nothing, which is what keeps this from becoming digest spam on top of
the 32,991 metacog digests already in `journal_entries`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.journaler.schemas import JournalEntryWriteV1
from orion.schemas.self_study import SelfWritebackStatusV1

AnalysisSource = Literal[
    "concept_induction",
    "vision_events",
    "affective_state",
    "cocreation_signals",
]

ANALYSIS_SOURCES: tuple[str, ...] = (
    "concept_induction",
    "vision_events",
    "affective_state",
    "cocreation_signals",
)

AnalysisStatus = Literal[
    # A notability rule fired and a journal entry was published.
    "journaled",
    # The analysis ran on real rows and nothing crossed a bar. Correct, quiet,
    # and deliberately NOT a failure -- see the module docstring.
    "skipped_not_notable",
    # The source could not be read at all (no DSN, query error, unknown
    # source). Distinct from `skipped_not_notable`: "unknown" is not "quiet".
    "unavailable",
    # A rule fired, but an entry with an identical finding-set for this same
    # source was already written inside the cooldown. Repeating it would be
    # the digest spam this action exists to avoid.
    "skipped_recently_journaled",
    # A rule fired but the journal publish itself failed.
    "journal_failed",
]


class AnalysisMetricV1(BaseModel):
    """One summarised quantity, recent window against the window before it.

    `baseline` is None when the baseline window held no rows to summarise --
    never 0.0, because "no observations" and "observed zero" are different
    claims and the notability rules below depend on telling them apart.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    recent: float | None = None
    baseline: float | None = None
    unit: str = "count"


class AnalysisFindingV1(BaseModel):
    """One notability rule that fired, with the numbers that fired it."""

    model_config = ConfigDict(extra="forbid")

    rule: str
    detail: str
    metric: str | None = None
    recent: float | None = None
    baseline: float | None = None


class SelfStudyAnalysisResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    source: AnalysisSource
    status: AnalysisStatus
    window_hours: float
    # The column the windows were cut on. Recorded because occurrence-time and
    # write-time columns are on different clocks in this schema (see
    # `SOURCE_SPECS`), and a reader cannot re-derive which one was used.
    time_column: str
    recent_since: datetime
    recent_until: datetime
    baseline_since: datetime
    recent_rows: int = 0
    baseline_rows: int = 0
    metrics: list[AnalysisMetricV1] = Field(default_factory=list)
    findings: list[AnalysisFindingV1] = Field(default_factory=list)
    unavailable_reason: str | None = None
    # Stable digest of the fired rule-set. Doubles as the journal entry's
    # `source_ref` suffix, which is what makes the cooldown dedup a plain
    # indexed-column lookup rather than a new table.
    finding_digest: str | None = None
    # Rules that were evaluated and did NOT fire. Rendered into the journal
    # body: the negative space is what makes this an analysis rather than a
    # highlight reel, and it is what lets a reader tell "nothing happened"
    # from "the rule was never checked".
    rules_not_fired: list[str] = Field(default_factory=list)
    journal_write: SelfWritebackStatusV1 | None = None
    journal_entry: JournalEntryWriteV1 | None = None
